from uicsmodels.bayesianmodels import BayesianModel, GibbsState, ArrayTree
from uicsmodels.sampling.inference import update_correlated_gaussian, update_metropolis
from uicsmodels.gaussianprocesses.meanfunctions import Zero
from uicsmodels.gaussianprocesses.likelihoods import AbstractLikelihood, Gaussian
from uicsmodels.gaussianprocesses.fullgp import FullGPModel
from uicsmodels.gaussianprocesses.gputil import sample_predictive

from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping

ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

import jax
import distrax as dx
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd
from blackjax import elliptical_slice, rmh

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

jitter = 1e-6

from icecream import ic


class SparseGPModel(FullGPModel):  
    """The latent Gaussian process model.
    
    The latent Gaussian process model consists of observations (y), generated by
    an observation model that takes the latent Gaussian process (f) and optional
    hyperparameters (phi) as input. The latent GP itself is parametrized by a
    mean function (mu) and a covariance function (cov). These can have optional
    hyperparameters (psi) and (theta).

    The generative model is given by:

    .. math::
        psi     &\sim p(\psi)\\
        theta   &\sim p(\theta) \\
        phi     &\sim p(\phi) \\
        f       &\sim GP(mu, cov) \\
        y       &\sim p(y \mid T(f), phi)

    """
    # NOTE: Update description
    # TODO: Change to factorize out f. 

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None,
                 likelihood: AbstractLikelihood = None,
                 num_inducing_points: int = None,
                 f_true = None):  # TODO: Remove after debugging
        if likelihood is None:
            likelihood = Gaussian()
        self.likelihood = likelihood
        self.m = num_inducing_points  # TODO: Instead of passing, infer from prior over inducing inputs
        self.f_true = f_true
        super().__init__(X, y, cov_fn, mean_fn, priors)        
        

    #
    def __get_component_parameters(self, position, component):
        """Extract parameter sampled values per model component for current
        position.

        """
        return {param: position[param] for param in
                self.param_priors[component]} if component in self.param_priors else {}

    #
    def _compute_sparse_gp(
            self, cov_params, x, samples_Z, samples_u, add_jitter=True):
        """
        Returns mean and covariance matrix of sparse gp
        """

        cov_XX = self.cov_fn.cross_covariance(
            params=cov_params,
            x=x, y=x) 
        cov_XX += jitter * jnp.eye(*cov_XX.shape)
        
        cov_ZZ = self.cov_fn.cross_covariance(
            params=cov_params,
            x=samples_Z, y=samples_Z)
        cov_ZZ += jitter * jnp.eye(*cov_ZZ.shape)

        cov_XZ = self.cov_fn.cross_covariance(
            params=cov_params,
            x=x, y=samples_Z)

        mean_gp = jnp.dot(cov_XZ, jnp.linalg.solve(cov_ZZ, samples_u))

        ZZ_ZX = jnp.linalg.solve(cov_ZZ, jnp.transpose(cov_XZ))
        cov_gp = cov_XX - jnp.dot(cov_XZ, ZZ_ZX)

        if add_jitter:
            cov_gp += jitter * jnp.eye(*cov_gp.shape)

        return mean_gp, cov_gp

    #   
    def init_fn(self, key, num_particles=1):
        """Initialization of the Gibbs state.

        The initial state is determined by sampling all variables from their
        priors, and then constructing one sample for (f) using these. All
        variables together are stored in a dict() in the GibbsState object.

        When num_particles > 1, each dict element contains num_particles random
        initial values.

        Args:
            key:
                The jax.random.PRNGKey
            num_particles: int
                The number of parallel particles for sequential Monte Carlo
        Returns:
            GibbsState
        """
        
        key, key_super_init = jrnd.split(key)

        # sample from hyperparameter priors
        initial_state = super().init_fn(key_super_init, num_particles)

        def sample_latent(key, cov_params, Z_params):
            """
            Sample Z and u, as well as f from resulting sparse GP
            """
            _, *sub_key = jrnd.split(key, num=4)
            key_sample_z = sub_key[0]
            key_sample_u = sub_key[1]
            key_sample_f = sub_key[2]

            # sample M inducing inputs Z
            # samples_Z = dx.Normal(   
            #     loc=Z_params['mean'],
            #     scale=Z_params['scale']).sample(seed=key_sample_z)

            # true evenly-spaced Z in x domain allows drawing true u samples
            lin_Z = jnp.linspace(
                jnp.min(self.X), 
                jnp.max(self.X), 
                self.m)
            # find and select closest values in X-domain
            samples_Z_idx = jnp.searchsorted(
                self.X.flatten(), lin_Z)
            samples_Z = self.X.flatten()[samples_Z_idx]


            # Sample inducing variables u
            mean_u = jnp.zeros(samples_Z.shape[0])
            cov_ZZ = self.cov_fn.cross_covariance(
                params=cov_params,
                x=samples_Z, y=samples_Z) 
            cov_ZZ = cov_ZZ + jitter * jnp.eye(cov_ZZ.shape[0])
            samples_u = jnp.asarray(mean_u + jnp.dot(
                jnp.linalg.cholesky(cov_ZZ),
                jrnd.normal(key_sample_u, shape=[samples_Z.shape[0]])))

            # True u samples based on true Z's
            # samples_u = self.f_true[samples_Z_idx]
            

            # compute mean and cov. function of sparse GP
            mean_gp, cov_gp = self._compute_sparse_gp(
                    cov_params=cov_params, 
                    x=self.X,
                    samples_Z=samples_Z, 
                    samples_u=samples_u)

            # Sample from GP
            # NOTE: tell cholesky that the cov. is diagonal, if I set it too. 
            # NOTE: To account for SMC particles, add dimension in `jrnd.normal`
            L = jnp.linalg.cholesky(cov_gp)
            z = jrnd.normal(key_sample_f, shape=[self.n])
            samples_f = jnp.asarray(mean_gp + jnp.dot(L, z))
            
            return samples_Z.flatten(), samples_u, samples_f.flatten()

        #
        initial_position = initial_state.position

        cov_params = initial_position.get('kernel', {})
        Z_params = initial_position.get('inducing_inputs_Z', {})

        if num_particles > 1:
            # We vmap across the first dimension of the elements *in* the
            # dictionary, rather than over the dictionary itself.
            key_sample_particles = jrnd.split(key, num_particles)

            cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)
            Z_param_in_axes = jax.tree_map(lambda l: 0, Z_params)

            samples_Z, samples_u, samples_f = jax.vmap(
                sample_latent,
                in_axes=(0, cov_param_in_axes, Z_param_in_axes)
                )(key_sample_particles, cov_params, Z_params)
                
        else:
            # NOTE: Not tested / modified after pulling latest changes.
            _, key_sample_latents = jrnd.split(key)

            samples_Z, samples_u, samples_f = sample_latent(
                key_sample_latents, cov_params, Z_params)

        initial_position['Z'] = samples_Z
        initial_position['u'] = samples_u
        initial_position['f'] = samples_f
        
        return GibbsState(initial_position)

        #

    def gibbs_fn(
            self, key, state, 
            loglik_fn__=None, temperature=1.0, **mcmc_parameters): 
        # HACK: Gave loglik_fn None as default, can it actually be removed?
        """The Gibbs MCMC kernel.

        The Gibbs kernel step function takes a state and returns a new state. In
        the latent GP model, the latent GP (f) is first updated, then the
        parameters of the mean (psi) and covariance function (theta), and lastly
        the parameters of the observation model (phi).

        Args:
            key:
                The jax.random.PRNGKey
            state: GibbsState
                The current state in the MCMC sampler
        Returns:
            GibbsState

        """
    
        # get current hypter-parameters and samples from gibbs-state
        position = state.position.copy()

        # define likelihood function
        # NOTE: The function might use old parameter values if they are updated.
        #   That means the u update uses an outdated likelihood...
        #   - Either do the likelihood update at the end
        #   - or use a location function that returns an updated likelihood function. 
        likelihood_params = position.get('likelihood', {})
        loglikelihood_fn_ = lambda f_: temperature * jnp.sum(
                self.likelihood.log_prob(params=likelihood_params, f=f_, y=self.y))
        
        def sample_f(key, position): # -> jrnd.key, gibbs_state
            """Sample the latent GP using:

            p(f | theta, psi, y) \propto p(y | f, phi) p(f | psi, theta)

            """
            cov_params = position.get('kernel', {}) 

            # update f
            mean, cov = self._compute_sparse_gp(
                        cov_params=cov_params, 
                        x=self.X,
                        samples_Z=position['Z'], 
                        samples_u=position['u'],
                        add_jitter=True)

            # sample f directly
            # key, subkey = jrnd.split(key)
            # sub_state = dx.MultivariateNormalFullCovariance(
            #     mean, 
            #     cov).sample(seed=subkey, sample_shape=1).flatten()
            # jax.debug.print('new f {d}', d=sub_state.shape)
            
            # sample f with ellipical slice sampling
            key, subkey = jrnd.split(key)
            sub_state, f_info = update_correlated_gaussian(
                subkey, 
                position['f'], 
                loglikelihood_fn_, 
                mean, cov)
            #jax.debug.print('old f {d}', d=sub_state.shape)
            
            return key, sub_state  # TODO: Don't return key, e.g. splitting needs to happen outside of function.
        
        key, position['f'] = sample_f(key, position)


        # update cov parameters 
        # if len(cov_params):  # theta
        if True:
            """Sample parameters of the kernel function using: 

            p(theta | u, Z, f, X) \propto 
                p(f | u, Z, theta, X) p(u | Z, theta) p(theta)
            """

            cov_params = position.get('kernel', {})

            def logdensity_fn_(theta_):
                log_pdf = 0
                
                # p(theta) | cov. kernel parameter
                for param, val in theta_.items():
                    # jax.debug.print('val {d}', d=val)
                    log_pdf += jnp.sum(self.param_priors['kernel'][param].log_prob(val))
                
                # p(u | Z, theta)
                mean_u = self.mean_fn.mean(params=None, x=position['Z'])
                cov_u = self.cov_fn.cross_covariance(
                    params=theta_,
                    x=position['Z'],
                    y=position['Z'])
                log_pdf += dx.MultivariateNormalFullCovariance(mean_u, cov_u).log_prob(position['u'])

                # p(f | u, Z, theta, X)
                mean_gp, cov_gp = self._compute_sparse_gp(
                    cov_params=theta_, 
                    x=self.X,
                    samples_Z=position['Z'],
                    samples_u=position['u'])
                
                log_pdf += dx.MultivariateNormalFullCovariance(
                    mean_gp, cov_gp).log_prob(position['f'])
                
                return log_pdf

            #
            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_metropolis(
                subkey, 
                logdensity_fn_, 
                cov_params, 
                stepsize=mcmc_parameters.get(
                    'stepsizes', 
                    dict()).get('kernel', 0.1))

            position['kernel'] = sub_state
        
        # update Z  # TODO: needs to be adapted to changes in code-base after merge with main
        if True:
            # Z_params = self.__get_component_parameters(
            #     position, 
            #     'inducing_inputs_Z')
            Z_params = position.get('inducing_inputs_Z', {})

            # Get updated cov. parameters und u samples
            # mean_params = self.__get_component_parameters(position, 'mean')
            # cov_params = self.__get_component_parameters(position, 'kernel')
            cov_params = position.get('kernel', {})

            def logdensity_fn_(Z_):
                curr_Z = Z_

                log_pdf = 0
        
                # P(Z)
                log_pdf += jnp.sum(dx.Normal(
                    loc=Z_params['mean'],
                    scale=Z_params['scale']).log_prob(curr_Z))

                # p(f | u, Z, theta, X)
                mean_gp, cov_gp = self._compute_sparse_gp(
                    cov_params=cov_params, 
                    x=self.X,
                    samples_Z=curr_Z,
                    samples_u=position['u'])
                cov_gp += jitter * jnp.eye(self.n)
                log_pdf += dx.MultivariateNormalFullCovariance(mean_gp, cov_gp).log_prob(position['f'])

                # p(u | Z, theta)
                mean_u = self.mean_fn.mean(params=None, x=curr_Z)
                cov_u = self.cov_fn.cross_covariance(
                    params=cov_params,
                    x=curr_Z,
                    y=curr_Z)
                cov_u += jitter * jnp.eye(curr_Z.shape[0])
                log_pdf += dx.MultivariateNormalFullCovariance(
                    mean_u, 
                    cov_u).log_prob(position['u'])

                return log_pdf

            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_metropolis(  
                subkey, 
                logdensity_fn_, 
                # NOTE: Packing Z samples into dict for metropolis update
                # how does that work for other variables? -> They get a whole gibbs_state
                position['Z'],  
                stepsize=mcmc_parameters.get(
                    'stepsizes', 
                    dict()).get('kernel', 0.1))
            
            position['Z'] = sub_state

        # jax.debug.print('z {z}', z=position['Z'])

        # update u 
        if True:
            # get updated cov. parameters theta
            cov_params = position.get('kernel', {}) 

            mean_u = self.mean_fn.mean(params=None, x=position['Z'])
            cov_u = self.cov_fn.cross_covariance(
                params=cov_params,
                x=position['Z'],
                y=position['Z'])
            cov_u += jitter * jnp.eye(*cov_u.shape)

            def loglikelihood_fn_(u_):
                mean, cov = self._compute_sparse_gp(
                    cov_params=cov_params, 
                    x=self.X,
                    samples_Z=position['Z'],
                    samples_u=u_)

                return dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position['f'])

            key, subkey = jrnd.split(key)
            position['u'], f_info = update_correlated_gaussian(
                subkey, 
                position['u'], 
                loglikelihood_fn_, 
                mean_u, cov_u)

        # update likelihood
        if True:
            """Sample parameters of the likelihood using: 

            p(\phi | y, f) \propto p(y | f, phi)p(phi)

            """

            def logdensity_fn_(phi_):
                log_pdf = 0
                for param, val in phi_.items():
                    log_pdf += jnp.sum(self.param_priors['likelihood'][param].log_prob(val))
                log_pdf += temperature*jnp.sum(self.likelihood.log_prob(params=phi_, f=position['f'], y=self.y))
                return log_pdf

            #
            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_metropolis(
                subkey, 
                logdensity_fn_, 
                likelihood_params, 
                stepsize=mcmc_parameters.get(
                    'stepsizes', dict()
                    ).get('likelihood', 0.1))
            
            position['likelihood'] = sub_state

        return GibbsState(position=position), None 
            # We return None to satisfy SMC; this needs to be filled with acceptance information

    def loglikelihood_fn(self) -> Callable:
        """Returns the log-likelihood function for the model given a state.

        Args:
            None

        Returns:
            A function that computes the log-likelihood of the model given a
            state.
        """
        def loglikelihood_fn_(state: GibbsState) -> Float:
            # position = state.position
            # jax.debug.print('Using loglikelihood_fn!!!')
            position = getattr(state, 'position', state)
            phi = state.get('likelihood', {})
            f = position['f']
            log_pdf = jnp.sum(self.likelihood.log_prob(params=phi, f=f, y=self.y))
            return log_pdf

        #
        return loglikelihood_fn_

    def logprior_fn(self) -> Callable:
        """Returns the log-prior function for the model given a state.

        Args:
            None
        Returns:
            A function that computes the log-prior of the model given a state.

        """
        
        def logprior_fn_(state: GibbsState) -> Float:
            jax.debug.print('Using logprior_fn!!!')
            position = getattr(state, 'position', state)  # to work in both Blackjax' MCMC and SMC environments
            logprob = 0
            for component, params in self.param_priors.items():
                # mean, kernel, likelihood
                for param, dist in params.items():
                    # parameters per component
                    logprob += jnp.sum(dist.log_prob(position[param]))

            # plus the logprob of the latent f itself
            psi = {param: position[param] for param in self.param_priors['mean']} if 'mean' in self.param_priors else {}
            theta = {param: position[param] for param in
                     self.param_priors['kernel']} if 'kernel' in self.param_priors else {}
            mean = self.mean_fn.mean(params=psi, x=self.X).flatten()
            cov = self.cov_fn.cross_covariance(params=theta,
                                               x=self.X,
                                               y=self.X) + jitter * jnp.eye(self.n)
            logprob += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position['f'])
            return logprob

        #
        return logprior_fn_

    #
    def predict_f(self, key: PRNGKey, x_pred: ArrayTree):
        samples = self.get_monte_carlo_samples()
        num_particles = samples['f'].shape[0]
        key_samples = jrnd.split(key, num_particles)

        mean_params = samples.get('mean', {})
        cov_params = samples.get('kernel', {})
        mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)

        sample_fun = lambda key, mean_params, cov_params, target: sample_predictive(
            key, 
            mean_params=mean_params, 
            cov_params=cov_params, 
            mean_fn=self.mean_fn,
            cov_fn=self.cov_fn, 
            x=self.X, 
            z=x_pred, 
            target=target)

        keys = jrnd.split(key, num_particles)
        target_pred = jax.vmap(
            jax.jit(sample_fun), 
            in_axes=(0, mean_param_in_axes, cov_param_in_axes, 0))(
                keys,
                mean_params,
                cov_params, 
                samples['f'])

        return target_pred


    def predict_f_from_u(self, key: PRNGKey, x_pred: ArrayTree):
        samples = self.get_monte_carlo_samples()
        num_particles = samples['f'].shape[0]
        key_samples = jrnd.split(key, num_particles)

        mean_params = samples.get('mean', {})
        cov_params = samples.get('kernel', {})
        mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)
        
        Z = samples.get('Z')
        Z = jnp.mean(Z, axis=0)  # HACK: This is only valid as long as all Z's are the same accross particles

        sample_fun = lambda key, mean_params, cov_params, target: sample_predictive(
            key,
            x=Z, 
            z=x_pred,
            target=target,
            mean_params=mean_params, 
            cov_params=cov_params, 
            mean_fn=self.mean_fn,
            cov_fn=self.cov_fn,
            obs_noise=None
            )

        keys = jrnd.split(key, num_particles)
        target_pred = jax.vmap(
            jax.jit(sample_fun), 
            in_axes=(0, mean_param_in_axes, cov_param_in_axes, 0))(
                keys,
                mean_params,
                cov_params, 
                samples['u'])

        return target_pred

    #
    def predict_y(self, key, x_pred):
        # todo; call predict_f first, then the dx random from the appropriate likelihood
        pass

    #
#