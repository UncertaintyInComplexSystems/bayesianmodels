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
from jax.debug import print

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

JITTER = 1e-6

from icecream import ic

class SparseGPModel(FullGPModel):  
    """The latent Gaussian process model.  # TODO: Update description
    
    The latent Gaussian process model consists of observations (y), generated by
    an observation model that takes the latent Gaussian process (f) and optional
    hyperparameters (phi) as input. The latent GP itself is parametrized by a
    mean function (mu) and a covariance function (cov). These can have optional
    hyperparameters (psi) and (theta).

    The generative model is given by:

    # TODO: update generative model documentation
    .. math::
        psi     &\sim p(\psi)\\
        theta   &\sim p(\theta) \\
        phi     &\sim p(\phi) \\
        f       &\sim GP(mu, cov) \\
        y       &\sim p(y \mid T(f), phi)

    """ 

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None,
                 likelihood: AbstractLikelihood = None,
                 num_inducing_points: int = None):
        
        if likelihood is None:
            likelihood = Gaussian()

        # self.likelihood = likelihood  # NOTE: I am actually not using it. Should I? 
        self.m = num_inducing_points  # TODO: Instead of passing, infer from prior over inducing inputs

        super().__init__(X, y, cov_fn, mean_fn, priors)

    #   
    def init_fn(self, key, num_particles=1):
        """Initialization of the Gibbs state.  # TODO: Update description

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
            Sample Z and u
            """

            samples_Z = Z_params['Z']

            # Sample inducing variables u
            mean_u = jnp.zeros(samples_Z.shape[0])
            cov_ZZ = self.cov_fn.cross_covariance(
                params=cov_params,
                x=samples_Z, y=samples_Z) 
            cov_ZZ = cov_ZZ + JITTER * jnp.eye(cov_ZZ.shape[0])

            samples_u = jnp.asarray(mean_u + jnp.dot(
                jnp.linalg.cholesky(cov_ZZ),
                jrnd.normal(key, shape=[samples_Z.shape[0]])))
            
            return samples_u

        #
        initial_position = initial_state.position

        cov_params = initial_position.get('kernel', {})
        Z_params = initial_position.get('inducing_points', {})

        if num_particles > 1:
            # We vmap across the first dimension of the elements *in* the
            # dictionary, rather than over the dictionary itself.
            key_sample_particles = jrnd.split(key, num_particles)

            cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)
            Z_param_in_axes = jax.tree_map(lambda l: 0, Z_params)

            samples_u = jax.vmap(
                sample_latent,
                in_axes=(0, cov_param_in_axes, Z_param_in_axes)
                )(key_sample_particles, cov_params, Z_params)
                
        else:
            _, key_sample_latents = jrnd.split(key)

            samples_u = sample_latent(
                key_sample_latents, cov_params, Z_params)

        initial_position['u'] = samples_u
        

        return GibbsState(initial_position)


    def _loglikelihood_fn_fitc(self, u, Z, theta, sigma):
        """
        log-density log p(y | u, Z, theta), used in very Gibbs update step
        defined in the Rossi et al. 2021, eq. 14, 15, 17

        Args:
            u (_type_): ?
            Z (_type_): ?
            cov_params (_type_): covariance function parameters, denoted as theta.
            sigma (_type_): noise parameter of observation model 

        Returns:
            (_type_): sum of log p(y | u, Z, theta) 
        """
        
        # compute needed covariance matricies 
        cov_XX = self.cov_fn.cross_covariance(
            params=theta,
            x=self.X, y=self.X)
        cov_XX += JITTER * jnp.eye(*cov_XX.shape)

        cov_ZZ = self.cov_fn.cross_covariance(
            params=theta,
            x=Z, y=Z)
        cov_ZZ += JITTER * jnp.eye(*cov_ZZ.shape)

        cov_XZ = self.cov_fn.cross_covariance(
            params=theta,
            x=self.X, y=Z)  # shape: (N, M), (x.shape[0], Z.shape[0])
        cov_XZ += JITTER * jnp.eye(*cov_XZ.shape)

        # compute mean, eq. 14
        means = jnp.dot(cov_XZ, jnp.linalg.solve(cov_ZZ, u))  # shape: (N)

        # compute variance, eq. 15
        #   instead of computing for each x_n 
        #   we can compute for all x and extract the diagonal 
        ZZ_ZX = jnp.linalg.solve(cov_ZZ, jnp.transpose(cov_XZ))
        vars = jnp.diag(cov_XX - jnp.dot(cov_XZ, ZZ_ZX))

        # eval. pdf
        log_prob = dx.Normal(means, vars + sigma).log_prob(self.y) 

        # TODO use vmap to get the diag of the cross-covariance instead of computing the whole cross-covariance myself. Other solutions are also fine.

        return jnp.sum(log_prob)

    
    def gibbs_fn(
            self, key, state, 
            loglik_fn__=None, temperature=1.0, **mcmc_parameters): 
        # HACK: Gave loglik_fn None as default, can it actually be removed?
        """The Gibbs MCMC kernel.

        The Gibbs kernel step function takes a state and returns a new state. 
        In the latent GP model, the latent GP (f) is first updated, then the
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
        # get current gibbs-state
        position = state.position.copy()

        def sample_z(key, position_):
            inducing_points = position_.get('inducing_points', {})
            cov_params = position_.get('kernel', {})
            likelihood_params = position_.get('likelihood', {})

            def logdensity_fn_Z(z):
                log_pdf = 0
        
                log_pdf += jnp.sum(
                    self.param_priors['inducing_points']['Z'].log_prob(inducing_points['Z']))

                # p(u | Z, theta)
                mean_u = self.mean_fn.mean(params=None, x=z)
                cov_u = self.cov_fn.cross_covariance(
                    params=cov_params,
                    x=z,
                    y=z)
                cov_u += JITTER * jnp.eye(z.shape[0])
                log_pdf += dx.MultivariateNormalFullCovariance(
                    mean_u, 
                    cov_u).log_prob(position_['u'])

                # p(y | u, Z, theta)
                log_pdf += temperature*self._loglikelihood_fn_fitc(
                    u=position_['u'],
                    Z=z, 
                    theta=cov_params, 
                    sigma=likelihood_params['obs_noise'])

                return log_pdf

            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_metropolis(  
                subkey, 
                logdensity_fn_Z, 
                position_['inducing_points']['Z'],  
                stepsize=mcmc_parameters.get(
                    'stepsizes', 
                    dict()).get('kernel', 0.1))
            
            return sub_state
        

        def sample_u(key, position_:GibbsState):
            """
            Sample inducing points u. 
            As us is defined as a GP we use Elleptical Slice Sampling.
            Gibbs update equation:

            p(u | theta, Z, y) \propto 
                    p(u | Z, theta) p(y | u, Z, theta)

            Args:
                key (_type_): JAX random key
                position_ (_GibbsState_): Full GibbsState that contains quantites updated in previous GibbsSteps

            Returns:
                GibbsState: Partial GibbsState containing only the updated quantity
            """
            # get updated cov. parameters theta
            cov_params = position_.get('kernel', {})
            likelihood_params = position_.get('likelihood', {})

            # u is a GP in itself.
            # Define u's mean and covariance function
            mean_u = self.mean_fn.mean(params=None, x=position_['inducing_points']['Z'])
            cov_u = self.cov_fn.cross_covariance(
                params=cov_params,
                x=position_['inducing_points']['Z'],
                y=position_['inducing_points']['Z'])
            cov_u += JITTER * jnp.eye(*cov_u.shape)

            def logdensity_fn_u(u_): 
                return temperature*self._loglikelihood_fn_fitc(
                    u=u_,
                    Z=position_['inducing_points']['Z'], 
                    theta=cov_params, 
                    sigma=likelihood_params['obs_noise'])

            key, subkey = jrnd.split(key)
            sub_state, f_info = update_correlated_gaussian(
                subkey, 
                position_['u'], 
                logdensity_fn_u, 
                mean_u, cov_u)
            
            return sub_state


        def sample_theta(key, position_): 
            """Sample parameters of the kernel function using: 

            p(theta | u, Z, y) \propto 
                p(theta) p(u | Z, theta) p(y | u, Z, theta)
            """
            cov_params = position_.get('kernel', {})
            inducing_points = position_.get('inducing_points', {})
            likelihood_params = position_.get('likelihood', {})

            def logdensity_fn_cov(theta_):
                log_pdf = 0
                
                # p(theta) | cov. kernel parameter
                for param, val in theta_.items():
                    # jax.debug.print('val {d}', d=val)
                    log_pdf += jnp.sum(self.param_priors['kernel'][param].log_prob(val))
                
                # p(u | Z, theta)
                mean_u = self.mean_fn.mean(params=None, x=inducing_points['Z'])
                cov_u = self.cov_fn.cross_covariance(
                    params=theta_,
                    x=inducing_points['Z'],
                    y=inducing_points['Z'])
                cov_u = JITTER * jnp.eye(*cov_u.shape)
                log_pdf += dx.MultivariateNormalFullCovariance(mean_u, cov_u).log_prob(position_['u'])

                # p(y | u, Z, theta)
                log_pdf += temperature*self._loglikelihood_fn_fitc(
                    u=position_['u'],
                    Z=inducing_points['Z'], 
                    theta=theta_, 
                    sigma=likelihood_params['obs_noise'])
                
                return log_pdf

            #
            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_metropolis(
                subkey, 
                logdensity_fn_cov, 
                cov_params, 
                stepsize=mcmc_parameters.get(
                    'stepsizes', 
                    dict()).get('kernel', 0.1))
            
            return sub_state

        
        def sample_likelihood(key, position_):
            """Sample parameters of the likelihood using: 

            p(sigma | u, Z, theta, y) \propto p(phi) p(y | u, Z, theta)

            """
            cov_params = position_.get('kernel', {})
            likelihood_params = position_.get('likelihood', {})

            def logdensity_fn_(sigma_):
                log_pdf = 0

                # p(sigma^2)
                for param, val in sigma_.items():
                    log_pdf += jnp.sum(self.param_priors['likelihood'][param].log_prob(val))

                # p(y | u, Z, theta)
                log_pdf += temperature*self._loglikelihood_fn_fitc(
                    u=position_['u'],
                    Z=position_['inducing_points']['Z'], 
                    theta=cov_params, 
                    sigma=sigma_['obs_noise'])

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
            
            return sub_state


        key, subkey = jrnd.split(key)
        position['inducing_points']['Z'] = sample_z(subkey, position)

        key, subkey = jrnd.split(key)
        position['u'] = sample_u(key, position)

        key, subkey = jrnd.split(key)
        position['kernel'] = sample_theta(subkey, position)
        
        key, subkey = jrnd.split(key)
        position['likelihood'] = sample_likelihood(key, position)


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
            phi = state.get('likelihood', {})  # QUESTION: How is this different to getattr?

            return self._loglikelihood_fn_fitc(
                    u=position['u'],
                    Z=position['inducing_points']['Z'], 
                    theta=position.get('kernel', {}), 
                    sigma=position.get('likelihood', {})['obs_noise'])

        #
        return loglikelihood_fn_


    # TODO somethingn needs to change here, its using f.
    # TODO: Is this ever called? 
    #       -> its references in SMC
    #       -> but I never see the debug print I put below. 
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
                                               y=self.X) + JITTER * jnp.eye(self.n)
            logprob += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position['f'])
            return logprob

        #
        return logprior_fn_


    # TODO rename? 
    def predict_f(self, key: PRNGKey, x_pred: ArrayTree):
        """ see Rossi eq. 16
        """

        # jax.debug.print('\n\n')
        # jax.debug.print('predict_f sample keys {d}', d=list(samples.keys()))
        # for k in samples.keys():
        #     if isinstance(samples[k], dict):
        #         jax.debug.print('{k}: {d}', k=k, d=list(samples[k].keys()))
        #         for kk in samples[k].keys():
        #             jax.debug.print('    {k}: {d}', k=kk, d=samples[k][kk].shape)
        #     else:
        #         jax.debug.print('{k}: {d}', k=k, d=samples[k].shape)
        # jax.debug.print('\n\n')



        def sample_predictive(
                key: PRNGKey,
                x: Array,
                y: Array,
                z: Array,
                xs: Array,  # x*
                cov_params: Dict = None,
                likelihood = None):
            """Sample latent f for new points x_pred given one posterior sample.
            """
            
            def compute_cov(in1, in2, jitter=True):
                """ Helper function for more consise code down the line
                """
                cov = self.cov_fn.cross_covariance(
                    params=cov_params, x=in1, y=in2)
                if jitter:
                    cov += JITTER * jnp.eye(*cov.shape)
                return cov

            def compute_ab_invbb_ba(ab, bb, use_cholesky:bool = False):
                if use_cholesky:
                    L = jnp.linalg.cholesky(bb)
                    v = jnp.linalg.solve(L, ab.T)
                    return jnp.dot(v.T, v)
                else:
                    return jnp.dot(ab, jnp.linalg.solve(bb, ab.T))

            # compute needed covariance matricies 
            cov_XX = compute_cov(x, x)  # shape: (N, N)
            cov_ZZ = compute_cov(z, z)  # shape: (M, M)
            cov_XZ = compute_cov(x, z)  # shape: (N, M)
            cov_XsXs = compute_cov(xs, xs)  # shape: (num_targets, num_targets)
            cov_XsZ = compute_cov(xs, z)  # shape: (num_targets, M)

            # compute alpha
            diag_noise = likelihood['obs_noise'] * jnp.eye(*cov_XX.shape)

            XZ_ZZ_ZX = jnp.dot(
                cov_XZ, 
                jnp.linalg.solve(cov_ZZ, jnp.transpose(cov_XZ)))
            alpha = (cov_XX - XZ_ZZ_ZX + diag_noise) * jnp.eye(*cov_XX.shape)  # keeping only the values along the diagonal  # NOTE: Used jnp.eye instead of jnp.diag.
            # print('alpha any NaNs: {b}', b=jnp.any(jnp.isnan(alpha)))

            # compute sigma fitc
            sigma_fitc = cov_ZZ + jnp.dot(
                jnp.transpose(cov_XZ), 
                jnp.linalg.solve(alpha, cov_XZ))  # NOTE: in the paper this equation is inverted. I left it out to compute the inverse implicitly when Sigma is used. 

            # compute mu fitc
            mu_fitc = jnp.dot(
                cov_XsZ,
                jnp.dot(
                    jnp.linalg.solve(sigma_fitc, jnp.transpose(cov_XZ)),
                    jnp.linalg.solve(alpha, jnp.transpose(y))
                )
            )  # shape (num_targets, )

            # compute variance (sigma^2) fitc

            # XsZ_ZZ_ZXs = jnp.dot(
            #     cov_XsZ,
            #     jnp.linalg.solve(cov_ZZ, jnp.transpose(cov_XsZ)))
            # XsZ_ZZ_ZXs += JITTER * jnp.eye(*Xs_ZZ_ZXs.shape)
            XsZ_ZZ_ZXs = compute_ab_invbb_ba(
                cov_XsZ, cov_ZZ, use_cholesky=False)            

            # XsZ_sigma_ZXs = jnp.dot(
            #     cov_XsZ,
            #     jnp.linalg.solve(sigma_fitc, jnp.transpose(cov_XsZ)))
            # Xs_sigma_ZXs += JITTER * jnp.eye(*Xs_sigma_ZXs.shape)
            XsZ_sigma_ZXs = compute_ab_invbb_ba(
                cov_XsZ, sigma_fitc, use_cholesky=False) 
            
            var_fitc = cov_XsXs - XsZ_ZZ_ZXs + XsZ_sigma_ZXs  # shape (num_targets, num_targets)
            var_fitc += JITTER * jnp.eye(*var_fitc.shape)

            # draw samples
            if jnp.ndim(xs) == 1:
                L = jnp.linalg.cholesky(var_fitc)
                u = jrnd.normal(key, shape=(len(xs),))
                pred = mu_fitc + jnp.dot(L, u)
            else:
                raise NotImplementedError(f'Shape of target must be (n,)',
                f'but {xs.shape} was provided.')

            return pred

        # extract parameters and samples from data structure
        samples = self.get_monte_carlo_samples()
        
        cov_params = samples['kernel']
        Z = samples['inducing_points']['Z']
        likelihood = samples['likelihood']

        num_particles = Z.shape[0]

        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)
        likelihood_in_axes = jax.tree_map(lambda l: 0, likelihood)

        sample_fun = lambda key, cov_params, z, noise: sample_predictive(
            key = key,
            x = self.X,
            y = self.y,
            z = z,
            cov_params = cov_params,
            likelihood = noise,
            xs = x_pred  # x*
            )
        
        keys = jrnd.split(key, num_particles)
        y_pred = jax.vmap(
            jax.jit(sample_fun), 
            in_axes=(0, cov_param_in_axes, 0, likelihood_in_axes))(
                keys,
                cov_params,
                Z,
                likelihood)

        return y_pred

    #
    def predict_y(self, key, x_pred):
        # todo; call predict_f first, then the dx random from the appropriate likelihood
        pass

    #
#