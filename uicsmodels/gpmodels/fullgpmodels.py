from uicsmodels.bayesianmodels import AbstractModel, GibbsState, ArrayTree
from uicsmodels.gpmodels.meanfunctions import Zero
from uicsmodels.gpmodels.likelihoods import AbstractLikelihood, Gaussian
from uicsmodels.sampling.inference import inference_loop, smc_inference_loop

from jax import Array
from jax.typing import ArrayLike
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

import jax
import distrax as dx
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd
import blackjax
from blackjax import elliptical_slice, rmh, adaptive_tempered_smc
import blackjax.smc.resampling as resampling

jitter = 1e-6


class FullGPModel(AbstractModel):

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None):
        if jnp.ndim(X) == 1:
            X = X[:, jnp.newaxis]
        self.X, self.y = X, y
        self.n = self.X.shape[0]
        if mean_fn is None:
            mean_fn = Zero()
        self.mean_fn = mean_fn
        self.kernel = cov_fn
        self.param_priors = priors
        # TODO:
        # - assert whether all trainable parameters have been assigned priors
        # - add defaults/fixed values for parameters without prior

    #
    def predict_f(self, key: PRNGKey, x_pred: ArrayTree):
        raise NotImplementedError

    #
    def predict_y(self, key: PRNGKey, x_pred: ArrayTree):
        raise NotImplementedError

    #
    def init_fn(self, key: PRNGKey, num_particles=1):
        """Initial state for MCMC/SMC.

        """

        initial_position = dict()
        for component, comp_priors in self.param_priors.items():
            for param, param_dist in comp_priors.items():
                key, _ = jrnd.split(key)
                if num_particles > 1:
                    initial_position[param] = param_dist.sample(seed=key, sample_shape=(num_particles,))
                else:
                    initial_position[param] = param_dist.sample(seed=key)
        return GibbsState(position=initial_position)

    # 
    def gibbs_fn(self, key: PRNGKey, state: GibbsState, **kwars):
        raise NotImplementedError
    
    #
    def loglikelihood_fn(self) -> Callable:
        raise NotImplementedError
    
    #
    def logprior_fn(self) -> Callable:
        raise NotImplementedError
    
    #
    def plot_priors(self, axes=None):
        raise NotImplementedError

    #
    def inference(self, key, mode='smc', sampling_parameters: Dict=None):
        """A wrapper for training the GP model.

        An interface to Blackjax' MCMC or SMC inference loops, tailored to the
        current latent GP model.

        Args:
            key: jrnd.KeyArray
                The random number seed, will be split into initialisation and
                inference.
            mode: {'mcmc', 'smc'}
                The desired inference approach. Defaults to SMC, which is
                generally preffered.
            sampling_parameters: dict
                Optional settings with defaults for the inference procedure.

        Returns:
            Depending on 'mode':
                smc:
                    num_iter: int
                        Number of tempering iterations.
                    particles:
                        The final states the SMC particles (at T=1).
                    marginal_likelihood: float
                        The approximated marginal likelihood of the model.
                mcmc:
                    states:
                        The MCMC states (including burn-in states).

        """
        if sampling_parameters is None:
            sampling_parameters = dict()

        if mode == 'smc':
            smc = adaptive_tempered_smc(
                logprior_fn=self.logprior_fn(),
                loglikelihood_fn=self.loglikelihood_fn(),
                mcmc_step_fn=self.gibbs_fn,
                mcmc_init_fn=self.smc_init_fn,
                mcmc_parameters=sampling_parameters.get('gibbs_parameters', dict()),
                resampling_fn=resampling.systematic,
                target_ess=sampling_parameters.get('target_ess', 0.5),
                num_mcmc_steps=sampling_parameters.get('num_mcmc_steps', 50)
            )

            num_particles = sampling_parameters.get('num_particles', 10_000)
            key_init, key_smc = jrnd.split(key, 2)

            initial_particles = self.init_fn(key_init,
                                             num_particles=num_particles)
            # We need to initialize SMC with a dictionary
            initial_smc_state = smc.init(initial_particles.position)

            num_iter, particles, marginal_likelihood = smc_inference_loop(key_smc,
                                                                          smc.step,
                                                                          initial_smc_state)

            self.particles = particles
            self.marginal_likelihood = marginal_likelihood
            return particles, num_iter, marginal_likelihood
        elif mode == 'mcmc':
            key_init, key_mcmc = jrnd.split(key, 2)
            # Is there a smarter initialization than from the prior for the MCMC case>
            initial_state = self.init_fn(key_init)

            num_burn = sampling_parameters.get('num_burn', 10_000)
            num_samples = sampling_parameters.get('num_samples', 10_000)

            states = inference_loop(key_mcmc,
                                    self.gibbs_fn,
                                    initial_state,
                                    num_burn + num_samples)
            self.states = states
            return states
        else:
            raise NotImplementedError(f'{mode} is not implemented as inference method')
        
    #
    def smc_init_fn(self, position: ArrayTree, kwargs):
        """Simply wrap the position dictionary in a GibbsState object. 

        Args:
            position: dict
                Current assignment of the state values
            kwargs: not used in our Gibbs kernel
        Returns:
            A Gibbs state object.
        """
        return GibbsState(position)

    #

#
        
class FullLatentGPModel(FullGPModel):

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

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None,
                 likelihood: AbstractLikelihood = None):
        if likelihood is None:
            likelihood = Gaussian()
        self.likelihood = likelihood
        super().__init__(X, y, cov_fn, mean_fn, priors)        
        # TODO:
        # - assert whether all trainable parameters have been assigned priors
        # - add defaults/fixed values for parameters without prior

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

        initial_state = super().init_fn(key, num_particles)

        def sample_latent(key, initial_position_):
            if 'mean_function' in self.param_priors.keys():
                mean_params = {param: initial_position_[param] for param in self.param_priors['mean_function']}
                mean = self.mean_fn.mean(params=mean_params, x=self.X)
            else:
#                 mean = jnp.zeros_like(self.X)
                mean = jnp.zeros((self.X.shape[0], ))
            if jnp.ndim(mean) == 1:
                mean = mean[:, jnp.newaxis]

            if 'kernel' in self.param_priors.keys():
                cov_params = {param: initial_position_[param] for param in self.param_priors['kernel']}
                cov = self.kernel.cross_covariance(params=cov_params,
                                                   x=self.X,
                                                   y=self.X) + jitter * jnp.eye(self.n)
            else:
                cov = jnp.eye(self.n)

            L = jnp.linalg.cholesky(cov)
            z = jrnd.normal(key, shape=(self.n, 1))
            f = jnp.asarray(mean + jnp.dot(L, z))
            return f.flatten()

        #
        initial_position = initial_state.position

        if num_particles > 1:
            keys = jrnd.split(key, num_particles)
            # We vmap across the first dimension of the elements *in* the
            # dictionary, rather than over the dictionary itself.
            initial_position['f'] = jax.vmap(sample_latent,
                                             in_axes=(0, {k: 0 for k in initial_position}))(keys, initial_position)
        else:
            key, _ = jrnd.split(key)
            initial_position['f'] = sample_latent(key, initial_position)

        return GibbsState(initial_position)

        #

    def gibbs_fn(self, key, state, **kwargs):
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

        def mcmc_step(key, logdensity: Callable, variables: Dict, stepsize: float = 0.01):
            """The MCMC step for sampling hyperparameters.

            This updates the hyperparameters of the mean, covariance function
            and likelihood, if any. Currently, this uses a random-walk
            Metropolis step function, but other Blackjax options are available.

            Args:
                key:
                    The jax.random.PRNGKey
                logdensity: Callable
                    Function that returns a logdensity for a given set of variables
                variables: Dict
                    The set of variables to sample and their current values
                stepsize: float
                    The stepsize of the random walk
            Returns:
                RMHState, RMHInfo

            """
            key, _ = jrnd.split(key)
            m = 0
            for varname, varval in variables.items():
                m += varval.shape[0] if varval.shape else 1

            kernel = rmh(logdensity, sigma=stepsize * jnp.eye(m))
            substate = kernel.init(variables)
            substate, info_ = kernel.step(key, substate)
            return substate.position, info_

        #
        def get_parameters_for(component):
            """Extract parameter sampled values per model component for current
            position.

            """
            return {param: position[param] for param in
                    self.param_priors[component]} if component in self.param_priors else {}

        #
        
        position = state.position.copy()
        temperature = kwargs.get('temperature', 1.0)

        """Sample the latent GP using:

        p(f | theta, psi, y) \propto p(y | f, phi) p(f | psi, theta)

        """
        phi = get_parameters_for('likelihood')
        loglikelihood_fn_ = lambda f_: temperature * jnp.sum(self.likelihood.log_prob(params=phi, f=f_, y=self.y))

        psi = get_parameters_for('mean')
        mean = self.mean_fn.mean(params=psi, x=self.X).flatten()

        theta = get_parameters_for('kernel')
        cov = self.kernel.cross_covariance(params=theta,
                                           x=self.X, y=self.X) + jitter * jnp.eye(self.n)

        latent_sampler = elliptical_slice(loglikelihood_fn_,
                                          mean=mean,
                                          cov=cov)
        f = position['f']
        state_f = latent_sampler.init(f)
        key, _ = jrnd.split(key)
        state_f, info_f = latent_sampler.step(key, state_f)
        f = state_f.position
        position['f'] = state_f.position

        if len(psi):
            """Sample parameters of the mean function using: 

            p(psi | f, theta) \propto p(f | psi, theta)p(psi)

            """

            def logdensity_fn_(psi_):
                log_pdf = 0
                for param, val in psi_.items():
                    log_pdf += jnp.sum(self.param_priors['mean'][param].log_prob(val))
                mean = self.mean_fn.mean(params=psi_, x=self.X).flatten()
                log_pdf += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(f)
                return log_pdf

            #
            sub_state, _ = mcmc_step(key, logdensity_fn_, psi, stepsize=0.1)
            for param, val in sub_state.items():
                position[param] = val

            mean = self.mean_fn.mean(params=sub_state, x=self.X).flatten()
        #

        if len(theta):
            """Sample parameters of the kernel function using: 

            p(theta | f, psi) \propto p(f | psi, theta)p(theta)

            """

            def logdensity_fn_(theta_):
                log_pdf = 0
                for param, val in theta_.items():
                    log_pdf += jnp.sum(self.param_priors['kernel'][param].log_prob(val))
                cov_ = self.kernel.cross_covariance(params=theta_, x=self.X, y=self.X) + jitter * jnp.eye(self.n)
                log_pdf += dx.MultivariateNormalFullCovariance(mean, cov_).log_prob(f)
                return log_pdf

            #
            sub_state, _ = mcmc_step(key, logdensity_fn_, theta, stepsize=0.1)
            for param, val in sub_state.items():
                position[param] = val
        #

        if len(phi):
            """Sample parameters of the likelihood using: 

            p(\phi | y, f) \propto p(y | f, phi)p(phi)

            """

            def logdensity_fn_(phi_):
                log_pdf = 0
                for param, val in phi_.items():
                    log_pdf += jnp.sum(self.param_priors['likelihood'][param].log_prob(val))
                log_pdf += jnp.sum(self.likelihood.log_prob(params=phi_, f=f, y=self.y))
                return log_pdf

            #
            sub_state, _ = mcmc_step(key, logdensity_fn_, phi, stepsize=0.1)
            for param, val in sub_state.items():
                position[param] = val
        #

        return GibbsState(
            position=position), None  # We return None to satisfy SMC; this needs to be filled with acceptance information

    #
    def loglikelihood_fn(self) -> Callable:
        """Returns the log-likelihood function for the model given a state.

        Args:
            None

        Returns:
            A function that computes the log-likelihood of the model given a
            state.
        """
        def loglikelihood_fn_(state: GibbsState) -> float:
            # position = state.position
            position = getattr(state, 'position', state)
            phi = {param: position[param] for param in
                   self.param_priors['likelihood']} if 'likelihood' in self.param_priors else {}
            f = position['f']
            log_pdf = jnp.sum(self.likelihood.log_prob(params=phi, f=f, y=self.y))
            return log_pdf

        #
        return loglikelihood_fn_

    #
    def logprior_fn(self) -> Callable:
        """Returns the log-prior function for the model given a state.

        Args:
            None
        Returns:
            A function that computes the log-prior of the model given a state.

        """

        def logprior_fn_(state: GibbsState) -> float:
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
            cov = self.kernel.cross_covariance(params=theta,
                                               x=self.X,
                                               y=self.X) + jitter * jnp.eye(self.n)
            logprob += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position['f'])
            return logprob

        #
        return logprior_fn_

    #
    def predict_f(self, key: PRNGKey, x_pred: ArrayTree, num_subsample=-1):
        """Predict the latent f on unseen pointsand

        This function takes the approximated posterior (either by MCMC or SMC)
        and predicts new latent function evaluations f^*.

        Args:
            key: PRNGKey
            x_pred: x^*; the queried locations.
            num_subsample: By default, we return one predictive sample for each
            posterior sample. While accurate, this can be memory-intensive; this
            parameter can be used to thin the MC output to every n-th sample.

        Returns:
            f_samples: An array of samples of f^* from p(f^* | x^*, x, y)


        todo:
        - predict using either SMC or MCMC output
        - predict from prior if desired
        """

        @jax.jit
        def sample_predictive_f(key, x_pred: ArrayTree, **state_variables):
            """Sample latent f for new points x_pred given one posterior sample.

            See Rasmussen & Williams. We are sampling from the posterior predictive for
            the latent GP f, at this point not concerned with an observation model yet.

            We have [f, f*]^T ~ N(0, KK), where KK is a block matrix:

            KK = [[K(x, x), K(x, x*)], [K(x, x*)^T, K(x*, x*)]]

            This results in the conditional

            f* | x, x*, f ~ N(mu, cov), where

            mu = K(x*, x)K(x,x)^-1 f
            cov = K(x*, x*) - K(x*, x) K(x, x)^-1 K(x, x*)

            Args:
                key: The jrnd.PRNGKey object
                x_pred: The prediction locations x*
                state_variables: A sample from the posterior

            Returns:
                A single posterior predictive sample f*

            """

            def get_parameters_for(component):
                """Extract parameter sampled values per model component for current
                position.

                """
                return {param: state_variables[param] for param in
                        self.param_priors[component]} if component in self.param_priors else {}

            #

            f = state_variables['f']

            # to implement!
            psi = get_parameters_for('mean')
            mean = self.mean_fn.mean(params=psi, x=self.X).flatten()
            theta = get_parameters_for('kernel')

            kXX = self.kernel.cross_covariance(params=theta, x=self.X, y=self.X)
            kxX = self.kernel.cross_covariance(params=theta, x=self.X, y=x_pred)
            kxx = self.kernel.cross_covariance(params=theta, x=x_pred, y=x_pred)
            for k in [kXX, kxX, kxx]:
                k += jitter * jnp.eye(*k.shape)

            L = jnp.linalg.cholesky(kXX + jitter * jnp.eye(self.n))
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, f))
            v = jnp.linalg.solve(L, kxX)
            predictive_mean = jnp.dot(kxX.T, alpha)
            predictive_var = kxx - jnp.dot(v.T, v)

            # if self.X.shape[0] < 20:
            # heuristic for numerical stability
            predictive_var += jitter * jnp.eye(*kxx.shape)

            C = jnp.linalg.cholesky(predictive_var)
            z = jrnd.normal(key, shape=(len(x_pred),))

            f_samples = predictive_mean + jnp.dot(C, z)
            return f_samples

        #

        num_particles = self.particles.particles['f'].shape[0]
        key_samples = jrnd.split(key, num_particles)

        f_pred = jax.vmap(sample_predictive_f,
                          in_axes=(0, None))(key_samples, x_pred, **self.particles.particles)
        return f_pred

    #
    def predict_y(self, key, x_pred):
        # todo; call predict_f first, then the dx random from the appropriate likelihood
        pass

    #
#
