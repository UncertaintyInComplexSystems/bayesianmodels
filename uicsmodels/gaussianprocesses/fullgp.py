from uicsmodels.bayesianmodels import BayesianModel, GibbsState, ArrayTree
from uicsmodels.sampling.inference import update_correlated_gaussian, update_metropolis
from uicsmodels.gaussianprocesses.meanfunctions import Zero
from uicsmodels.gaussianprocesses.likelihoods import AbstractLikelihood, Gaussian

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
#from blackjax import elliptical_slice, rmh


jitter = 1e-6


class FullGPModel(BayesianModel):

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
    def __get_component_parameters(self, position, component):
        """Extract parameter sampled values per model component for current
        position.

        """
        return {param: position[param] for param in
                self.param_priors[component]} if component in self.param_priors else {}

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

    def gibbs_fn(self, key, state, temperature=1.0, **mcmc_parameters):
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
        
        position = state.position.copy()

        """Sample the latent GP using:

        p(f | theta, psi, y) \propto p(y | f, phi) p(f | psi, theta)

        """
        likelihood_params = self.__get_component_parameters(position, 'likelihood')
        loglikelihood_fn_ = lambda f_: temperature * jnp.sum(self.likelihood.log_prob(params=likelihood_params, f=f_, y=self.y))

        mean_params = self.__get_component_parameters(position, 'mean')
        mean = self.mean_fn.mean(params=mean_params, x=self.X).flatten()

        cov_params = self.__get_component_parameters(position, 'kernel')
        cov = self.kernel.cross_covariance(params=cov_params,
                                            x=self.X, y=self.X) + jitter * jnp.eye(self.n)
        
        key, subkey = jrnd.split(key)
        position['f'] = update_correlated_gaussian(subkey, state, position['f'], 
                                                   loglikelihood_fn_, mean, cov)

        if len(mean_params):
            """Sample parameters of the mean function using: 

            p(psi | f, theta) \propto p(f | psi, theta)p(psi)

            """
            key, subkey = jrnd.split(key)

            def logdensity_fn_(psi_):
                log_pdf = 0
                for param, val in psi_.items():
                    log_pdf += jnp.sum(self.param_priors['mean'][param].log_prob(val))
                mean = self.mean_fn.mean(params=psi_, x=self.X).flatten()
                log_pdf += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position['f'])
                return log_pdf

            #
            sub_state, _ = update_metropolis(subkey, logdensity_fn_, mean_params, stepsize=0.1)
            for param, val in sub_state.items():
                position[param] = val

            mean = self.mean_fn.mean(params=sub_state, x=self.X).flatten()
        #

        if len(cov_params):
            """Sample parameters of the kernel function using: 

            p(theta | f, psi) \propto p(f | psi, theta)p(theta)

            """
            key, subkey = jrnd.split(key)

            def logdensity_fn_(theta_):
                log_pdf = 0
                for param, val in theta_.items():
                    log_pdf += jnp.sum(self.param_priors['kernel'][param].log_prob(val))
                cov_ = self.kernel.cross_covariance(params=theta_, x=self.X, y=self.X) + jitter * jnp.eye(self.n)
                log_pdf += dx.MultivariateNormalFullCovariance(mean, cov_).log_prob(position['f'])
                return log_pdf

            #
            sub_state, _ = update_metropolis(subkey, logdensity_fn_, cov_params, stepsize=0.1)
            for param, val in sub_state.items():
                position[param] = val
        #

        if len(likelihood_params):
            """Sample parameters of the likelihood using: 

            p(\phi | y, f) \propto p(y | f, phi)p(phi)

            """
            key, subkey = jrnd.split(key)

            def logdensity_fn_(phi_):
                log_pdf = 0
                for param, val in phi_.items():
                    log_pdf += jnp.sum(self.param_priors['likelihood'][param].log_prob(val))
                log_pdf += temperature*jnp.sum(self.likelihood.log_prob(params=phi_, f=position['f'], y=self.y))
                return log_pdf

            #
            sub_state, _ = update_metropolis(subkey, logdensity_fn_, likelihood_params, stepsize=0.1)
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
        def loglikelihood_fn_(state: GibbsState) -> Float:
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

        def logprior_fn_(state: GibbsState) -> Float:
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
        if hasattr(self, 'particles'):
            samples = self.particles.particles
        elif hasattr(self, 'states'):
            samples = self.states.position
                
        num_samples = samples['f'].shape[0]
        key_samples = jrnd.split(key, num_samples)

        f_pred = jax.vmap(sample_predictive_f,
                          in_axes=(0, None))(key_samples, x_pred, **samples)
        return f_pred

    #
    def predict_y(self, key, x_pred):
        # todo; call predict_f first, then the dx random from the appropriate likelihood
        pass

    #
#

class FullMarginalGPModel(FullGPModel):
    """The marginal Gaussian process model.

    In case the likelihood of the GP is Gaussian, we marginalize out the latent
    GP f for (much) more efficient inference.

    The marginal Gaussian process model consists of observations (y), generated
    by a Gaussian observation model with hyperparameter sigma as input. The
    latent GP itself is parametrized by a mean function (mu) and a covariance
    function (cov). These can have optional hyperparameters (psi) and (theta).

    The generative model is given by:

    .. math::
        psi     &\sim p(\psi)\\
        theta   &\sim p(\theta) \\
        sigma     &\sim p(\sigma) \\
        y       &\sim N(mu, cov + \sigma^2 I_n)

    """

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None):
        super().__init__(X, y, cov_fn, mean_fn, priors)

    #
    def gibbs_fn(self, key, state, temperature=1.0, **mcmc_parameters):
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

        position = state.position.copy()

        loglikelihood_fn_ = self.loglikelihood_fn()
        logprior_fn_ = self.logprior_fn()

        logdensity = lambda state: temperature * loglikelihood_fn_(state) + logprior_fn_(state)
        new_position, info_ = update_metropolis(key, logdensity, position)

        return GibbsState(
            position=new_position), None  # We return None to satisfy SMC; this needs to be filled with acceptance information

    #
    def loglikelihood_fn(self) -> Callable:
        """Returns the log-likelihood function for the model given a state.

        Args:
            None

        Returns:
            A function that computes the log-likelihood of the model given a
            state.
        """
        jitter = 1e-6

        def loglikelihood_fn_(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)
            psi = {param: position[param] for param in self.param_priors['mean']} if 'mean' in self.param_priors else {}
            theta = {param: position[param] for param in
                     self.param_priors['kernel']} if 'kernel' in self.param_priors else {}

            sigma = state['obs_noise']
            mean = self.mean_fn.mean(params=psi, x=self.X).flatten()
            cov = self.kernel.cross_covariance(params=theta,
                                               x=self.X,
                                               y=self.X) + (sigma ** 2 + jitter) * jnp.eye(self.n)
            logprob = dx.MultivariateNormalFullCovariance(mean, cov).log_prob(self.y)
            return logprob

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

        def logprior_fn_(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)  # to work in both Blackjax' MCMC and SMC environments
            logprob = 0
            for component, params in self.param_priors.items():
                # mean, kernel, likelihood
                for param, dist in params.items():
                    # parameters per component
                    logprob += jnp.sum(dist.log_prob(position[param]))
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

            jitter = 1e-6

            # to implement!
            psi = get_parameters_for('mean')
            mean = self.mean_fn.mean(params=psi, x=self.X).flatten()
            theta = get_parameters_for('kernel')
            sigma = get_parameters_for('likelihood')['obs_noise']

            kXX = self.kernel.cross_covariance(params=theta, x=self.X, y=self.X)
            kXX += sigma ** 2 * jnp.eye(*kXX.shape)  # add observation noise
            kxX = self.kernel.cross_covariance(params=theta, x=self.X, y=x_pred)
            kxx = self.kernel.cross_covariance(params=theta, x=x_pred, y=x_pred)
            for k in [kXX, kxX, kxx]:
                k += jitter * jnp.eye(*k.shape)

            L = jnp.linalg.cholesky(kXX + jitter * jnp.eye(self.n))
            alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, self.y))
            v = jnp.linalg.solve(L, kxX)
            predictive_mean = jnp.dot(kxX.T, alpha)
            predictive_var = kxx - jnp.dot(v.T, v) + jitter * jnp.eye(*kxx.shape)

            C = jnp.linalg.cholesky(predictive_var)
            z = jrnd.normal(key, shape=(len(x_pred),))

            f_samples = predictive_mean + jnp.dot(C, z)
            return f_samples

        #
        if hasattr(self, 'particles'):
            samples = self.particles.particles
        elif hasattr(self, 'states'):
            samples = self.states.position
                
        num_samples = samples['obs_noise'].shape[0]
        key_samples = jrnd.split(key, num_samples)

        f_pred = jax.vmap(sample_predictive_f,
                          in_axes=(0, None))(key_samples, x_pred, **samples)
        return f_pred

    #


#
