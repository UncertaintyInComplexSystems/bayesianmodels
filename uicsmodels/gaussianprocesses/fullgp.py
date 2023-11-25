from uicsmodels.bayesianmodels import BayesianModel, GibbsState, ArrayTree
from uicsmodels.sampling.inference import update_correlated_gaussian, update_metropolis
from uicsmodels.gaussianprocesses.gputil import sample_prior, sample_predictive, update_gaussian_process, update_gaussian_process_cov_params, update_gaussian_process_mean_params, update_gaussian_process_obs_params
from uicsmodels.gaussianprocesses.meanfunctions import Zero
from uicsmodels.gaussianprocesses.likelihoods import AbstractLikelihood, Gaussian, RepeatedObsLikelihood

from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
from jax.tree_util import tree_flatten, tree_unflatten

from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

import jax
import distrax as dx
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd
from blackjax import elliptical_slice, rmh


jitter = 1e-6


class FullGPModel(BayesianModel):

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None,
                 duplicate_input=False):
        if jnp.ndim(X) == 1:
            X = X[:, jnp.newaxis]        
        # Validate arguments
        if X.shape[0] > len(y):
            raise ValueError(
                f'X and y should have the same leading dimension, '
                f'but X has shape {X.shape} and y has shape {y.shape}')
        self.X, self.y = X, y        
        self.n = self.X.shape[0]        
        if mean_fn is None:
            mean_fn = Zero()
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn
        self.param_priors = priors

    #
    def predict_f(self, key: PRNGKey, x_pred: ArrayTree):
        raise NotImplementedError

    #
    def predict_y(self, key: PRNGKey, x_pred: ArrayTree):
        raise NotImplementedError

    #
    def init_fn(self, key: Array, num_particles: int = 1):
        """Initial state for MCMC/SMC.

        This function initializes all highest level latent variables. Children
        of this class need to implement initialization of intermediate latent
        variables according to the structure of the hierarchical model.

        Args:
            key: PRNGKey
            num_particles: int
                Number of particles to initialize a state for
        Returns:
            GibbsState

        """

        priors_flat, priors_treedef = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
        samples = list()
        for prior in priors_flat:
            key, subkey = jrnd.split(key)
            samples.append(prior.sample(seed=subkey, sample_shape=(num_particles,)))

        initial_position = jax.tree_util.tree_unflatten(priors_treedef, samples)
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

    Here, the scalar parameters are sampled using Gaussian random walk MCMC, 
    while the latent function f (or rather its evaluations) is sampled using
    Elliptical Slice Sampling.

    """

    def __init__(self, X, y,
                 cov_fn: Callable,
                 mean_fn: Optional[Callable] = None,
                 priors: Dict = None,
                 likelihood: AbstractLikelihood = None,
                 **kwargs):
        if likelihood is None:
            likelihood = Gaussian()
        self.likelihood = likelihood
        super().__init__(X, y, cov_fn, mean_fn, priors, **kwargs)      

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
        initial_position = initial_state.position

        mean_params = initial_position.get('mean', {})
        cov_params = initial_position.get('kernel', {})
        mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)

        if num_particles > 1:
            keys = jrnd.split(key, num_particles)                
            sample_fun = lambda key_, mean_params_, cov_params_: sample_prior(key=key_, 
                                                                                mean_params=mean_params_,
                                                                                cov_params=cov_params_,
                                                                                mean_fn=self.mean_fn,
                                                                                cov_fn=self.cov_fn, 
                                                                                x=self.X)
            initial_position['f'] = jax.vmap(sample_fun,
                                             in_axes=(0,
                                                      mean_param_in_axes,
                                                      cov_param_in_axes))(keys, mean_params, cov_params)
        else:
            key, subkey = jrnd.split(key)
            initial_position['f'] = sample_prior(subkey, 
                                                 self.mean_fn, 
                                                 self.cov_fn, 
                                                 mean_params, 
                                                 cov_params, 
                                                 self.X)

        return GibbsState(initial_position)

        #

    def gibbs_fn(self, key: PRNGKey, state: GibbsState, temperature: Float= 1.0, **mcmc_parameters):
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
            temperature: Float
                The likeihood temperature, \beta in p_\beta(x | y) \propto p(x) p(y | x)^\beta
            mcmc_parameters: Dict
                A dictionary with optional settings for the MCMC-within-Gibbs 
                steps. TODO
        Returns:
            GibbsState

        """
        position = state.position.copy()

        # Sample the latent GP using:   
        # p(f | theta, psi, y) \propto p(y | f, phi) p(f | psi, theta)

        likelihood_params = position.get('likelihood', {})
        mean_params = position.get('mean', {}) 
        cov_params = position.get('kernel', {}) 

        loglikelihood_fn_ = lambda f_: temperature * jnp.sum(self.likelihood.log_prob(params=likelihood_params, f=f_, y=self.y))

        key, subkey = jrnd.split(key)
        position['f'], f_info = update_gaussian_process(subkey,
                                                        position['f'],
                                                        loglikelihood_fn_,
                                                        self.X,
                                                        mean_fn=self.mean_fn,
                                                        cov_fn=self.cov_fn,
                                                        mean_params=mean_params,
                                                        cov_params=cov_params)

        if len(mean_params):
            # Sample parameters of the mean function using: 
            # p(psi | f, theta) \propto p(f | psi, theta)p(psi)      

            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_gaussian_process_mean_params(subkey, self.X,
                                       position['f'],
                                       mean_fn=self.mean_fn,
                                       cov_fn=self.cov_fn,
                                       mean_params=mean_params,
                                       cov_params=cov_params,
                                       hyperpriors=self.param_priors['mean'])
            position['mean'] = sub_state
            
        #

        if len(cov_params):
            # Sample parameters of the kernel function using: 
            # p(theta | f, psi) \propto p(f | psi, theta)p(theta)

            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_gaussian_process_cov_params(subkey, self.X,
                                       position['f'],
                                       mean_fn=self.mean_fn,
                                       cov_fn=self.cov_fn,
                                       mean_params=mean_params,
                                       cov_params=cov_params,
                                       hyperpriors=self.param_priors['kernel'])
            position['kernel'] = sub_state
        #

        if len(likelihood_params):
            # Sample parameters of the likelihood using: 
            # p(\phi | y, f) \propto p(y | f, phi)p(phi)

            key, subkey = jrnd.split(key)
            sub_state, sub_info = update_gaussian_process_obs_params(subkey, self.y,
                                       position['f'],
                                       temperature=temperature,
                                       likelihood=self.likelihood,
                                       obs_params=likelihood_params,
                                       hyperpriors=self.param_priors['likelihood'])
            position['likelihood'] = sub_state
        
        #
        return GibbsState(position=position), None  # We return None to satisfy SMC; this needs to be filled with acceptance information

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
            position = getattr(state, 'position', state)
            phi = state.get('likelihood', {})
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

        # todo: add 2D vmap

        """
        def logprior_fn_(state: GibbsState) -> Float:
            # This function isn't reached??
            # to work in both Blackjax' MCMC and SMC environments
            position = getattr(state, 'position', state) 
            logprob = 0
            for component, params in self.param_priors.items():
                for param, dist in params.items():
                    logprob += jnp.sum(dist.log_prob(position[param]))
            psi = {param: position[param] for param in self.param_priors['mean']} if 'mean' in self.param_priors else {}
            theta = {param: position[param] for param in
                     self.param_priors['kernel']} if 'kernel' in self.param_priors else {}
            mean = self.mean_fn.mean(params=psi, x=self.X).flatten()
            cov = self.cov_fn.cross_covariance(params=theta,
                                               x=self.X,
                                               y=self.X) + jitter * jnp.eye(self.n)
            f = position['f']
            if jnp.ndim(f) == 1:
                logprob += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(f)
            elif jnp.ndim(f) == 3:
                log_pdf += jnp.sum(jax.vmap(jax.vmap(dx.MultivariateNormalFullCovariance(mean, cov).log_prob, in_axes=1), in_axes=1)(f))
            else:
                raise NotImplementedError(f'Expected f to be of size (n,) or (n, nu, d),',
                                          f'but size {f.shape} was provided.')
            return logprob

        #
        return logprior_fn_

    #
    def predict_f(self, key: PRNGKey, x_pred: ArrayTree):
        """Samples from the posterior predictive of the latent f

        Args:
            key: PRNGKey
            x_pred: Array
                The test locatons
        Returns:
            Returns samples from the posterior predictive distribution:

            f* \sim p(f* | f, X, y x*) = \int p(f* | x*, f) p(f | X, y) df

        """
        if jnp.ndim(x_pred) == 1:
            x_pred = x_pred[:, jnp.newaxis]

        samples = self.get_monte_carlo_samples()
        if samples is None:
            raise AssertionError(
                f'The posterior predictive distribution can only be called after training.')

        num_particles = samples['f'].shape[0]
        key_samples = jrnd.split(key, num_particles)

        mean_params = samples.get('mean', {})
        cov_params = samples.get('kernel', {})
        mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)

        sample_fun = lambda key, mean_params, cov_params, target: sample_predictive(key, 
                                                                            mean_params=mean_params, 
                                                                            cov_params=cov_params, 
                                                                            mean_fn=self.mean_fn,
                                                                            cov_fn=self.cov_fn, 
                                                                            x=self.X, 
                                                                            z=x_pred, 
                                                                            target=target)
        keys = jrnd.split(key, num_particles)
        target_pred = jax.vmap(jax.jit(sample_fun), 
                        in_axes=(0, 
                                 mean_param_in_axes, 
                                 cov_param_in_axes, 
                                 0))(keys, 
                                    mean_params, 
                                    cov_params, 
                                    samples['f'])

        return target_pred

    #
    def forward(self, key, params, f):
        return self.likelihood.likelihood(params, f).sample(seed=key)

    #
    def predict_y(self, key: PRNGKey, x_pred: Array):
        """Samples from the posterior predictive distribution

        Args:
            key: PRNGKey
            x_pred: Array
                The test locatons
        Returns:
            Returns samples from the posterior predictive distribution:

            y* \sim p(y* | X, y x*) = \int p(y* | f*)p(f* | f)p(f | X, y) df

        """
        if jnp.ndim(x_pred) == 1:
            x_pred = x_pred[:, jnp.newaxis]

        samples = self.get_monte_carlo_samples()
        if samples is None:
            raise AssertionError(
                f'The posterior predictive distribution can only be called after training.')
        
        key, key_f, key_y = jrnd.split(key, 3)
        f_pred = self.predict_f(key_f, x_pred)        
        num_particles = samples['f'].shape[0]
        keys_y = jrnd.split(key_y, num_particles)
        likelihood_params = samples.get('likelihood', {}) #{param: samples[param] for param in self.param_priors['likelihood']}
        obs_param_in_axes = jax.tree_map(lambda l: 0, likelihood_params)
        y_pred = jax.vmap(jax.jit(self.forward), 
                          in_axes=(0, 
                                   obs_param_in_axes, 
                                   0))(keys_y, 
                                    likelihood_params, 
                                    f_pred)
        return y_pred

    #
#

class FullLatentGPModelRepeatedObs(FullLatentGPModel):
    """An implementation of the full latent GP model that supports repeated 
    observations at a single input location. This class mostly inherits the 
    FullLatentGPModel, but identifies unique input locations and ensures these 
    are duplicated at likelihood evaluations.

    """

    def __init__(self, X, y, 
                 cov_fn: Callable,
                 mean_fn: Optional[Callable] = None,
                 priors: Dict = None,
                 likelihood: AbstractLikelihood = None):  
        if jnp.ndim(X) > 1:
            raise NotImplementedError(f'Repeated input models are only implemented for 1D input, ',
                                      f'but X is of shape {X.shape}')
        X = jnp.squeeze(X)
        # sort observations
        sort_idx = jnp.argsort(X, axis=0)
        X = X[sort_idx]
        y = y[sort_idx]

        # get unique values and reverse indices
        self.X, self.ix, self.rev_ix = jnp.unique(X, 
                                                  return_index=True, 
                                                  return_inverse=True)
        self.X = self.X[:, jnp.newaxis] 
        self.y = y

        # self.n = len(self.X)
        if likelihood is None:
            likelihood = Gaussian()
        self.likelihood = RepeatedObsLikelihood(base_likelihood=likelihood,
                                                inv_i=self.rev_ix)  # not unique
        self.param_priors = priors
        if mean_fn is None:
            mean_fn = Zero()
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn               

    #
    def forward(self, key, params, f):
        return self.likelihood.likelihood(params, 
                                          f, 
                                          do_reverse=False).sample(seed=key)

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
        sigma   &\sim p(\sigma) \\
        y       &\sim N(mu, cov + \sigma^2 I_n)

    All scalar parameters are sampled using Gaussian random walk MCMC.

    """

    def __init__(self, X, y,
                 cov_fn: Optional[Callable],
                 mean_fn: Callable = None,
                 priors: Dict = None,
                 **kwargs):
        self.likelihood = Gaussian()
        super().__init__(X, y, cov_fn, mean_fn, priors, **kwargs)        

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
        new_position, info_ = update_metropolis(key, logdensity, position, stepsize=mcmc_parameters.get('stepsize', 0.01))

        return GibbsState(position=new_position), None  # We return None to satisfy SMC; this needs to be filled with acceptance information

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
            psi = state.get('mean', {})
            theta = state['kernel']
            sigma = state['likelihood']['obs_noise']
            mean = self.mean_fn.mean(params=psi, x=self.X).flatten()
            cov = self.cov_fn.cross_covariance(params=theta,
                                               x=self.X,
                                               y=self.X) + (sigma ** 2 + jitter) * jnp.eye(self.X.shape[0])
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

        def logprior_fn_(state: GibbsState):
            position = getattr(state, 'position', state)
            logprob = 0
            priors_flat, _ = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
            values_flat, _ = tree_flatten(position)
            for value, dist in zip(values_flat, priors_flat):
                logprob += jnp.sum(dist.log_prob(value))
            return logprob

        #
        return logprior_fn_

    #
    def predict_f(self, key: Array, x_pred: ArrayTree, num_subsample=-1):
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
        if jnp.ndim(x_pred) == 1:
            x_pred = x_pred[:, jnp.newaxis]

        samples = self.get_monte_carlo_samples()
        flat_particles, _ = tree_flatten(samples)
        num_particles = flat_particles[0].shape[0]
        key_samples = jrnd.split(key, num_particles)

        mean_params = samples.get('mean', {})
        cov_params = samples['kernel']
        mean_params_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)
        sample_fun = lambda key, mean_params_, cov_params_, obs_noise_: sample_predictive(key,
                                                                            mean_params=mean_params_,
                                                                            cov_params=cov_params_,
                                                                            mean_fn=self.mean_fn,
                                                                            cov_fn=self.cov_fn,
                                                                            x=self.X,
                                                                            z=x_pred,
                                                                            target=self.y,
                                                                            obs_noise=obs_noise_)
        keys = jrnd.split(key, num_particles)
        target_pred = jax.vmap(jax.jit(sample_fun),
                        in_axes=(0,
                                {k: 0 for k in mean_params},
                                cov_param_in_axes,
                                0))(keys,
                                        mean_params,
                                        cov_params,
                                        samples['likelihood']['obs_noise'])

        return target_pred

    #
    def predict_y(self, key: PRNGKey, x_pred: Array):
        """Samples from the posterior predictive distribution

        Args:
            key: PRNGKey
            x_pred: Array
                The test locatons
        Returns:
            Returns samples from the posterior predictive distribution:

            y* \sim p(y* | X, y x*) = \int p(y* | f*)p(f* | f)p(f | X, y) df

        """
        if jnp.ndim(x_pred) == 1:
            x_pred = x_pred[:, jnp.newaxis]

        samples = self.get_monte_carlo_samples()
        if samples is None:
            raise AssertionError(
                f'The posterior predictive distribution can only be called after training.')

        def forward(key, params, f):
            return self.likelihood.likelihood(params, f).sample(seed=key)

        #
        key, key_f, key_y = jrnd.split(key, 3)
        f_pred = self.predict_f(key_f, x_pred)
        flat_particles, _ = tree_flatten(samples)
        num_particles = flat_particles[0].shape[0]
        keys_y = jrnd.split(key_y, num_particles)
        likelihood_params = samples['likelihood']
        y_pred = jax.vmap(jax.jit(forward),
                            in_axes=(0,
                                    0,
                                    0))(keys_y,
                                    likelihood_params,
                                    f_pred)
        return y_pred

    #


#
