import jax
import blackjax
import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

from blackjax import elliptical_slice, rmh

__all__ = ['inference_loop', 'smc_inference_loop', 'update_correlated_gaussian', 'update_gaussian_process_cov_params']


def smc_inference_loop(rng_key: PRNGKey, smc_kernel: Callable, initial_state):
    """The sequential Monte Carlo loop.

    Args:
        key: 
            The jax.random.PRNGKey
        smc_kernel: 
            The SMC kernel object (e.g. SMC, tempered SMC or 
                    adaptive-tempered SMC)
        initial_state: 
            The initial state for each particle
    Returns:
        n_iter: int
            The number of tempering steps
        final_state: 
            The final state of each of the particles
        info: SMCinfo
            the SMC info object which contains the log marginal likelihood of 
              the model (for model comparison)
        
    """

    def cond(carry):
        _, state, *_k = carry
        return state.lmbda < 1

    #
    @jax.jit
    def one_step(carry):                
        i, state, k, curr_log_likelihood = carry
        k, subk = jax.random.split(k, 2)
        state, info = smc_kernel(subk, state)
        return i + 1, state, k, curr_log_likelihood + info.log_likelihood_increment

    #
    n_iter, final_state, _, info = jax.lax.while_loop(cond, one_step, 
                                                      (0, initial_state, rng_key, 0))
    return n_iter, final_state, info

#
def inference_loop(rng_key: PRNGKey, kernel: Callable, initial_state, num_samples: int):
    """The MCMC inference loop.

    The inference loop takes an initial state, a step function, and the desired
    number of samples. It returns a list of states.
    
    Args:
        rng_key: 
            The jax.random.PRNGKey
        kernel: Callable
            A step function that takes a state and returns a new state
        initial_state: 
            The initial state of the sampler
        num_samples: int
            The number of samples to obtain
    Returns: 
        GibbsState [List, "num_samples"]

    """
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

#
def update_correlated_gaussian(key, f_current, loglikelihood_fn_, mean, cov):
    elliptical_slice_sampler = elliptical_slice(loglikelihood_fn_,
                                    mean=mean,
                                    cov=cov)

    ess_state = elliptical_slice_sampler.init(f_current)    
    ess_state, ess_info = elliptical_slice_sampler.step(key, ess_state)
    return ess_state.position, ess_info

#
def update_metropolis(key, logdensity: Callable, variables: Dict, stepsize: Float = 0.01):
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
    m = 0
    for varval in variables.values():
        m += varval.shape[0] if varval.shape else 1

    kernel = rmh(logdensity, sigma=stepsize * jnp.eye(m))
    rmh_state = kernel.init(variables)
    rmh_state, rmh_info = kernel.step(key, rmh_state)
    return rmh_state.position, rmh_info

#
def update_gaussian_process(key: PRNGKey, f_current: Array, loglikelihood_fn: Callable, X: Array,
                            mean_fn: Callable = Zero(),
                            cov_fn: Callable = jk.RBF(),
                            mean_params: Dict = None,
                            cov_params: Dict = None):
    n = X.shape[0]
    mean = mean_fn.mean(params=mean_params, x=X)
    cov = cov_fn.cross_covariance(params=cov_params, x=X, y=X) + jitter * jnp.eye(n)
    return update_correlated_gaussian(key, f_current, loglikelihood_fn, mean, cov)

#
def update_gaussian_process_cov_params(key: PRNGKey, X: Array,
                                       f: Array,
                                       mean_fn: Callable = Zero(),
                                       cov_fn: Callable = jk.RBF(),
                                       mean_params: Dict = None,
                                       cov_params: Dict = None,
                                       hyperpriors: Dict = None):


    n = X.shape[0]
    mu = mean_fn.mean(params=mean_params, x=X)
    def logdensity_fn_(cov_params_):
        log_pdf = 0
        for param, val in cov_params_.items():
            log_pdf += jnp.sum(hyperpriors[param].log_prob(val))
        cov_ = cov_fn.cross_covariance(params=cov_params_, x=X, y=X) + jitter * jnp.eye(n)
        log_pdf += dx.MultivariateNormalFullCovariance(mu, cov_).log_prob(f)
        return log_pdf

    #
    return update_metropolis(key, logdensity_fn_, cov_params, stepsize=0.1)

#