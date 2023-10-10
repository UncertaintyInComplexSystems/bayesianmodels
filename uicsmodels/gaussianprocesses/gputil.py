from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
from jaxtyping import Float, Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax.random as jrnd
from uicsmodels.sampling.inference import update_correlated_gaussian, update_metropolis
from uicsmodels.gaussianprocesses.meanfunctions import Zero


def sample_predictive(key: PRNGKey,
                      mean_params: Dict,
                      cov_params: Dict,
                      mean_fn: Callable,
                      cov_fn: Callable,
                      x: Array,
                      z: Array,
                      target: Array,
                      obs_noise = None):
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
    jitter = 1e-6    

    if obs_noise is not None:
        if jnp.isscalar(obs_noise):
            diagonal_noise = obs_noise * jnp.eye(nx)
        else:
            diagonal_noise = jnp.diagflat(obs_noise)
    else:
        diagonal_noise = 0

    nx = x.shape[0]
    mean = mean_fn.mean(params=mean_params, x=z)
    Kxx = cov_fn.cross_covariance(params=cov_params, x=x, y=x)
    Kzx = cov_fn.cross_covariance(params=cov_params, x=z, y=x)
    Kzz = cov_fn.cross_covariance(params=cov_params, x=z, y=z)

    Kxx += jitter * jnp.eye(*Kxx.shape)
    Kzx += jitter * jnp.eye(*Kzx.shape)
    Kzz += jitter * jnp.eye(*Kzz.shape)

    L = jnp.linalg.cholesky(Kxx + diagonal_noise)    
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, target))   
    v = jnp.linalg.solve(L, Kzx.T)

    predictive_mean = mean + jnp.dot(Kzx, alpha)
    predictive_var = Kzz - jnp.dot(v.T, v)
    predictive_var += jitter * jnp.eye(*Kzz.shape)

    C = jnp.linalg.cholesky(predictive_var)
    u = jrnd.normal(key, shape=(len(z), ))
    samples = predictive_mean + jnp.dot(C, u)

    return samples

#
def sample_prior(key: PRNGKey, mean_params: Dict, cov_params: Dict, mean_fn: Callable, cov_fn: Callable, x: Array):
    n = x.shape[0]
    mu = mean_fn.mean(params=mean_params, x=x)
    cov = cov_fn.cross_covariance(params=cov_params,
                                    x=x,
                                    y=x) + jitter * jnp.eye(n)
    L = jnp.linalg.cholesky(cov)
    z = jrnd.normal(key, shape=(n, ))
    f = jnp.asarray(mu + jnp.dot(L, z))
    return f.flatten()

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