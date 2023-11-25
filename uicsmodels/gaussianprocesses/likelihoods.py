from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax.scipy as jsp
import distrax as dx

import jax
from jax import Array
from jaxtyping import Float
from typing import Union, Dict, Any, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
from distrax._src.distributions.distribution_from_tfp import distribution_from_tfp
from tensorflow_probability.substrates import jax as tfp

def inv_probit(x) -> Float:
    """Compute the inverse probit function.

    Args:
        x (Float[Array, "N 1"]): 
            A vector of values.
    Returns:
        Float[Array, "N 1"]: 
            The inverse probit of the input vector.
        
    """
    jitter = 1e-3  # To ensure output is in interval (0, 1).
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0))) * (1 - 2 * jitter) + jitter

#
def vec2tril(v, d):
    m = len(v)
    L_sample = jnp.zeros((d, d))
    return L_sample.at[jnp.tril_indices(d, 0)].set(v)

#
def tril2vec(L):
    d = L.shape[0]
    return L[jnp.tril_indices(d, 0)]

#
def outer_self_sum(x):
    def outer_self(x):
        return jnp.outer(x, x)
    return jnp.sum(jax.vmap(outer_self, in_axes=(0))(x), axis=0)

#
def construct_wishart(F, L=None):
    n, nu, d = F.shape
    if L is None:
        L = jnp.eye(d)

    FF = jax.vmap(outer_self_sum, in_axes=(0))(F)
    LFFL = jax.vmap(jnp.matmul,
                    in_axes=(0, None))(jax.vmap(jnp.matmul,
                                                in_axes=(None, 0))(L, FF), L.T)
    return LFFL

#
def construct_wishart_Lvec(F, L_vec):
    _, _, d = F.shape
    L = vec2tril(L_vec, d)
    return construct_wishart(F, L)

#


class AbstractLikelihood(ABC):

    @abstractmethod
    def link_function(self, f):
        pass

    #
    @abstractmethod
    def likelihood(self, params, f):
        pass

    #
    # @abstractmethod
    # def log_prob(self, params, f, y):
    #     pass

    # #
    def log_prob(self, params, f, y):
        return self.likelihood(params, f).log_prob(y)

    #
#
class RepeatedObsLikelihood(AbstractLikelihood):

    def __init__(self, base_likelihood, inv_i):
        # Note: checking if x is sorted takes O(n); this seems expensive as a check here
        self.base_likelihood = base_likelihood
        self.inv_i = inv_i

    #
    def link_function(self, f):
        return self.base_likelihood.link_function()

    #
    def likelihood(self, params, f, do_reverse=True):
        if do_reverse:
            f = f[self.inv_i]
        return self.base_likelihood.likelihood(params, f)

    #

#
class Gaussian(AbstractLikelihood):

    def link_function(self, f):
        """Identity function

        """
        return f
    
    #
    def likelihood(self, params, f):
        return dx.Normal(loc=self.link_function(f), scale=params['obs_noise'])

    #    
#
class Wishart(AbstractLikelihood):

    def __init__(self, nu, d):
        self.nu = nu
        self.d = d
    
    #

    def link_function(self, f):
        """Identity function

        """
        return f

    #
    def likelihood(self, params, f=None, Sigma=None):
        assert f is not None or Sigma is not None, 'Provide either f or Sigma'
        if Sigma is None:
            if jnp.ndim(f):
                f = jnp.reshape(f, (-1, self.nu, self.d))
            L_vec = params['L_vec']
            L = vec2tril(L_vec, self.d)
            Sigma = construct_wishart(F=f, L=L)
        mean = params.get('likelihood.mean', jnp.zeros((self.d, )))
        return dx.MultivariateNormalFullCovariance(loc=mean,
                                                   covariance_matrix=Sigma)

    #
#

class Bernoulli(AbstractLikelihood):

    def link_function(self, f):
        """Transform f \in R^D to [0,1]^D

        """
        return inv_probit(f)

    #
    def likelihood(self, params, f):
        return dx.Bernoulli(probs=self.link_function(f))

    #
#

class Poisson(AbstractLikelihood):

    def link_function(self, f):
        """Transform from f \in R to R+

        """
        return jnp.exp(f)

    #
    def likelihood(self, params, f):
        """Distrax wrapper around a Tensorflow Probability distribution

        """
        return distribution_from_tfp(tfp.distributions.Poisson(rate=self.link_function(f)))

    #