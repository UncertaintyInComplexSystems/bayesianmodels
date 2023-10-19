from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax.scipy as jsp
import distrax as dx

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


class AbstractLikelihood(ABC):

    @abstractmethod
    def link_function(self, f):
        pass

    #
    @abstractmethod
    def likelihood(self, params, f):
        pass

    #
    @abstractmethod
    def log_prob(self, params, f, y):
        pass

    #
    def log_prob(self, params, f, y):
        return self.likelihood(params, f).log_prob(y)

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