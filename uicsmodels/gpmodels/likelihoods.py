from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax.scipy as jsp
import distrax as dx
from blackjax.types import Array, PRNGKey, ArrayTree

from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional
from jaxtyping import Array, Float


def inv_probit(x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
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