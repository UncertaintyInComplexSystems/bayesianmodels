from abc import ABC, abstractmethod

from typing import Any, Iterable, Mapping, Union
import jax
from jax import Array
from jax.typing import ArrayLike
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

class GibbsState(NamedTuple):

    position: ArrayTree


#

class AbstractModel(ABC):
    
    @abstractmethod
    def init_fn(self, key: PRNGKey):
        pass

    #
    @abstractmethod
    def gibbs_fn(self, key: PRNGKey, state: GibbsState, **kwargs):
        pass

    #
    def loglikelihood_fn(self):
        pass

    #
    def logprior_fn(self):
        pass

    #
    def inference(self, key: PRNGKey, mode='smc', sampling_parameters: Dict = None):
        pass

    #
    def plot_priors(self, axes=None):
        pass

    #

#
