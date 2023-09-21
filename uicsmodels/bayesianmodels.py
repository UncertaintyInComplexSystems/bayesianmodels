from abc import ABC, abstractmethod

from typing import Any, Iterable, Mapping, Union
from jax import Array
from jax.random import PRNGKeyArray as PRNGKey
from typing import Union, NamedTuple, Dict, Any, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

class GibbsState(NamedTuple):

    position: ArrayTree


#

class BayesianModel(ABC):
    
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
