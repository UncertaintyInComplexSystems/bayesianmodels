from uicsmodels.bayesianmodels import AbstractModel
from uicsmodels.gpmodels.meanfunctions import Zero

from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional

import jax.numpy as jnp



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
        print('GP model initialized')
        # TODO:
        # - assert whether all trainable parameters have been assigned priors
        # - add defaults/fixed values for parameters without prior