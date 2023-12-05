# Copyright 2023- The Uncertainty in Complex Systems contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
import jaxkern as jk
import jax.numpy as jnp
import jax.random as jrnd
from typing import Callable, Tuple, Union, NamedTuple, Dict, Any, Optional, List
from jaxtyping import Array, Float
from jax.nn import softmax

__all__ = ['DefaultingKernel', 'Brownian', 'SpectralMixture', 'Discontinuous']

def centered_softmax(beta):
    return softmax(jnp.insert(beta, 0, 0.0))

#
class DefaultingKernel(jk.base.AbstractKernel):
    """Constructs a kernel with default values for a subset of its hyperparameters.

    For example, we might use a jk.RBF() base kernel, but force its variance
    to be 1.0. Using 
    
    kernel = DefaultingKernel(base_kernel=jk.RBF(), defaults=dict(variance=1.0))

    this can be accomplished.
    
    """

    def __init__(self, base_kernel, defaults: Dict) -> None:
        self.base_kernel = base_kernel
        self.defaults = defaults
        self._stationary = base_kernel._stationary
        self.name = 'Default ' + base_kernel.name

    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all
        correlations between observations on different sides of the threshold
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        params = {**params, **self.defaults}
        K = self.base_kernel.cross_covariance(params, x, y)
        return K

    #
    def init_params(self, key: Array) -> dict:
        self.base_kernel.init_params(key)

    #
#

class Brownian(jk.base.AbstractKernel):

    """A GP covariance function for Brownian motion

    """

    def __init__(self, active_dims: Optional[List[int]] = None, name: Optional[str] = "Brownian motion") -> None:
        self.active_dims = active_dims
        self.name = name
        self._stationary = False
        self.ndims = 1 if not self.active_dims else len(self.active_dims)

    #
    def __call__(self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y) -> Float[Array, "1"]:
        # see https://github.com/SheffieldML/GPy/blob/devel/GPy/kern/src/brownian.py
        x = self.slice_input(x)
        y = self.slice_input(y)
        n_x = x.shape[0]
        n_y = y.shape[0]
        x_mat = jnp.tile(jnp.squeeze(x), (n_y, 1))
        y_mat = jnp.tile(jnp.squeeze(y), (n_x, 1)).T        
        return jnp.squeeze((params['variance'] * jnp.where(jnp.sign(x_mat)==jnp.sign(y_mat), jnp.fmin(jnp.abs(x_mat), jnp.abs(y_mat)), 0)).T)

    #
    def init_params(self, key: Array) -> dict:
        params = {           
            "variance": jnp.array([1.0]),
        }
        return jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x), params)

    #
#

class SpectralMixture(jk.base.AbstractKernel):

    def __init__(self, 
                 active_dims: Optional[List[int]] = None, 
                 name: Optional[str] = 'Spectral mixture') -> None:
        # Note: we don't want to inherit here.
        self.name = name
        self._stationary = True
        self.active_dims = active_dims
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def __euclidean_distance_einsum(self, X, Y):
        """Efficiently calculates the euclidean distance
        between two vectors using Numpys einsum function.

        Parameters
        ----------
        X : array, (n_samples x d_dimensions)
        Y : array, (n_samples x d_dimensions)

        Returns
        -------
        D : array, (n_samples, n_samples)
        """
        
        XX = jnp.einsum('ij,ij->i', X, X)[:, jnp.newaxis]
        YY = jnp.einsum('ij,ij->i', Y, Y)
        XY = 2 * jnp.dot(X, Y.T)
        return  XX + YY - XY

    #

    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The spectral mixture kernel is defined as

        .. math::

            \mu_q ~ N(.,.), for q = 1..Q
            log \nu_q ~ N(., .), for q = 1..Q
            beta_q ~ N(., .), for q = 2..Q
            w = softmax_centered(beta)
            k(tau) = \sum_{q=1}^Q w_q \prod_{i=1}^D \exp[-2pi^2 tau_i^2 \nu_q^({i})] cos(2pi tau_i \mu_q^{(i)}),

            with tau = x - y.        

        Importantly, we enforce identifiability of the posterior of these 
        parameters in two ways. First, w is drawn from a centered softmax, which
        ensures w_q > 0 and \sum w_q = 1, but in addition the weights are 
        anchored around the first element which is always forced to zero (i.e. 
        we sample only beta_2, ..., beta_Q, and set beta_1 = 0). Second, we sort
        the vector of means so that the smallest frequency component is always
        the first.

        This does not yet work in higher dimensions, as the sorting needs to be 
        defined there.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an n x m matrix of cross covariances (n = len(x), m = len(y))
        """

        def compsum(res, el):
            w_, mu_, nu_ = el
            res = res + w_ * jnp.exp(-2*jnp.pi**2 * tau**2 * nu_) * jnp.cos(2*jnp.pi * tau * mu_)
            return res, el

        #        
        tau = jnp.sqrt(self.__euclidean_distance_einsum(x, y))
        beta = params['beta']
        w = centered_softmax(beta)
        mu = params['mu']
        # To solve the identifiability issue in mixture models, we sort according to the means:
        mu = jnp.sort(mu)
        nu = params['nu']     

        K, _ = jax.lax.scan(compsum, jnp.zeros((x.shape[0], y.shape[0])), (w, mu, nu))        
        return K

    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #

#


class Discontinuous(jk.base.AbstractKernel):
    # todo: this implementation contains redundant computation and doesn't scale 
    # well to multiple change points; refactor

    def __init__(self, base_kernel, x0: Float = 0.0) -> None:
        self.base_kernel = base_kernel
        self.x0 = x0
        self.name = 'Discontinuous'
        
    #
    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        return self.cross_covariance(params, x, y)

    #
    def cross_covariance(self, params: Dict, x, y):
        """Computes the discontinuous cross-covariance.

        The bread-and-butter of the discontinuity analysis removes all 
        correlations between observations on different sides of the threshold 
        x0.

        Args:
            params: Parameters of the base kernel.
            x, y: points to determine covariance for
        Returns:
            an nxm matrix of cross covariances (n = len(x), m = len(y))
        """
        
        def check_side(x_, y_):
            return 1.0*jnp.logical_or(jnp.logical_and(jnp.less(x_, self.x0), 
                                                      jnp.less(y_, self.x0)), 
                                      jnp.logical_and(jnp.greater_equal(x_, self.x0), 
                                                      jnp.greater_equal(y_, self.x0)))

        #
        K = self.base_kernel.cross_covariance(params, x, y)
        mask = jax.vmap(lambda x_: jax.vmap(lambda y_: check_side(x_, y_))(y))(x)
        return jnp.multiply(K, mask.squeeze())

    #
    def init_params(self, key: jrnd.KeyArray) -> dict:
        self.base_kernel.init_params(key)

    #

#