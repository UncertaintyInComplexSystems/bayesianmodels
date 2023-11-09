import matplotlib.pyplot as plt

import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk

from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes
config.update("jax_default_device", jax.devices()[0])

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from uicsmodels.gaussianprocesses.kernels import Discontinuous
from uicsmodels.gaussianprocesses.fullgp import FullMarginalGPModel

class DiscontinuityDesign():

    def __init__(self, X: Array, y: Array, base_cov_fn: Callable, x0: Float, priors: Dict):
        if jnp.ndim(X) == 1:
            X = X[:, jnp.newaxis]
        self.X = X
        self.y = y
        if not isinstance(base_cov_fn, list):
            base_cov_fn = [base_cov_fn]
        self.base_cov_fns = base_cov_fn
        self.x0 = x0
        self.priors = priors

    #
#


class RegressionDiscontinuity(DiscontinuityDesign):


    def __init__(self, X: Array, y: Array, base_cov_fn: Callable, x0: Float, priors: Dict):
        super().__init__(X, y, base_cov_fn, x0, priors)

    #
    def train(self, key, num_particles: int = 1000, num_mcmc_steps: int = 100):
        self.m1 = list()
        self.m0 = list()
        for priors, base_cov_fn in zip(self.priors, self.base_cov_fns):
            key, key_m1, key_m0 = jrnd.split(key, 3)
            disc_cov_fn = Discontinuous(base_cov_fn, x0=self.x0)
            m1 = FullMarginalGPModel(self.X, self.y, cov_fn=disc_cov_fn, priors=priors)
            _ = m1.inference(key_m1,
                             mode='gibbs-in-smc',
                             sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

            m0 = FullMarginalGPModel(self.X, self.y, cov_fn=base_cov_fn, priors=priors)
            _ = m0.inference(key_m0,
                             mode='gibbs-in-smc',
                             sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

            self.m1.append(m1)
            self.m0.append(m0)

        if len(self.base_cov_fns) == 1:
            self.m1 = self.m1[0]
            self.m0 = self.m0[0]

    #
    def effect_size_distribution(self, key):
        def get_effect_size(model):
            epsilon = 1e-6
            x_at_intervention = jnp.array([self.x0 - epsilon, self.x0 + epsilon])[:, jnp.newaxis]
            pred_at_intervention = model.predict_f(key, x_pred=x_at_intervention)
            return pred_at_intervention[:, 1] - pred_at_intervention[:, 0]

        #
        if not hasattr(self, 'm1'):
            raise AssertionError('The model must be trained first.')
        
        if isinstance(self.m1, list):
            effect_size_samples = list()
            for m1 in self.m1:
                effect_size_samples.append(get_effect_size(m1))
            return effect_size_samples
        else:
            return get_effect_size(self.m1)

    #
#

class InterruptedTimeseries(DiscontinuityDesign):

    def __init__(self, X: Array, y: Array, cov_fn: Callable, x0: Float, priors: Dict):
        super().__init__(X, y, cov_fn, x0, priors)

    #
    def train(self, key, num_particles: int = 1000, num_mcmc_steps: int = 100):
        self.particles = list()
        self.lml = list()
        self.m = list()
        for cov_fn, priors in zip(self.base_cov_fns, self.priors):
            key, subkey = jrnd.split(key)
            ix = self.X[:, 0] < self.x0
            m = FullMarginalGPModel(self.X[ix, :], self.y[ix], cov_fn=cov_fn, priors=priors)
            _ = m.inference(subkey,
                            mode='gibbs-in-smc',
                            sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))        
            self.m.append(m)

        if len(self.base_cov_fns) == 1:
            self.m = self.m[0]

    #
    def counterfactual(self, key: PRNGKey, x_pred: Array, mode: str ='y'):
        if mode not in ['f', 'y']:
            raise ValueError(f'Mode must be \'f\' or \'y\', but {mode} was provided.')
        predictions = list()
        for i, cov_fn in enumerate(self.base_cov_fns):
            key, subkey = jrnd.split(key)
            if mode == 'f':
                predictions.append(self.m[i].predict_f(key, x_pred))
            elif mode == 'y':
                predictions.append(self.m[i].predict_y(key, x_pred))
        if len(self.base_cov_fns) == 1:
            predictions = predictions[0]
        
        return predictions

    #


#

            










