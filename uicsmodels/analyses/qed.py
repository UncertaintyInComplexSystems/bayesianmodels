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
        self.m1_particles = list()
        self.m0_particles = list()
        self.m1_lml = list()
        self.m0_lml = list()
        self.m1 = list()
        self.m0 = list()
        for base_cov_fn in self.base_cov_fns:
            key, key_m1, key_m0 = jrnd.split(key, 3)
            disc_cov_fn = Discontinuous(base_cov_fn, x0=self.x0)
            m1 = FullMarginalGPModel(self.X, self.y, cov_fn=disc_cov_fn, priors=self.priors)
            m1_particles, _, m1_lml = m1.inference(key_m1,
                                                   mode='gibbs-in-smc',
                                                   sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

            m0 = FullMarginalGPModel(self.X, self.y, cov_fn=base_cov_fn, priors=self.priors)
            m0_particles, _, m0_lml = m0.inference(key_m0,
                                                   mode='gibbs-in-smc',
                                                   sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

            self.m1.append(m1)
            self.m0.append(m0)
            self.m1_particles.append(m1_particles)
            self.m1_lml.append(m1_lml)
            self.m0_particles.append(m0_particles)
            self.m0_lml.append(m0_lml)

        if len(self.base_cov_fns) == 1:
            self.m1_particles = self.m1_particles[0]
            self.m0_particles = self.m0_particles[0]
            self.m1_lml = self.m1_lml[0]
            self.m0_lml = self.m0_lml[0]
            self.m1 = self.m1[0]
            self.m0 = self.m0[0]

        return (self.m0_particles, self.m1_particles), (self.m0_lml, self.m1_lml)

    #
#

class InterruptedTimeseries(DiscontinuityDesign):

    def __init__(self, X: Array, y: Array, base_cov_fn: Callable, x0: Float, priors: Dict):
        super().__init__(X, y, base_cov_fn, x0, priors)

    #



            










