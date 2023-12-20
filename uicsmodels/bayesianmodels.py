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

from uicsmodels.sampling.inference import inference_loop, smc_inference_loop
from uicsmodels.sampling.inference import update_metropolis

from abc import ABC, abstractmethod


import jax
import jax.numpy as jnp
from jax import Array
import jax.random as jrnd
from jax.random import PRNGKeyArray as PRNGKey
from typing import Any, Union, NamedTuple, Dict, Any, Iterable, Mapping, Callable
from jaxtyping import Float
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

from blackjax import adaptive_tempered_smc, rmh
import blackjax.smc.resampling as resampling

from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

__all__ = ['GibbsState', 'BayesianModel']

class GibbsState(NamedTuple):

    position: ArrayTree


#

class BayesianModel(ABC):
    
    def init_fn(self, key: Array, num_particles: int = 1):
        """Initial state for MCMC/SMC.

        This function initializes all highest level latent variables. Children
        of this class need to implement initialization of intermediate latent
        variables according to the structure of the hierarchical model.

        Args:
            key: PRNGKey
            num_particles: int
                Number of particles to initialize a state for
        Returns:
            GibbsState

        """

        priors_flat, priors_treedef = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
        samples = list()
        for prior in priors_flat:
            key, subkey = jrnd.split(key)
            samples.append(prior.sample(seed=subkey, sample_shape=(num_particles,)))

        initial_position = jax.tree_util.tree_unflatten(priors_treedef, samples)
        if num_particles == 1:
            initial_position = tree_map(lambda x: jnp.squeeze(x), initial_position)
        return GibbsState(position=initial_position)

    #  
    def sample_from_prior(self, key, num_samples=1):
        return self.init_fn(key, num_particles=num_samples)

    #
    def smc_init_fn(self, position: ArrayTree, kwargs):
        """Simply wrap the position dictionary in a GibbsState object. 

        Args:
            position: dict
                Current assignment of the state values
            kwargs: not used in our Gibbs kernel
        Returns:
            A Gibbs state object.
        """
        if isinstance(position, GibbsState):
            return position
        return GibbsState(position)

    #
    def loglikelihood_fn(self):
        pass

    #
    def logprior_fn(self) -> Callable:
        """Returns the log-prior function for the model given a state.

        This default logprior assumes a non-hierarchical model. If a 
        hierarchical model is used, the mode should implement its own 
        logprior_fn.

        Args:
            None
        Returns:
            A function that computes the log-prior of the model given a state.

        """

        def logprior_fn_(state: GibbsState):
            position = getattr(state, 'position', state)
            logprob = 0
            priors_flat, _ = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
            values_flat, _ = tree_flatten(position)
            for value, dist in zip(values_flat, priors_flat):
                logprob += jnp.sum(dist.log_prob(value))
            return logprob

        #
        return logprior_fn_

    #
    def inference(self, key: PRNGKey, mode='gibbs-in-smc', sampling_parameters: Dict = None):
        """A wrapper for training the GP model.

        An interface to Blackjax' MCMC or SMC inference loops, tailored to the
        current Bayesian model.

        Args:
            key: jrnd.KeyArray
                The random number seed, will be split into initialisation and
                inference.
            mode: {'mcmc', 'smc'}
                The desired inference approach. Defaults to SMC, which is
                generally prefered.
            sampling_parameters: dict
                Optional settings with defaults for the inference procedure.

        Returns:
            Depending on 'mode':
                smc:
                    num_iter: int
                        Number of tempering iterations.
                    particles:
                        The final states the SMC particles (at T=1).
                    marginal_likelihood: float
                        The approximated marginal likelihood of the model.
                mcmc:
                    states:
                        The MCMC states (including burn-in states).

        """
        if sampling_parameters is None:
            sampling_parameters = dict()

        key, key_init, key_inference = jrnd.split(key, 3)

        if mode == 'gibbs-in-smc' and not (hasattr(self, 'gibbs_fn') and callable(self.gibbs_fn)):
            sigma = 0.01
            print(f'No Gibbs kernel available, defaulting to Random Walk Metropolis MCMC, sigma = {sigma:.2f}') 
            mode = 'mcmc-in-smc'
            priors_flat, _ = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
            m = 0            
            for prior in priors_flat:
                m += jnp.prod(jnp.asarray(prior.batch_shape)) if prior.batch_shape else 1
            sampling_parameters['kernel'] = rmh
            sampling_parameters['kernel_parameters'] = dict(sigma=sigma*jnp.eye(m))


        if mode == 'gibbs-in-smc' or mode == 'mcmc-in-smc':
            if mode == 'gibbs-in-smc':
                mcmc_step_fn = self.gibbs_fn
                mcmc_init_fn = self.smc_init_fn
            elif mode == 'mcmc-in-smc':

                # Set up tempered MCMC kernel
                def mcmc_step_fn(key, state, temperature, **mcmc_parameters):
                    def apply_mcmc_kernel(key, logdensity, pos):
                        kernel = kernel_type(logdensity, **kernel_parameters)
                        state_ = kernel.init(pos)
                        state_, info = kernel.step(key, state_)
                        return state_.position, info
                    
                    #
                    position = state.position.copy()
                    loglikelihood_fn_ = self.loglikelihood_fn()
                    logprior_fn_ = self.logprior_fn()
                    logdensity = lambda state: temperature * loglikelihood_fn_(state) + logprior_fn_(state)
                    new_position, info_ = apply_mcmc_kernel(key, logdensity, position)
                    return GibbsState(position=new_position), None  

                #
                kernel_type = sampling_parameters.get('kernel')
                kernel_parameters = sampling_parameters.get('kernel_parameters')
                mcmc_init_fn = self.smc_init_fn
            
            #Set up adaptive tempered SMC
            smc = adaptive_tempered_smc(
                logprior_fn=self.logprior_fn(),
                loglikelihood_fn=self.loglikelihood_fn(),
                mcmc_step_fn=mcmc_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                mcmc_parameters=sampling_parameters.get('mcmc_parameters', dict()),
                resampling_fn=resampling.systematic,
                target_ess=sampling_parameters.get('target_ess', 0.5),
                num_mcmc_steps=sampling_parameters.get('num_mcmc_steps', 100)
            )
            num_particles = sampling_parameters.get('num_particles', 1_000)
            initial_particles = self.init_fn(key_init,
                                             num_particles=num_particles)
            initial_smc_state = smc.init(initial_particles.position)
            num_iter, particles, marginal_likelihood = smc_inference_loop(key_inference,
                                                                          smc.step,
                                                                          initial_smc_state)
            self.particles = particles
            self.marginal_likelihood = marginal_likelihood
            return particles, num_iter, marginal_likelihood
        elif mode == 'gibbs' or mode == 'mcmc':
            num_burn = sampling_parameters.get('num_burn', 10_000)
            num_samples = sampling_parameters.get('num_samples', 10_000)
            num_thin = sampling_parameters.get('num_thin', 1)

            if mode == 'gibbs':
                step_fn = self.gibbs_fn
                initial_state = self.init_fn(key_init)
            elif mode == 'mcmc':
                kernel_type = sampling_parameters.get('kernel')
                kernel_parameters = sampling_parameters.get('kernel_parameters')
                loglikelihood_fn = self.loglikelihood_fn()
                logprior_fn = self.logprior_fn()

                logdensity_fn = lambda state: loglikelihood_fn(state) + logprior_fn(state)
                kernel = kernel_type(logdensity_fn, **kernel_parameters)
                step_fn = kernel.step
                initial_state = sampling_parameters.get('initial_state', kernel.init(self.init_fn(key_init).position))

            states = inference_loop(key_inference,
                                    step_fn,
                                    initial_state,
                                    num_burn + num_samples)

            # remove burn-in
            self.states = tree_map(lambda x: x[num_burn::num_thin], states)
            return states
        else:
            raise NotImplementedError(f'{mode} is not implemented as inference method. Valid options are:\ngibbs-in-smc\ngibbs\nmcmc-in-smc\nmcmc')

    #
    def get_monte_carlo_samples(self, mode='smc'):
        if mode == 'smc' and hasattr(self, 'particles'):
            return self.particles.particles
        elif mode == 'mcmc' and hasattr(self, 'states'):
            return self.states.position
        raise ValueError('No inference has been performed')

    #
    def plot_priors(self, axes=None):
        pass

    #

#
