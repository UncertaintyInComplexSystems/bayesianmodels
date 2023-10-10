from abc import ABC, abstractmethod

from uicsmodels.sampling.inference import inference_loop, smc_inference_loop

from typing import Any, Iterable, Mapping, Union
from jax import Array
import jax.random as jrnd
from jax.random import PRNGKeyArray as PRNGKey
from typing import Union, NamedTuple, Dict, Any, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

from blackjax import adaptive_tempered_smc
import blackjax.smc.resampling as resampling

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
    def smc_init_fn(self, position: ArrayTree, kwargs):
        """Simply wrap the position dictionary in a GibbsState object. 

        Args:
            position: dict
                Current assignment of the state values
            kwargs: not used in our Gibbs kernel
        Returns:
            A Gibbs state object.
        """
        return GibbsState(position)

    #
    def loglikelihood_fn(self):
        pass

    #
    def logprior_fn(self):
        pass

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

        if mode == 'gibbs-in-smc' or mode == 'mcmc-in-smc':
            if mode == 'gibbs-in-smc':
                mcmc_step_fn = self.gibbs_fn
                mcmc_init_fn = self.smc_init_fn
            elif mode == 'mcmc-in-smc':
                kernel_type = sampling_parameters.get('kernel')
                kernel_parameters = sampling_parameters.get('kernel_parameters')
                mcmc_step_fn = kernel_type.build_kerel(),
                mcmc_init_fn = kernel_type.init,
            
            smc = adaptive_tempered_smc(
                logprior_fn=self.logprior_fn(),
                loglikelihood_fn=self.loglikelihood_fn(),
                mcmc_step_fn=mcmc_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                mcmc_parameters=sampling_parameters.get('mcmc_parameters', dict()),
                resampling_fn=resampling.systematic,
                target_ess=sampling_parameters.get('target_ess', 0.5),
                num_mcmc_steps=sampling_parameters.get('num_mcmc_steps', 50)
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
                initial_state = kernel.init(self.init_fn(key_init).position)

            states = inference_loop(key_inference,
                                    step_fn,
                                    initial_state,
                                    num_burn + num_samples)
            self.states = states #if mode == 'gibbs' else states.position
            return states
        else:
            raise NotImplementedError(f'{mode} is not implemented as inference method. Valid options are:\ngibbs-in-smc\ngibbs\nmcmc-in-smc\nmcmc')

    #
    def get_monte_carlo_samples(self):
        if hasattr(self, 'particles'):
            return self.particles.particles
        elif hasattr(self, 'states'):
            return self.states.initial_position
        else:
            raise NotImplementedError('Please first run inference')

    #
    def plot_priors(self, axes=None):
        pass

    #

#
