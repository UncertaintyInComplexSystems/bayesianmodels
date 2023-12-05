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

from uicsmodels.gaussianprocesses.likelihoods import AbstractLikelihood, Gaussian
from uicsmodels.gaussianprocesses.meanfunctions import Zero
from uicsmodels.bayesianmodels import GibbsState, BayesianModel
from uicsmodels.gaussianprocesses.gputil import sample_prior, sample_predictive, update_gaussian_process, update_gaussian_process_cov_params, update_gaussian_process_mean_params

from jax import Array
from jaxtyping import Float
# from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

import jax
import distrax as dx
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd

from jax.tree_util import tree_flatten, tree_unflatten

from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

__all__ = ['FullLatentHSGPModel']

class FullLatentHSGPModel(BayesianModel):

    """A Gaussian process model for heteroskedastic noise.

    In this model, the observation noise of a Gaussian likelihood is itself
    assumed to follow a (transformed) Gaussian process. The generative model is
    based on the model by Benton et al. (2022). In brief, it is defined as 
    follows:

    v ~ GP(0, k_v)
    f ~ GP(0, k_f)
    \sigma_i = sqrt(exp(v(x_i)))
    y_i  ~ N(f(x_i), \sigma_i)


    Benton, G., Maddox, W. J., & Wilson, A. G. (2022). Volatility based kernels 
    and moving average means for accurate forecasting with Gaussian processes. 
    http://arxiv.org/abs/2207.06544



    """

    def __init__(self, X, y,
                 cov_fns: Dict = None,
                 mean_fns: Dict = None,
                 priors: Dict = None,
                 likelihood: AbstractLikelihood = None) -> None:
        if jnp.ndim(X) == 1:
            X = X[:, jnp.newaxis]
        self.X, self.y = X, y
        self.n = self.X.shape[0]
        if mean_fns is None:
            mean_fns = dict()
            for name in ['f', 'v']:
                mean_fns[name] = Zero()
        self.mean_fns = mean_fns
        self.cov_fns = cov_fns
        self.param_priors = priors
        if likelihood is None:
            likelihood = Gaussian()
        self.likelihood = likelihood

    # 
    def init_fn(self, key: PRNGKey, num_particles=1):
        
        initial_position = dict()

        priors_flat, priors_treedef = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
        samples = list()
        for prior in priors_flat:
            key, subkey = jrnd.split(key)
            samples.append(prior.sample(seed=subkey, sample_shape=(num_particles,)))

        initial_position = jax.tree_util.tree_unflatten(priors_treedef, samples) 

        # sample latent gps
        for name in ['v', 'f']:
            mean_params = initial_position[name].get('mean', {})
            cov_params = initial_position[name].get('kernel', {})
            mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
            cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)

            if num_particles > 1:
                keys = jrnd.split(key, num_particles)                
                sample_fun = lambda key_, mean_params_, cov_params_: sample_prior(key=key_, 
                                                                                  mean_params=mean_params_,
                                                                                  cov_params=cov_params_,
                                                                                  mean_fn=self.mean_fns[name],
                                                                                  cov_fn=self.cov_fns[name], 
                                                                                  x=self.X)
                initial_position[name]['gp'] = jax.vmap(sample_fun,
                                                  in_axes=(0,
                                                           mean_param_in_axes,
                                                           cov_param_in_axes))(keys, mean_params, cov_params)
            else:
                key, subkey = jrnd.split(key)
                initial_position[name]['gp'] = sample_prior(subkey, self.mean_fns[name], 
                                                       self.cov_fns[name], mean_params, 
                                                       cov_params, self.X)

        return GibbsState(initial_position)

    #
    def gibbs_fn(self, key: PRNGKey, state: GibbsState, temperature: Float = 1.0, **mcmc_parameters):
        """The Gibbs sweep updating all parameters once and returning a new state


        """
        position = state.position.copy()        
        def loglikelihood_fn_v_(v_):
            return temperature * jnp.sum(jnp.array([self.likelihood.log_prob(params=dict(obs_noise=jnp.sqrt(jnp.exp(v_[i]))),
                                                                             f=position['f']['gp'][i],
                                                                             y=self.y[i]) for i in range(self.n)]))
        
        def loglikelihood_fn_f_(f_):
            return temperature * jnp.sum(jnp.array([self.likelihood.log_prob(params=dict(obs_noise=jnp.sqrt(jnp.exp(position['v']['gp'][i]))),
                                                                             f=f_[i],
                                                                             y=self.y[i]) for i in range(self.n)]))  

        loglikelihood_fns = dict(v=loglikelihood_fn_v_, f=loglikelihood_fn_f_)  

        # Sequentially update v and f and their respective covariance parameters
        for name in ['v', 'f']:   
            mean_params = position[name].get('mean', {})
            cov_params = position[name].get('kernel', {})

            key, subkey = jrnd.split(key)
            position[name]['gp'], gp_info = update_gaussian_process(subkey,
                                                            position[name]['gp'],
                                                            loglikelihood_fns[name],
                                                            self.X,
                                                            mean_fn=self.mean_fns[name],
                                                            cov_fn=self.cov_fns[name],
                                                            mean_params=mean_params,
                                                            cov_params=cov_params)

            # update hyperparameters for v: p(v | theta_v)p(\theta_v)
            if len(cov_params):
                key, subkey = jrnd.split(key)
                sub_state, sub_info = update_gaussian_process_cov_params(subkey,
                                                                         self.X,
                                                                         position[name]['gp'],
                                                                         mean_fn=self.mean_fns[name],
                                                                         cov_fn=self.cov_fns[name],
                                                                         mean_params=mean_params,
                                                                         cov_params=cov_params,
                                                                         hyperpriors=self.param_priors[name]['kernel'])
                position[name]['kernel'] = sub_state

            #
            if len(mean_params):
                cov_params = position[name].get('kernel', {})
                key, subkey = jrnd.split(key)
                sub_state, sub_info = update_gaussian_process_mean_params(subkey,
                                                                        self.X,
                                                                        position[name]['gp'],
                                                                        mean_fn=self.mean_fns[name],
                                                                        cov_fn=self.cov_fns[name],
                                                                        mean_params=mean_params,
                                                                        cov_params=cov_params,
                                                                        hyperpriors=self.param_priors[f'mean_{name}'])

                position[name]['mean'] = sub_state

            #
        #
        return GibbsState(position=position), None  # todo: incorporate sampling info

    #
    def loglikelihood_fn(self) -> Callable:
        """Returns the log-likelihood function for the model given a state.

        Args:
            None

        Returns:
            A function that computes the log-likelihood of the model given a
            state.
        """

        def loglikelihood_fn_(state: GibbsState) -> Float:
            """The log-likelihood is given by

            \log p(Y | F, V)    &= \sum_{t=1}^T \log p(y(t) | f(t), sqrt(V(t)))
                                &= \sum_{t=1}^T \log \mathcal{N}(y(t) | f(t), sqrt(V(t)))

            """
            position = getattr(state, 'position', state)
            f, v = position['f']['gp'], position['v']['gp']
            return jnp.sum(jnp.array([self.likelihood.log_prob(params=dict(obs_noise=jnp.sqrt(jnp.exp(v[i]))), f=f[i], y=self.y[i]) for i in range(self.n)]))

        #
        return loglikelihood_fn_

    #
    def logprior_fn(self) -> Callable:
        jitter = 1e-6
        def logprior_fn_(state: GibbsState) -> Float:
            position = getattr(state, 'position', state)
            logprob = 0
            for component, params in self.param_priors.items():
                for param, dist in params.items():
                    logprob += jnp.sum(dist.log_prob(position[param]))

            for name in ['f', 'v']:
                mu = self.mean_fns[name]
                kernel = self.cov_fns[name]                
                mean_params = position[name].get('mean', {})
                mean = mu.mean(params=mean_params, x=self.X).flatten()  # this will break with multitask GPs
                theta = position[name]['kernel']
                cov = kernel.cross_covariance(params=theta,
                                                   x=self.X,
                                                   y=self.X)

                logprob += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position[name]['gp'])

        #
        return logprior_fn_

    #
    def __predict_latent(self, key: PRNGKey, x_pred: Array, latent):

        num_particles = self.get_monte_carlo_samples()[latent]['gp'].shape[0]
        key_samples = jrnd.split(key, num_particles)

        mean_params = self.get_monte_carlo_samples()[latent].get('mean', {})
        cov_params = self.get_monte_carlo_samples()[latent]['kernel']
        mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)

        sample_fun = lambda key, mean_params, cov_params, target: sample_predictive(key, 
                                                                            mean_params=mean_params, 
                                                                            cov_params=cov_params, 
                                                                            mean_fn=self.mean_fns[latent], # TODO: how to get this from 'latent'?
                                                                            cov_fn=self.cov_fns[latent], 
                                                                            x=self.X, 
                                                                            z=x_pred, 
                                                                            target=target)
        keys = jrnd.split(key, num_particles)
        target_pred = jax.vmap(jax.jit(sample_fun), 
                        in_axes=(0, 
                        mean_param_in_axes, 
                            cov_param_in_axes, 
                                0))(keys, 
                                    mean_params, 
                                    cov_params, 
                                    self.get_monte_carlo_samples()[latent]['gp'])

        return target_pred
    
    #
    def predict_f(self, key: PRNGKey, x_pred: Array):
        return self.__predict_latent(key, x_pred, 'f')

    #
    def predict_v(self, key: PRNGKey, x_pred: Array):
        return self.__predict_latent(key, x_pred, 'v')

    #
    def predict_y(self, key: PRNGKey, x_pred: Array):
        """

        """
        assert hasattr(self, 'particles'), 'No particles available'

        def forward(key, f, V):
            return dx.Normal(loc=f, scale=V).sample(seed=key)

        #
        key, key_f, key_v, key_y = jrnd.split(key, 4)
        f_pred = self.predict_f(key_f, x_pred)
        v_pred = jnp.sqrt(jnp.exp(self.predict_v(key_v, x_pred)))
        num_particles = self.get_monte_carlo_samples()['f']['gp'].shape[0]
        keys_y = jrnd.split(key_y, num_particles)
        y_pred = jax.vmap(forward, in_axes=(0, 0, 0))(keys_y, f_pred, v_pred)
        return y_pred

    #
#