from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

import jax
import distrax as dx
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd

from uicsmodels.gaussianprocesses.likelihoods import AbstractLikelihood, Gaussian
from uicsmodels.gaussianprocesses.meanfunctions import Zero
from uicsmodels.bayesianmodels import GibbsState, BayesianModel
from uicsmodels.gaussianprocesses.gputil import sample_prior, sample_predictive, update_gaussian_process, update_gaussian_process_cov_params

class FullLatentHSGPModel(BayesianModel):

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
    def get_component_parameters(self, position, component):
        """Extract parameter sampled values per model component for current
        position.

        """
        if component.startswith('kernel_'):
            return {param: position[f'{component}.{param}'] for param in
                self.param_priors[component]} if component in self.param_priors else {}
        return {param: position[param] for param in
                self.param_priors[component]} if component in self.param_priors else {}

    #
    def init_fn(self, key: PRNGKey, num_particles=1):
        
        initial_position = dict()

        # sample from all priors
        for component, comp_priors in self.param_priors.items():
            for param, param_dist in comp_priors.items():
                key, subkey = jrnd.split(key)
                param_name = f'{component}.{param}'
                if num_particles > 1:
                    initial_position[param_name] = param_dist.sample(seed=subkey,
                                                                     sample_shape=(num_particles,))
                else:
                    initial_position[param_name] = param_dist.sample(seed=subkey)


        # sample latent gps
        for name in ['v', 'f']:
            mean_params = {param: initial_position[f'kernel_{name}.{param}'] for param in self.param_priors.get(f'mean_{name}', {})}
            cov_params = {param: initial_position[f'kernel_{name}.{param}'] for param in self.param_priors[f'kernel_{name}']}

            if num_particles > 1:
                keys = jrnd.split(key, num_particles)                
                sample_fun = lambda key_, mean_params_, cov_params_: sample_prior(key=key_, 
                                                                                  mean_params=mean_params_,
                                                                                  cov_params=cov_params_,
                                                                                  mean_fn=self.mean_fns[name],
                                                                                  cov_fn=self.cov_fns[name], 
                                                                                  x=self.X)
                initial_position[name] = jax.vmap(sample_fun,
                                                  in_axes=(0,
                                                           {k: 0 for k in mean_params},
                                                           {k: 0 for k in cov_params}))(keys, mean_params, cov_params)
            else:
                key, subkey = jrnd.split(key)
                initial_position[name] = sample_prior(subkey, self.mean_fns[name], 
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
                                                                             f=position['f'][i],
                                                                             y=self.y[i]) for i in range(self.n)]))
        
        def loglikelihood_fn_f_(f_):
            return temperature * jnp.sum(jnp.array([self.likelihood.log_prob(params=dict(obs_noise=jnp.sqrt(jnp.exp(position['v'][i]))),
                                                                             f=f_[i],
                                                                             y=self.y[i]) for i in range(self.n)]))  

        loglikelihood_fns = dict(v=loglikelihood_fn_v_, f=loglikelihood_fn_f_)  

        # Sequentially update v and f and their respective covariance parameters
        for name in ['v', 'f']:   
            mean_params = self.get_component_parameters(position, f'mean_{name}')
            cov_params = self.get_component_parameters(position, f'kernel_{name}')

            key, subkey = jrnd.split(key)
            position[name], v_info = update_gaussian_process(subkey,
                                                            position[name],
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
                                                                        position[name],
                                                                        mean_fn=self.mean_fns[name],
                                                                        cov_fn=self.cov_fns[name],
                                                                        mean_params=mean_params,
                                                                        cov_params=cov_params,
                                                                        hyperpriors=self.param_priors[f'kernel_{name}'])

                for param, val in sub_state.items():
                    position[f'kernel_{name}.{param}'] = val

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
            f, v = position['f'], position['v']
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
                psi = {param: position[param] for param in self.param_priors[f'mean_{name}']} if f'mean_{name}' in self.param_priors else {}
                mean = mu.mean(params=psi, x=self.X).flatten()  # this will break with multitask GPs
                theta = {param: position[param] for param in
                     self.param_priors[f'kernel_{name}']} if f'kernel_{name}' in self.param_priors else {}
                cov = kernel.cross_covariance(params=theta,
                                                   x=self.X,
                                                   y=self.X)

                logprob += dx.MultivariateNormalFullCovariance(mean, cov).log_prob(position[name])

        #
        return logprior_fn_

    #
    def __predict_latent(self, key: PRNGKey, x_pred: Array, latent):

        num_particles = self.get_monte_carlo_samples()[latent].shape[0]
        key_samples = jrnd.split(key, num_particles)

        mean_params = {param: self.get_monte_carlo_samples()[f'kernel_{latent}.{param}'] for param in self.param_priors.get(f'mean_{latent}', {})}
        cov_params = {param: self.get_monte_carlo_samples()[f'kernel_{latent}.{param}'] for param in self.param_priors[f'kernel_{latent}']}

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
                        {k: 0 for k in mean_params}, 
                            {k: 0 for k in cov_params}, 
                                0))(keys, 
                                    mean_params, 
                                    cov_params, 
                                    self.get_monte_carlo_samples()[latent])

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
        num_particles = self.get_monte_carlo_samples()['f'].shape[0]
        keys_y = jrnd.split(key_y, num_particles)
        y_pred = jax.vmap(forward, in_axes=(0, 0, 0))(keys_y, f_pred, v_pred)
        return y_pred

    #
#