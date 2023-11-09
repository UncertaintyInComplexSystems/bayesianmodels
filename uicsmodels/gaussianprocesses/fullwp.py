from uicsmodels.bayesianmodels import GibbsState, ArrayTree
from uicsmodels.gaussianprocesses.gputil import sample_prior
from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel
from uicsmodels.gaussianprocesses.likelihoods import Wishart, construct_wishart_Lvec
from uicsmodels.gaussianprocesses.kernels import DefaultingKernel

from jax import Array
from jaxtyping import Float
from jax.random import PRNGKeyArray as PRNGKey
from typing import Callable, Union, Dict, Any, Optional, Iterable, Mapping
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

from jax.tree_util import tree_flatten, tree_unflatten

from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

from copy import deepcopy
import jax
import distrax as dx
import jax.numpy as jnp
from jax.random import PRNGKey
import jax.random as jrnd

def cov_default_recursive(cov_fn, defaults = None):
    if hasattr(cov_fn, 'kernel_set'):
        cov_fn_defaults = deepcopy(cov_fn)
        cov_fn_defaults.kernel_set = [cov_default_recursive(cov_fn_) for cov_fn_ in cov_fn.kernel_set]
        return cov_fn_defaults
    else:
        if isinstance(cov_fn, DefaultingKernel):
            return cov_fn
        if defaults is None:
            defaults = dict(variance=1.0)
        return DefaultingKernel(cov_fn, defaults)

#


class FullLatentWishartModel(FullLatentGPModel):

    """The latent Wishart process model.

    The generative model is given by:

    .. math::
        psi     &\sim p(\psi)\\
        theta   &\sim p(\theta) && s.t. \tau=1\\ 
        phi     &\sim p(\phi) \\
        f_id    &\sim GP(mu, cov) \\
        f_i     &= (f_i1, ..., f_iD)^T
        Sigma   &= \sum_{i=1}^\nu L f_i f_i^T L^T
        y_t     &\sim MVN(0, Sigma_t)

    Here, the scalar parameters are sampled using Gaussian random walk MCMC,
    while the latent function f (or rather its evaluations) is sampled using
    Elliptical Slice Sampling.

    Since the Wishart process is just a Gaussian process with multiple 
    independent latent GPs, and a new likelihood, we inherit most functionality 
    from FullLatentGPModel.

    """

    def __init__(self, X, Y,
                 cov_fn: Callable,
                 mean_fn: Optional[Callable] = None,
                 priors: Dict = None):
        self.D = Y.shape[1]
        self.nu = self.D + 1
        self.output_shape = (self.nu, self.D) # nu x d; note that JAX must know the number of elements in this tuple
        likelihood = Wishart(nu=self.nu, d=self.D)
        self.likelihood = likelihood    
        cov_fn_defaults = cov_default_recursive(cov_fn)
        super().__init__(X,
                         Y,
                         cov_fn=cov_fn_defaults,
                         mean_fn=mean_fn,
                         priors=priors,
                         likelihood=self.likelihood)

    #
    def init_fn(self, key, num_particles=1):
        """Initialization of the Gibbs state.

        The initial state is determined by sampling all variables from their
        priors, and then constructing one sample for (f) using these. All
        variables together are stored in a dict() in the GibbsState object.

        When num_particles > 1, each dict element contains num_particles random
        initial values.

        For the Wishart process, f is of shape (n, nu, d). The same mean and
        covariance functions are broadcasted over all trailing dimensions.

        Args:
            key:
                The jax.random.PRNGKey
            num_particles: int
                The number of parallel particles for sequential Monte Carlo
        Returns:
            GibbsState

        """

        priors_flat, priors_treedef = tree_flatten(self.param_priors, lambda l: isinstance(l, (Distribution, Bijector)))
        samples = list()
        for prior in priors_flat:
            key, subkey = jrnd.split(key)
            samples.append(prior.sample(seed=subkey, sample_shape=(num_particles,)))

        initial_position = jax.tree_util.tree_unflatten(priors_treedef, samples)

        mean_params = initial_position.get('mean', {})
        cov_params = initial_position.get('kernel', {})
        mean_param_in_axes = jax.tree_map(lambda l: 0, mean_params)
        cov_param_in_axes = jax.tree_map(lambda l: 0, cov_params)

        if num_particles > 1:
            keys = jrnd.split(key, num_particles)
            sample_fun = lambda key_, mean_params_, cov_params_: sample_prior(key=key_,
                                                                                 mean_params=mean_params_,
                                                                                 cov_params=cov_params_,
                                                                                 mean_fn=self.mean_fn,
                                                                                 cov_fn=self.cov_fn,
                                                                                 x=self.X,
                                                                                 nd=self.output_shape)
            initial_position['f'] = jax.vmap(sample_fun,
                                             in_axes=(0,
                                                      mean_param_in_axes,
                                                      cov_param_in_axes))(keys, mean_params, cov_params)
        else:
            key, subkey = jrnd.split(key)
            initial_position['f'] = sample_prior(key=subkey,
                                                    mean_fn=self.mean_fn,
                                                    cov_fn=self.cov_fn,
                                                    mean_params=mean_params,
                                                    cov_params=cov_params,
                                                    x=self.X,
                                                    nd=self.output_shape)

        return GibbsState(initial_position)

    #
    def predict_Sigma(self, key: PRNGKey, x_pred: Array):
        """Get posterior predictive of covariance process.

        We first determine posterior predictive samples of the latent functions
        f, and then construct the Wishart process using these.

        """ 
        samples = self.get_monte_carlo_samples()
        f_pred = self.predict_f(key, x_pred)
        Sigma_pred = jax.vmap(construct_wishart_Lvec, in_axes=(0, 0))(f_pred,
                                                                      samples['likelihood']['L_vec'])
        return Sigma_pred

    #
#