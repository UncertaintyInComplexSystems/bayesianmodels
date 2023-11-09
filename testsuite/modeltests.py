import matplotlib.pyplot as plt

import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk
import blackjax as bjx

from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes
config.update("jax_default_device", jax.devices()[0])

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from collections.abc import MutableMapping

from jax.tree_util import tree_flatten, tree_unflatten, tree_flatten_with_path

from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

from uicsmodels.gaussianprocesses.gputil import sample_prior, plot_dist
from uicsmodels.gaussianprocesses.hsgp import FullLatentHSGPModel
from uicsmodels.gaussianprocesses.kernels import Brownian, SpectralMixture, centered_softmax, DefaultingKernel
from uicsmodels.gaussianprocesses.meanfunctions import Constant
from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel
from uicsmodels.gaussianprocesses.likelihoods import Wishart, construct_wishart, tril2vec, construct_wishart_Lvec
from uicsmodels.gaussianprocesses.fullwp import FullLatentWishartModel

#### PLOTTING


def plot_wishart(x, Sigma, add_title=False, **kwargs):
    n, d, _ = Sigma.shape
    _, axes = plt.subplots(nrows=d, ncols=d, sharex=True, sharey=True,
                           constrained_layout=True)
    for i in range(d):
        for j in range(d):
            if i <= j:
                axes[i, j].plot(x, Sigma[:, i, j], **kwargs)
                if add_title:
                    axes[i, j].set_title(r'$\Sigma_{{{:d}{:d}}}(t)$'.format(i, j))
            else:
                axes[i, j].axis('off')            
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$x$')
    return axes

#
def plot_wishart_dist(x, Sigma_samples, axes=None, add_title=False, **kwargs):
    _, n, d, _ = Sigma_samples.shape
    color = kwargs.get('color', 'tab:blue')
    if axes is None:
        _, axes = plt.subplots(nrows=d, ncols=d, sharex=True, sharey=True,
                            constrained_layout=True)
    for i in range(d):
        for j in range(d):
            if i <= j:
                plot_dist(axes[i, j],
                          x,
                          Sigma_samples[:, :, i, j],
                          color=color)
                if add_title:
                    axes[i, j].set_title(r'$\Sigma_{{{:d}{:d}}}(t)$'.format(i, j))
            else:
                axes[i, j].axis('off')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$x$')
    return axes

#
def plot_latents(x, f, axes=None):
    n, nu, d = f.shape
    if axes is None:
        _, axes = plt.subplots(nrows=nu, ncols=d, figsize=(8, 6), sharex=True,
                            sharey=True, constrained_layout=True)
    for i in range(nu):
        for j in range(d):
            axes[i, j].plot(x, f[:, i, j])
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$x$')
    return axes

#

#### UTILITY

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

#

#### TEST CASES

def test_gwp(seed=42, show_latents=False):
    key = jrnd.PRNGKey(seed)
    key, key_F, key_Y = jrnd.split(key, 3)

    cov_fn = DefaultingKernel(jk.RBF(), dict(variance=1.0)) * DefaultingKernel(jk.Periodic(), dict(variance=1.0))

    params_rbf = dict(lengthscale=2.0)  # ls: how much fluctation between periods
    params_periodic = dict(period=0.3, lengthscale=0.5)  # ls: how much fluctuation within one period
    cov_params = [params_rbf, params_periodic]

    n = 100
    d = 3
    nu = d + 1
    x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

    V_gt = jnp.eye(d)
    L_gt = jnp.linalg.cholesky(V_gt)

    F_gt = sample_prior(key_F, x, cov_params=cov_params, cov_fn=cov_fn, nd=(nu, d))
    Sigma_gt = construct_wishart(F_gt, L_gt)

    plt.figure()
    plot_wishart(x, Sigma_gt, add_title=True);
    plt.suptitle('Ground truth Wishart process (RBF $\times$ Periodic)')

    print('Create observations')
    wishart_likelihood = Wishart(nu=nu, d=d)

    params = dict(L_vec=tril2vec(L_gt))
    Y = wishart_likelihood.likelihood(params, F_gt).sample(seed=key_Y)

    plt.figure(figsize=(8, 4))
    plt.plot(x, Y)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$Y$')
    plt.xlim([x[0], x[-1]])
    plt.title(r'Observations $Y$');

    print('Set up models')
    m = int(d*(d+1)/2)

    priors = dict(kernel=[dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp())),
                        dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp()),
                            period=dx.Uniform(0.0, 1.0))],
                    likelihood=dict(L_vec=dx.Normal(loc=jnp.zeros((m, )),
                                                    scale=jnp.ones((m, )))))

    cov_fn = jk.RBF() * jk.Periodic()
    wp = FullLatentWishartModel(x, Y, cov_fn=cov_fn, priors=priors)

    print('Do inference with SMC')
    key, subkey = jrnd.split(key)
    particles, _, lml = wp.inference(subkey,
                                    mode='gibbs-in-smc',
                                    sampling_parameters=dict(num_particles=1000,
                                                            num_mcmc_steps=100))

    def construct_wishart_samples(samples):
        F_samples = samples['f']
        L_vec_samples = samples['likelihood']['L_vec']
        return jax.vmap(construct_wishart_Lvec, in_axes=(0, 0))(F_samples,
                                                                L_vec_samples)

    #
    samples = wp.get_monte_carlo_samples()
    Sigma_samples = construct_wishart_samples(samples)

    # plot ground truth
    axes = plot_wishart(x, Sigma_gt, add_title=True, color='k', ls='--')

    # superposition the sampled distribution on the same axes
    plot_wishart_dist(x, Sigma_samples, axes=axes);
    plt.suptitle(r'Estimation of $\Sigma(t)$')

    print('Extrapolate ')

    key, subkey = jrnd.split(key)
    z = jnp.linspace(1.0, 2.0, num=100)
    Sigma_pred = wp.predict_Sigma(subkey, z)
    axes = plot_wishart_dist(x, Sigma_samples)
    _ = plot_wishart_dist(z, Sigma_pred, axes, add_title=True, color='tab:orange')
    plt.suptitle(r'Extrapolate the Wishart process to $x^* \neq x$')

    if show_latents:
        sample_ix = 100
        key, subkey = jrnd.split(key)
        axes = plot_latents(x, samples['f'][sample_ix]);
        f_pred = wp.predict_f(subkey, z)
        plot_latents(z, f_pred[sample_ix], axes=axes);
        plt.suptitle(r'Extrapolation of one sample $f$ (sanity check)')

    return wp

#
def test_compound_kernel(seed=42):
    key = jrnd.PRNGKey(42)

    key, key_f, key_obs = jrnd.split(key, 3)
    cov_fn = DefaultingKernel(jk.RBF(), dict(variance=1.0)) * DefaultingKernel(jk.Periodic(), dict(variance=1.0))

    obs_noise_ = 0.2
    params_rbf = dict(lengthscale=2.0)  # ls: how much fluctation between periods
    params_periodic = dict(period=0.3, lengthscale=0.5)  # ls: how much fluctuation within one period
    cov_params = [params_rbf, params_periodic]

    gt = [2.0, 0.5, 0.3, 0.2]

    n = 200
    x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

    f_true = sample_prior(key_f, x, cov_params=cov_params, cov_fn=cov_fn)

    y = f_true + obs_noise_*jrnd.normal(key_obs, shape=(n,))

    plt.figure(figsize=(8, 4))
    plt.plot(x, f_true, 'k', label=r'$f$')
    plt.plot(x, y, 'rx', label='obs')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0., 1.])
    plt.legend()
    plt.title(r'Locally period kernel (RBF $\times$ Periodic)')
    plt.show();    


    priors = dict(kernel=[dict(lengthscale=dx.Transformed(dx.Normal(loc=-1.,
                                                                        scale=1.),
                                                            tfb.Exp())),
                            dict(lengthscale=dx.Transformed(dx.Normal(loc=-1.,
                                                                        scale=1.),
                                                            tfb.Exp()),
                                period=dx.Uniform(0.0, 1.0))],
                likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                        tfb.Exp())))


    print('Inference')
    gp = FullMarginalGPModel(x, y, cov_fn=cov_fn, priors=priors) 

    num_particles = 1_000
    num_mcmc_steps = 100

    key, gpm_key = jrnd.split(key)
    particles, _, lml = gp.inference(gpm_key,
                                    mode='gibbs-in-smc',
                                    sampling_parameters=dict(num_particles=num_particles,
                                                            num_mcmc_steps=num_mcmc_steps))

    flat_priors, _ = tree_flatten_with_path(priors, 
                                        lambda l: isinstance(l, (Distribution, Bijector)))
    M = len(flat_priors)

    _, axes = plt.subplots(nrows=1, ncols=M, figsize=(12, 3), 
                        constrained_layout=True)

    flat_particles, _ = tree_flatten(particles)
    for i, (keyleaf, part) in enumerate(zip(flat_priors, flat_particles)):

        axes[i].hist(part, density=True, bins=30)
        axes[i].axvline(x=gt[i], ls=':', c='k')

    plt.suptitle('Marginal posteriors of hyperparameters')

    x_pred = jnp.linspace(0, 2.0, num=200)
    key, key_pred = jrnd.split(key)

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    f_pred = gp.predict_f(key_pred, x_pred)     
        
    plot_dist(ax, x_pred, f_pred, color='tab:blue')

    ax.plot(x, f_true, 'k', label=r'$f$')
    ax.plot(x, y, 'rx', label='obs')
    ax.axvline(x=1.0, ls=':', c='k')
    ax.set_xlim([0, 1.5])
    # ax.set_ylim([-2., 0.25])
    ax.set_xlabel(r'$x$')
    ax.set_title(r'Posterior predictive $f$')

    return gp


#                                                          
def test_smk(seed=42):
    print('Generate data')

    n = 200
    obs_noise = 0.3

    key = jrnd.PRNGKey(seed)
    key, key_x, key_y = jrnd.split(key, 3)

    x = jnp.sort(jrnd.uniform(key=key_x, minval=-3.0, maxval=3.0, shape=(n,))).reshape(-1, 1)
    f = lambda x: 10*jnp.sin(x) + jnp.cos(15*x)
    signal = f(x)
    y = (signal + jrnd.normal(key_y, shape=signal.shape) * obs_noise).flatten()

    plt.figure(figsize=(12, 4))
    plt.plot(jnp.linspace(-3, 3, num=300).reshape(-1, 1),
            f(jnp.linspace(-3, 3, num=300).reshape(-1, 1)),
            color='tab:green')
    plt.plot(x, y, 'o', c='tab:orange')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Observations')
    plt.xlim([-3, 3]);

    Q = 2

    priors = dict(kernel=dict(beta=dx.Normal(loc=jnp.zeros((Q-1, )),
                                            scale=jnp.ones((Q-1, ))),
                              mu=dx.Normal(loc=jnp.zeros((Q, )),
                                        scale=jnp.ones((Q, ))),
                              nu=dx.Transformed(dx.Normal(loc=jnp.zeros((Q, )),
                                                        scale=jnp.ones((Q, ))),
                                                tfb.Exp())),
                likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                        tfb.Exp())))

    print('Inference')
    key, subkey = jrnd.split(key)
    num_particles = 1_000
    num_mcmc_steps = 100

    gp_smk = FullMarginalGPModel(x, y, cov_fn=SpectralMixture(), priors=priors)
    smk_particles, _, _ = gp_smk.inference(subkey, mode='gibbs-in-smc',
                                           sampling_parameters=dict(num_particles=num_particles,
                                                                    num_mcmc_steps=num_mcmc_steps))

    w = jax.vmap(centered_softmax, in_axes=0)(smk_particles.particles['kernel']['beta'])
    _, axes = plt.subplots(nrows=3, ncols=Q, figsize=(12, 9), constrained_layout=True)
    for q in range(Q):
        axes[0, q].hist(jnp.abs(smk_particles.particles['kernel']['mu'][:,q]) * (2*jnp.pi), 
                 density=True, bins=30, alpha=0.5)
        axes[0, q].set_title(r'$\mu_{:d}$'.format(q+1))
        axes[0, q].set_xlabel(r'$\omega$')
        axes[1, q].hist(smk_particles.particles['kernel']['nu'][:,q], 
                 density=True, bins=30, alpha=0.5)
        axes[1, q].set_title(r'$\nu_{:d}$'.format(q+1))
        axes[1, q].set_xlabel(r'$\omega$')
        axes[2, q].hist(w[:, q], 
                 density=True, bins=30, alpha=0.5)
        axes[2, q].set_title(r'$w_{:d}$'.format(q+1))

    plt.suptitle('Component parameters')

    print('Prediction')

    plt.figure(figsize=(12, 3))
    ax = plt.gca()

    x_pred = jnp.linspace(-6, 6, num=300)[:, jnp.newaxis]
    key, key_f, key_y = jrnd.split(key, 3)
    f_pred = gp_smk.predict_f(key_f, x_pred)  
    y_pred = gp_smk.predict_y(key_y, x_pred)      

    for i in jnp.arange(0, num_particles, step=50):
        ax.plot(x_pred, f_pred[i, :], alpha=0.1, color='tab:blue')

    ax.plot(jnp.linspace(-3, 3, num=300).reshape(-1, 1),
            f(jnp.linspace(-3, 3, num=300).reshape(-1, 1)),
            color='tab:green')
    ax.plot(x, y, 'o', c='tab:orange')
    for lim in [-3, 3]:
        ax.axvline(x=lim, ls='--', color='k')
    ax.set_xlim([-6, 6])
    ax.set_title('Extrapolation')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
       
    return smk_particles

#
def test_hsgp(seed=1234):
    print('Generate data')
    key = jrnd.PRNGKey(seed)
    key, key_v, key_f, key_y = jrnd.split(key, 4)

    jitter = 1e-6

    n = 100
    x = jnp.linspace(2, 3, n)[:, jnp.newaxis]

    ground_truth = dict(v={'lengthscale': 0.3, 'variance': 4.0}, f={'variance': 10.0})

    kernel_v = jk.RBF()    
    v = sample_prior(key_v,
                    x=x, 
                    cov_fn=kernel_v, 
                    cov_params=dict(lengthscale=ground_truth['v']['lengthscale'], 
                                    variance=ground_truth['v']['variance']))
    V = jnp.exp(v)

    kernel_f = Brownian()
    f = sample_prior(key_f, 
                    x=x, 
                    cov_fn=kernel_f, 
                    cov_params=dict(variance=ground_truth['f']['variance']))

    y = f + jnp.sqrt(V)*jrnd.normal(key_y, shape=(n, ))

    _, axes = plt.subplots(figsize=(12, 3), nrows=1, ncols=2, sharex=True, constrained_layout=True)
    axes[0].plot(x, V)
    axes[0].set_title(r'Heteroskedastic observation variance $V(t)$')
    axes[1].plot(x, f)
    axes[1].plot(x, y, 'x')
    axes[1].set_title(r'Latent $f(t)$ and observations $y(t)$')
    for ax in axes:
        ax.set_xlim([2., 3.])
        ax.set_xlabel(r'$t$')

    plt.suptitle('A draw from the prior')

    print('Set up heteroskedastic GP model')

    priors = dict(v=dict(kernel=dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                        scale=1.),
                                                                tfb.Exp()),
                                    variance=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp()))),
                f=dict(kernel=dict(variance=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp()))))

    hsgp = FullLatentHSGPModel(x, y,
                            cov_fns=dict(v=jk.RBF(), 
                                            f=Brownian()),
                            priors=priors)

    print('Inference')
    num_particles = 1_000
    num_mcmc_steps = 100
    key, subkey = jrnd.split(key)
    results = hsgp.inference(subkey,
                            mode='gibbs-in-smc',
                            sampling_parameters=dict(num_particles=num_particles,
                                                    num_mcmc_steps=num_mcmc_steps))

    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), constrained_layout=True)
    symbols = dict(v={'lengthscale': r'\ell_v', 'variance':  r'\tau_v'},
                    f={'variance': r'\tau_f'})

    flat_priors = flatten_dict(priors)

    for i, (ax, param_path) in enumerate(zip(axes, flat_priors.keys())):
        path = param_path.split('.')    
        ax.hist(results[0].particles[path[0]][path[1]][path[2]], density=True, bins=30)
        ax.axvline(x=ground_truth[path[0]][path[2]], ls='--', color='k')
        ax.set_xlabel(r'${:s}$'.format(symbols[path[0]][path[2]]))

    plt.suptitle('Marginal posteriors of hyperparameters')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), sharex=True,
                            constrained_layout=True)

    v_samples = jnp.exp(results[0].particles['v']['gp'])
    f_samples = results[0].particles['f']['gp']

    plot_dist(axes[0], x, v_samples)
    plot_dist(axes[1], x, f_samples)

    axes[0].plot(x, V, color='tab:red')
    axes[1].plot(x, f, color='tab:red')
    axes[1].plot(x, y, 'x', color='tab:orange')

    axes[0].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xlim([2., 3.])
        ax.set_xlabel('t')

    plt.suptitle('Posterior estimates of $V(t)$ and $f(t)$')

    print('Predictive')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4), sharex=True,
                            constrained_layout=True)

    x_pred = jnp.linspace(-1.5, 3.5, num=233)[:, jnp.newaxis]

    v_samples = jnp.exp(results[0].particles['v']['gp'])
    f_samples = results[0].particles['f']['gp']

    key, key_v, key_f, key_y = jrnd.split(key, 4)

    v_pred = jnp.exp(hsgp.predict_v(key_v, x_pred))
    f_pred = hsgp.predict_f(key_f, x_pred)
    y_pred = hsgp.predict_y(key_y, x_pred)

    plot_dist(axes[0], x, v_samples, color='tab:green')
    plot_dist(axes[0], x_pred, v_pred, color='tab:green', label=r'$V(t)$')
    plot_dist(axes[1], x, f_samples, color='tab:green')
    plot_dist(axes[1], x_pred, f_pred, color='tab:green', label=r'$f(t*)$')

    axes[0].plot(x, V, color='tab:red', label=r'True $V$')
    axes[1].plot(x, f, color='tab:red', label=r'True $f$')
    axes[1].plot(x, y, 'x', color='tab:orange', label='Obs')

    axes[0].set_ylim(bottom=0, top=30)
    axes[1].set_ylim(bottom=0, top=15)

    axes[0].set_title(r'Heteroskedastic variance $V(t)$')
    axes[1].set_title(r'Brownian motion $f(t)$ and observations $y(t)$')

    for ax in axes:
        ax.set_xlim([1.5, 3.5])
        ax.axvline(x=2.0, ls='--', c='k')
        ax.axvline(x=3.0, ls='--', c='k')
        ax.set_xlabel('t')

    plot_dist(axes[1], x_pred, y_pred, color='tab:orange', label=r'$y(t*)$')
    axes[0].legend()
    axes[1].legend()
    plt.suptitle('Posterior predictive distributions')

    return hsgp

#
def test_fullgp(seed=42):
    print('Generate data')
    key = jrnd.PRNGKey(seed)

    lengthscale_ = 0.1
    output_scale_ = 5.0
    obs_noise_ = 0.8
    n = 100
    x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

    key, key_f, key_obs = jrnd.split(key, 3)
    f_true = sample_prior(key_f, x=x, cov_fn=jk.RBF(), cov_params=dict(lengthscale=lengthscale_,
                                            variance=output_scale_))

    y = f_true + obs_noise_*jrnd.normal(key_obs, shape=(n,))

    ground_truth = dict(f=f_true,
                        lengthscale=lengthscale_,
                        variance=output_scale_,
                        obs_noise=obs_noise_)

    plt.figure(figsize=(12, 4))
    plt.plot(x, f_true, 'k', label=r'$f$')
    plt.plot(x, y, 'rx', label='obs')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0., 1.])
    plt.legend()
    plt.show();

    print('Set up full and marginal GP models')
    priors = dict(kernel=dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp()),
                                variance=dx.Transformed(dx.Normal(loc=0.,
                                                                scale=1.),
                                                        tfb.Exp())),
                    likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                        scale=1.),
                                                            tfb.Exp())))

    gp_marginal = FullMarginalGPModel(x, y, cov_fn=jk.RBF(), priors=priors)  
    gp_latent = FullLatentGPModel(x, y, cov_fn=jk.RBF(), priors=priors)
    print('Inference')

    num_particles = 1_000
    num_mcmc_steps = 100

    key, gpm_key = jrnd.split(key)
    mgp_particles, _, mgp_marginal_likelihood = gp_marginal.inference(gpm_key,
                                                                    mode='gibbs-in-smc',
                                                                    sampling_parameters=dict(num_particles=num_particles, 
                                                                                            num_mcmc_steps=num_mcmc_steps))

    key, gpl_key = jrnd.split(key)
    lgp_particles, _, lgp_marginal_likelihood = gp_latent.inference(gpl_key,
                                                                    mode='gibbs-in-smc',
                                                                    sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    trainables = ['lengthscale', 'variance', 'obs_noise']

    num_params = len(trainables)
    show_samples = jnp.array([int(i) for i in num_particles*jnp.linspace(0, 1, num=500)])

    symbols = dict(lengthscale='\ell',
                obs_noise='\sigma',
                variance=r'\tau')

    _, axes = plt.subplots(nrows=2, ncols=num_params, constrained_layout=True,
                        sharex='col', sharey='col', figsize=(12, 6))

    for m, particles in enumerate([mgp_particles, lgp_particles]):
        tr = dict(lengthscale=particles.particles['kernel']['lengthscale'], 
                variance=particles.particles['kernel']['variance'], 
                obs_noise=particles.particles['likelihood']['obs_noise'])
        for j, var in enumerate(trainables):
            ax = axes[m, j]
            pd = tr[var]
            # There are some outliers that skew the axis
            pd_u, pd_l = jnp.percentile(pd, q=99.9), jnp.percentile(pd, q=0.1)
            pd_filtered = jnp.extract(pd>pd_l, pd)
            pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
            ax.hist(pd_filtered, bins=30, density=True, color='tab:blue')
            if var in symbols and m==1:
                ax.set_xlabel(r'${:s}$'.format(symbols[var]))

    plt.suptitle(f'Posterior estimate of Bayesian GP ({num_particles} particles)');

    axes[0, 0].set_ylabel('Marginal GP', rotation=0, ha='right')
    axes[1, 0].set_ylabel('Latent GP', rotation=0, ha='right')

    if len(ground_truth):
        for j, var in enumerate(trainables):
            axes[0, j].axvline(x=ground_truth[var], ls='--', c='k');
            axes[1, j].axvline(x=ground_truth[var], ls='--', c='k');

    print('Predictive')

    x_pred = jnp.linspace(-0.25, 1.25, num=150)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=True,
                                sharey=True, constrained_layout=True)

    for j, gp in enumerate([gp_marginal, gp_latent]):
        key, key_f, key_y = jrnd.split(key, 3)
        f_pred = gp.predict_f(key_f, x_pred)  
        y_pred = gp.predict_y(key_y, x_pred)      

        ax = axes[j, 0]
        for i in jnp.arange(0, num_particles, step=10):
            ax.plot(x_pred, f_pred[i, :], alpha=0.1, color='tab:blue')

        ax = axes[j, 1]        
        plot_dist(ax, x_pred, y_pred, color='tab:red')
        plot_dist(ax, x_pred, f_pred, color='tab:red')

    for ax in axes.flatten():
        ax.plot(x, f_true, 'k', label=r'$f$')
        ax.plot(x, y, 'rx', label='obs')
        ax.set_xlim([-0.25, 1.25])
        ax.set_ylim([-4., 6.])
        ax.set_xlabel(r'$x$')

    axes[0, 0].set_title('SMC particles')
    axes[0, 1].set_title('Posterior 95% HDI')

    axes[0, 0].set_ylabel('Marginal GP', rotation=0, ha='right')
    axes[1, 0].set_ylabel('Latent GP', rotation=0, ha='right');

    return gp_marginal, gp_latent

#
def test_fullgp_mcmc_within_smc(seed=42):
    print('Generate data')
    key = jrnd.PRNGKey(seed)

    lengthscale_ = 0.1
    output_scale_ = 5.0
    obs_noise_ = 0.8
    n = 100
    x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

    key, key_f, key_obs = jrnd.split(key, 3)
    f_true = sample_prior(key_f, x=x, cov_fn=jk.RBF(), cov_params=dict(lengthscale=lengthscale_,
                                            variance=output_scale_))
    
    y = f_true + obs_noise_*jrnd.normal(key_obs, shape=(n,))

    ground_truth = dict(f=f_true,
                        lengthscale=lengthscale_,
                        variance=output_scale_,
                        obs_noise=obs_noise_)

    plt.figure(figsize=(12, 4))
    plt.plot(x, f_true, 'k', label=r'$f$')
    plt.plot(x, y, 'rx', label='obs')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0., 1.])
    plt.legend()
    plt.show();

    print('Set up full and marginal GP models')
    priors = dict(kernel=dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp()),
                                variance=dx.Transformed(dx.Normal(loc=0.,
                                                                scale=1.),
                                                        tfb.Exp())),
                    likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                        scale=1.),
                                                            tfb.Exp())))

    model = FullMarginalGPModel(x, y, cov_fn=jk.RBF(), priors=priors)  # Implies likelihood=Gaussian()
    # gp_latent = FullLatentGPModel(x, y, cov_fn=jk.RBF(), priors=priors)  # Defaults to likelihood=Gaussian()

    print('Inference')

    # kernels = list()
    # kernel_parameters = list()
    # kernels.append(bjx.rmh)
    
    # num_particles = 1_000
    # num_mcmc_steps = 100

    # key, gpm_key = jrnd.split(key)
    # mgp_particles, _, mgp_marginal_likelihood = gp_marginal.inference(gpm_key,
    #                                                               mode='gibbs-in-smc',
    #                                                               sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    # key, gpl_key = jrnd.split(key)
    # lgp_particles, _, lgp_marginal_likelihood = gp_latent.inference(gpl_key,
    #                                                             mode='gibbs-in-smc',
    #                                                             sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    # trainables = list()
    # for component, val in priors.items():
    #     trainables.extend(list(val.keys()))

    # num_params = len(trainables)
    # show_samples = jnp.array([int(i) for i in num_particles*jnp.linspace(0, 1, num=500)])

    # symbols = dict(lengthscale='\ell',
    #             obs_noise='\sigma',
    #             variance=r'\tau')

    # _, axes = plt.subplots(nrows=2, ncols=num_params, constrained_layout=True,
    #                     sharex='col', sharey='col', figsize=(12, 6))

    # for m, particles in enumerate([mgp_particles, lgp_particles]):
    #     for j, var in enumerate(trainables):
    #         ax = axes[m, j]
    #         pd = particles.particles[var]
    #         # There are some outliers that skew the axis
    #         pd_u, pd_l = jnp.percentile(pd, q=99.9), jnp.percentile(pd, q=0.1)
    #         pd_filtered = jnp.extract(pd>pd_l, pd)
    #         pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
    #         ax.hist(pd_filtered, bins=30, density=True, color='tab:blue')
    #         if var in symbols and m==1:
    #             ax.set_xlabel(r'${:s}$'.format(symbols[var]))

    # plt.suptitle(f'Posterior estimate of Bayesian GP ({num_particles} particles)');

    # axes[0, 0].set_ylabel('Marginal GP', rotation=0, ha='right')
    # axes[1, 0].set_ylabel('Latent GP', rotation=0, ha='right')

    # if len(ground_truth):
    #     for j, var in enumerate(trainables):
    #         axes[0, j].axvline(x=ground_truth[var], ls='--', c='k');
    #         axes[1, j].axvline(x=ground_truth[var], ls='--', c='k');

    # print('Predictive')

    # x_pred = jnp.linspace(-0.25, 1.25, num=150)

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=True,
    #                             sharey=True, constrained_layout=True)

    # for j, gp in enumerate([gp_marginal, gp_latent]):
    #     key, key_f, key_y = jrnd.split(key, 3)
    #     f_pred = gp.predict_f(key_f, x_pred)  
    #     y_pred = gp.predict_y(key_y, x_pred)      

    #     ax = axes[j, 0]
    #     for i in jnp.arange(0, num_particles, step=10):
    #         ax.plot(x_pred, f_pred[i, :], alpha=0.1, color='tab:blue')

    #     ax = axes[j, 1]        
    #     plot_dist(ax, x_pred, y_pred, color='tab:red')
    #     plot_dist(ax, x_pred, f_pred, color='tab:red')

    # for ax in axes.flatten():
    #     ax.plot(x, f_true, 'k', label=r'$f$')
    #     ax.plot(x, y, 'rx', label='obs')
    #     ax.set_xlim([-0.25, 1.25])
    #     ax.set_ylim([-4., 6.])
    #     ax.set_xlabel(r'$x$')

    # axes[0, 0].set_title('SMC particles')
    # axes[0, 1].set_title('Posterior 95% HDI')

    # axes[0, 0].set_ylabel('Marginal GP', rotation=0, ha='right')
    # axes[1, 0].set_ylabel('Latent GP', rotation=0, ha='right');

    # return gp_marginal, gp_latent