import matplotlib.pyplot as plt

import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk

from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes
config.update("jax_default_device", jax.devices()[0])
print('Jax version:', jax.__version__)
print('Jax default backend:', jax.default_backend())  # check if we're using the GPU

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from uicsmodels.gaussianprocesses.gputil import sample_prior
from uicsmodels.gaussianprocesses.hsgp import FullLatentHSGPModel
from uicsmodels.gaussianprocesses.kernels import Brownian
from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel

def plot_dist(ax, x, samples, **kwargs):
    f_mean = jnp.mean(samples, axis=0)
    f_hdi_lower = jnp.percentile(samples, q=2.5, axis=0)
    f_hdi_upper = jnp.percentile(samples, q=97.5, axis=0)
    color = kwargs.get('color', 'tab:blue')
    ax.plot(x, f_mean, lw=2, **kwargs)
    ax.fill_between(x.flatten(), f_hdi_lower, f_hdi_upper,
                    alpha=0.2, lw=0, color=color)

#

def test_hsgp(seed=42):
    print('Generate data')
    key = jrnd.PRNGKey(seed)
    key, key_v, key_f, key_y = jrnd.split(key, 4)

    jitter = 1e-6

    n = 100
    x = jnp.linspace(2, 3, n)[:, jnp.newaxis]

    kernel_v = jk.RBF()
    v = sample_prior(key_v, x=x, cov_fn=kernel_v, cov_params=dict(lengthscale=0.3, variance=4.))
    V = jnp.exp(v)

    kernel_f = Brownian()
    f = sample_prior(key_f, x=x, cov_fn=kernel_f, cov_params=dict(variance=10.0))

    y = f + jnp.sqrt(V)*jrnd.normal(key_y, shape=(n, ))

    _, axes = plt.subplots(figsize=(12, 3), nrows=1, ncols=2, sharex=True)
    axes[0].plot(x, V)
    axes[0].set_title(r'Heteroskedastic observation variance $V(t)$')
    axes[1].plot(x, f)
    axes[1].plot(x, y, 'x')
    axes[1].set_title(r'Latent $f(t)$ and observations $y(t)$')
    for ax in axes:
        ax.set_xlim([2., 3.])
        ax.set_xlabel(r'$t$')

    print('Set up heteroskedastic GP model')

    priors = dict(kernel_v=dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                scale=1.),
                                                        tfb.Exp()),
                                variance=dx.Transformed(dx.Normal(loc=0.,
                                                                scale=1.),
                                                    tfb.Exp())),
                kernel_f=dict(variance=dx.Transformed(dx.Normal(loc=0.,
                                                                scale=1.),
                                                        tfb.Exp())))

    hsgp = FullLatentHSGPModel(x, y,
                            cov_fns=dict(v=jk.RBF(), f=Brownian()),
                            priors=priors)

    print('Inference')
    num_particles = 1_000
    num_mcmc_steps = 100
    key, subkey = jrnd.split(key)
    results = hsgp.inference(subkey,
                            mode='gibbs-in-smc',
                            sampling_parameters=dict(num_particles=num_particles,
                                                    num_mcmc_steps=num_mcmc_steps))


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), sharex=True,
                            constrained_layout=True)

    v_samples = jnp.exp(results[0].particles['v'])
    f_samples = results[0].particles['f']

    plot_dist(axes[0], x, v_samples)
    plot_dist(axes[1], x, f_samples)

    axes[0].plot(x, V, color='tab:red')
    axes[1].plot(x, f, color='tab:red')
    axes[1].plot(x, y, 'x', color='tab:orange')

    axes[0].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xlim([2., 3.])
        ax.set_xlabel('t')

    print('Predictive')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4), sharex=True,
                            constrained_layout=True)

    x_pred = jnp.linspace(-1.5, 3.5, num=233)[:, jnp.newaxis]

    v_samples = jnp.exp(results[0].particles['v'])
    f_samples = results[0].particles['f']

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
    axes[1].set_ylim(bottom=-5, top=10)

    axes[0].set_title(r'Heteroskedastic variance $V(t)$')
    axes[1].set_title(r'Brownian motion $f(t)$ and observations $y(t)$')

    for ax in axes:
        ax.set_xlim([1.5, 3.5])
        ax.axvline(x=2.0, ls='--', c='k')
        ax.axvline(x=3.0, ls='--', c='k')
        ax.set_xlabel('t')

    plot_dist(axes[1], x_pred, y_pred, color='tab:orange', label=r'$y(t*)$')
    axes[0].legend();
    axes[1].legend();

#
def test_fullgp(seed=42):
    print('Generate data')
    key = jrnd.PRNGKey(seed)

    lengthscale_ = 0.1
    output_scale_ = 5.0
    obs_noise_ = 0.3
    n = 100
    x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

    kernel = jk.RBF()
    K = kernel.cross_covariance(params=dict(lengthscale=lengthscale_,
                                            variance=output_scale_),
                                x=x, y=x) + 1e-6*jnp.eye(n)

    L = jnp.linalg.cholesky(K)
    z = jrnd.normal(key, shape=(n,))

    f_true = jnp.dot(L, z) + jnp.ones_like(z)
    key, obs_key = jrnd.split(key)
    y = f_true + obs_noise_*jrnd.normal(obs_key, shape=(n,))

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

    gp_marginal = FullMarginalGPModel(x, y, cov_fn=jk.RBF(), priors=priors)  # Implies likelihood=Gaussian()
    gp_latent = FullLatentGPModel(x, y, cov_fn=jk.RBF(), priors=priors)  # Defaults to likelihood=Gaussian()

    print('Inference')
    
    num_particles = 1_000
    num_mcmc_steps = 100

    key, gpm_key = jrnd.split(key)
    mgp_particles, _, mgp_marginal_likelihood = gp_marginal.inference(gpm_key,
                                                                  mode='gibbs-in-smc',
                                                                  sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    key, gpl_key = jrnd.split(key)
    lgp_particles, _, lgp_marginal_likelihood = gp_latent.inference(gpl_key,
                                                                mode='gibbs-in-smc',
                                                                sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    trainables = list()
    for component, val in priors.items():
        trainables.extend(list(val.keys()))

    num_params = len(trainables)
    show_samples = jnp.array([int(i) for i in num_particles*jnp.linspace(0, 1, num=500)])

    symbols = dict(lengthscale='\ell',
                obs_noise='\sigma',
                variance=r'\tau')

    _, axes = plt.subplots(nrows=2, ncols=num_params, constrained_layout=True,
                        sharex='col', sharey='col', figsize=(12, 6))

    for m, particles in enumerate([mgp_particles, lgp_particles]):
        for j, var in enumerate(trainables):
            ax = axes[m, j]
            pd = particles.particles[var]
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
        key, key_pred = jrnd.split(key)
        f_pred = gp.predict_f(key_pred, x_pred)

        ax = axes[j, 0]
        for i in jnp.arange(0, num_particles, step=10):
            ax.plot(x_pred, f_pred[i, :], alpha=0.1, color='tab:blue')

        ax = axes[j, 1]
        f_mean = jnp.mean(f_pred, axis=0)
        f_hdi_lower = jnp.percentile(f_pred, q=2.5, axis=0)
        f_hdi_upper = jnp.percentile(f_pred, q=97.5, axis=0)

        ax.plot(x_pred, f_mean, color='tab:blue', lw=2)
        ax.fill_between(x_pred, f_hdi_lower, f_hdi_upper,
                        alpha=0.2, color='tab:blue', lw=0)

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
