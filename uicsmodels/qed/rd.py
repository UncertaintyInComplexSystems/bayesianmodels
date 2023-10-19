import matplotlib.pyplot as plt

import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk

from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes
config.update("jax_default_device", jax.devices()[0])

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from uicsmodels.gaussianprocesses.gputil import sample_prior
from uicsmodels.gaussianprocesses.kernels import Brownian, SpectralMixture, Discontinuous
from uicsmodels.gaussianprocesses.meanfunctions import Constant
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


def regression_discontinuity_test(seed=42):
    print('Generate data')

    key = jrnd.PRNGKey(seed)
    key, key_f, key_y = jrnd.split(key, 3)

    x0 = 0.5
    base_kernel = jk.RBF()
    kernel = Discontinuous(base_kernel, x0=x0)

    n = 101
    x = jnp.linspace(0, 1, num=n)
    f = sample_prior(key_f, 
                    x=x, 
                    cov_fn=kernel, 
                    cov_params=dict(variance=1.0, 
                                    lengthscale=0.1))
    obs_noise = 0.5
    y = f + obs_noise*jrnd.normal(key=key_y, shape=(n, ))

    plt.figure(figsize=(12, 3))
    ax = plt.gca()
    ax.plot(x[x < x0], f[x < x0], color='k')
    ax.plot(x[x >= x0], f[x >= x0], color='k')
    ax.plot(x, y, 'x', color='tab:blue')
    ax.axvline(x=x0, color='k', ls=':')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim([0.0, 1.0])

    priors = dict(kernel=dict(lengthscale=dx.Transformed(dx.Normal(loc=0.,
                                                                    scale=1.),
                                                            tfb.Exp()),
                                variance=dx.Transformed(dx.Normal(loc=0.,
                                                                scale=1.),
                                                        tfb.Exp())),
                    likelihood=dict(obs_noise=dx.Transformed(dx.Normal(loc=0.,
                                                                        scale=1.),
                                                            tfb.Exp())))

    gp_cont = FullMarginalGPModel(x, y, cov_fn=base_kernel, priors=priors)
    gp_disc = FullMarginalGPModel(x, y, cov_fn=kernel, priors=priors)

    num_particles = 1_000
    num_mcmc_steps = 100

    key, key_m1, key_m0 = jrnd.split(key, 3)
    print('Inference of M1')
    m1_particles, _, m1_lml = gp_disc.inference(key_m1,
                                                mode='gibbs-in-smc',
                                                sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    print('Inference of M0')
    m0_particles, _, m0_lml = gp_cont.inference(key_m0,
                                                mode='gibbs-in-smc',
                                                sampling_parameters=dict(num_particles=num_particles, num_mcmc_steps=num_mcmc_steps))

    symbols = dict(lengthscale='\ell',
                obs_noise='\sigma',
                variance=r'\tau')

    trainables = list()
    for component, val in priors.items():
        trainables.extend(list(val.keys()))

    num_params = len(trainables)

    print('Visualization')
    _, axes = plt.subplots(nrows=2, ncols=num_params, constrained_layout=True,
                        sharex='col', sharey='col', figsize=(12, 6))

    for m, particles in enumerate([m0_particles, m1_particles]):
        axes[m, 0].set_ylabel(r'$M_{:d}$'.format(m))
        for j, var in enumerate(trainables):
            ax = axes[m, j]
            pd = particles.particles[var]
            pd_u, pd_l = jnp.percentile(pd, q=99.9), jnp.percentile(pd, q=0.1)
            pd_filtered = jnp.extract(pd>pd_l, pd)
            pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
            ax.hist(pd_filtered, bins=30, density=True, color='tab:blue')
            if var in symbols and m==1:
                ax.set_xlabel(r'${:s}$'.format(symbols[var]))

    _, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True, 
                        sharey=True, figsize=(12, 6))
    ax = axes[0]
    for ax in axes:
        ax.plot(x[x < x0], f[x < x0], color='k')
        ax.plot(x[x >= x0], f[x >= x0], color='k')
        ax.axvline(x=x0, color='k', ls=':')
        ax.plot(x, y, 'x', color='tab:blue')
        ax.set_ylabel(r'$y$')
        ax.set_xlim([0.0, 1.0])
    axes[1].set_xlabel(r'$x$')    
    key, key_f_disc, key_f_cont = jrnd.split(key, 3)
    predf_m1 = gp_disc.predict_f(key_f_disc, x[:, jnp.newaxis])
    predf_m0 = gp_cont.predict_f(key_f_cont, x[:, jnp.newaxis])
    plot_dist(axes[0], x, predf_m0, color='tab:red')
    plot_dist(axes[1], x[x < x0], predf_m1[:, x < x0], color='tab:red')
    plot_dist(axes[1], x[x >= x0], predf_m1[:, x >= x0], color='tab:red')

    print(f'log Bayes factor: {m1_lml - m0_lml}')
    return (m1_particles, m1_lml), (m0_particles, m0_lml)



