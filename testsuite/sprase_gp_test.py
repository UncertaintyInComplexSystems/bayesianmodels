import os
import sys
import datetime
import logging
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import jax
from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector
import jaxkern as jk
from jax.tree_util import tree_flatten

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from uicsmodels.gaussianprocesses.sparsegp import SparseGPModel
from uicsmodels.gaussianprocesses.fullgp import FullLatentGPModel, FullMarginalGPModel

from icecream import ic
ic.configureOutput(includeContext=True)

# solarized colors
colors = {
    'base01': '#586e75',
    'yellow':'#b58900',
    'orange': '#cb4b16',
    'red':'#dc322f',
    'magenta':'#d33682',
    'violet':'#6c71c4',
    'blue': '#268bd2',
    'cyan': '#2aa198',
    'green':'#859900'}

def plt_stylelize():
    plt.rc('axes', titlesize=18)        # fontsize of the axes title
    plt.rc('axes', labelsize=16)        # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)       # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)       # fontsize of the tick labels
    plt.rc('legend', fontsize=16)       # legend fontsize
    plt.rc('figure', titlesize=20)      # fontsize of the figure title
    plt.style.use('Solarize_Light2')
plt_stylelize()

def setup_results_folder_and_logging(
        static_folder=False, 
        log_level=logging.DEBUG, 
        subfolders=None):
    """_summary_

    Args:
        - static_folder (bool, optional): creates unique folder based on date and time if False, if true the folder is named 'debug'. Defaults to False.
        - log_level (_type_, optional): _description_. Defaults to logging.DEBUG.
        - sub_folder (list(string), optional): creates additional sub_folder if not empty. Defaults to None.

    Returns:
        string: relative filepath
    """

    # create unique folder for log files and other output
    timestamp = 'debug' if static_folder else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = f'./results_sparseGP/{timestamp}/'

    # create subfolders
    paths_sub = []
    for folder in subfolders:
        paths_sub.append(path + f'seed_{folder}/')
        os.makedirs(paths_sub[-1], exist_ok = True)

    # disable logging of other modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)

    # setup logging to file and stdout
    logging.basicConfig(
        handlers=[
            logging.FileHandler(path + 'log.log'), 
            logging.StreamHandler(sys.stdout)],
        encoding='utf-8', level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s")
    
    # log a first entry
    logging.info('Hello cruel world.')
    if static_folder:
        logging.warning('using debug logging folder. Files might be overwritten!')

    return path, paths_sub



def generate_data(path_plot=None, seed=12345):
    key_data = jrnd.PRNGKey(seed)  # 1106, 5368, 8928, 5609

    lengthscale_ = 0.2
    output_scale_ = 5.0
    obs_noise_ = 0.7
    n = 100
    x = jnp.linspace(0, 1, n)[:, jnp.newaxis]

    kernel = jk.RBF()
    K = kernel.cross_covariance(params=dict(lengthscale=lengthscale_,
                                            variance=output_scale_),
                                x=x, y=x) + 1e-6*jnp.eye(n)
    # plot_cov(K, 'true cov')

    L = jnp.linalg.cholesky(K)
    z = jrnd.normal(key_data, shape=(n,))
    f_true = jnp.dot(L, z) + jnp.zeros_like(z)  # NOTE: True GP had mean=1

    _, obs_key = jrnd.split(key_data)
    y = f_true + obs_noise_*jrnd.normal(obs_key, shape=(n,))

    ground_truth = dict(
        f=f_true,
        lengthscale=lengthscale_,
        variance=output_scale_,
        obs_noise=obs_noise_)

    if path_plot is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(x, f_true, 'k', label=r'$f$')
        plt.plot(x, y, 'rx', label='obs')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.xlim([0., 1.])
        plt.legend()
        plt.savefig(f'./{path_plot}/data.png')
        plt.close()

    return x, y, ground_truth


def plot_smc(x, y, particles, ground_truth, title, folder):
    def particle_avg_hdi(particles):
        # generate average and highest density intervals of SMC particles
        mean = jnp.mean(particles, axis=0)
        hdi_lower = jnp.percentile(particles, q=2.5, axis=0)
        hdi_upper = jnp.percentile(particles, q=97.5, axis=0)
        return mean, hdi_lower, hdi_upper
    
    def plot_smc_posterior(title, figsize=(12, 4)):
        plt.figure(figsize=figsize)

        # observations and true function
        plt.plot(x, ground_truth['f'], color=colors['base01'], label=r'true $f$', lw=1.5, alpha=1, zorder=-1)
        plt.plot(x, y, 'x', label='obs', color=colors['base01'], alpha=0.5, zorder=-1)

        # f | particle average
        fp_mean, fp_hdi_lower, fp_hdi_upper = particle_avg_hdi(particles['f'])
        ax = plt.gca()
        ax.plot(x, fp_mean, label='posterior f', color=colors['cyan'], lw=2.5, zorder=2)
        ax.fill_between(
            x.flatten(), fp_hdi_lower, fp_hdi_upper, 
            alpha=0.5, color=colors['green'], lw=0,
            zorder=1)
        
        # Plot inducing points
        if 'u' in particles.keys():
            up_mean, up_hdi_lower, up_hdi_upper = particle_avg_hdi(particles['u'])
            zp_mean, zp_hdi_lower, zp_hdi_upper = particle_avg_hdi(particles['Z'])
            plt.plot(
                zp_mean, up_mean,
                'x', 
                label='posterior\ninducing variables', 
                color=colors['red'],
                alpha=1.0,
                zorder=2)
        
        plt.legend()
        plt.title(title)
        plt.savefig(f'./{folder}/' + title.replace(' ', '_').replace('\n', '_'))
        plt.close()

    def plot_hyperparameter_histograms(particles, title='', prior_values=None):
        params = list(particles.keys())

        _, axs = plt.subplots(
            nrows=1, ncols=len(params), 
            constrained_layout=True,
            sharex='col', sharey='col', figsize=(12, 4))

        for i, p in enumerate(params):
            
            ax = axs if type(axs) == plt.Axes else axs[i]

            pd = particles[p]
            # There are some outliers that skew the axis
            pd_u, pd_l = jnp.percentile(pd, q=99.9), jnp.percentile(pd, q=0.1)
            pd_filtered = jnp.extract(pd>pd_l, pd)
            pd_filtered = jnp.extract(pd_filtered<pd_u, pd_filtered)
            if pd_filtered.shape[0] == 0:
                pd_filtered = pd

            ax.hist(pd_filtered, bins=30, density=True, color=colors['blue'])
            ax.set_xlabel(p)
            
            if p in ground_truth.keys():
                ax.axvline(
                    x=ground_truth[p], 
                    ls='--', lw=1.7, c=colors['base01'], 
                    label=f'true ({ground_truth[p]})')
            
            mean = jnp.mean(pd_filtered)
            ax.axvline(
                x=mean, 
                ls='--', c=colors['cyan'], lw=2,
                label=f'sample\nmean ({jnp.round(mean, decimals=3)})')
            
            if prior_values:
                ax.axvline(
                x=prior_values[i], 
                ls='--', c=colors['magenta'],
                lw=1.7,
                label=f'prior mean')
                    
            ax.legend()

        plt.suptitle(title)
        plt.savefig(
            f'./{folder}/' + title.replace(' ', '_').replace('\n', '_').replace('$', ''))
        plt.close()

    def plot_inducing_point_histograms(particles, plots_per_row, title=''):
        num_particles, num_inducing_points = particles.shape
        num_rows = num_inducing_points / plots_per_row
        num_rows = int(num_rows if num_inducing_points % plots_per_row == 0 else np.round(num_rows + 0.3))

        fig, axs = plt.subplots(
                nrows=num_rows, ncols=plots_per_row, 
                constrained_layout=True,
                sharex=None, sharey=None, figsize=(12, 2 + (num_rows*2)))
        axs = axs.flatten()

        colors = plt.colormaps.get_cmap('brg')(np.linspace(0, 1, num_inducing_points))

        for i, ax in enumerate(axs):
            if i < num_inducing_points:
                ax.hist(
                    particles[:, i], bins=None, density=True, 
                    color=colors[i], alpha=0.5)
                ax.set_title(i)
            else:
                fig.delaxes(ax)  # delete empty axis

        plt.suptitle(title)
        plt.savefig(
            f'./{folder}/' + title.replace(' ', '_').replace('\n', '_').replace('$', ''))
        plt.close()


    plot_smc_posterior(title=title+'\nposterior f')

    plot_hyperparameter_histograms(
    particles=particles.get('kernel', {}),
    title=title+f'\nposterior cov')
    # prior_values=[prior_lengthscale_mean, prior_variance_mean])

    plot_hyperparameter_histograms(
    particles=particles.get('likelihood', {}),
    title=title+f'\nposterior likelihood')
    #prior_values=[prior_obs_noise_mean])

    plot_inducing_point_histograms(
        particles=particles.get('u'),
        plots_per_row=5,
        title=title+f'\nposterior inducing points $u$')
    plot_inducing_point_histograms(
        particles=particles.get('Z'),
        plots_per_row=5,
        title=title+f'\nposterior inducing points $Z$')


def sparse_gp_inference(seed, path):
    key = jrnd.PRNGKey(seed)

    # generate data
    x, y, ground_truth = generate_data(path_plot=path)

    # setup model parameter
    model_parameter = dict(
        num_inducing_points = 20)
    sampling_parameter = dict(  # SMC parameter
        num_particles = 10,
        num_mcmc_steps = 2)
    logging.info(f'model parameter: {model_parameter}')
    logging.info(f'sampling parameter: {sampling_parameter}')

    # prior
    priors = dict(
        kernel=dict(
            lengthscale = dx.Transformed( 
                dx.Normal(loc=0.0, scale=1.0),
                tfb.Exp()), 

            variance = dx.Transformed( 
                dx.Normal(loc=0.0, scale=1.0),
                tfb.Exp())),

        likelihood=dict(
            obs_noise = dx.Transformed(
                dx.Normal(loc=0.0, scale=1.0), 
                tfb.Exp())),
            
        inducing_inputs_Z=dict( 
            mean=dx.Deterministic(
                loc=jnp.ones(shape=model_parameter['num_inducing_points']) * jnp.mean(x)),
            scale=dx.Deterministic(
                loc=jnp.ones(shape=model_parameter['num_inducing_points']) * jnp.std(x)*.5)))

    # setup model
    gp_sparse = SparseGPModel(
        x, y, 
        cov_fn=jk.RBF(), 
        priors=priors, 
        num_inducing_points=model_parameter['num_inducing_points'])  

    # inference
    key, key_inference = jrnd.split(key)
    start = timer()
    with jax.disable_jit(disable=False):
        initial_particles, particles, _, marginal_likelihood = gp_sparse.inference(
            key_inference, 
            mode='gibbs-in-smc', 
            sampling_parameters=sampling_parameter)
    logging.info(f'execution time: {(timer() - start):9.3f} seconds')
    
    # plot results
    sub_title = ''
    plot_smc(
        x, y, particles.particles, ground_truth, 
        title=f'Sparse GP' + sub_title,
        folder=path)
    
    # TODO: pickle data and infernece output for combining the results later
    

def main():
    num_runs = 3

    random_random_seeds = np.random.randint(0, 10000 + 1, num_runs)

    # setup results folder and logging
    path, paths_sub = setup_results_folder_and_logging(
        static_folder=True, 
        log_level=logging.DEBUG,
        subfolders=random_random_seeds)
    
    for i in range(num_runs):
        logging.info(f'sparse GP run {i}')
        sparse_gp_inference(random_random_seeds[i], path=paths_sub[i])


    # TODO: less smooth data, e.g. box car
    # TODO: add full or marginal latent gp
    # TODO: run over multiple seeds