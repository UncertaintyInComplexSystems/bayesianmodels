import os
import sys
import datetime
import logging
import pickle
import csv
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# optional package for debugging
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
        root_folder=False, 
        sub_folders=None,
        id=None,
        log_level=logging.DEBUG):
    """_summary_

    Args:
        - root_folder (string): root folder of experiment
        - log_level (_type_, optional): _description_. Defaults to logging.DEBUG.
        - sub_folder (list(string), optional): creates additional sub_folder if not empty. Defaults to None.

    Returns:
        string: relative filepath
    """

    # create subfolders
    paths_sub = []
    for folder in sub_folders:
        paths_sub.append(root_folder + f'{folder}/')
        os.makedirs(paths_sub[-1], exist_ok = True)

    # disable logging of other modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)

    # setup logging to file and stdout
    logging.basicConfig(
        handlers=[
            logging.FileHandler(root_folder + f'{id}.log'), 
            logging.StreamHandler(sys.stdout)],
        encoding='utf-8', level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s")
    
    # log a first entry
    logging.info('Hello cruel world.')

    return paths_sub



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


def plot_predictive_f(key, x, y, x_true, f_true, predictive_fn, x_pred, title, folder):
    _, key_pred = jrnd.split(key)
    
    f_pred = predictive_fn(key_pred, x_pred)

    # setup plotting
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), 
        sharex=True, sharey=True, 
        constrained_layout=True)

    # plot each particle
    num_particles = f_pred.shape[0]
    ax = axes[0]
    for i in jnp.arange(0, num_particles, step=10):
        ax.plot(
            x_pred, f_pred[i, :], 
            alpha=0.1, color='tab:blue', zorder=2)

    # mean and HDI over particles
    ax = axes[1]
    f_mean = jnp.mean(f_pred, axis=0)
    f_hdi_lower = jnp.percentile(f_pred, q=2.5, axis=0)
    f_hdi_upper = jnp.percentile(f_pred, q=97.5, axis=0)
    ax.plot(
        x_pred, f_mean, 
        color='tab:blue', lw=2, zorder=2, alpha=0.3,
        label='posterior f')
    ax.fill_between(
        x_pred, f_hdi_lower, f_hdi_upper,
        alpha=0.2, color='tab:blue', lw=0)

    # True f, observations and others for all axis
    for ax in axes.flatten():
        ax.plot(x_true, f_true, 'k', label=r'true $f$', zorder=-1, alpha=0.5)
        ax.plot(x, y, 'x', label='obs / inducing points', color='black', alpha=0.5)
        # ax.set_xlim([-10, 10])
        ax.set_ylim([-5., 5.])
        ax.set_xlabel(r'$x$')
        ax.legend()

    axes[0].set_title('SMC particles', fontsize=12)
    axes[1].set_title('Posterior 95% HDI', fontsize=12)
    fig.suptitle(title, fontsize=15)
    plt.savefig(
            f'./{folder}/' + title.replace(' ', '_').replace('\n', '_').replace('$', ''))
    plt.close()
    #axes[0].set_ylabel('Latent GP', rotation=0, ha='right');




def summary_stats_from_log(path_logfile):
    # parse log-file
    import ast

    def extract_single_dict_entris_from_log(file_path, var_name):
        file = open(file_path,'r')

        results = []
        while True:
            next_line = file.readline()
            if not next_line:
                break
            curr_line = next_line.strip()

            if var_name in curr_line:
                # get message from log entry
                d = curr_line.split('[INFO]')[-1].strip()
                # generate dict from message
                dict = ast.literal_eval(d)
                results.append(dict.get(var_name))
        
        file.close()
        return results

    exec_times = extract_single_dict_entris_from_log(
        path_logfile, 'execution_time_sec')
    mse = extract_single_dict_entris_from_log(
        path_logfile, 'mean_squared_error')
    
    summary_stats = dict(
        execution_time=dict(
            mean = np.mean(exec_times),
            variance = np.var(exec_times)
            ),
        mean_squared_error=dict(
            mean = np.mean(mse),
            variance = np.var(mse)
            ),
        )
    
    return summary_stats


def sparse_gp_inference(seed, path):
    key = jrnd.PRNGKey(seed)

    # generate data
    x, y, ground_truth = generate_data(path_plot=path)

    # setup model parameter
    model_parameter = dict(
        num_inducing_points = 20)
    sampling_parameter = dict(  # SMC parameter
        num_particles = 1000,
        num_mcmc_steps = 50)
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
    logging.info('run inference')
    key, key_inference = jrnd.split(key)
    start = timer()
    with jax.disable_jit(disable=False): #, jax.debug_nans():
        initial_particles, particles, _, marginal_likelihood = gp_sparse.inference(
            key_inference, 
            mode='gibbs-in-smc', 
            sampling_parameters=sampling_parameter)
    logging.info(
        '{\'execution_time_sec\': ' + f'{(timer() - start)}' + '}')
    
    # plot results
    logging.info('generate plots')
    sub_title = ''
    plot_smc(
        x, y, particles.particles, ground_truth, 
        title=f'Sparse GP' + sub_title,
        folder=path)
    plot_predictive_f(
        key=key, x=x, y=y, 
        x_true=x, f_true=ground_truth.get('f'),
        predictive_fn=gp_sparse.predict_f,
        x_pred=jnp.linspace(-0.5, 1.5, num=150),
        title='Sparse GP\npredictive f',
        folder=path)
    z = jnp.mean(particles.particles['Z'], axis=0)
    u = jnp.mean(particles.particles['u'], axis=0)
    plot_predictive_f(
        key=key, x=z, y=u, 
        x_true=x, f_true=ground_truth.get('f'),
        predictive_fn=gp_sparse.predict_f_from_u,
        x_pred=jnp.linspace(-0.5, 1.5, num=150),
        title='Sparse GP\npredictive f from u',
        folder=path)
    
    # pickle data and infernece output for combining the results later
    logging.info('pickle data and inference output')
    to_pickle = dict(
        x = x,
        y = y,
        ground_truth = ground_truth,
        initial_particles = initial_particles,
        particles = particles,
        marginal_likelihood=marginal_likelihood)
    for dkey in to_pickle:
        logging.debug('pickle ' + path+f'{dkey}.pickle')
        with open(path+f'{dkey}.pickle', 'wb') as file_handle:
            pickle.dump(
                to_pickle[dkey], file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # compute mean squared error between f particle mean and true f
    def mse(approx, true):
        return jnp.mean(jnp.square(jnp.subtract(approx, true)))

    mse = mse(jnp.mean(particles.particles['f'], axis=0), ground_truth.get('f'))
    logging.info('{\'mean_squared_error\': ' + f'{mse}' + '}')
    

def main():
    note = f'freeZ_20-inducing'

    # create unique folder name for log files and other output
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = f'./results_sparse_gp_test/{timestamp}_samplingF_{note}/'

    # parameters and random seeds  
    num_runs = 3
    random_random_seeds = jrnd.randint(
        jrnd.PRNGKey(23), [num_runs], 0, jnp.iinfo(jnp.int32).max)

    # run sparse gp
    id = 'sparseGP'
    ## setup results folder and logging
    sub_folders = [f'{id}_seed_{s}' for s in random_random_seeds]
    paths = setup_results_folder_and_logging(
        root_folder = path, 
        sub_folders = sub_folders,
        id = id,
        log_level = logging.INFO)
    logging.info(f'experiment id: {id} | {note}')
    logging.info(f'number_runs: {num_runs}')
    
    ## inference for each seed
    for i in range(num_runs):
        logging.info('')  # intentionally left blank
        logging.info(f'run: {i}')
        logging.info(f'seed: {random_random_seeds[i]}')
        sparse_gp_inference(random_random_seeds[i], path=paths[i])

    ## summary stats
    summary_stats = summary_stats_from_log(path + id + '.log')
    logging.info(f'summary_stats: {summary_stats}')

    print(f'\n{id} | summary stats')
    for var in summary_stats:
        print(f' {var}')
        for stat in summary_stats[var]:
            print(f'    {stat}: {summary_stats[var][stat]:0.3f}')

    # run latent gp
    id = 'latentGP'
    #   TODO: add full or marginal latent gp, either is fine as switching is easy
