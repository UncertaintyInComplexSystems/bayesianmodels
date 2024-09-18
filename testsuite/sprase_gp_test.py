import os
import sys
import logging
import datetime
from timeit import default_timer as timer
import configparser
import pickle
from typing import Callable

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import signal

import jax
from jax.config import config
config.update("jax_enable_x64", True)  # crucial for Gaussian processes
config.update("jax_debug_nans", False)
config.update("jax_debug_infs", True)
config.update("jax_disable_jit", False)

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
        root_path=False, 
        sub_folder_names=None,
        log_file_name=None,
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
    for folder in sub_folder_names:
        paths_sub.append(root_path + f'{folder}/')
        os.makedirs(paths_sub[-1], exist_ok = True)

    # disable logging of other modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('jax').setLevel(logging.WARNING)

    # setup logging to file and stdout
    #   using force=True to allow overwriting settings from previous runs 
    logging.basicConfig(
        handlers=[
            logging.FileHandler(root_path + f'{log_file_name}.log'), 
            logging.StreamHandler(sys.stdout)],
        encoding='utf-8', level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True)
    
    # log a first entry
    logging.info('Hello cruel world.')

    return paths_sub


## generate toy data

def generate_smooth_gp(
        path_plot=None, n=100, 
        lengthscale=0.05, scale=5.0, obs_noise=0.3, seed=12345):
    key_data = jrnd.PRNGKey(seed)  # 1106, 5368, 8928, 5609

    lengthscale_ = lengthscale
    output_scale_ = scale
    x = jnp.linspace(-1, 1, n)[:, jnp.newaxis]

    kernel = jk.RBF()
    K = kernel.cross_covariance(
        params=dict(
            lengthscale=lengthscale_,
            variance=output_scale_),
        x=x, y=x) + 1e-6*jnp.eye(n)

    L = jnp.linalg.cholesky(K)
    z = jrnd.normal(key_data, shape=(n,))
    f_true = jnp.dot(L, z) + jnp.zeros_like(z)

    # standerdize data
    # f_true = (f_true - jnp.mean(f_true)) / jnp.std(f_true) 
    _, obs_key = jrnd.split(key_data)
    y = f_true + obs_noise*jrnd.normal(obs_key, shape=(n,))
    # standerdize data
    #x = (x - jnp.mean(x)) / jnp.std(x)

    ground_truth = dict(
        f=f_true,
        lengthscale=lengthscale_,
        variance=output_scale_,
        obs_noise=obs_noise)

    if path_plot is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(x, f_true, 'k', label=r'$f$')
        plt.plot(x, y, 'rx', label='obs')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.xlim([0., 1.])
        plt.title(f'GP\nobs_noise {obs_noise}')
        plt.legend()
        plt.savefig(f'{path_plot}data.png')
        plt.close()

    return dict(x=x, y=y, ground_truth=ground_truth)


def generate_square_data(
        path_plot=None, n=100, obs_noise=0.3, num_periods=2, seed=12345):
        key_data = jrnd.PRNGKey(seed)

        x = jnp.linspace(0, 1, n)[:, jnp.newaxis]
        f_true = signal.square(2*np.pi * num_periods * x).flatten()
        # standerdize data
        f_true = (f_true - jnp.mean(f_true)) / jnp.std(f_true)
        _, obs_key = jrnd.split(key_data)
        y = f_true + obs_noise*jrnd.normal(obs_key, shape=(n,))
        x = (x - jnp.mean(x)) / jnp.std(x)  # standerdize data

        ground_truth = dict(
            f=f_true,
            obs_noise=obs_noise)

        if path_plot is not None:
            plt.figure(figsize=(12, 4))
            plt.plot(x, f_true, 'k', label=r'$f$')
            plt.plot(x, y, 'rx', label='obs')
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.xlim([0., 1.])
            plt.legend()
            plt.title(f'obs_noise {obs_noise}')
            plt.savefig(f'{path_plot}data.png')
            plt.close()

        return dict(x=x, y=y, ground_truth=ground_truth)


def generate_chirp_data(path_plot=None, n=100, obs_noise=0.3, f0=1, f1=3, seed=12345):
        key_data = jrnd.PRNGKey(seed)

        x = jnp.linspace(0, 1, n)[:, jnp.newaxis]
        f_true = signal.chirp(x, f0=f0, f1=f1, t1=1, method='logarithmic').flatten()
        # normalize
        # f_true = (f_true - f_true.min())/ (f_true.max() - f_true.min())
        # f_true *= 4  # HACK: sclae data up for testing purposes
         # standerdize data
        f_true = (f_true - jnp.mean(f_true)) / jnp.std(f_true)
        _, obs_key = jrnd.split(key_data)
        y = f_true + obs_noise*jrnd.normal(obs_key, shape=(n,))

        x = (x - jnp.mean(x)) / jnp.std(x)  # standerdize data

        ground_truth = dict(
            f=f_true,
            obs_noise=obs_noise)

        if path_plot is not None:
            plt.figure(figsize=(12, 4))
            plt.plot(x, f_true, 'k', label=r'$f$')
            plt.plot(x, y, 'rx', label='obs')
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.xlim([0., 1.])
            plt.legend()
            plt.title(f'Chirp {f0}Hz to {f1}Hz, logarithmic\nobs_noise {obs_noise}')
            plt.savefig(f'{path_plot}data.png')
            plt.close()

        return dict(x=x, y=y, ground_truth=ground_truth)


## plotting functions

def plot_smc(x, y, particles, ground_truth, title, folder, inducing_points=True):
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
        # fp_mean, fp_hdi_lower, fp_hdi_upper = particle_avg_hdi(particles['f'])
        # ax = plt.gca()
        # ax.plot(x, fp_mean, label='posterior f', color=colors['cyan'], lw=2.5, zorder=2)
        # ax.fill_between(
        #     x.flatten(), fp_hdi_lower, fp_hdi_upper, 
        #     alpha=0.5, color=colors['green'], lw=0,
        #     zorder=1)
        
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

            ax.hist(pd_filtered, bins=None, density=True, color=colors['blue'])
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
                sharex='all', sharey=None,
                figsize=(2 + (plots_per_row*2), 2 + (num_rows*2)))
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

    def plot_each_particles(particles, title=''):
        num_particles, num_inducing_points = particles.shape

        plt.suptitle(title)
        plt.savefig(
            f'./{folder}/' + title.replace(' ', '_').replace('\n', '_').replace('$', ''))
        plt.close()


    # plot_smc_posterior(title=title+'\nposterior f')

    plot_hyperparameter_histograms(
    particles=particles.get('kernel', {}),
    title=title+f'\nposterior cov')
    # prior_values=[prior_lengthscale_mean, prior_variance_mean])

    plot_hyperparameter_histograms(
    particles=particles.get('likelihood', {}),
    title=title+f'\nposterior likelihood')
    #prior_values=[prior_obs_noise_mean])

    # if inducing_points:  # HACK: only needed cause this function is not model agnostic.
    #     plot_inducing_point_histograms(
    #         particles=particles.get('u'),
    #         plots_per_row=7,
    #         title=title+f'\nposterior inducing points $u$')
    #     plot_inducing_point_histograms(
    #         particles=particles.get('Z'),
    #         plots_per_row=7,
    #         title=title+f'\nposterior inducing points $Z$')


def plot_predictive_f(
        particles,
        points_x, points_y, points_label, points_color,
        x_true, f_true, x_pred, y_pred, title, folder):

    # setup plotting
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(12, 10.5), 
        sharex=True, sharey=True, 
        constrained_layout=True)

    # ax0; plot each particle
    num_particles = y_pred.shape[0]
    ax = axes[0]
    for i in jnp.arange(0, num_particles, step=10):
        ax.plot(
            x_pred, y_pred[i, :], color='tab:blue', 
            alpha=0.1, zorder=2, label='' if i>0 else 'particle')
    ax.plot(points_x, points_y, 'x', label=points_label, color=points_color, alpha=0.7)

    # ax1; mean and HDI over particles
    ax = axes[1]
    ax.plot(points_x, points_y, 'x', label=points_label, color=points_color, alpha=0.7)
    f_mean = jnp.mean(y_pred, axis=0)
    f_hdi_lower = jnp.percentile(y_pred, q=2.5, axis=0)
    f_hdi_upper = jnp.percentile(y_pred, q=97.5, axis=0)
    ax.plot(
        x_pred, f_mean, 
        color='tab:blue', lw=2, zorder=2, alpha=0.3,
        label='posterior f')
    ax.fill_between(
        x_pred, f_hdi_lower, f_hdi_upper,
        alpha=0.2, color='tab:blue', lw=0)
    
    # ax2; inducing points each particle
    ax = axes[2]
    ax.errorbar(
        points_x, points_y, 
        xerr= jnp.std(particles['inducing_points']['Z'], axis=0), 
        yerr= jnp.std(particles['u'], axis=0), 
        color=colors['magenta'],
        fmt='None',
        label='inducing point\nstandard deviation')
    # for i in jnp.arange(0, num_particles, step=10):
    #     ax.plot(
    #         particles['Z'][i, :], particles['u'][i, :],
    #         'x', 
    #         label='' if i > 0 else 'particles\ninducing variables', 
    #         color=colors['red'],
    #         alpha=0.8,
    #         zorder=2)

    # True f, observations and others for all axis
    for ax in axes.flatten():
        ax.plot(x_true, f_true, 'k', label=r'true $f$', zorder=-1, alpha=0.5)
        ax.set_ylim([jnp.min(f_true)-0.5, jnp.max(f_true)+0.5])
        ax.set_xlabel(r'$x$')
        ax.legend()

    axes[0].set_title('SMC particles', fontsize=12)
    axes[1].set_title('Posterior 95% HDI', fontsize=12)
    fig.suptitle(title, fontsize=15)
    plt.savefig(
            f'./{folder}/' + title.replace(' ', '_').replace('\n', '_').replace('$', ''))
    plt.close()
    #axes[0].set_ylabel('Latent GP', rotation=0, ha='right');

    return y_pred


## run algorithms / automatic testing

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
    # mse = extract_single_dict_entris_from_log(
    #     path_logfile, 'mean_squared_error')
    
    summary_stats = dict(
        execution_time=dict(
            mean = np.mean(exec_times),
            variance = np.var(exec_times)
            ),
        # mean_squared_error=dict(
        #     mean = np.mean(mse),
        #     variance = np.var(mse)
        #     ),
        )
    
    return summary_stats


def latent_gp_inference(
    seed,
    model_parameter:dict, 
    sampling_parameter:dict,
    data:dict, path:str):

    key = jrnd.PRNGKey(seed)
    x = data['x']
    y = data['y']
    ground_truth = data['ground_truth']
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
                tfb.Exp())))

    # setup model
    model = FullLatentGPModel(
        x, y, 
        cov_fn=jk.RBF(), 
        priors=priors)  

    # inference
    logging.info('run inference')
    key, key_inference = jrnd.split(key)
    start = timer()
    initial_particles, particles, _, marginal_likelihood = model.inference(
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
        title=f'Latent GP' + sub_title,
        folder=path,
        inducing_points=False)
    plot_predictive_f(
        key=key, points_x=x, points_y=y, 
        x_true=x, f_true=ground_truth.get('f'),
        predictive_fn=model.predict_f,
        x_pred=jnp.linspace(-2.5, 2.5, num=250),
        title='Latent GP\npredictive f',
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


def sparse_gp_inference(
    seed,
    model_parameter:dict, 
    sampling_parameter:dict,
    data:dict, path:str):


    key = jrnd.PRNGKey(seed)
    x = data['x']
    y = data['y']
    ground_truth = data['ground_truth']
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
                
        inducing_points=dict(
            Z=dx.Normal(
                loc=jnp.zeros(shape=model_parameter['num_inducing_points']),
                scale=jnp.ones(
                    shape=model_parameter['num_inducing_points']) * jnp.var(x))  #NOTE: using jnp.var instead of jnp.std to produce Z's within the data range (-1, 1). 
                    )
        
                )
    
    

    # setup model
    gp_sparse = SparseGPModel(
        x, y, 
        cov_fn=jk.RBF(), 
        priors=priors, 
        num_inducing_points=model_parameter['num_inducing_points'],
        f_true=ground_truth.get('f'))  

    # inference
    logging.info('run inference')
    key, key_inference = jrnd.split(key)
    start = timer()
    initial_particles, particles, _, marginal_likelihood = gp_sparse.inference(
        key_inference, 
        mode='gibbs-in-smc', 
        sampling_parameters=sampling_parameter)
    logging.info(
        '{\'execution_time_sec\': ' + f'{(timer() - start)}' + '}')
    logging.info(
        'execution_time_min: ' + f'{(timer() - start)/60}')
    
    # Sort Z and u following Z.
    sorted_inducing_indices = jnp.argsort(particles.particles['inducing_points']['Z'], axis=1)
    particles.particles['inducing_points']['Z'] = jnp.take_along_axis(
        particles.particles['inducing_points']['Z'], sorted_inducing_indices, axis=1)
    particles.particles['u'] = jnp.take_along_axis(
        particles.particles['u'], sorted_inducing_indices, axis=1)


    logging.info('generate predictive')
    key, key_pred = jrnd.split(key)
    x_pred = jnp.linspace(-1.5, 1.5, num=250)
    y_pred = gp_sparse.predict_f(key_pred, x_pred)

    # plot results
    logging.info('generate plots')

    logging.info('call plot_smc')
    sub_title = ''
    plot_smc(
        x, y, particles.particles, ground_truth, 
        title=f'Sparse GP' + sub_title,
        folder=path)
    
    logging.info('call plot_predictive')
    z = jnp.mean(particles.particles['inducing_points']['Z'], axis=0)
    u = jnp.mean(particles.particles['u'], axis=0)
    y_pred = plot_predictive_f(
        particles = particles.particles,
        points_x=z, points_y=u, 
        points_label='inducing points (mean)', points_color=colors['red'],
        x_true=x, f_true=ground_truth.get('f'),
        x_pred=x_pred,
        y_pred=y_pred,
        title='Sparse GP\npredictive',
        folder=path)

    z = jnp.mean(particles.particles['inducing_points']['Z'], axis=0)
    u = jnp.mean(particles.particles['u'], axis=0)
    
    # pickle data and infernece output for combining the results later
    logging.info('pickle data and inference output')
    to_pickle = dict(
        x = x,
        y = y,
        x_pred = x_pred,
        y_pred = y_pred,
        ground_truth = ground_truth,
        initial_particles = initial_particles,
        particles = particles,
        marginal_likelihood = marginal_likelihood)
    
    for dkey in to_pickle:
        logging.debug('pickle ' + path+f'{dkey}.pickle')
        with open(path+f'{dkey}.pickle', 'wb') as file_handle:
            pickle.dump(
                to_pickle[dkey], file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # compute mean squared error between predictive and true f
    def mse(approx, true):
        return jnp.mean(jnp.square(jnp.subtract(approx, true)))

    #mse = mse(jnp.mean(y_pred, axis=0), ground_truth.get('f'))
    #logging.info('{\'mean_squared_error\': ' + f'{mse}' + '}')


def main(args):
    # parse config file
    config = configparser.ConfigParser()
    config.read(args.CONFIG_FILE)

    # parameters 
    num_runs = int(config['DEFAULT']['num_runs'])
    data_type = config['DEFAULT']['data']
    # note = config['DEFAULT']['note']
    note = config['DEFAULT'].get('Note', '')

    # Everything in the config file is encoded as a string.
    # Thus any numerical parameters need to be translated into their correct data type. 
    # TODO: Do automatically by iterating over dict that defines keys to be converted to int values.
    model_parameter = dict( 
        num_inducing_points = int(
            config['model_parameter']['num_inducing_points']))
    sampling_parameter = dict(  # SMC parameter
        num_particles = int(
            config['sampling_parameter']['num_particles']),
        num_mcmc_steps = int(
            config['sampling_parameter']['num_mcmc_steps']))


    # add a 'note' to the run-folder name
    # note = 'implementing'
    #note += f'_{model_parameter["num_inducing_points"]}-inducing'
    #note += f'_{sampling_parameter["num_particles"]}_{sampling_parameter["num_mcmc_steps"]}-smc'
    #note += f'_{data_type}'

    # create root folder name for log files and other output
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = f'./results_sparse_gp_test/{timestamp}_{note}/'
    os.makedirs(path, exist_ok = True)


    # copy used config into run folder for reference
    with open(f'{path}config.ini', 'w') as f:
        config.write(f)

    def gen_data(path, seed=12345):
        # generate data 
        if data_type == 'gp':
            return generate_smooth_gp(
                path_plot=path, 
                lengthscale=float(config['data.gp']['lengthscale']),
                scale=float(config['data.gp']['scale']),
                obs_noise=float(config['data.gp']['obs_noise']),
                seed=seed)
        if data_type == 'square':
            return generate_square_data(path_plot=path)
        if data_type == 'chirp':
            return generate_chirp_data(f0=1, f1=6, path_plot=path)


    # data = gen_data(path=path)
    
    # generate random seeds for inference
    random_random_seeds = jrnd.randint(
        jrnd.PRNGKey(23), [num_runs], 0, jnp.iinfo(jnp.int32).max)
    
    def run_model(
            seeds, id: str, 
            num_runs: int, 
            inference_fn: Callable,
            root_path:str):

        ## setup results folder and logging
        sub_folder_names = [f'{id}_seed_{s}' for s in seeds]
        paths = setup_results_folder_and_logging(
            root_path = root_path, 
            sub_folder_names = sub_folder_names,
            log_file_name = id,
            log_level = logging.INFO)
        logging.info(f'experiment id: {id} | {note}')
        logging.info(f'number_runs: {num_runs}')
        logging.info(f'selected data: {data_type}')

        ## inference for each seed
        for i in range(num_runs):
            print('')  # intentionally left blank
            data = gen_data(path=paths[i], seed=seeds[i])
            logging.info(f'run: {i} | seed: {seeds[i]}')
            inference_fn(
                seeds[i], 
                model_parameter=model_parameter,
                sampling_parameter=sampling_parameter,
                data=data,
                path=paths[i])

        ## summary stats
        summary_stats = summary_stats_from_log(root_path + id + '.log')
        logging.info(f'summary_stats: {summary_stats}')

        print(f'\n{id} | summary stats')
        for var in summary_stats:
            print(f' {var}')
            for stat in summary_stats[var]:
                print(f'    {stat}: {summary_stats[var][stat]:0.3f}')


    # sparse gp
    run_model(
        seeds=random_random_seeds,
        id='sparseGP',
        num_runs = num_runs,
        inference_fn=sparse_gp_inference,
        root_path=path)

    # run latent gp
    # run_model(
    #     seeds=random_random_seeds,
    #     id='latentGP',
    #     num_runs = num_runs,
    #     inference_fn=latent_gp_inference,
    #     data=data,
    #     root_path=path)

