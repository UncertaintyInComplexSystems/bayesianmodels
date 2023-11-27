import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_dist(ax, 
              x, 
              samples, 
              **kwargs):
    if jnp.ndim(x) > 1:
        x = x.flatten()
    f_mean = jnp.mean(samples, axis=0)
    f_hdi_lower = jnp.percentile(samples, q=2.5, axis=0)
    f_hdi_upper = jnp.percentile(samples, q=97.5, axis=0)
    color = kwargs.get('color', 'tab:blue')
    ax.plot(x, f_mean, lw=2, **kwargs)
    ax.fill_between(x, f_hdi_lower, f_hdi_upper,
                    alpha=0.2, lw=0, color=color)

#

def plot_wishart(x, 
                 Sigma, 
                 axes=None, 
                 figsize=(8, 6), 
                 include_diagonal=True, 
                 add_title=False, 
                 **kwargs):
    n, d, _ = Sigma.shape

    if axes is None:
        _, axes = plt.subplots(nrows=d - 1 + int(include_diagonal), ncols=d - 1 + int(include_diagonal), sharex=True, sharey=True,
                               constrained_layout=True, figsize=figsize)
    for i in range(1 - int(include_diagonal), d):
        for j in range(1 - int(include_diagonal), d):
            if i <= j:
                axes[i, j].plot(x, Sigma[:, i, j], **kwargs)
                if add_title:
                    axes[i, j].set_title(r'$\Sigma_{{{:d}{:d}}}(x)$'.format(i, j))
            else:
                axes[i, j].axis('off')            
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$x$')
    return axes

#
def plot_wishart_dist(x, 
                      Sigma_samples, 
                      axes=None, 
                      figsize=(8, 6), 
                      include_diagonal=True, 
                      add_title=False, 
                      **kwargs):
    _, n, d, _ = Sigma_samples.shape
    color = kwargs.get('color', 'tab:blue')
    offset = 1 - int(include_diagonal)
    if axes is None:
        _, axes = plt.subplots(nrows=d - offset, ncols=d - offset, sharex=True, sharey=True,
                            constrained_layout=True, figsize=figsize)

    for i in range(0, d - offset):
        for j in range(offset, d):
            ax = axes[i, j - offset]
            if i <= j - offset:
                plot_dist(ax,
                          x,
                          Sigma_samples[:, :, i, j],
                          color=color)
                if add_title:
                    ax.set_title(r'$\Sigma_{{{:d}{:d}}}(x)$'.format(i, j))
            else:
                ax.axis('off')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$x$')
    return axes

#
def plot_latents(x, 
                 f, 
                 axes=None):
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