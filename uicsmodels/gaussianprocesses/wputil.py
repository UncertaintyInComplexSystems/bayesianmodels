import jax.numpy as jnp
import jax

def vec2tril(v, d):
    m = len(v)
    L_sample = jnp.zeros((d, d))
    return L_sample.at[jnp.tril_indices(d, 0)].set(v)

#
def tril2vec(L):
    d = L.shape[0]
    return L[jnp.tril_indices(d, 0)]

#
def outer_self_sum(x):
    def outer_self(x):
        return jnp.outer(x, x)
    return jnp.sum(jax.vmap(outer_self, in_axes=(0))(x), axis=0)

#
def construct_wishart(F, L=None):
    n, nu, d = F.shape
    if L is None:
        L = jnp.eye(d)

    FF = jax.vmap(outer_self_sum, in_axes=(0))(F)
    LFFL = jax.vmap(jnp.matmul,
                    in_axes=(0, None))(jax.vmap(jnp.matmul,
                                                in_axes=(None, 0))(L, FF), L.T)
    return LFFL

#
def construct_wishart_Lvec(F, L_vec):
    _, _, d = F.shape
    L = vec2tril(L_vec, d)
    return construct_wishart(F, L)

#
def cov2corr(Sigma):
    v = jnp.sqrt(jnp.diag(Sigma))
    outer_v = jnp.outer(v, v)
    correlation = Sigma / outer_v
    return correlation

#