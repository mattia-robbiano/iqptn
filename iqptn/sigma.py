import jax
import jax.numpy as jnp

@jax.jit
def median_heuristic(X: jnp.ndarray) -> jnp.ndarray:
    """
    Computes an estimate of the median heuristic used to decide the bandwidth of the RBF kernels; see
    https://arxiv.org/abs/1707.07269
    Args:
        X (jnp.ndarray): dataset
    
    Returns:
        float: median heuristic estimate
    """
    # Compute pairwise squared Euclidean distances using broadcasting
    # (m, 1, d) - (1, m, d) -> (m, m, d)
    diffSq = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    distances = jnp.sqrt(diffSq)
    return jnp.median(distances)


def sigma_spectrum(X: jnp.ndarray, n_sigmas: int = 1) -> list[float]:
    """
    Returns a list of sigmas between the median heuristic and sigma_2, corresponding to an average operator weight of 2.
    This should compose a broad significant spectrum of bandwidths to test
    
    Args:
        X (jnp.ndarray): dataset
        n_sigmas (int): number of sigma values to generate
    
    Returns:
        list[float]: list of bandwidths
    """
    med = median_heuristic(X[:2000])
    sigma_2 = jnp.sqrt(-1 / (2 * jnp.log(1 - 4 / X.shape[-1])))  # has a mean operator weight of 2
    lower_bound = jnp.maximum(sigma_2, 1e-4)
    upper_bound = jnp.maximum(med, 1e-4)
    return [float(s) for s in jnp.linspace(lower_bound, upper_bound, n_sigmas)]
