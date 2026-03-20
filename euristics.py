import jax.numpy as jnp
import numpy as np

def gaussian_kernel(sigma: float, x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Calculates the value for the gaussian kernel between two vectors x, y

    Args:
        sigma (float): sigma parameter, the width of the kernel
        x (jnp.ndarray): one of the vectors
        y (jnp.ndarray): the other vector

    Returns:
        float: Result value of the gaussian kernel
    """
    return jnp.exp(-((x-y)**2).sum()/2/sigma**2)


def median_heuristic(X):
    """
    Computes an estimate of the median heuristic used to decide the bandwidth of the RBF kernels; see
    https://arxiv.org/abs/1707.07269
    :param X (array): Dataset of interest
    :return (float): median heuristic estimate
    """
    m = len(X)
    X = np.array(X)
    med = np.median([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for i in range(m) for j in range(m)])
    return med


def sigma_heuristic(X, n_sigmas=1):
    """
    heuristic method used to set the training bandwidths. The bandwidths are chosen to be within
    the SQRT of the median heuristic and the value of sigma that corresponds to an average
    operator weight of 2.
    :param X: training dataset
    :return: list of bandwidths
    """
    med = np.sqrt(median_heuristic(X[:1000]))
    sigma_2 = np.sqrt(-1/(2*np.log(1-4/X.shape[-1])))  # has a mean operator weight of 2
    return list(np.linspace(sigma_2, med, n_sigmas))
