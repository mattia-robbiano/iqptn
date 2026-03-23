import jax
import jax.numpy as jnp
from scipy.optimize import brentq


def generate_distribution_with_target_entropy(n_states: int, target_entropy: float, key: jax.Array) -> jnp.ndarray:
    """Generates a discrete probability distribution with a specific target Shannon entropy.

    This function uses a Boltzmann distribution (softmax) over a random energy landscape.
    It solves for the inverse temperature beta such that the resulting distribution 
    matches the requested entropy. This is useful for benchmarking IQP-based 
    generative models under different data complexity regimes.

    Args:
        n_states (int): The number of possible outcomes (size of the Hilbert space 
            or bitstring space).
        target_entropy (float): Desired Shannon entropy in nats. Must be in the 
            range [0, log(n_states)].
        key (jax.Array): A JAX PRNG key used to generate the underlying 
            random energy landscape.

    Returns:
        jnp.ndarray: A 1D array of shape (n_states,) representing the 
            normalized probability distribution.

    Raises:
        ValueError: If target_entropy is greater than the maximum possible 
            entropy for the given n_states.

    Notes:
        The implementation uses Brent's root-finding method to find the optimal 
        inverse temperature. For target_entropy = log(n_states), a uniform 
        distribution is returned directly.
    """
    max_entropy = float(jnp.log(n_states))
    if target_entropy >= max_entropy:
        return jnp.ones(n_states) / n_states
    elif target_entropy <= 0.0:
        dist = jnp.zeros(n_states)
        # Se entropia 0, collassiamo tutto su un singolo stato a caso
        idx = jax.random.randint(key, shape=(), minval=0, maxval=n_states)
        return dist.at[idx].set(1.0)
    
    # Generiamo un panorama di "energie" casuali
    energies = jax.random.normal(key, (n_states,))
    
    # Shift delle energie per stabilità numerica (log-sum-exp trick)
    shifted_energies = energies - jnp.min(energies)
    
    # Funzione per la distribuzione di Boltzmann
    def boltzmann_dist(beta: float) -> jnp.ndarray:
        exp_terms = jnp.exp(-beta * shifted_energies)
        return exp_terms / jnp.sum(exp_terms)
    
    # Funzione obiettivo per il root-finding: S(beta) - target = 0
    def entropy_diff(beta: float) -> float:
        p = boltzmann_dist(beta)
        # Mask per evitare warning su log(0) se beta è molto grande
        p_safe = jnp.where(p > 0, p, 1e-12)
        current_entropy = -jnp.sum(p * jnp.log(p_safe))
        return float(current_entropy - target_entropy)
    
    # Ricerca dello zero: S(beta) decresce da max_entropy (a beta=0) a 0 (a beta=inf)
    # Cerchiamo il beta ottimale nell'intervallo [0, 1000]
    try:
        beta_opt = brentq(entropy_diff, 0.0, 1000.0)
    except ValueError:
        # Se 1000 non basta per raggiungere energie abbastanza fredde, allarghiamo il bound
        beta_opt = brentq(entropy_diff, 0.0, 100000.0)
        
    return boltzmann_dist(beta_opt)


def boltzmann_batch(
    n_dist: int,
    n_qubits: int,
    key: jax.Array,
    *,
    sample_within_bins: bool = True,
):
    """Generate n_dist distributions with entropies uniformly covering [0, log(n_qubits)]."""
    if n_dist <= 0:
        raise ValueError("n_dist must be > 0")
    if n_qubits <= 1:
        raise ValueError("n_qubits must be > 1 to have a non-zero entropy range")

    n_states = 2 ** n_qubits
    h_max = float(jnp.log(n_qubits))

    # Uniform coverage of entropy interval
    if sample_within_bins:
        # Stratified sampling: one target entropy per equal-width bin
        edges = jnp.linspace(0.0, h_max, n_dist + 1)
        k_main, k_ent, k_dist = jax.random.split(key, 3)
        u = jax.random.uniform(k_ent, shape=(n_dist,))
        target_entropies = edges[:-1] + u * (edges[1:] - edges[:-1])
        target_entropies = target_entropies[jax.random.permutation(k_main, n_dist)]
    else:
        # Deterministic uniform grid
        target_entropies = jnp.linspace(0.0, h_max, n_dist)

    keys = jax.random.split(k_dist if sample_within_bins else key, n_dist)
    distributions = jnp.stack(
        [
            generate_distribution_with_target_entropy(n_states, float(h), k)
            for h, k in zip(target_entropies, keys)
        ]
    )
    return distributions, target_entropies