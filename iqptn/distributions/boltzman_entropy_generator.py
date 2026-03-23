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

import jax
import jax.numpy as jnp

def sample_dataset_from_distribution(probabilities: jnp.ndarray, n_qubits: int, n_samples: int, key: jax.Array) -> jnp.ndarray:
    """
    Campiona un dataset di training di bitstrings a partire da una distribuzione di probabilità.
    
    Args:
        probabilities (jnp.ndarray): La distribuzione di probabilità esatta su 2^n_qubits stati 
                                     (shape: (2^n_qubits,)).
        n_qubits (int): Il numero di qubit (ovvero il numero di features).
        n_samples (int): Il numero di campioni da estrarre (grandezza del dataset).
        key (jax.Array): Chiave PRNG di JAX.
        
    Returns:
        jnp.ndarray: Il dataset di training di shape (n_samples, n_qubits) contenente 0 e 1.
    """
    n_states = probabilities.shape[0]
    
    # 1. Campioniamo gli interi (da 0 a 2^n_qubits - 1) usando le probabilità target
    sampled_indices = jax.random.choice(key, jnp.arange(n_states), shape=(n_samples,), p=probabilities)
    
    # 2. Decodifichiamo gli indici interi in bitstrings
    # Creiamo un array di shift per isolare ogni bit tramite operatori bit-a-bit.
    # Esempio per n_qubits=3: shifts = [2, 1, 0]
    shifts = jnp.arange(n_qubits - 1, -1, -1)
    
    # Utilizziamo il broadcasting: (n_samples, 1) >> (n_qubits,) -> (n_samples, n_qubits)
    # L'AND bit a bit con 1 estrae l'n-esimo bit.
    bitstrings = (sampled_indices[:, None] >> shifts) & 1
    
    return bitstrings