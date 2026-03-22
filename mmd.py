import jax
import jax.numpy as jnp
from jax import Array
from typing import List, Optional

@jax.jit(static_argnames=['n_samples'])
def _mmd_mc_core(
    params: jnp.ndarray, 
    generators: jnp.ndarray, 
    ground_truth: jnp.ndarray, 
    visible_ops: jnp.ndarray, 
    all_ops: jnp.ndarray, 
    key: Array, 
    n_samples: int
) -> float:
    """
    Core JIT-compiled mathematical function to estimate the MMD^2 loss.
    
    Args:
        params: Phases of the IQP gates (n_generators,).
        generators: Binary matrix of IQP generators (n_generators, n_qubits).
        ground_truth: Training samples (m_train, n_visible_qubits).
        visible_ops: Binary operators acting on the visible subsystem (n_ops, n_visible_qubits).
        all_ops: Binary operators acting on the full system, padded with zeros (n_ops, n_qubits).
        key: JAX PRNG key for Monte Carlo sampling.
        n_samples: Static integer for the number of MC samples.
        
    Returns:
        float: The unbiased estimate of the squared MMD loss.
    """
    n_qubits = generators.shape[1]
    m_train = ground_truth.shape[0]

    # 1. IQP Model Expectations via Monte Carlo Sampling
    # Draw classical random bitstrings
    samples = jax.random.randint(key, shape=(n_samples, n_qubits), minval=0, maxval=2)
    
    # Compute parity and phases
    ops_gen = (all_ops @ generators.T) % 2
    samples_gates = 1 - 2 * ((samples @ generators.T) % 2)
    par_ops_gates = 2 * params * ops_gen
    
    # tr_iqp_samples shape: (n_ops, n_samples)
    tr_iqp_samples = jnp.cos(par_ops_gates @ samples_gates.T)
    
    # Estimate expectation values and the statistical correction for the unbiased variance
    tr_iqp = jnp.mean(tr_iqp_samples, axis=-1)
    correction = jnp.mean(tr_iqp_samples**2, axis=-1) / n_samples

    # 2. Ground Truth Expectations
    # Compute the expectation values of the visible ops on the empirical data
    # Shape: (n_ops,)
    tr_train = jnp.mean(1 - 2 * ((ground_truth @ visible_ops.T) % 2), axis=0)

    # 3. Unbiased MMD^2 Formulation
    # As derived in the paper, we apply finite-sample corrections to avoid biased norms
    term_iqp = (tr_iqp**2 - correction) * n_samples / (n_samples - 1)
    term_cross = -2 * tr_iqp * tr_train
    term_train = (tr_train**2 * m_train - 1) / (m_train - 1)

    res = term_iqp + term_cross + term_train

    # Return the average loss over the sampled operators
    return jnp.mean(res)


# THEORETICAL NOTE: STOCHASTIC APPROXIMATION LEVELS (n_ops vs n_samples)
"""
    The MMD squared loss estimation involves two distinct levels of Monte Carlo 
    approximation, each targeting a different source of exponential complexity.

    1. n_ops (Approximation of the Kernel/Loss Space):
       As derived in Eq. (111) of arXiv:2503.02934, the exact MMD loss with a Gaussian 
       kernel requires a summation over all 2^N_visible Pauli-Z strings. 
       `n_ops` defines the number of operators randomly sampled from this space 
       according to the probability distribution p_sigma. 
       - Low n_ops: High gradient noise but significantly reduced computational overhead.
       - High n_ops: Lower variance in the loss landscape, approaching the exact 
         analytical MMD.

    2. n_samples (Approximation of the Quantum State):
       For each of the chosen `n_ops` operators, we must estimate the quantum 
       expectation value <O>. Calculating this exactly requires full contraction 
       of the 2^N Hilbert space. 
       `n_samples` defines the number of classical bitstrings drawn to compute 
       the Monte Carlo average via the cosine estimator (Eq. 14).
       - This parameter controls the statistical precision of each individual 
         operator measurement ("shots").
"""
def mmd_mc(
    params: jnp.ndarray, 
    generators: jnp.ndarray, 
    ground_truth: jnp.ndarray, 
    sigma: float, 
    n_ops: int, 
    n_samples: int, 
    key: Array, 
    wires: Optional[List[int]] = None
) -> float:
    """
    Returns an unbiased estimate of the squared MMD Loss of an IQP circuit 
    with respect to a classical ground truth distribution.
    
    This function acts as a setup wrapper to generate the random Pauli-Z 
    operators according to the Gaussian kernel bandwidth (sigma), and then 
    dispatches the heavy computation to a JIT-compiled core.

    Args:
        params: The parameters (phases) of the IQP gates.
        generators: The binary matrix defining the IQP circuit structure.
        ground_truth: Array containing training samples as rows (0s and 1s).
        sigma: The bandwidth of the Gaussian kernel.
        n_ops: Number of random operators to sample for the MMD estimation.
        n_samples: Number of MC samples used to estimate the IQP expectation values.
        key: JAX PRNG key.
        wires: List of qubit indices that correspond to the visible data. 
               If None, assumes all qubits are visible.

    Returns:
        float: The estimated MMD^2 loss.
    """
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1 for unbiased estimation.")

    n_qubits = generators.shape[1]
    wires = wires if wires is not None else list(range(n_qubits))
    
    # Split key for operator generation and MC sampling
    key, subkey = jax.random.split(key, 2)
    
    # 1. Generate random observable masks based on the Gaussian kernel mapping
    # Probability of an operator acting on a given visible qubit
    p_MMD = (1 - jnp.exp(-1 / (2 * sigma**2))) / 2
    
    # visible_ops shape: (n_ops, len(wires))
    visible_ops = jax.random.binomial(subkey, 1, p_MMD, shape=(n_ops, len(wires))).astype(jnp.float32)

    # 2. Pad operators to match the full Hilbert space size (n_qubits)
    # This maps the visible subsystem operators to the global IQP architecture
    all_ops_list = []
    wire_idx = 0
    for q in range(n_qubits):
        if q in wires:
            all_ops_list.append(visible_ops[:, wire_idx])
            wire_idx += 1
        else:
            all_ops_list.append(jnp.zeros(n_ops))
            
    all_ops = jnp.array(all_ops_list, dtype=jnp.float32).T
    
    # To match iqpopt randomness management and perform tests
    # TODO remove in the future
    key, subkey_samples = jax.random.split(key, 2)

    # 3. Call the JIT-compiled mathematical core
    loss = _mmd_mc_core(
        params=params, 
        generators=generators, 
        ground_truth=ground_truth, 
        visible_ops=visible_ops, 
        all_ops=all_ops, 
        key=subkey_samples, 
        n_samples=n_samples
    )

    return float(loss)