import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from quimb.tensor import Circuit
from quimb import pauli

def expvals_contraction(circ: Circuit, sites):
        """
        Calcola i valori di aspettazione per una lista di operatori di Pauli Z.
        tn: il tensor network (già contratto come MPS o PEPS)
        ops: lista di indici, es. [(), [j], [i, j]]
        """
        expvals = []
        for op in sites:
            if len(op) == 1:
                expvals.append(circ.local_expectation(G=pauli("Z"), where=op[0]).real)
            elif len(op) == 2:
                expvals.append(circ.local_expectation(G=pauli("Z")&pauli("Z"), where=op).real)
        return jnp.array(expvals)

def expvals_sampling(circuit: Circuit, ops: jnp.ndarray, n_samples: int, seed: int = None) -> tuple:
    """
    Estimate the expectation values of a batch of Pauli-Z type operators 
    by directly sampling the final state of a quimb Circuit.
    
    Args:
        circuit (qtn.Circuit): L'oggetto Circuit o CircuitMPS da cui campionare.
        ops (jnp.ndarray): Array di shape (l, n_qubits) dove ogni riga è un vettore 
                           binario che indica su quali qubit agisce l'operatore Z.
        n_samples (int): Numero di samples estratti per la statistica.
        seed (int): Seed per il random number generator di quimb.
        
    Returns:
        tuple (jnp.ndarray, jnp.ndarray): I valori di aspettazione stimati e la 
                                          loro deviazione standard (errore standard).
    """

    # Convert list of tuples to jnp array
    if isinstance(ops, list):
        ops_binary = np.zeros((len(ops), circuit.N), dtype=int)
        for i, sites in enumerate(ops):
            for s in sites:
                ops_binary[i, s] = 1
        ops = jnp.array(ops_binary)
    
    samples_raw = circuit.sample(
        n_samples, 
        qubits=None, 
        order=None, 
        group_size=10, 
        max_marginal_storage=2**20, 
        seed=seed, 
        optimize='auto-hq', 
        backend="jax", 
        dtype='complex64', 
        simplify_sequence='ADCRS', 
        simplify_atol=1e-06, 
        simplify_equalize_norms=True
    )
    samples_mat = jnp.array([[int(bit) for bit in s] for s in samples_raw])
    
    # 2. Computazione degli autovalori per il batch di operatori Z
    # ops ha shape (l, n_qubits), samples_mat ha shape (n_samples, n_qubits).
    # Il prodotto (ops @ samples_mat.T) conta quanti operatori Z agiscono su qubit nello stato |1>
    # Applicando il modulo 2 si ottiene la parità: 0 se il numero è pari, 1 se dispari.
    parity = (ops @ samples_mat.T) % 2
    
    # Mappiamo la parità binaria {0, 1} negli autovalori dell'operatore {+1, -1}
    eigenvalues = 1 - 2 * parity
    
    # 3. Calcolo dello stimatore Monte Carlo (media) e del suo standard error
    expvals = jnp.mean(eigenvalues, axis=1)
    
    # La deviazione standard della media (ddof=1 fornisce un estimatore un-biased della varianza campionaria)
    std_devs = jnp.std(eigenvalues, axis=1, ddof=1) / jnp.sqrt(n_samples)
    
    return expvals, std_devs

def expvals_mc(
    params: jnp.ndarray,
    ops: jnp.ndarray,
    generators: jnp.ndarray,
    n_samples: int,
    key: Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Estimate the expectation values of a batch of Pauli-Z operators for an IQP circuit
    using a classical Monte Carlo estimator defined in arXiv:2503.02934.
    
    Args:
        params (jnp.ndarray): The effective parameters (phases) of the IQP gates.
            Shape: (n_generators,)
        ops (jnp.ndarray): A binary matrix specifying the Pauli-Z operators to measure.
            Shape: (n_ops, n_qubits), where 1 indicates a Z operator acting on a qubit.
        generators (jnp.ndarray): A binary matrix specifying the IQP circuit generators.
            Shape: (n_generators, n_qubits).
        n_samples (int): The number of classical Monte Carlo samples to draw.
        key (jax.Array): The JAX PRNG key used to control randomness.
        
    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - mean_expvals (jnp.ndarray): The estimated expectation values for each operator.
              Shape: (n_ops,).
            - std_error (jnp.ndarray): The standard error of the mean for each estimate.
              Shape: (n_ops,).
    """
    n_qubits = generators.shape[1]
    
    # Generate classical random bitstrings (samples from the computational basis)
    samples = jax.random.randint(key, shape=(n_samples, n_qubits), minval=0, maxval=2)
    
    # Compute the parity of the operators with respect to the generators
    # This determines the commutation relations between the observables and the gates
    ops_gen = (ops @ generators.T) % 2
    
    # Compute the parity of the samples with respect to the generators 
    #    and map the boolean domain {0, 1} to physical eigenvalues {+1, -1}
    samples_gates = 1 - 2 * ((samples @ generators.T) % 2)
    
    # Compute the phase angles for the estimator
    par_ops_gates = 2 * params * ops_gen
    
    # Evaluate the Monte Carlo estimator (cosine of the phases)
    # The dot product inherently sums over the generator contributions for each sample
    expvals = jnp.cos(par_ops_gates @ samples_gates.T)
    
    # Compute statistics: expected value (mean) and standard error
    mean_expvals = jnp.mean(expvals, axis=-1)
    std_error = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
    
    return mean_expvals, std_error