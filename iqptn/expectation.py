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
            else: raise ValueError("Only 1 or 2-site operators are supported.")

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


# TECHNICAL NOTE: COMMUTATION RULES AND THE '@' OPERATOR
"""
    The mathematical logic follows the randomized method for estimating IQP expectation 
    values as derived in Eq. (14) of arXiv:2503.02934.

    The instruction `ops_gen = (ops @ generators.T) % 2`:

    Measuring a Pauli-Z string (an 'op') at the end of an IQP circuit is equivalent
    to measuring a Pauli-X string before the Hadamard layer. Let's call this O_X.
    The expectation value depends on how O_X evolves under the diagonal unitary U:
    U† O_X U = exp(-i Σ θ_k Z_gk) O_X exp(i Σ θ_k Z_gk)

    - If a generator Z_gk COMMUTES with O_X, they bypass each other: the gate 
        has no effect on that specific observable.
    - If a generator Z_gk ANTI-COMMUTES with O_X, the observable picks up a phase:
        exp(-i θ Z) O_X exp(i θ Z) = O_X exp(2i θ Z).

    A Pauli-X string and a Pauli-Z string anti-commute if and only if they share
    an ODD number of active qubits (sites where both have a non-identity operator).

    - `ops @ generators.T`: This matrix multiplication performs a dot product between 
        each observable and each generator, effectively counting the overlapping qubits.
    - `% 2`: This filtering operation extracts the parity of the overlap.
        * 0 (Even) -> Commutation: The phase is 0, the gate is ignored.
        * 1 (Odd)  -> Anti-commutation: The phase is 2*θ, the gate contributes.

    EIGENVALUE MAPPING:
    The line `samples_gates = 1 - 2 * ((samples @ generators.T) % 2)` evaluates the 
    eigenvalues of the generators Z_gk on the classical bitstrings |s>. 
    Mathematically, it computes (-1)^{s · g_k}, mapping the binary parity {0, 1} 
    to the physical spectral domain {+1, -1}.

    THE COSINE ESTIMATOR (Eq. 14):
    The final step `expvals = jnp.cos(par_ops_gates @ samples_gates.T)` implements the 
    analytical expectation value. 
    The term `par_ops_gates @ samples_gates.T` calculates the total accumulated phase 
    Φ_O(s) for each sample, summing only the contributions from anti-commuting gates 
    weighted by their eigenvalues.
    Since the expectation value is the real part of the interference sum over 
    computational basis states, the cosine of the total phase provides an unbiased 
    Monte Carlo estimator of the quantum observable:
    ⟨O⟩ = E_s [ cos( Σ_k 2θ_k · δ_anticomm(O, g_k) · (-1)^{s · g_k} ) ]

"""
# -----------------------------------------------------------------------------------------
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
            params (jnp.ndarray): The effective parameters (phases $\theta_k$) of the IQP evolution.
                Shape: (n_generators,). These are the trainable weights of the quantum circuit.
            ops (jnp.ndarray): A binary matrix specifying the physical observables (Pauli-Z strings) 
                to measure at the end of the circuit. Shape: (n_ops, n_qubits). 
                These act as the "queries" we make to the final quantum state.
            generators (jnp.ndarray): A binary matrix specifying the IQP circuit's architecture 
                (the Pauli-Z strings $g_k$ that generate the unitaries). Shape: (n_generators, n_qubits).
                These define the physical interactions within the system.
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
    # and map the boolean domain {0, 1} to physical eigenvalues {+1, -1}
    samples_gates = 1 - 2 * ((samples @ generators.T) % 2)
    
    # Apply phases
    par_ops_gates = 2 * params * ops_gen
    
    # Evaluate the Monte Carlo estimator (cosine of the phases)
    # The dot product sums over the generator contributions for each sample
    expvals = jnp.cos(par_ops_gates @ samples_gates.T)
    
    mean_expvals = jnp.mean(expvals, axis=-1)
    std_error = jnp.std(expvals, axis=-1, ddof=1) / jnp.sqrt(n_samples)
    
    return mean_expvals, std_error