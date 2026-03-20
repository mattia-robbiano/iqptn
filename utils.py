import jax.numpy as jnp
import numpy as np

def convert_to_jnp_ndarray(gate_list: list, n_qubits: int) -> jnp.ndarray:
    """
    Manually convert a list of gate indices into a binary generator matrix 
    compatible with IQPOptimizer.
    """
    n_generators = len(gate_list)
    matrix = np.zeros((n_generators, n_qubits), dtype=int)
    
    for i, gate in enumerate(gate_list):
        for qubit_idx in gate:
            matrix[i, qubit_idx] = 1
            
    return jnp.array(matrix)
