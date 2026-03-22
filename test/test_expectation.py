import jax.numpy as jnp
import jax
import numpy as np
from iqpopt import IqpSimulator
from iqpopt.utils import local_gates
from iqptn.expectation import expvals_mc

def convert_to_binary_matrix(gate_list: list, n_qubits: int) -> jnp.ndarray:
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


def test_consistency():
    n_qubits = 4
    max_weight = 2

    quimb_gates = local_gates(n_qubits, max_weight) # this is iqpopt function!! with 1 more nesting layer
    print(f"Number of generators: {len(quimb_gates)}")
    generators_matrix = convert_to_binary_matrix(quimb_gates, n_qubits)

    iqp = IqpSimulator(
        n_qubits=n_qubits,
        gates=quimb_gates,
)
    key = jax.random.PRNGKey(0)
    params = jax.random.uniform(key, (len(quimb_gates),))
    ops = generators_matrix[0:10]
    
    res_iqp, _ = iqp.op_expval_batch(params, ops, n_samples=1000, key=key)    
    res_manual, _ = expvals_mc(params, ops, generators_matrix, n_samples=1000, key=key)
    
    np.testing.assert_allclose(res_iqp, res_manual, atol=1e-7)