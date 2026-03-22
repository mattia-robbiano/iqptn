import jax.random as random
import jax.numpy as jnp
import numpy as np
from models import IQPTensorNetwork, local_gates
from euristics import sigma_heuristic
from utils import convert_to_jnp_ndarray
from mmd import mmd_mc
from iqpopt.gen_qml import mmd_loss_iqp
from iqpopt import IqpSimulator
from iqpopt.utils import local_gates as local_gates_iqpopt

def test_mmd():
    """Test MMD computation between two implementations on 4x4 Ising dataset."""
    nqubits = 25
    key = random.PRNGKey(42)
    
    # Setup gate lists
    gate_list = local_gates(nqubits, max_weight=3)
    params = random.uniform(key, shape=(len(gate_list),), minval=0, maxval=2*jnp.pi)
    
    # Load ground truth (4x4 Ising)
    ground_truth = jnp.asarray(np.load(f"datasets/ising_L{5}_T2.4_h0.08.npy"))[:1000]
    
    # Compute sigma
    sigma = sigma_heuristic(X=ground_truth)
    
    # Test 1: Custom implementation
    generators = convert_to_jnp_ndarray(gate_list, n_qubits=nqubits)
    mmd_mc_result = mmd_mc(
        params=params,
        generators=generators,
        ground_truth=ground_truth,
        sigma=sigma[0],
        n_ops=min(8, 2**nqubits),
        n_samples=1000,
        key=key,
    )
    
    # Test 2: IQPOpt implementation
    gate_list_iqpopt = local_gates_iqpopt(nqubits, max_weight=3)
    circuit_iqpopt = IqpSimulator(n_qubits=nqubits, gates=gate_list_iqpopt)
    
    mmd_iqpopt_result = mmd_loss_iqp(
        params=params,
        iqp_circuit=circuit_iqpopt,
        ground_truth=ground_truth,
        sigma=sigma[0],
        n_ops=min(8, 2**nqubits),
        n_samples=1000,
        key=key
    )
    
    print("MMD (Custom):", mmd_mc_result)
    print("MMD (IQPOpt):", mmd_iqpopt_result)
    assert jnp.allclose(mmd_mc_result, mmd_iqpopt_result, rtol=0.1), "MMD values differ significantly"

if __name__ == "__main__":
    test_mmd()
