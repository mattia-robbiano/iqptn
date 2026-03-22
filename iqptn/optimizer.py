import jax
import jax.numpy as jnp
import optax
from typing import List, Optional
from iqptn.mmd import _mmd_mc_core 

def setup_training(
    init_params: jnp.ndarray, 
    generators: jnp.ndarray, 
    ground_truth: jnp.ndarray, 
    sigma: float, 
    n_ops: int, 
    n_samples: int, 
    lr: float = 0.01,
    wires: Optional[List[int]] = None
):
    """
    Inizializza l'optimizer state e definisce lo step di training JIT-compilato.
    """
    n_qubits = generators.shape[1]
    wires_arr = jnp.arange(n_qubits) if wires is None else jnp.array(wires)
    p_MMD = (1 - jnp.exp(-1 / (2 * sigma**2))) / 2

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(init_params)

    def loss_fn(params, key):
        key_ops, key_mc = jax.random.split(key, 2)
        all_ops = jnp.zeros((n_ops, n_qubits), dtype=jnp.float32)
        visible_ops = jax.random.binomial(key_ops, 1, p_MMD, shape=(n_ops, len(wires_arr))).astype(jnp.float32)
        all_ops = all_ops.at[:, wires_arr].set(visible_ops)
        
        return _mmd_mc_core(
            params=params, generators=generators, ground_truth=ground_truth,
            visible_ops=visible_ops, all_ops=all_ops, key=key_mc, n_samples=n_samples
        )

    @jax.jit
    def train_step(params, opt_state, key):
        loss_val, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    return opt_state, train_step