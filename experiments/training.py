import time
import json

import numpy as np
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from tqdm import tqdm

from iqptn.euristics import sigma_heuristic
from iqptn.utils import convert_to_jnp_ndarray
from iqptn.optimizer import setup_training
from iqptn.models import IQPTensorNetwork, local_gates
from iqptn.distributions.boltzman_entropy_generator import generate_distribution_with_target_entropy, sample_dataset_from_distribution

with open('training_parameters.json', 'r') as f:
    params = json.load(f)

n_qubits = params['model']['n_qubits']
max_weight = params['model']['max_weight']
n_training_samples = params['distribution']['n_training_samples']
entropy_rate = params['distribution']['entropy_rate']
ops_covarage = params['training']['ops_covarage']
lr = params['training']['lr']
n_samples = params['training']['n_samples']
epochs = params['training']['epochs']


key = jax.random.PRNGKey(int(time.time_ns() & 0xFFFFFFFF))
interactions = local_gates(N = n_qubits, max_weight=max_weight)
parameters = jax.random.uniform(key, shape=(len(interactions),), minval=0, maxval=2*jnp.pi)
iqp = IQPTensorNetwork(nqubits=n_qubits,interactions=interactions)
circuit = iqp.build_circuit(parameters)

n_states = 2**n_qubits
target_entropy = jnp.log(n_training_samples) * entropy_rate
key_dist, key_sample = jax.random.split(key)
ground_truth_dist = generate_distribution_with_target_entropy(n_states, target_entropy, key_dist)
training_data = sample_dataset_from_distribution(ground_truth_dist, n_qubits, n_training_samples, key_sample)


n_ops = int(n_states * 0.5)
generators = convert_to_jnp_ndarray(interactions, n_qubits=n_qubits)
sigma = sigma_heuristic(X=training_data)[0]
opt_state, train_step = setup_training(
    init_params=parameters, 
    generators=generators, 
    ground_truth=training_data, 
    sigma=sigma, 
    n_ops=n_ops, 
    n_samples=n_samples, 
    lr=lr,
)

loss_history = []
for _ in tqdm(range(epochs), desc="Training", leave=False):
    key, subkey = jax.random.split(key)
    parameters, opt_state, loss_val = train_step(parameters, opt_state, subkey)
    loss_history.append(loss_val)
loss_history_np = np.asarray(jnp.stack(loss_history))

print("Settings:")
print(f"n_qubits      : {n_qubits}")
print(f"n_ops         : {n_ops},    covarage:   {ops_covarage}")
print(f"lr            : {lr}")
print(f"n_samples     : {n_samples}")
print(f"epochs        : {epochs}")
print(f"sigma         : {float(sigma):.6f}")
print(f"n_interactions: {len(interactions)}")
print(f"n_parameters  : {parameters.shape[0]}")
print(f"n_training_samples: {n_training_samples}")
print(f"entropy_rate : {entropy_rate}")
print(f"max_weight   : {max_weight}")


plt.figure(figsize=(7, 4))
plt.plot(loss_history_np)
plt.xlabel("Epoch")
plt.ylabel("MMD Loss")
plt.title("Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

save = input("Do you want to save training results? (yes/no): ").lower() in ['yes', 'y']
if save:
    params_np = np.asarray(parameters)
    np.savez(
        f"training_results{key}.npz",
        loss_history=loss_history_np,
        params=params_np,
        epochs=epochs,
        nqubits=n_qubits,
        sigma=float(sigma),
        n_ops=n_ops,
        lr=lr,
        n_samples=n_samples,
        n_training_samples=n_training_samples,
        entropy_rate=entropy_rate,
        ops_coverage=ops_covarage,
        max_weight=max_weight,
    )


