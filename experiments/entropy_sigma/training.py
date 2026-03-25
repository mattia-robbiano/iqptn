import os
import time
import json
import argparse

import numpy as np
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from tqdm import tqdm

from iqptn.sigma import median_heuristic
from iqptn.utils import convert_to_jnp_ndarray
from iqptn.optimizer import setup_training
from iqptn.models import IQPTensorNetwork, local_gates

key = jax.random.PRNGKey(int(time.time_ns() & 0xFFFFFFFF))


with open('training_parameters.json', 'r') as f:
    params = json.load(f)

parser = argparse.ArgumentParser(description='IQP Tensor Network Training')
parser.add_argument('--identifier')
parser.add_argument('--n_qubits', type=int, default=params['model']['n_qubits'], help='Number of qubits')
parser.add_argument('--max_weight', type=int, default=params['model']['max_weight'], help='Maximum weight of interactions')
parser.add_argument('--n_training_samples', type=int, default=params['distribution']['n_training_samples'], help='Number of training samples')
parser.add_argument('--entropy_rate', type=float, default=params['distribution']['entropy_rate'], help='Entropy rate of the target distribution')
parser.add_argument('--ops_covarage', type=float, default=params['training']['ops_covarage'], help='Coverage of operators')
parser.add_argument('--n_samples', type=int, default=params['training']['n_samples'], help='Number of samples for MMD estimation')
parser.add_argument('--sigma', type=float, default=params['training']['sigma'], help='Sigma for MMD kernel (float or "e")')
parser.add_argument('--lr', type=float, default=params['training']['lr'], help='Learning rate')
parser.add_argument('--epochs', type=int, default=params['training']['epochs'], help='Number of training epochs')

args = parser.parse_args()

id = args.identifier # run indentifier for saving and loading reference
n_qubits     = args.n_qubits
max_weight   = args.max_weight

n_training_samples = args.n_training_samples
entropy_rate       = args.entropy_rate

ops_covarage = args.ops_covarage
n_samples    = args.n_samples
sigma        = args.sigma
lr           = args.lr
epochs       = args.epochs


# Build model
interactions = local_gates(N = n_qubits, max_weight=max_weight)
parameters = jax.random.uniform(key, shape=(len(interactions),), minval=0, maxval=2*jnp.pi)
iqp = IQPTensorNetwork(nqubits=n_qubits,interactions=interactions)
circuit = iqp.build_circuit(parameters)


# Files path handling
file_path = os.path.join("datasets", f"{id}_entropy_rate_{entropy_rate}.dat")

os.makedirs("figures", exist_ok=True)
figure_path = os.path.join("figures", f"{id}_entropy_rate_{entropy_rate}_sigma_{sigma}.pdf")

os.makedirs("trained_models", exist_ok=True)
model_path = os.path.join("trained_models", f"{id}_entropy_rate_{entropy_rate}_sigma_{sigma}.npz")


# Load dataset
training_data = np.loadtxt(file_path, dtype=int, delimiter=',')
print(training_data.shape)

# Optimization
n_states = 2**n_qubits
n_ops = int(n_states * 0.5)
generators = convert_to_jnp_ndarray(interactions, n_qubits=n_qubits)
if sigma=="e": sigma = jnp.sqrt(median_heuristic(X=training_data))

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
# plt.show()


# save = input("Do you want to save training results? (yes/no): ").lower() in ['yes', 'y']
save = True
if save:
    plt.savefig(figure_path)
    print(f"Figure saved to {figure_path}")

    params_np = np.asarray(parameters)
    np.savez(
        file=model_path,
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


