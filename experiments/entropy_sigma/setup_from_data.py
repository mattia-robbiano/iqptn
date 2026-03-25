import time
import argparse
import os
import numpy as np

import jax.numpy as jnp
import jax.random

from iqptn.sigma import sigma_spectrum
from iqptn.distributions.boltzman_entropy_generator import generate_distribution_with_target_entropy, sample_dataset_from_distribution

key = jax.random.PRNGKey(int(time.time_ns() & 0xFFFFFFFF))


parser = argparse.ArgumentParser(description='Setup data handling')
parser.add_argument('--identifier')
parser.add_argument('--n_qubits',           type=int,   help='Number of qubits')
parser.add_argument('--n_training_samples', type=int,   help='Number of training samples')
parser.add_argument('--entropy_rate',       type=float, nargs='+', help='Entropy rate of the target distribution')
parser.add_argument('--spectrum_lenght',    type=int, help='Lenght of list of bandwidths')
args = parser.parse_args()

n_qubits            = args.n_qubits
n_training_samples  = args.n_training_samples
entropy_rate        = args.entropy_rate
spectrum_lenght     = args.spectrum_lenght

sigmas_ub=[]
sigmas_lb=[]
for entropy_rate_value in entropy_rate:

    n_states = 2**n_qubits
    target_entropy = jnp.log(n_training_samples) * entropy_rate_value
    key_dist, key_sample, key = jax.random.split(key, 3)
    ground_truth_dist = generate_distribution_with_target_entropy(n_states, target_entropy, key_dist)
    training_data = sample_dataset_from_distribution(ground_truth_dist, n_qubits, n_training_samples, key_sample)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    # Save the training data
    file_path = os.path.join(datasets_dir, f"{args.identifier}_entropy_rate_{entropy_rate_value}.dat")
    np.savetxt(file_path, training_data, fmt='%d', delimiter=',')

    sigmas = sigma_spectrum(X=training_data, n_sigmas=2)
    sigmas_lb.append(min(sigmas))
    sigmas_ub.append(max(sigmas))

lower_bound = max(min(sigmas_lb)*0.5, 1e-4)
upper_bound = max(sigmas_ub) + 0.5 * max(sigmas_ub)
sigmas_overall=[float(s) for s in jnp.linspace(lower_bound, upper_bound, spectrum_lenght)]
print(' '.join(map(str, sigmas_overall)))
