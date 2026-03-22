# IQP Tensor Network (iqptn)

A high-performance Python package for pseudo-simulation of **Instantaneous Quantum Polynomial (IQP)** circuits using Tensor Network, Monte Carlo and BMS sampling.

## Overview

`iqptn` is designed for estimating Maximum Mean Discrepancy loss, optimization and large-scale sampling. It leverages `quimb` for tensor network operations and `jax` for high-performance, differentiable classical simulations.

## Project Structure

- `iqptn/models.py`: Core `IQPTensorNetwork` class and circuit building utilities.
- `iqptn/mmd.py`: JAX implementation of MMD loss and Monte Carlo estimators.
- `iqptn/expectation.py`: Tools for calculating expectation values via TN contraction and sampling.
- `iqptn/ising_generator.py`: Numba-accelerated Ising model sampler for dataset generation.
- `iqptn/euristics.py`: Statistical heuristics for kernel parameter selection.

## Installation

### Prerequisites

- Python >= 3.10
- JAX
- quimb
- numpy
- numba

### Setup

```bash
git clone https://github.com/your-repo/iqptn.git
cd iqptn
pip install -e .
```

## Quick Start

### 1. Define an IQP Circuit

```python
from iqptn.models import IQPTensorNetwork, local_gates
import jax.numpy as jnp

n_qubits = 10
interactions = local_gates(n_qubits, max_weight=2)
params = jnp.zeros(len(interactions))

model = IQPTensorNetwork(n_qubits, interactions)
circuit = model.build_circuit(params)
```

### 2. Compute MMD Loss (JAX)

```python
from iqptn.mmd import mmd_mc
import jax

key = jax.random.PRNGKey(42)
ground_truth = jnp.array(...) # Your training data (bitstrings)

loss = mmd_mc(
    params=params,
    generators=jnp.array(interactions_binary), # Binary matrix of generators
    ground_truth=ground_truth,
    sigma=1.0,
    n_ops=1000,     # Number of operators to sample for MMD
    n_samples=2048, # Number of MC samples per operator
    key=key
)
```

### 3. Estimate Expectation Values

```python
from iqptn.expectation import expvals_sampling

# Estimate Z-type expectations by sampling the TN
ops = jnp.array([[1, 1, 0, ...], [0, 1, 1, ...]]) # Pauli-Z strings
expvals, std_errors = expvals_sampling(circuit, ops, n_samples=1024)
```

## Theory

The package implements the stochastic approximation levels described in [arXiv:2503.02934](https://arxiv.org/abs/2503.02934), specifically:
1. **Operator Sampling (`n_ops`)**: Approximating the Kernel/Loss space (Eq. 111).
2. **State Sampling (`n_samples`)**: Approximating quantum expectation values via the cosine estimator (Eq. 14).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
