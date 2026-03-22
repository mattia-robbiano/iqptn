# IQP Tensor Network (iqptn)

A high-performance Python package for simulating, training, and evaluating **Instantaneous Quantum Polynomial (IQP)** circuits using Tensor Network (TN) contraction, Monte Carlo (MC) sampling, and JAX-accelerated statistical methods.

## Overview

`iqptn` is designed for efficient research on IQP circuits, particularly for tasks involving Maximum Mean Discrepancy (MMD) loss optimization and large-scale sampling. It leverages `quimb` for tensor network operations and `jax` for high-performance, differentiable classical simulations.

### Key Features

- **Efficient Simulation**: Build and simulate IQP circuits using `quimb` Tensor Networks.
- **MMD Loss Calculation**: JIT-compiled MMD loss estimation with unbiased finite-sample corrections, optimized for training generative models.
- **Monte Carlo Estimation**: Randomized estimators for expectation values of Pauli-Z strings, avoiding exponential Hilbert space overhead.
- **Tensor Network Contraction**: Exact and approximate expectation value calculation via TN contraction (supporting MPS/PEPS backends).
- **Ising Data Generation**: Metropolis-Hastings sampler for 2D Ising models to generate training datasets.
- **Statistical Heuristics**: Automatic bandwidth selection (median heuristic) for RBF kernels in MMD.

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
