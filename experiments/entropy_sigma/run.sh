#!/bin/bash
set -e

# Unique identifier of the run
RUN_ID=$(date +"%Y%m%d_%H%M%S")_$$
echo "--- Running: $RUN_ID ---"

# Parameters
N_QUBITS=9
MAX_WEIGHT=2

N_SAMPLES=10000
ENTROPY_RATES=(0.1 0.3 0.5 0.7 0.9)

SPECTRUM_LENGTH=10
OPS_COVERAGE=0.5
N_TRAINING_SAMPLES_MMD=1000
LR=0.01
EPOCHS=2000

echo "Running setup_from_data..."
SIGMAS_STRING=$(python3 setup_from_data.py \
    --identifier "$RUN_ID" \
    --n_qubits "$N_QUBITS" \
    --n_training_samples "$N_SAMPLES" \
    --entropy_rate "${ENTROPY_RATES[@]}" \
    --spectrum_lenght "$SPECTRUM_LENGTH"
    )

sigmas=($SIGMAS_STRING)

echo "setup_from_data successful for ID: $RUN_ID"
echo "Bandwidths ${#sigmas[@]}"
for s in "${sigmas[@]}"; do
    echo " -> Sigma: $s"
done

for e in "${ENTROPY_RATES[@]}"; do
    for s in "${sigmas[@]}"; do
        echo "Running training.py for sigma=$s and entropy=$e"
        python3 training.py \
            --identifier "$RUN_ID" \
            --n_qubits "$N_QUBITS" \
            --max_weight "$MAX_WEIGHT" \
            --n_training_samples "$N_SAMPLES" \
            --entropy_rate "$e}" \
            --ops_covarage "$OPS_COVERAGE" \
            --n_samples "$N_TRAINING_SAMPLES_MMD" \
            --sigma "$s" \
            --lr "$LR" \
            --epochs "$EPOCHS"
    done
done