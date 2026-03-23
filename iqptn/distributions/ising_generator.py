import numpy as np
from numba import njit
import time
import argparse
from pathlib import Path

@njit
def energetic_cost(grid, x, y, L, h, fields):
    spin = grid[x, y]
    
    # Periodic boundary conditions
    n_up    = grid[(x - 1) % L, y]
    n_down  = grid[(x + 1) % L, y]
    n_left  = grid[x, (y - 1) % L]
    n_right = grid[x, (y + 1) % L]
    
    neighbor_sum = n_up + n_down + n_left + n_right
    
    # If fields is None, use checkerboard pattern
    if fields is not None:
        field = fields[x, y]
    else:
        field = 1 if (x + y) % 2 == 0 else -1
        
    local_field = h * field
    dE = 2 * spin * (neighbor_sum + local_field)
    
    return dE

@njit
def run_metropolis(L, init_steps, all_steps, temperature, h=0.0, fields=None):
    # Initialize Random Grid (-1 and 1)
    grid = np.empty((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            grid[i, j] = 1 if np.random.random() > 0.5 else -1
    
    # Pre-calculate snapshots size to avoid dynamic list resizing in Numba
    num_samples = (all_steps - init_steps + 1999) // 2000
    if num_samples < 0: num_samples = 0
    collection = np.empty((num_samples, L * L), dtype=np.int8)
    sample_idx = 0

    for step in range(all_steps):
        # Pick random site
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        
        dE = energetic_cost(grid, x, y, L, h, fields)
        
        # Metropolis Acceptance Criterion
        if dE <= 0:
            grid[x, y] *= -1
        else:
            if np.random.random() < np.exp(-dE / temperature):
                grid[x, y] *= -1

        # Collect samples after equilibration
        if step >= init_steps and step % 2000 == 0:
            collection[sample_idx] = grid.ravel().copy()
            sample_idx += 1
                
    return (collection + 1) // 2

def main():
    parser = argparse.ArgumentParser(description="Generate Ising dataset using Metropolis algorithm.")
    parser.add_argument("L", type=int, help="Grid size (LxL)")
    parser.add_argument("--init_steps", type=int, help="Initial equilibration steps")
    parser.add_argument("--all_steps", type=int, default=50_000_000, help="Total simulation steps")
    parser.add_argument("--temp", type=float, default=2.4, help="Temperature")
    parser.add_argument("--h", type=float, default=0.08, help="External field strength")
    parser.add_argument("--out_dir", type=str, default="datasets", help="Output directory")
    
    args = parser.parse_args()
    
    L = args.L
    INIT_STEPS = args.init_steps if args.init_steps is not None else L**5
    ALL_STEPS = args.all_steps
    TEMP = args.temp
    h = args.h
    fields = None # Default from notebook

    print(f"Starting Metropolis simulation: L={L}, INIT_STEPS={INIT_STEPS}, ALL_STEPS={ALL_STEPS}, TEMP={TEMP}, h={h}")

    # --- Generate ---
    start = time.perf_counter()
    ising_samples = run_metropolis(L, INIT_STEPS, ALL_STEPS, TEMP, h=h, fields=fields)
    elapsed = time.perf_counter() - start

    print(f"run_metropolis took {elapsed:.2f} seconds")
    print(f"Generated {ising_samples.shape[0]} samples.")

    # --- Save ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"ising_L{L}_T{TEMP:g}_h{h:g}"
    path = out_dir / f"{tag}.npy"

    np.save(path, ising_samples)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
