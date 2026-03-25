import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scienceplots

# Assumo che tu importi la tua mmd_mc e i generatori da iqpopt
# from iqpopt.gen_qml.sample_methods import mmd_mc 

def run_samples_experiment(
    model_path: str, 
    ground_truth: jnp.ndarray, # Passa il ground truth rigenerato con la stessa key/entropia
    generators: jnp.ndarray,   # Passa i generatori dell'architettura
    n_ops: int = 10000,        # Molto alto per sopprimere la varianza MC sugli operatori
    min_samples: int = 10,
    max_samples: int = 100000,
    n_points: int = 20
):
    print(f"Caricamento modello da: {model_path}")
    data = np.load(model_path)
    params = jnp.array(data['params'])
    sigma = float(data['sigma'])
    
    # Generiamo un array logaritmico per i samples
    samples_array = np.logspace(np.log10(min_samples), np.log10(max_samples), n_points).astype(int)
    
    mmd_values = []
    key = jax.random.PRNGKey(42)
    
    print("Inizio valutazione MMD...")
    for s in samples_array:
        key, subkey = jax.random.split(key)
        
        # Valutiamo la MMD con il numero corrente di samples s
        loss_val = mmd_mc(
            params=params,
            generators=generators,
            ground_truth=ground_truth,
            sigma=sigma,
            n_ops=n_ops,
            n_samples=int(s),
            key=subkey
        )
        
        # Poiché stiamo plottando in log-log, gestiamo eventuali fluttuazioni negative
        # prendendo il valore assoluto (ci interessa la magnitudo del bias/errore)
        mmd_values.append(float(jnp.abs(loss_val)))
        print(f"Samples: {s:6d} | MMD^2: {mmd_values[-1]:.2e}")

    return samples_array, np.array(mmd_values), sigma

def plot_shot_noise_scaling(samples_array, mmd_values, sigma, output_path="shot_noise_scaling.pdf"):
    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot empirico
    ax.loglog(
        samples_array, 
        mmd_values, 
        marker='o', 
        markersize=3, 
        linestyle='-', 
        color='blue',
        label=r'Empirical $|\widehat{\text{MMD}}^2|$'
    )
    
    # Linea teorica 1/s
    # Fissiamo l'intercetta della linea teorica per matchare il primo punto empirico
    c = mmd_values[0] * samples_array[0]
    theoretical_line = c / samples_array
    
    ax.loglog(
        samples_array, 
        theoretical_line, 
        linestyle='--', 
        color='red', 
        linewidth=1.2,
        label=r'Theoretical scaling $\sim \mathcal{O}(s^{-1})$'
    )
    
    ax.set_xlabel(r'Number of samples $s$ (Log scale)')
    ax.set_ylabel(r'$|\widehat{\text{MMD}}^2|$ (Log scale)')
    
    ax.legend(fontsize=7, loc='upper right')
    
    # Annotazione
    text_str = f'$\sigma={sigma:.2f}$'
    ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# Seleziona il modello convergente
target_npz = "../entropy_sigma/trained_model/..._entropy_1.0_sigma_1.5.npz"
ground_truth = 
generators = 
s_arr, mmd_vals, sig = run_samples_experiment(target_npz, ground_truth, generators)
plot_shot_noise_scaling(s_arr, mmd_vals, sig)