import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # Assicurati di averlo installato: pip install SciencePlots

def load_experiment_data(models_dir: str) -> pd.DataFrame:
    """
    Legge tutti i file .npz nella directory e ne estrae i parametri rilevanti
    e l'ultimo valore della loss history.
    """
    file_pattern = os.path.join(models_dir, "*.npz")
    files = glob.glob(file_pattern)
    
    if not files:
        raise FileNotFoundError(f"Nessun file trovato in {models_dir}.")

    data_records = []
    
    for file in files:
        try:
            with np.load(file, allow_pickle=True) as data:
                # Estraiamo la final loss
                loss_history = data['loss_history']
                final_loss = loss_history[-1]
                
                # Raccogliamo i metadati
                record = {
                    'sigma': float(data['sigma']),
                    'entropy_rate': float(data['entropy_rate']),
                    'final_loss': float(final_loss),
                    'n_qubits': int(data['nqubits']),
                    'max_weight': int(data['max_weight'])
                }
                data_records.append(record)
        except Exception as e:
            print(f"Errore nella lettura del file {file}: {e}")
            
    # Convertiamo in un DataFrame per manipolarlo facilmente
    df = pd.DataFrame(data_records)
    return df

def plot_mmd_landscape_symlog(df: pd.DataFrame, output_path: str = "mmd_vs_sigma_symlog.pdf"):
    """
    Genera un plot publication-ready usando symlog per gestire il rumore statistico
    attorno allo zero della MMD^2.
    """
    plt.style.use(['science', 'ieee'])
# Aumentiamo leggermente la larghezza per dare respiro
    fig, ax = plt.subplots(figsize=(4.0, 3.0), layout='constrained')
    
    # Ordiniamo le entropie per una legenda coerente
    entropies = sorted(df['entropy_rate'].unique())
    cmap = plt.get_cmap('viridis')
    
    for i, entropy in enumerate(entropies):
        subset = df[df['entropy_rate'] == entropy].sort_values(by='sigma')
        color = cmap(i / max(1, len(entropies) - 1))
        
        ax.plot(
            subset['sigma'], 
            subset['final_loss'], 
            marker='o', 
            markersize=2.5,
            linestyle='-',
            linewidth=1.0,
            color=color,
            label=f'$H/H_{{max}} = {entropy:.2f}$'
        )

    # Configurazione Assi
    ax.set_xscale('log')
    
    # Symlog per gestire i valori vicino a zero o negativi (unbiased estimator)
    # linthresh deve essere circa l'ordine di grandezza del rumore previsto
    n_samples = df['n_samples'].iloc[0] if 'n_samples' in df.columns else 5000
    noise_floor = 1.0 / n_samples
    ax.set_yscale('symlog', linthresh=1e-4) 

    # Linea teorica dello Shot Noise (Note Giulio)
    ax.axhline(noise_floor, color='red', linestyle='--', linewidth=0.7, alpha=0.6, label='Shot Noise $1/s$')

    ax.set_xlabel(r'Kernel Bandwidth $\sigma$')
    ax.set_ylabel(r'Final $\widehat{\text{MMD}}^2$')
    
    # Legenda esterna o meglio posizionata
    ax.legend(
        title='Entropy Rate', 
        fontsize=6, 
        title_fontsize=7, 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), # Sposta la legenda fuori a destra
        borderaxespad=0
    )

    # Annotazione parametri in un angolo libero
    n_qubits = df['nqubits'].iloc[0] if 'nqubits' in df.columns else df['n_qubits'].iloc[0]
    max_w = df['max_weight'].iloc[0]
    ax.text(0.05, 0.05, f'$n={n_qubits}, w_{{max}}={max_w}$\n$s={n_samples}$', 
            transform=ax.transAxes, fontsize=6, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    plt.savefig(output_path, dpi=300)

    
if __name__ == "__main__":
    # Assicurati di puntare alla cartella corretta
    MODELS_DIR = "trained_models" 
    
    print("Caricamento dei dati degli esperimenti...")
    results_df = load_experiment_data(MODELS_DIR)
    
    print("Generazione del plot...")
    plot_mmd_landscape_symlog(results_df, output_path="mmd_sigma_entropy_landscape.pdf")