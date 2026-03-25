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

    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots(figsize=(4, 3))
    
    grouped = df.groupby('entropy_rate')
    cmap = plt.get_cmap('viridis')
    entropies = sorted(df['entropy_rate'].unique())
    
    for i, entropy in enumerate(entropies):
        subset = grouped.get_group(entropy).sort_values(by='sigma')
        color = cmap(i / max(1, len(entropies) - 1))
        
        ax.plot(
            subset['sigma'], 
            subset['final_loss'], 
            marker='o', 
            markersize=3,
            linestyle='-',
            linewidth=1.2,
            color=color,
            label=f'$H/H_{{max}} = {entropy:.2f}$'
        )

    # 1. Impostiamo asse X logaritmico normale
    ax.set_xscale('log')
    
    # 2. Impostiamo asse Y su SYMLOG
    # linthresh è la soglia sotto la quale la scala diventa lineare. 
    # 1e-4 è un buon valore, ma puoi aggiustarlo a 1e-5 se necessario.
    noise_threshold = 1e-4 
    ax.set_yscale('symlog', linthresh=noise_threshold)
    
    # 3. Aggiungiamo la linea teorica del limite statistico (Shot Noise Floor)
    # Assumiamo che n_samples sia costante per tutti i run
    # Se hai salvato 'n_samples' nel dataframe, usalo:
    n_samples = df.iloc[0].get('n_samples', 5000) # Fallback a 5000 se non lo trovi
    shot_noise_floor = 1.0 / n_samples
    
    ax.axhline(
        shot_noise_floor, 
        color='red', 
        linestyle='--', 
        linewidth=0.8, 
        alpha=0.7, 
        label=r'Shot Noise Limit ($\sim 1/s$)'
    )
    
    # Riempiamo l'area sotto il noise floor per far capire che è "zona cieca"
    ax.axhspan(-noise_threshold, shot_noise_floor, color='red', alpha=0.05)
    
    ax.set_xlabel(r'Kernel Bandwidth $\sigma$')
    ax.set_ylabel(r'Final Unbiased $\widehat{\text{MMD}}^2$')
    
    # Ottimizziamo la legenda per non coprire il plot
    ax.legend(title='Entropy Rate', fontsize=5, title_fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Annotazione parametri
    n_qubits = df['n_qubits'].iloc[0]
    max_weight = df['max_weight'].iloc[0]
    text_str = f'$n={n_qubits}$\n$w_{{max}}={max_weight}$\n$s={n_samples}$'
    ax.text(0.8, 0.95, text_str, 
            transform=ax.transAxes, 
            fontsize=6,
            verticalalignment='top',
            # horizontalalignment='right',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8)
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot salvato con successo in: {output_path}")

    
if __name__ == "__main__":
    # Assicurati di puntare alla cartella corretta
    MODELS_DIR = "trained_models" 
    
    print("Caricamento dei dati degli esperimenti...")
    results_df = load_experiment_data(MODELS_DIR)
    
    print("Generazione del plot...")
    plot_mmd_landscape_symlog(results_df, output_path="mmd_sigma_entropy_landscape.pdf")