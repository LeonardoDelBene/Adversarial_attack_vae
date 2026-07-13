"""
Script per creare due grafici che metteno in relazione LPIPS del subject
con due metriche di Qwen:
    1) lo score medio di attack success (1-7)
    2) l'attack success rate (frazione di campioni con score >= soglia)

Entrambi i grafici mostrano un piano cartesiano con pallini colorati per
configurazione. Ogni METODO (VAE_MSE, VAE_MSE_FT, VAE_MSE_FT_2_STAGE,
DiffVax, ...) ha un colore di base proprio (es. rosso, blu, verde...).
All'interno dello stesso metodo, le tre PIPELINE (SD_Inpainting,
SD_Img2Img, InstructionPix2Pix) sono rappresentate con sfumature
diverse dello stesso colore base:
    - SD_Inpainting       -> sfumatura CHIARA
    - SD_Img2Img          -> sfumatura INTERMEDIA
    - InstructionPix2Pix  -> sfumatura SCURA

In questo modo, ad esempio, tutti i pallini "VAE_MSE" sono nella stessa
famiglia di rosso, ma puoi distinguere a colpo d'occhio se un punto è
Inpainting (rosso chiaro), Img2Img (rosso medio) o InstructionPix2Pix
(rosso scuro).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import re


# ============================================================================
# CONFIGURAZIONE: Lista dei percorsi ai file global_summary.txt (o CSV)
# ============================================================================
# Modifica questa lista aggiungendo i percorsi ai tuoi file
DATA_FILES = [
    'output/SD_Inpainting/full_dataset/VAE_MSE_FT_2_STAGE/global_summary.txt',
    'output/SD_Img2Img/full_dataset/VAE_MSE_FT_2_STAGE/global_summary.txt',
    'output/InstructionPix2Pix/full_dataset/VAE_MSE_FT_2_STAGE/global_summary.txt',
    'output/SD_Inpainting/full_dataset/DiffVax/global_summary.txt',
    'output/SD_Img2Img/full_dataset/DiffVax/global_summary.txt',
    'output/InstructionPix2Pix/full_dataset/DiffVax/global_summary.txt',
    'output/SD_Inpainting/full_dataset/VAE_MSE_FT/global_summary.txt',
    'output/SD_Img2Img/full_dataset/VAE_MSE_FT/global_summary.txt',
    'output/InstructionPix2Pix/full_dataset/VAE_MSE_FT/global_summary.txt',
    'output/SD_Inpainting/full_dataset/VAE_MSE/global_summary.txt',
    'output/SD_Img2Img/full_dataset/VAE_MSE/global_summary.txt',
    'output/InstructionPix2Pix/full_dataset/VAE_MSE/global_summary.txt',
]


# ============================================================================
# COLORI BASE DISPONIBILI (uno per ogni METODO, es. VAE_MSE, DiffVax, ...)
# ============================================================================
BASE_COLORS = [
    '#1f77b4',  # blu
    '#ff7f0e',  # arancione
    '#2ca02c',  # verde
    '#d62728',  # rosso
    '#9467bd',  # viola
    '#8c564b',  # marrone
    '#e377c2',  # rosa
    '#7f7f7f',  # grigio
    '#bcbd22',  # giallo-verde
    '#17becf',  # ciano
]

# ============================================================================
# PIPELINE RICONOSCIUTE E LORO LUMINOSITA' RELATIVA
# ============================================================================
# Valori di "lightness" (in scala HLS, 0=nero, 1=bianco) usati per
# generare la sfumatura di ciascuna pipeline a partire dal colore base
# del metodo. SD_Inpainting = chiaro, SD_Img2Img = intermedio,
# InstructionPix2Pix = scuro.
PIPELINE_LIGHTNESS = {
    'SD_Inpainting': 0.78,        # chiaro
    'SD_Img2Img': 0.55,           # intermedio (vicino al colore base originale)
    'InstructionPix2Pix': 0.32,   # scuro
}

# Ordine di disegno/legenda delle pipeline (facoltativo, solo estetico)
PIPELINE_ORDER = ['SD_Inpainting', 'SD_Img2Img', 'InstructionPix2Pix']


def shade_color(hex_color: str, lightness: float) -> str:
    """
    Genera una variante più chiara/scura di un colore esadecimale,
    mantenendo hue e saturazione invariati e modificando solo la
    luminosità (modello HLS). Questo garantisce che tutte le sfumature
    di un metodo restino nella stessa "famiglia" di colore.

    Args:
        hex_color: colore base in formato '#rrggbb'
        lightness: nuova luminosità desiderata, in [0, 1]
                    (0 = nero, 1 = bianco)

    Returns:
        Colore esadecimale risultante.
    """
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return mcolors.to_hex((r2, g2, b2))


def extract_pipeline_from_path(filepath: str) -> str:
    """
    Estrae il nome della pipeline (SD_Inpainting, SD_Img2Img,
    InstructionPix2Pix, ...) dal percorso del file, cercando una
    qualsiasi delle chiavi conosciute in PIPELINE_LIGHTNESS tra le
    componenti del path.
    """
    parts = Path(filepath).parts
    for part in parts:
        if part in PIPELINE_LIGHTNESS:
            return part
    # Fallback: nessuna pipeline conosciuta trovata nel path
    return 'Unknown'


def extract_method_from_path(filepath: str) -> str:
    """
    Estrae il nome del metodo (es. VAE_MSE, VAE_MSE_FT,
    VAE_MSE_FT_2_STAGE, DiffVax, ...) dal percorso del file.
    Per convenzione nello script originale, è il nome della
    directory che contiene il file (parent immediato).
    """
    return Path(filepath).parent.name


def read_metrics_from_csv(filepath: str) -> pd.DataFrame:
    """
    Legge le metriche da un file CSV nel formato metriche_globali.csv.

    Colonne rilevanti:
    - name: nome della configurazione
    - subject_lpips_orig: LPIPS del subject (immagine originale)
    - subject_lpips_edited: LPIPS del subject (immagine immunizzata)
    - attack_success_score: score medio di attack success (Qwen)
    - attack_success_rate (opzionale): frazione di campioni con score >= soglia
    """
    df = pd.read_csv(filepath)

    # Calcola la differenza LPIPS tra originale e immunizzata
    df['lpips_difference'] = df['subject_lpips_edited'] - df['subject_lpips_orig']

    # Rinomina per chiarezza
    df['qwen_score'] = df['attack_success_score']

    # Se il CSV ha già una colonna 'attack_success_rate', viene mantenuta
    # com'è; altrimenti la lasciamo assente (verrà gestita più avanti).

    return df


def read_metrics_from_global_summary(filepath: str) -> pd.DataFrame:
    """
    Legge le metriche da un file global_summary.txt.

    Estrae:
    - Subject LPIPS dalla sezione "Original vs Immunized LPIPS"
    - Average attack success score dalla sezione "Qwen Attack Evaluation Summary"
    """
    data = {
        'lpips_subject': [],
        'qwen_score': [],
        'attack_success_rate': [],
        'name': [],
        'method': [],
        'pipeline': [],
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Estrai Subject LPIPS dalla sezione "Original vs Immunized LPIPS"
        lpips_start = content.find("---- Original vs Immunized LPIPS ----")
        if lpips_start != -1:
            lpips_section = content[lpips_start:lpips_start + 500]
            # Cerca "Subject LPIPS: X.XXXX"
            lpips_match = re.search(r'Subject LPIPS:\s+([\d.]+)', lpips_section)
            if lpips_match:
                lpips_value = float(lpips_match.group(1))
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        # Estrai Average attack success score e Attack success rate dalla
        # sezione "Qwen Attack Evaluation Summary"
        qwen_start = content.find("=== Qwen Attack Evaluation Summary ===")
        if qwen_start != -1:
            qwen_section = content[qwen_start:qwen_start + 500]

            qwen_match = re.search(r'Average attack success score.*?:\s+([\d.]+)', qwen_section)
            if qwen_match:
                qwen_value = float(qwen_match.group(1))
            else:
                return pd.DataFrame()

            # Cerca "Attack success rate: X.XXXX" (es. 0.1800)
            rate_match = re.search(r'Attack success rate:\s+([\d.]+)', qwen_section)
            if rate_match:
                rate_value = float(rate_match.group(1))
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        # Nome del metodo = nome della directory che contiene il file
        method_name = extract_method_from_path(filepath)
        # Nome della pipeline = componente del path (SD_Inpainting, ecc.)
        pipeline_name = extract_pipeline_from_path(filepath)
        # Nome di configurazione completo, usato come etichetta unica
        config_name = f"{method_name} ({pipeline_name})"

        data['lpips_subject'].append(lpips_value)
        data['qwen_score'].append(qwen_value)
        data['attack_success_rate'].append(rate_value)
        data['name'].append(config_name)
        data['method'].append(method_name)
        data['pipeline'].append(pipeline_name)

        df = pd.DataFrame(data)
        return df

    except Exception as e:
        print(f"Errore nel parsing di {filepath}: {e}")
        return pd.DataFrame()


def load_all_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Carica i dati da tutti i file forniti.
    """
    all_data = []

    for filepath in file_paths:
        if not Path(filepath).exists():
            print(f"⚠️  Attenzione: Il file {filepath} non esiste, skippato.")
            continue

        try:
            if filepath.endswith('.csv'):
                df = read_metrics_from_csv(filepath)
                # Se il CSV non porta già method/pipeline, provo a derivarli dal path
                if 'method' not in df.columns:
                    df['method'] = extract_method_from_path(filepath)
                if 'pipeline' not in df.columns:
                    df['pipeline'] = extract_pipeline_from_path(filepath)
            elif filepath.endswith('global_summary.txt'):
                df = read_metrics_from_global_summary(filepath)
            elif filepath.endswith('.txt'):
                df = read_metrics_from_global_summary(filepath)
            else:
                print(f"⚠️  Formato non riconosciuto: {filepath}")
                continue

            if df.empty:
                print(f"⚠️  Nessun dato valido trovato in: {filepath}")
                continue

            print(f"✓ Caricato {filepath} ({len(df)} righe)")
            all_data.append(df)

        except Exception as e:
            print(f"❌ Errore nel caricamento di {filepath}: {e}")
            continue

    if not all_data:
        raise ValueError("Nessun dato caricato. Verifica i percorsi!")

    return pd.concat(all_data, ignore_index=True)


def build_color_map(methods: List[str]) -> Dict[str, str]:
    """
    Assegna un colore base di BASE_COLORS a ciascun metodo, in ordine
    di apparizione, riciclando la palette se ci sono più metodi che
    colori disponibili.
    """
    color_map = {}
    for idx, method in enumerate(methods):
        color_map[method] = BASE_COLORS[idx % len(BASE_COLORS)]
    return color_map


def create_plot(
    df: pd.DataFrame,
    y_column: str = 'qwen_score',
    y_label: str = 'Qwen Attack Success Score (1-7)',
    title: str = 'Relazione tra LPIPS del Subject e Qwen Score',
    output_path: str = "lpips_vs_qwen.png",
):
    """
    Crea un grafico che mostra la relazione tra LPIPS e una metrica di
    Qwen a scelta (score medio oppure attack success rate).

    X-axis: Subject LPIPS (dagli originali immunizzati)
    Y-axis: colonna indicata da y_column (es. 'qwen_score' oppure
            'attack_success_rate')
    Colori: un colore di base per ogni METODO; sfumatura (chiaro /
            medio / scuro) in base alla PIPELINE (Inpainting / Img2Img
            / InstructionPix2Pix).
    """

    # Rimuovi righe con valori mancanti
    df_clean = df.dropna(subset=['lpips_subject', y_column])

    if len(df_clean) == 0:
        raise ValueError("Nessun dato valido per il grafico!")

    # Crea la figura
    fig, ax = plt.subplots(figsize=(12, 8))

    # Metodi in ordine di apparizione (determina il colore base)
    methods = list(dict.fromkeys(df_clean['method']))
    color_map = build_color_map(methods)

    # Pipeline presenti nei dati, ordinate secondo PIPELINE_ORDER quando possibile
    pipelines_present = list(dict.fromkeys(df_clean['pipeline']))
    pipelines_sorted = [p for p in PIPELINE_ORDER if p in pipelines_present] + \
                        [p for p in pipelines_present if p not in PIPELINE_ORDER]

    # Disegna un gruppo di punti per ogni combinazione (metodo, pipeline)
    for method in methods:
        base_color = color_map[method]
        for pipeline in pipelines_sorted:
            subset = df_clean[(df_clean['method'] == method) & (df_clean['pipeline'] == pipeline)]
            if subset.empty:
                continue

            lightness = PIPELINE_LIGHTNESS.get(pipeline, 0.5)
            point_color = shade_color(base_color, lightness)

            ax.scatter(
                subset['lpips_subject'],
                subset[y_column],
                c=point_color,
                label=f"{method} - {pipeline}",
                s=150,
                alpha=0.85,
                edgecolors='black',
                linewidth=0.8
            )

    # Configura il grafico
    ax.set_xlabel('Subject LPIPS (Immunized)', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legenda: un colore per ogni metodo (colore base) + indicazione della sfumatura per pipeline
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 1:
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafico salvato: {output_path}")
    plt.show()


def print_statistics(df: pd.DataFrame):
    """
    Stampa statistiche dei dati caricati.
    """
    print("\n" + "=" * 70)
    print("STATISTICHE DEI DATI")
    print("=" * 70)
    print(f"Totale configurazioni caricate: {len(df)}")
    print(f"\nSubject LPIPS:")
    print(f"  Min:    {df['lpips_subject'].min():.6f}")
    print(f"  Max:    {df['lpips_subject'].max():.6f}")
    print(f"  Media:  {df['lpips_subject'].mean():.6f}")
    print(f"  Mediana: {df['lpips_subject'].median():.6f}")
    print(f"\nQwen Score:")
    print(f"  Min:    {df['qwen_score'].min():.4f}")
    print(f"  Max:    {df['qwen_score'].max():.4f}")
    print(f"  Media:  {df['qwen_score'].mean():.4f}")
    print(f"  Mediana: {df['qwen_score'].median():.4f}")
    if 'attack_success_rate' in df.columns and df['attack_success_rate'].notna().any():
        print(f"\nAttack Success Rate:")
        print(f"  Min:    {df['attack_success_rate'].min():.4f}")
        print(f"  Max:    {df['attack_success_rate'].max():.4f}")
        print(f"  Media:  {df['attack_success_rate'].mean():.4f}")
        print(f"  Mediana: {df['attack_success_rate'].median():.4f}")
    print("=" * 70 + "\n")


def main():
    """
    Funzione principale.
    """
    print("\n" + "=" * 70)
    print("GENERATORE GRAFICI: LPIPS vs QWEN SCORE / ATTACK SUCCESS RATE")
    print("=" * 70)

    # Verifica che i file siano configurati
    if not DATA_FILES:
        print("❌ ERRORE: Nessun file configurato in DATA_FILES!")
        print("Modifica lo script e aggiungi i percorsi ai tuoi file.")
        sys.exit(1)

    # Carica i dati
    print("\nCaricamento dati...")
    try:
        df = load_all_data(DATA_FILES)
    except Exception as e:
        print(f"❌ Errore: {e}")
        sys.exit(1)

    # Stampa statistiche
    print_statistics(df)

    # --- Grafico 1: LPIPS vs Qwen Average Score ---
    print("Creazione grafico 1/2: LPIPS vs Qwen Score medio...")
    try:
        create_plot(
            df,
            y_column='qwen_score',
            y_label='Qwen Attack Success Score (1-7)',
            title='Relazione tra LPIPS del Subject e Qwen Score',
            output_path="lpips_vs_qwen_score.png",
        )
    except Exception as e:
        print(f"❌ Errore nella creazione del grafico 1: {e}")
        sys.exit(1)

    # --- Grafico 2: LPIPS vs Attack Success Rate ---
    if 'attack_success_rate' in df.columns and df['attack_success_rate'].notna().any():
        print("\nCreazione grafico 2/2: LPIPS vs Attack Success Rate...")
        try:
            create_plot(
                df,
                y_column='attack_success_rate',
                y_label='Attack Success Rate',
                title='Relazione tra LPIPS del Subject e Attack Success Rate',
                output_path="lpips_vs_attack_success_rate.png",
            )
        except Exception as e:
            print(f"❌ Errore nella creazione del grafico 2: {e}")
            sys.exit(1)
    else:
        print("\n⚠️  Nessun dato di 'attack_success_rate' disponibile: grafico 2 saltato.")

    print("\n✅ Completato!")


if __name__ == "__main__":
    main()