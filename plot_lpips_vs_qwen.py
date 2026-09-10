"""
Script per creare un'unica figura composta da N grafici a barre affiancati
(uno per dataset tra DiffVax, MagicBrush, TedBench), che mettono in relazione,
per ogni METODO (VAE_MSE, VAE_MSE_FT, VAE_MSE_FT_2_STAGE, DiffVax, ...), le
tre PIPELINE (SD_Inpainting, SD_Img2Img, InstructionPix2Pix) rispetto a:
    1) lo score medio di attack success (1-7) di Qwen, oppure
    2) l'attack success rate (frazione di campioni con score >= soglia)

Per ogni metodo (configurazione) vengono disegnate 4 barre affiancate:
    - 3 barre, una per pipeline, con sfumature diverse dello stesso
      colore base del metodo (SD_Inpainting = chiaro, SD_Img2Img =
      intermedio, InstructionPix2Pix = scuro)
    - 1 barra aggiuntiva "Average", con il colore pieno (non sfumato) del
      metodo e un hatch per distinguerla a colpo d'occhio dalle altre tre

L'asse x mostra il Subject LPIPS (Original vs Immunized) di ciascun
metodo (nome per esteso incluso nell'etichetta), quindi non serve una
legenda "Configuration" separata: l'unica legenda condivisa da tutta la
figura è quella "Task / Aggregate" (pipeline + Average), identica per
tutti i sottografici.

NOTA: solo i dataset per cui e' stata popolata almeno una lista di file
(DIFFVAX_FILES / MAGICBRUSH_FILES / TEDBENCH_FILES) diventano un
sottografico della figura finale. Se e' popolato un solo dataset, la
figura risultante contiene un solo grafico (non 3, con 2 vuoti).

Tutti i testi disegnati nel grafico (titoli, assi, legenda) sono in
inglese.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ============================================================================
# CONFIGURAZIONE: un set di file (global_summary.txt) per ciascun dataset.
# ============================================================================
DIFFVAX_FILES: List[str] = [
    'output/SD_Inpainting/full_dataset/VAE_MSE_BLACK/global_summary.txt',
    'output/SD_Img2Img/full_dataset/VAE_MSE_BLACK/global_summary.txt',
    'output/InstructionPix2Pix/full_dataset/VAE_MSE_BLACK/global_summary.txt',

    'output/SD_Inpainting/full_dataset/VAE_MSE_WHITE/global_summary.txt',
    'output/SD_Img2Img/full_dataset/VAE_MSE_WHITE/global_summary.txt',
    'output/InstructionPix2Pix/full_dataset/VAE_MSE_WHITE/global_summary.txt',

    'output/SD_Inpainting/full_dataset/VAE_MSE_FT_2_STAGE/global_summary.txt',
        'output/SD_Img2Img/full_dataset/VAE_MSE_FT_2_STAGE/global_summary.txt',
        'output/InstructionPix2Pix/full_dataset/VAE_MSE_FT_2_STAGE/global_summary.txt',

    'output/SD_Inpainting/full_dataset/VAE_MSE_TARGET_OPT/global_summary.txt',
        'output/SD_Img2Img/full_dataset/VAE_MSE_TARGET_OPT/global_summary.txt',
        'output/InstructionPix2Pix/full_dataset/VAE_MSE_TARGET_OPT/global_summary.txt',

    'output/SD_Inpainting/full_dataset/VAE_MSE_MEAN/global_summary.txt',
        'output/SD_Img2Img/full_dataset/VAE_MSE_MEAN/global_summary.txt',
        'output/InstructionPix2Pix/full_dataset/VAE_MSE_MEAN/global_summary.txt',
    
]

MAGICBRUSH_FILES: List[str] = [
    
]

TEDBENCH_FILES: List[str] = [
    
]

# Ordine (e titolo) dei sottografici nella figura finale.
DATASETS: List[Dict[str, object]] = [
    {"name": "DiffVax", "files": DIFFVAX_FILES},
    {"name": "MagicBrush", "files": MAGICBRUSH_FILES},
    {"name": "TedBench", "files": TEDBENCH_FILES},
]


# ============================================================================
# NOMI VISUALIZZATI PER LE CONFIGURAZIONI (METODI)
# ============================================================================
METHOD_LABELS: Dict[str, str] = {
    'TedBench_diff_noise_mask_invert': 'Ours [train DiffVax, noise Mask]',
    'TedBench_diff_noise_all': 'Ours [train DiffVax, noise All]',
    'TedBench_photoguard': 'PhotoGuard',
    'TedBench_magic_noise_all': 'Ours [train MagicBrush, noise All]',
    'TedBench_magic_noise_mask': 'Ours [train MagicBrush, noise Mask]',
    'VAE_MSE_TARGET_OPT_NOISE_ALL': 'Ours [target Opt, noise All]',
    'VAE_MSE_FT_2_STAGE_NOSIE_ALL': 'Gray',
    'VAE_MSE_TARGET_OPT': 'Opt',
    'VAE_MSE_FT_2_STAGE': 'Gray',
    'MagicBrush_gray_NOISE_ALL': 'Ours [target Gray, noise All]',
    'MagicBrush_gray_FT': 'Ours [target Gray, noise Mask]',
    'MagicBrush_target_opt': 'Ours [target Opt, noise Mask]',
    'MagicBrush_TARGET_OPT_NOISE_ALL': 'Ours [target Opt, noise All]',
    'MagicBrush_photoguard': 'PhotoGuard',
    'VAE_MSE_MEAN': 'Mean',
    'VAE_MSE_WHITE': 'White',
    'VAE_MSE_BLACK': 'Black',
}


def display_name(method: str) -> str:
    """Restituisce il nome da mostrare in etichetta per un dato metodo."""
    return METHOD_LABELS.get(method, method)


BASE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

PIPELINE_LIGHTNESS = {
    'SD_Inpainting': 0.78,
    'SD_Img2Img': 0.55,
    'InstructionPix2Pix': 0.32,
}

PIPELINE_ORDER = ['SD_Inpainting', 'SD_Img2Img', 'InstructionPix2Pix']

LEGEND_NEUTRAL_COLOR = '#808080'


def shade_color(hex_color: str, lightness: float) -> str:
    """Genera una variante piu' chiara/scura di un colore esadecimale."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, s)
    return mcolors.to_hex((r2, g2, b2))


def extract_pipeline_from_path(filepath: str) -> str:
    parts = Path(filepath).parts
    for part in parts:
        if part in PIPELINE_LIGHTNESS:
            return part
    return 'Unknown'


def extract_method_from_path(filepath: str) -> str:
    return Path(filepath).parent.name


def read_metrics_from_global_summary(filepath: str) -> pd.DataFrame:
    data = {
        'lpips_subject': [], 'qwen_score': [], 'attack_success_rate': [],
        'name': [], 'method': [], 'pipeline': [],
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        lpips_start = content.find("---- Original vs Immunized LPIPS ----")
        if lpips_start != -1:
            lpips_section = content[lpips_start:lpips_start + 500]
            lpips_match = re.search(r'Subject LPIPS:\s+([\d.]+)', lpips_section)
            if lpips_match:
                lpips_value = float(lpips_match.group(1))
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        qwen_start = content.find("=== Qwen Attack Evaluation Summary ===")
        if qwen_start != -1:
            qwen_section = content[qwen_start:qwen_start + 500]

            qwen_match = re.search(r'Average attack success score.*?:\s+([\d.]+)', qwen_section)
            if qwen_match:
                qwen_value = float(qwen_match.group(1))
            else:
                return pd.DataFrame()

            rate_match = re.search(r'Attack success rate:\s+([\d.]+)', qwen_section)
            if rate_match:
                rate_value = float(rate_match.group(1))
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        method_name = extract_method_from_path(filepath)
        pipeline_name = extract_pipeline_from_path(filepath)
        config_name = f"{method_name} ({pipeline_name})"

        data['lpips_subject'].append(lpips_value)
        data['qwen_score'].append(qwen_value)
        data['attack_success_rate'].append(rate_value)
        data['name'].append(config_name)
        data['method'].append(method_name)
        data['pipeline'].append(pipeline_name)

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return pd.DataFrame()


def load_all_data(file_paths: List[str]) -> pd.DataFrame:
    all_data = []

    for filepath in file_paths:
        if not Path(filepath).exists():
            print(f"Warning: file not found, skipped: {filepath}")
            continue

        try:
            df = read_metrics_from_global_summary(filepath)

            if df.empty:
                print(f"Warning: no valid data found in: {filepath}")
                continue

            print(f"Loaded {filepath} ({len(df)} rows)")
            all_data.append(df)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def build_color_map(methods: List[str]) -> Dict[str, str]:
    return {method: BASE_COLORS[idx % len(BASE_COLORS)] for idx, method in enumerate(methods)}


def plot_dataset_on_axis(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    y_column: str,
    y_label: str,
) -> List[Patch]:
    """
    Disegna il bar plot di un singolo dataset sull'axis fornito.

    Ritorna la lista di handle della legenda "Task / Aggregate"
    (pipeline + Average), identica per costruzione in ogni dataset,
    cosi' che il chiamante possa riusarla una sola volta per l'intera
    figura.
    """
    df_clean = df.dropna(subset=[y_column, 'lpips_subject'])

    if df_clean.empty:
        ax.set_title(f"{title} (no data)", fontsize=13, fontweight='bold')
        ax.axis('off')
        return []

    lpips_by_method = df_clean.groupby('method')['lpips_subject'].mean()

    methods = sorted(
        list(dict.fromkeys(df_clean['method'])),
        key=lambda m: lpips_by_method[m]
    )
    color_map = build_color_map(methods)

    pipelines_present = list(dict.fromkeys(df_clean['pipeline']))
    pipelines_sorted = [p for p in PIPELINE_ORDER if p in pipelines_present] + \
                        [p for p in pipelines_present if p not in PIPELINE_ORDER]

    bar_names = pipelines_sorted + ['Average']
    n_bars = len(bar_names)
    n_groups = len(methods)

    x = np.arange(n_groups)
    group_width = 0.8
    bar_width = group_width / n_bars

    for j, bar_name in enumerate(bar_names):
        offset = (j - (n_bars - 1) / 2) * bar_width

        values = []
        colors = []
        for method in methods:
            base_color = color_map[method]

            if bar_name == 'Average':
                subset = df_clean[df_clean['method'] == method]
                value = subset[y_column].mean() if not subset.empty else np.nan
                color = base_color
            else:
                subset = df_clean[(df_clean['method'] == method) & (df_clean['pipeline'] == bar_name)]
                value = subset[y_column].mean() if not subset.empty else np.nan
                lightness = PIPELINE_LIGHTNESS.get(bar_name, 0.5)
                color = shade_color(base_color, lightness)

            values.append(value)
            colors.append(color)

        hatch = '////' if bar_name == 'Average' else None

        ax.bar(
            x + offset, values, bar_width * 0.9,
            color=colors, edgecolor='black', linewidth=0.8, hatch=hatch,
        )

    xtick_labels = [f"{display_name(m)}\n({lpips_by_method[m]:.3f})" for m in methods]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=25, ha='right', fontsize=9)
    ax.set_xlabel('Configuration (Mask LPIPS)', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    pipeline_handles = []
    for pipeline in pipelines_sorted:
        lightness = PIPELINE_LIGHTNESS.get(pipeline, 0.5)
        c = shade_color(LEGEND_NEUTRAL_COLOR, lightness)
        pipeline_handles.append(Patch(facecolor=c, edgecolor='black', label=pipeline))
    pipeline_handles.append(
        Patch(facecolor=LEGEND_NEUTRAL_COLOR, edgecolor='black', hatch='////', label='Average')
    )

    return pipeline_handles


def create_combined_figure(
    datasets: List[Dict[str, object]],
    y_column: str = 'attack_success_rate',
    y_label: str = 'Attack Success Rate',
    suptitle: str = 'Subject LPIPS vs Attack Success Rate',
    output_path: str = "lpips_vs_attack_success_rate_combined.png",
) -> None:
    """
    Crea una figura con un sottografico per ciascun dataset NON vuoto
    (in orizzontale) e un'unica legenda "Task / Aggregate" condivisa,
    in fondo alla figura.

    Se e' popolato un solo dataset, la figura contiene un solo
    sottografico (non vengono creati subplot vuoti per gli altri).
    """
    active_datasets = [d for d in datasets if d["files"]]

    if not active_datasets:
        print("ERROR: no dataset has any file configured. Nothing to plot.")
        return

    n_datasets = len(active_datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 8), squeeze=False)
    axes = axes[0]

    shared_legend_handles: List[Patch] = []

    for ax, dataset in zip(axes, active_datasets):
        name = dataset["name"]
        files = dataset["files"]

        df = load_all_data(files)

        if df.empty:
            ax.set_title(f"{name} (no data)", fontsize=13, fontweight='bold')
            ax.axis('off')
            continue

        handles = plot_dataset_on_axis(ax, df, title=name, y_column=y_column, y_label=y_label)
        if handles and not shared_legend_handles:
            shared_legend_handles = handles

    fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    if shared_legend_handles:
        fig.legend(
            handles=shared_legend_handles,
            title='Task / Aggregate',
            loc='lower center',
            ncol=len(shared_legend_handles),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=10,
        )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.show()


def main():
    print("\n" + "=" * 70)
    print("COMBINED CHART GENERATOR: LPIPS vs ATTACK SUCCESS RATE")
    print("=" * 70)

    if not any(d["files"] for d in DATASETS):
        print("ERROR: No files configured in DIFFVAX_FILES / MAGICBRUSH_FILES / TEDBENCH_FILES!")
        print("Edit the script and add the paths to your files.")
        sys.exit(1)

    create_combined_figure(
        DATASETS,
        y_column='attack_success_rate',
        y_label='Attack Success Rate',
        suptitle='Attack Success Rate vs Mask LPIPS',
        output_path="lpips_vs_attack_success_rate_combined.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()