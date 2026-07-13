import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

# Inserisci qui i percorsi ai file sweep_eps.csv che vuoi usare.
CSV_PATHS = [
    "./unet_best_4g2mrzt5/sweep_eps.csv", #8
    "./unet_best_fk2utznx/sweep_eps.csv", #16
    "./unet_best_2ji8vjn3/sweep_eps.csv", #32
    "./unet_best_uz0247gg/sweep_eps.csv", #64
    "./unet_best_o23oqvbx/sweep_eps.csv", #128
]

# Inserisci qui i nomi delle righe che vuoi usare nel CSV / heatmap.
# Deve avere la stessa lunghezza di CSV_PATHS.
MODEL_LABELS = [
    "eps 8/255",
    "eps 16/255",
    "eps 32/255",
    "eps 64/255",
    "eps 128/255",
]

# Inserisci qui i nomi delle colonne che vuoi usare nel CSV / heatmap.
# Deve avere la stessa lunghezza dei valori eps in ciascun sweep_eps.csv.
EPS_LABELS = [
    "eps 8/255",
    "eps 16/255",
    "eps 32/255",
    "eps 64/255",
    "eps 128/255",
]

# Scegli il valore da usare: "l_vae" o "subject_lpips".
METRIC_NAME = "editing_score_attacksd"

# Imposta a True se per questa metrica un valore più basso è migliore
# (es. LPIPS, l_vae), a False se un valore più alto è migliore
# (es. PSNR, SSIM, editing_score). Controlla sia la colormap che il
# testo del titolo/colorbar della heatmap.
LOWER_IS_BETTER = False

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), f"heatmap_{METRIC_NAME}.csv")
HEATMAP_IMAGE = os.path.join(os.path.dirname(__file__), f"heatmap_{METRIC_NAME}.png")


def load_metric_values(path: str, metric: str) -> pd.Series:
    df = pd.read_csv(path)
    if "eps" not in df.columns or metric not in df.columns:
        raise ValueError(f"Il file {path} non contiene le colonne richieste: eps, {metric}")
    return df.sort_values("eps").set_index("eps")[metric]


def build_heatmap_dataframe(csv_paths: List[str], row_labels: List[str], col_labels: List[str], metric_name: str) -> pd.DataFrame:
    if len(csv_paths) != len(row_labels):
        raise ValueError("CSV_PATHS e MODEL_LABELS devono avere la stessa lunghezza")

    rows = {}
    eps_index = None
    for path, label in zip(csv_paths, row_labels):
        series = load_metric_values(path, metric_name)
        if eps_index is None:
            eps_index = series.index
        elif not series.index.equals(eps_index):
            raise ValueError(
                f"I valori eps non corrispondono tra i file: {path} ha eps diversi"
            )
        rows[label] = series.values

    if len(col_labels) != len(eps_index):
        raise ValueError("EPS_LABELS deve avere la stessa lunghezza dei valori eps in ciascun file")

    df = pd.DataFrame(rows, index=col_labels).T
    df.index.name = "model"
    return df



def plot_heatmap(df: pd.DataFrame, output_path: str, lower_is_better: bool) -> None:
    # cmap "_r" (reversed): per lower_is_better i valori bassi sono
    # chiari/migliori; altrimenti sono i valori alti a esserlo.
    cmap = "viridis_r" if lower_is_better else "viridis"
    direction_label = "lower is better" if lower_is_better else "higher is better"

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    heatmap = ax.imshow(df.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    ax.set_xlabel("eps 2 stage")
    ax.set_ylabel("eps 1 stage")
    ax.set_title(f"Heatmap {METRIC_NAME} ({direction_label})")

    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label(METRIC_NAME)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    csv_paths = [os.path.join(os.path.dirname(__file__), path) for path in CSV_PATHS]

    for path in csv_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File non trovato: {path}")

    print(f"Uso i seguenti {len(csv_paths)} file sweep_eps.csv:")
    for path, label in zip(csv_paths, MODEL_LABELS):
        print(f" - {path} -> {label}")

    heatmap_df = build_heatmap_dataframe(csv_paths, MODEL_LABELS, EPS_LABELS, METRIC_NAME)
    heatmap_df.to_csv(OUTPUT_FILE)
    print(f"CSV generato: {OUTPUT_FILE}")

    plot_heatmap(heatmap_df, HEATMAP_IMAGE, LOWER_IS_BETTER)
    print(f"Heatmap generata: {HEATMAP_IMAGE}")


if __name__ == "__main__":
    main()
