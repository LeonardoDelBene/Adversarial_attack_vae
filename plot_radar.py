import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FACTOR_KEYS = [
    "background_infiltration",
    "global_coherence_ruin",
    "identity_erasure",
    "scale_distortion",
    "spatial_chaos",
    "texture_degradation",
    "distortion_effectiveness",
    "light_color_mismatch",
    "edge_visibility",
    "alignment_sabotage",
    "editing_incompleteness",
    "total_implausibility",
]

# Modelli di editing su cui generare un radar chart dedicato, più uno medio.
# La chiave è il nome della cartella usata nel path, il valore è il titolo
# da mostrare nel grafico.
EDITING_MODELS = {
    "SD_Inpainting": "SD Inpainting",
    "SD_Img2Img": "SD Img2Img",
    "InstructionPix2Pix": "InstructionPix2Pix",
}

BASE_OUTPUT_DIR = Path("/equilibrium/ldelbene/Immunization/output")
DATASET_SUBPATH = "full_dataset"

# Definisci qui le run da confrontare: ogni voce è (etichetta_legenda, nome_cartella_run).
# nome_cartella_run è lo stesso per tutti i modelli di editing (cambia solo il modello
# nel path), l'etichetta invece è quella che vuoi vedere in legenda ed è scelta da te.
RUNS = [
    ("PhotoGuard", "TedBench_photoguard"),
    ("Ours (train DiffVax, noise Mask)", "TedBench_diff_noise_mask_invert"),
    ("Ours (train Diffvax, noise All)", "TedBench_diff_noise_all"),
    ("Ours (train MagicBrush, noise Mask)", "TedBench_magic_noise_mask"),
    ("Ours (train MagicBrush, noise All)", "TedBench_magic_noise_all"),
]

OUTPUT_DIR = Path(".")
OUTPUT_PREFIX = "radar_chart_tedbench"


def parse_global_summary(summary_path: Path):
    text = summary_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]

    if "Average factor scores:" not in lines:
        raise ValueError(f"Cannot find 'Average factor scores:' in {summary_path}")

    start = lines.index("Average factor scores:") + 1
    values = {}
    pattern = re.compile(r"^([a-z_]+):\s*([0-9]+(?:\.[0-9]+)?)")

    for line in lines[start:]:
        if not line:
            break
        match = pattern.match(line)
        if not match:
            break
        key, value = match.groups()
        if key in FACTOR_KEYS:
            values[key] = float(value)

    missing = [key for key in FACTOR_KEYS if key not in values]
    if missing:
        raise ValueError(
            f"Missing factor values in {summary_path}: {', '.join(missing)}"
        )

    return [values[key] for key in FACTOR_KEYS]


def build_run_path(editing_model_dir: str, run_dir_name: str) -> Path:
    return BASE_OUTPUT_DIR / editing_model_dir / DATASET_SUBPATH / run_dir_name


def prepare_runs_for_model(editing_model_dir: str):
    """Ritorna una lista di (etichetta, values) per il modello di editing dato."""
    runs = []
    for label, run_dir_name in RUNS:
        base_dir = build_run_path(editing_model_dir, run_dir_name)
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        summary_path = base_dir / "global_summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing global_summary.txt in {base_dir}")

        values = parse_global_summary(summary_path)
        runs.append((label, values))
    return runs


def plot_radar(runs, title: str, output_path: Path):
    labels = FACTOR_KEYS
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for label, values in runs:
        values = values + values[:1]
        ax.plot(angles, values, label=label, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 7)

    ax.set_rlabel_position(180 / num_vars)
    ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color="gray", linestyle="--", linewidth=0.5)

    ax.set_title(title, va="bottom", fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def average_runs(per_model_runs):
    """
    per_model_runs: dict {editing_model_dir: [(label, values), ...]}
    Ritorna [(label, values_medi), ...] mediando, per ciascuna label,
    i valori sui modelli di editing.
    """
    labels = [label for label, _ in RUNS]
    averaged = []
    for label in labels:
        stacked = np.array(
            [
                dict(per_model_runs[model_dir])[label]
                for model_dir in EDITING_MODELS
            ]
        )
        mean_values = stacked.mean(axis=0).tolist()
        averaged.append((label, mean_values))
    return averaged


def main():
    if not RUNS:
        raise ValueError("RUNS è vuoto. Definisci le run da confrontare direttamente nel codice.")

    per_model_runs = {}

    for editing_model_dir, editing_model_title in EDITING_MODELS.items():
        runs = prepare_runs_for_model(editing_model_dir)
        per_model_runs[editing_model_dir] = runs

        output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{editing_model_dir.lower()}.png"
        plot_radar(
            runs,
            title=f"Qwen Attack Average Factor Scores — {editing_model_title}",
            output_path=output_path,
        )
        print(f"Salvato: {output_path}")

    averaged_runs = average_runs(per_model_runs)
    output_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_mean.png"
    plot_radar(
        averaged_runs,
        title="Qwen Attack Average Factor Scores — Media sui 3 modelli di editing",
        output_path=output_path,
    )
    print(f"Salvato: {output_path}")


if __name__ == "__main__":
    main()