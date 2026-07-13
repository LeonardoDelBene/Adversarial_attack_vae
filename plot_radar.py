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

# Aggiungi qui le directory delle run che vuoi confrontare.
# Ogni directory deve contenere global_summary.txt.
BASE_DIRS = [
    #Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/DiffVax"),
    #Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE"),
    #Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE_FT"),
    #Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE_FT_2_STAGE"),

    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/DiffVax"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE_FT"),
    Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE_FT_2_STAGE"),
    
    #Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/DiffVax"),
    #Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE"),
    #Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE_FT"),
    #Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE_FT_2_STAGE"),
    
]

OUTPUT_PATH = Path("sd_Img2Img_qwen_radar.png")



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


def prepare_runs(paths):
    runs = []
    for path in paths:
        base_dir = Path(path)
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        summary_path = base_dir / "global_summary.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing global_summary.txt in {base_dir}")

        values = parse_global_summary(summary_path)
        runs.append((base_dir.name, values))
    return runs


def plot_radar(runs, output_path: Path):
    labels = FACTOR_KEYS
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for run_name, values in runs:
        values = values + values[:1]
        ax.plot(angles, values, label=run_name, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 7)

    ax.set_rlabel_position(180 / num_vars)
    ax.yaxis.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color="gray", linestyle="--", linewidth=0.5)

    ax.set_title("Qwen Attack Average Factor Scores", va="bottom", fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.show()


def main():
    if not BASE_DIRS:
        raise ValueError("BASE_DIRS is empty. Definisci le directory da processare direttamente nel codice.")

    runs = prepare_runs(BASE_DIRS)
    plot_radar(runs, OUTPUT_PATH)


if __name__ == "__main__":
    main()
