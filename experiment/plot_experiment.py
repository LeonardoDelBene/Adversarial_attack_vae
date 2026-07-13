import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import cycle

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATHS = [
    "./vae_mse/sweep_lambda_noise.csv",
    "./vae_mse_ft/sweep_lambda_noise.csv",
    "./Stage_2/sweep_lambda_noise.csv",
]

# Optional: set to None to use filenames as labels
LABELS = [
    "Stage 1 + 2",
    "Stage 1 FT + 2",
    "STage 2",
]

OUTPUT = "Stage_lambda_noise.png"
# ─────────────────────────────────────────────────────────────────────────────

METRIC_GROUPS = [
    ["l_vae"],
    ["l_noise"],
    ["subject_lpips"],
    ["subject_lpips", "global_lpips"],
]

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LINESTYLES = ["-", "--", "-.", ":"]


def load_csv(path):
    df = pd.read_csv(path)
    return df


def get_x_param(df):
    return df.columns[0]


def plot_group(ax, metrics, dataframes, labels, x_param):
    color_cycle = cycle(COLORS)
    for i, (df, label) in enumerate(zip(dataframes, labels)):
        color = next(color_cycle)
        for j, metric in enumerate(metrics):
            if metric not in df.columns:
                print(f"Warning: '{metric}' not found in {label}, skipping.")
                continue
            ls = LINESTYLES[j % len(LINESTYLES)]
            curve_label = f"{label} — {metric}" if len(metrics) > 1 else label
            ax.plot(df[x_param], df[metric], marker="o", label=curve_label, color=color, linestyle=ls)

    ax.set_xlabel(x_param)
    ax.set_title(" & ".join(metrics))
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    dataframes = [load_csv(p) for p in CSV_PATHS]
    labels = LABELS if LABELS else [os.path.splitext(os.path.basename(p))[0] for p in CSV_PATHS]
 
    x_param = get_x_param(dataframes[0])
 
    # Filter groups to only those with at least one metric present in any df
    active_groups = []
    for group in METRIC_GROUPS:
        if any(m in df.columns for df in dataframes for m in group):
            active_groups.append(group)
 
    base, ext = os.path.splitext(OUTPUT)
    if not ext:
        ext = ".png"
 
    for group in active_groups:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_group(ax, group, dataframes, labels, x_param)
        fig.suptitle(f"Experiment comparison  (x = {x_param})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        name = f"{base}_{'_'.join(group)}{ext}"
        plt.savefig(name, dpi=150)
        plt.close(fig)
        print(f"Saved: {name}")


if __name__ == "__main__":
    main()