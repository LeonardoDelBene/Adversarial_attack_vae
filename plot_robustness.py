"""
Generates a 2x3 figure (rows = dataset, columns = transformation: JPEG /
Gaussian Blur / Random Crop) showing how S_attack behaves as the intensity
of each transformation increases, comparing immunized (solid) vs original
(dashed) images.

Data is taken from the robustness tables (columns "Imm." and "Orig.")
already present in the thesis. Update the DATA dict below if the tables
change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# Output configuration
# ----------------------------------------------------------------------------
OUTPUT_IMAGE = Path("Img/robustness_s_attack.png")

# ----------------------------------------------------------------------------
# Data: S_attack (Imm. / Orig.) for each transformation, dataset and method.
# ----------------------------------------------------------------------------
DATA = {
    "JPEG": {
        "x_label": "JPEG Quality",
        "x_ticks": ["Clean", "Q=85", "Q=60", "Q=30"],
        "DiffVax": {
            "DiffVax":     {"imm": [2.5845, 2.3356, 2.5611, 2.7600], "orig": [1.3522, 1.5933, 1.8022, 2.1000]},
            "PhotoGuard":  {"imm": [5.3678, 3.7689, 3.0755, 2.6022], "orig": [1.2278, 1.5389, 1.8445, 2.2033]},
            "Ours (Gray)": {"imm": [5.5155, 3.9344, 3.4722, 3.4778], "orig": [1.3567, 1.6511, 1.6689, 2.0022]},
            "Ours (Opt)":  {"imm": [4.8411, 4.2556, 3.6778, 3.2556], "orig": [1.2689, 1.6900, 1.7911, 2.1644]},
        },
        "MagicBrush": {
            "PhotoGuard":  {"imm": [4.3811, 2.3900, 2.4378, 2.5833], "orig": [1.6211, 1.9578, 2.1033, 2.3689]},
            "Ours (Gray)": {"imm": [5.4078, 5.2867, 5.1244, 4.6456], "orig": [1.5800, 2.0322, 2.2589, 2.3367]},
            "Ours (Opt)":  {"imm": [5.3567, 4.9611, 4.7067, 4.0822], "orig": [1.6578, 1.9689, 2.1822, 2.3900]},
        },
    },
    "Gaussian Blur": {
        "x_label": "Sigma",
        "x_ticks": ["Clean", "0.5", "1.0", "2.0", "4.0"],
        "DiffVax": {
            "DiffVax":     {"imm": [2.5845, 2.5122, 2.6144, 2.9556, 3.2355], "orig": [1.3522, 1.6400, 2.0822, 2.6722, 3.2167]},
            "PhotoGuard":  {"imm": [5.3678, 4.3189, 2.7189, 3.2533, 3.3200], "orig": [1.2278, 1.5656, 2.1133, 2.5378, 3.2300]},
            "Ours (Gray)": {"imm": [5.5155, 4.4522, 2.9589, 3.2189, 3.7811], "orig": [1.3567, 1.6756, 2.1489, 2.5389, 3.5400]},
            "Ours (Opt)":  {"imm": [4.8411, 4.5778, 2.9045, 3.4333, 3.8345], "orig": [1.2689, 1.6033, 2.0833, 2.6122, 3.2278]},
        },
        "MagicBrush": {
            "PhotoGuard":  {"imm": [4.3811, 3.8111, 2.1744, 2.4944, 3.5478], "orig": [1.6211, 1.5889, 2.1544, 2.6966, 3.5122]},
            "Ours (Gray)": {"imm": [5.4078, 4.9078, 3.3789, 3.6078, 4.1667], "orig": [1.5800, 1.7489, 2.2878, 2.7478, 3.5756]},
            "Ours (Opt)":  {"imm": [5.3567, 5.1156, 3.3878, 3.5889, 4.2966], "orig": [1.6578, 1.8167, 1.9989, 2.6933, 3.6822]},
        },
    },
    "Random Crop": {
        "x_label": "Crop Ratio",
        "x_ticks": ["Clean", "r=0.9", "r=0.75", "r=0.5"],
        "DiffVax": {
            "DiffVax":     {"imm": [2.5845, 2.6178, 2.8922, 3.4022], "orig": [1.3522, 2.6311, 2.9055, 3.1122]},
            "PhotoGuard":  {"imm": [5.3678, 5.1011, 6.0922, 6.5433], "orig": [1.2278, 2.5555, 2.9289, 3.1167]},
            "Ours (Gray)": {"imm": [5.5155, 5.2511, 5.3244, 5.9145], "orig": [1.3567, 2.5678, 2.8855, 3.1956]},
            "Ours (Opt)":  {"imm": [4.8411, 4.8911, 5.4900, 6.1422], "orig": [1.2689, 2.7222, 2.8633, 3.1733]},
        },
        "MagicBrush": {
            "PhotoGuard":  {"imm": [4.3811, 3.5645, 4.0678, 5.4656], "orig": [1.6211, 2.4667, 2.9022, 3.2044]},
            "Ours (Gray)": {"imm": [5.4078, 5.4611, 6.4067, 6.7744], "orig": [1.5800, 2.5089, 2.9622, 3.3044]},
            "Ours (Opt)":  {"imm": [5.3567, 5.9367, 6.3400, 6.7078], "orig": [1.6578, 2.5600, 3.1878, 3.2600]},
        },
    },
}

DATASETS = ["DiffVax", "MagicBrush"]
TRANSFORMS = ["JPEG", "Gaussian Blur", "Random Crop"]

# Fixed style per method, kept consistent across all subplots.
METHOD_STYLE = {
    "DiffVax":     {"color": "tab:green",  "marker": "s"},
    "PhotoGuard":  {"color": "tab:orange", "marker": "^"},
    "Ours (Gray)": {"color": "tab:blue",   "marker": "o"},
    "Ours (Opt)":  {"color": "tab:red",    "marker": "D"},
}


def plot_robustness(output_path: Path) -> None:
    figure, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(TRANSFORMS),
        figsize=(16, 8),
        sharey=False,
    )

    for row, dataset_name in enumerate(DATASETS):
        for col, transform_name in enumerate(TRANSFORMS):
            axis = axes[row, col]
            transform_data = DATA[transform_name]
            x_ticks = transform_data["x_ticks"]
            x_positions = range(len(x_ticks))
            methods = transform_data[dataset_name]

            for method_name, series in methods.items():
                style = METHOD_STYLE[method_name]
                # Immunized: solid line.
                axis.plot(
                    x_positions,
                    series["imm"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="-",
                    linewidth=1.8,
                    markersize=5,
                )
                # Original: dashed line, same color.
                axis.plot(
                    x_positions,
                    series["orig"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="--",
                    linewidth=1.3,
                    markersize=4,
                    alpha=0.6,
                )

            axis.set_xticks(list(x_positions))
            axis.set_xticklabels(x_ticks)
            axis.set_title(f"{transform_name} \u2014 {dataset_name}")
            axis.set_xlabel(transform_data["x_label"])
            axis.grid(True, alpha=0.3)

            if col == 0:
                axis.set_ylabel(r"$S_{\mathrm{attack}}$ ($\uparrow$)")

    # Legend: one entry per method (color) + one entry for line style meaning.
    method_handles = [
        Line2D([0], [0], color=style["color"], marker=style["marker"], linewidth=1.8, label=name)
        for name, style in METHOD_STYLE.items()
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Immunized"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.3, alpha=0.6, label="Original"),
    ]
    all_handles = method_handles + style_handles
    all_labels = [h.get_label() for h in all_handles]

    figure.legend(all_handles, all_labels, loc="lower center", ncol=len(all_labels), bbox_to_anchor=(0.5, -0.03))

    figure.tight_layout(rect=(0, 0.05, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    plot_robustness(OUTPUT_IMAGE)