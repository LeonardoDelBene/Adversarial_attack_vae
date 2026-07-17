import os
os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"
import csv
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data import ImmunizationDataset
from loss import vae_mse, noise_loss
from metrics.factory import create_metric, MetricType
from model import Attack, Immunization
from utils import load_image_from_path, prepare_mask_and_masked_image, recover_image, set_seed_lib


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().float()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    img = (img / 2.0 + 0.5).clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
    mask = mask.detach().cpu().float()
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    arr = (mask.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ─────────────────────────────────────────────────────────────────────────────
# Sample = (image_tensor_masked, mask_tensor, image_pil_originale, mask_pil, sample_idx)
# ─────────────────────────────────────────────────────────────────────────────
Sample = Tuple[torch.Tensor, torch.Tensor, Image.Image, Image.Image, int]


def load_experiment_samples(config: Dict[str, Any]) -> List[Sample]:
    """
    Carica piu' campioni dal dataset invece di uno singolo.

    Gli indici da usare sono presi da config['sample_indices'] se presente,
    altrimenti vengono usati i primi config['num_samples'] campioni
    (default 5), cioe' range(num_samples).

    Per ogni campione viene applicato lo stesso mascheramento (sfondo grigio)
    usato in immunize.py, in modo che la VAE veda lo stesso input in
    entrambi gli script.
    """
    dataset = ImmunizationDataset(
        dataset=config.get("dataset_type", "DiffVax"),
        split=config.get("dataset_split", "validation"),
        image_size=config.get("image_size", 512),
    )

    sample_indices = config.get("sample_indices")
    if sample_indices is None:
        num_samples = config.get("num_samples", 5)
        sample_indices = list(range(num_samples))

    samples: List[Sample] = []
    for idx in sample_indices:
        if idx >= len(dataset):
            raise IndexError(f"sample_idx {idx} out of range for dataset of size {len(dataset)}")

        image_t, mask_t = dataset[idx]
        image_t = image_t.unsqueeze(0)
        mask_t = mask_t.unsqueeze(0)

        image_pil = tensor_to_pil(image_t)
        mask_pil = mask_tensor_to_pil(mask_t)

        # Sfondo grigio nella regione mascherata, coerente con immunize.py
        mask_masked_t, masked_image_t, _ = prepare_mask_and_masked_image(image_pil, mask_pil)

        samples.append((masked_image_t, mask_masked_t, image_pil, mask_pil, idx))

    return samples


def load_models(
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    molt_filter: Optional[int] = None,
) -> Immunization:
    attack_model = Attack()
    immunization_model = Immunization(
        device=str(device),
        load_existing=config.get("load_existing", True),
        load_path=checkpoint_path if checkpoint_path is not None else config.get("checkpoint_path"),
        vae=attack_model.model.vae,
        molt_filter=molt_filter if molt_filter is not None else config.get("molt_filter", 1),
        tg=config.get("target", "gray"),
    )
    return immunization_model


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_filename_component(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".6f").rstrip("0").rstrip(".") or "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    text = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _build_image_filename(prefix: str, kwargs: Dict[str, Any], variable_name: str, param_value: float) -> str:
    parts = [prefix, f"{variable_name}_{_sanitize_filename_component(param_value)}"]
    for key, value in kwargs.items():
        if key in {variable_name, "img", "img_mask"}:
            continue
        parts.append(f"{key}_{_sanitize_filename_component(value)}")
    return "_".join(parts) + ".png"


def _save_image(image: Image.Image, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path


def _immunize_and_score_multi(
    immunization_model: Immunization,
    samples: List[Sample],
    immunize_kwargs: Dict[str, Any],
    variable_name: str,
    param_value: float,
    masked_metric,
    output_dir: str,
) -> Dict[str, float]:
    """
    Immunizza TUTTI i campioni in `samples` con gli stessi iperparametri
    (immunize_kwargs) e restituisce media e deviazione standard delle
    metriche calcolate sui singoli campioni.
    """
    per_sample_metrics: Dict[str, List[float]] = {
        "l_vae": [],
        "l_noise": [],
        "subject_lpips": [],
        "global_lpips": [],
    }

    for image_tensor, mask_tensor, image_pil, mask_pil, sample_idx in samples:
        img = image_tensor.to(immunization_model.device).float()
        mask = mask_tensor.to(immunization_model.device).float()

        img_final, l_vae, l_noise = immunization_model.immunize_img_targeted(
            img=img, img_mask=mask, **immunize_kwargs
        )

        img_final_pil = tensor_to_pil(img_final)
        img_final_recovered = recover_image(img_final_pil, image_pil, mask_pil, background=True)

        images_dir = os.path.join(output_dir, "saved_images")
        recovered_name = _build_image_filename(
            f"immunized_sample{sample_idx}", immunize_kwargs, variable_name, param_value
        )
        recovered_path = _save_image(img_final_recovered, images_dir, recovered_name)
        print(f"  [sample {sample_idx}] saved: {recovered_path}")

        lpips_scores = masked_metric.compute(image_pil, img_final_recovered, mask_pil)

        per_sample_metrics["l_vae"].append(float(l_vae))
        per_sample_metrics["l_noise"].append(float(l_noise))
        per_sample_metrics["subject_lpips"].append(float(lpips_scores["subject_lpips"]))
        per_sample_metrics["global_lpips"].append(float(lpips_scores["global_lpips"]))

    result: Dict[str, float] = {variable_name: param_value}
    for key, values in per_sample_metrics.items():
        arr = np.array(values, dtype=np.float64)
        result[key] = float(arr.mean())
        result[f"{key}_std"] = float(arr.std())
    return result


def _base_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed immunization kwargs shared across all sweeps."""
    return {
        "noise_mode": config.get("noise_mode", "mask"),
        "is_2_stage": config.get("is_2_stage", True),
        "lr": config.get("lr", 1e-4),
        "lambda_vae": config.get("lambda_vae", 1.0),
        "n_steps": config.get("n_steps", 300),
        "eps": config.get("eps", 64 / 255.0),
        "lambda_noise": config.get("lambda_noise", 100.0),
        "targeted": config.get("targeted", True),
    }


def _save_csv(results: List[Dict[str, float]], output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV: {csv_path}")


def _plot_results(
    results: List[Dict[str, float]],
    variable_name: str,
    output_dir: str,
    experiment_name: str,
    n_samples: Optional[int] = None,
) -> None:
    """
    Plotta media +/- deviazione standard (barre d'errore) sui campioni,
    per ciascun valore dell'iperparametro sweepato.
    """
    os.makedirs(output_dir, exist_ok=True)

    xs = [r[variable_name] for r in results]
    n_label = f" (media su {n_samples} campioni)" if n_samples else ""

    # -------------------------
    # VAE LOSS
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        xs,
        [r["l_vae"] for r in results],
        yerr=[r.get("l_vae_std", 0.0) for r in results],
        marker="o",
        capsize=3,
    )

    ax.set_title(f"{experiment_name} - VAE Loss{n_label}")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("VAE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(
        output_dir,
        f"vae_loss_{experiment_name}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

    # -------------------------
    # NOISE LOSS
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        xs,
        [r["l_noise"] for r in results],
        yerr=[r.get("l_noise_std", 0.0) for r in results],
        marker="s",
        capsize=3,
    )

    ax.set_title(f"{experiment_name} - Noise Loss")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Noise Loss")
    ax.grid(True, linestyle="--", alpha=0.4)

    path = os.path.join(
        output_dir,
        f"noise_loss_{experiment_name}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

    # -------------------------
    # LPIPS
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        xs,
        [r["subject_lpips"] for r in results],
        yerr=[r.get("subject_lpips_std", 0.0) for r in results],
        marker="o",
        capsize=3,
        label="Subject LPIPS",
    )

    ax.errorbar(
        xs,
        [r["global_lpips"] for r in results],
        yerr=[r.get("global_lpips_std", 0.0) for r in results],
        marker="s",
        capsize=3,
        label="Global LPIPS",
    )

    ax.set_title(f"{experiment_name} - LPIPS")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("LPIPS")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    path = os.path.join(
        output_dir,
        f"lpips_{experiment_name}.png",
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter experiment functions
# ─────────────────────────────────────────────────────────────────────────────

def experiment_n_steps(
    immunization_model: Immunization,
    samples: List[Sample],
    masked_metric,
    config: Dict[str, Any],
    values: Optional[List[int]] = None,
) -> List[Dict[str, float]]:
    """Sweep over n_steps, keeping eps e lambda_noise fissi. Media su tutti i samples."""
    values = values or config.get("sweep_n_steps", [10, 20, 50, 100])
    results = []
    for v in values:
        print(f"[n_steps] {v}  (su {len(samples)} campioni)")
        kwargs = {**_base_kwargs(config), "n_steps": v}
        results.append(
            _immunize_and_score_multi(
                immunization_model,
                samples,
                kwargs,
                "n_steps",
                v,
                masked_metric,
                config["output_dir"],
            )
        )
    _save_csv(results, config["output_dir"], "sweep_n_steps.csv")
    _plot_results(results, "n_steps", config["output_dir"], experiment_name="n_steps", n_samples=len(samples))
    return results


def experiment_eps(
    immunization_model: Immunization,
    samples: List[Sample],
    masked_metric,
    config: Dict[str, Any],
    values: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Sweep over eps, keeping n_steps e lambda_noise fissi. Media su tutti i samples."""
    values = values or config.get("sweep_eps", [8 / 255.0, 16 / 255.0, 32 / 255.0, 64 / 255.0])
    results = []
    for v in values:
        print(f"[eps] {v:.5f}  (su {len(samples)} campioni)")
        kwargs = {**_base_kwargs(config), "eps": v}
        results.append(
            _immunize_and_score_multi(
                immunization_model,
                samples,
                kwargs,
                "eps",
                v,
                masked_metric,
                config["output_dir"],
            )
        )
    _save_csv(results, config["output_dir"], "sweep_eps.csv")
    _plot_results(results, "eps", config["output_dir"], experiment_name="eps", n_samples=len(samples))
    return results


def experiment_lambda_noise(
    immunization_model: Immunization,
    samples: List[Sample],
    masked_metric,
    config: Dict[str, Any],
    values: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Sweep over lambda_noise, keeping n_steps ed eps fissi. Media su tutti i samples."""
    values = values or config.get("sweep_lambda_noise", [10.0, 50.0, 100.0, 200.0])
    results = []
    for v in values:
        print(f"[lambda_noise] {v}  (su {len(samples)} campioni)")
        kwargs = {**_base_kwargs(config), "lambda_noise": v}
        results.append(
            _immunize_and_score_multi(
                immunization_model,
                samples,
                kwargs,
                "lambda_noise",
                v,
                masked_metric,
                config["output_dir"],
            )
        )
    _save_csv(results, config["output_dir"], "sweep_lambda_noise.csv")
    _plot_results(
        results, "lambda_noise", config["output_dir"], "lambda_noise", n_samples=len(samples)
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — single run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    molt_filter: Optional[int] = None,
) -> Immunization:
    """
    Esegue gli esperimenti per UNA rete (un checkpoint), mediando ogni
    combinazione di iperparametri su config['num_samples'] campioni
    (default 5) del dataset di validazione.

    Ritorna il modello Immunization caricato, cosi' il chiamante (es. il loop
    multi-checkpoint) puo' liberare la VRAM dopo l'uso.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed_lib(config.get("seed", 42))

    immunization_model = load_models(config, device, checkpoint_path=checkpoint_path, molt_filter=molt_filter)
    samples = load_experiment_samples(config)
    masked_metric = create_metric(MetricType.MASKED, lpips_net="alex")

    os.makedirs(config["output_dir"], exist_ok=True)
    original_dir = os.path.join(config["output_dir"], "saved_images")
    for _, _, image_pil, _, sample_idx in samples:
        original_name = f"original_{sample_idx}.png"
        original_path = _save_image(image_pil, original_dir, original_name)
        print(f"Saved original image: {original_path}")

    shared = (immunization_model, samples, masked_metric, config)

    print("\n" + "=" * 60)
    print(f"Experiment 1/3 — n_steps sweep (media su {len(samples)} campioni)")
    #experiment_n_steps(*shared)

    print("\n" + "=" * 60)
    print(f"Experiment 2/3 — eps sweep (media su {len(samples)} campioni)")
    experiment_eps(*shared)

    print("\n" + "=" * 60)
    print(f"Experiment 3/3 — lambda_noise sweep (media su {len(samples)} campioni)")
    #experiment_lambda_noise(*shared)

    print("\nAll experiments completed.")
    return immunization_model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — multi checkpoint loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments_multi_checkpoint(
    base_config: Dict[str, Any],
    checkpoint_paths: List[str],
    molt_filters: Optional[List[Optional[int]]] = None,
) -> None:
    """
    Esegue run_experiments per ogni rete in `checkpoint_paths`.

    Il nome della sottocartella di output e' derivato automaticamente dal nome
    del file checkpoint (senza estensione), es:
        checkpoints/unet_best_fk2utznx.pth -> experiment/unet_best_fk2utznx/
        checkpoints/unet_best_fgh123.pth   -> experiment/unet_best_fgh123/

    `molt_filters`, se fornito, deve avere la stessa lunghezza di
    `checkpoint_paths` (un molt_filter per ogni checkpoint; usa None in una
    posizione per usare il default presente in base_config).

    A fine di ogni iterazione il modello viene eliminato e la VRAM liberata
    prima di passare al checkpoint successivo.
    """
    base_output_dir = base_config["output_dir"]

    if molt_filters is None:
        molt_filters = [None] * len(checkpoint_paths)
    elif len(molt_filters) != len(checkpoint_paths):
        raise ValueError("molt_filters deve avere la stessa lunghezza di checkpoint_paths")

    for i, (checkpoint_path, molt_filter) in enumerate(zip(checkpoint_paths, molt_filters)):
        name = os.path.splitext(os.path.basename(checkpoint_path))[0]

        print("\n" + "#" * 70)
        print(f"# Network {i + 1}/{len(checkpoint_paths)} — {name}")
        print(f"# checkpoint: {checkpoint_path}")
        if molt_filter is not None:
            print(f"# molt_filter: {molt_filter}")
        print("#" * 70)

        run_config = dict(base_config)
        run_config["checkpoint_path"] = checkpoint_path
        if molt_filter is not None:
            run_config["molt_filter"] = molt_filter
        run_config["output_dir"] = os.path.join(base_output_dir, name)

        immunization_model = run_experiments(run_config, checkpoint_path=checkpoint_path, molt_filter=molt_filter)

        # Libera la VRAM prima di passare alla rete successiva
        del immunization_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\nAll networks completed.")


def get_default_config() -> Dict[str, Any]:
    return {
        "dataset_type": "DiffVax",
        "dataset_split": "validation",
        "target": "gray",
        # Numero di campioni su cui mediare ogni punto dello sweep.
        # In alternativa, specifica esplicitamente "sample_indices": [0, 3, 7, 12, 20]
        "num_samples": 5,
        "sample_indices": [0,40,80,100,190],
        "image_size": 512,
        "noise_mode": "mask",
        "is_2_stage": True,
        "lr": 1e-4,
        "lambda_vae": 1.0,
        "targeted": True,
        # fixed defaults used when a parameter is NOT being swept
        "n_steps": 300,  # valore standard 300
        "eps": 64 / 255.0, # valore standard 64
        "lambda_noise": 150.0,  # valore standard 150
        "load_existing": True,
        "checkpoint_path": os.path.join("checkpoints", "Stage 2"),
        "molt_filter": 2,
        "seed": 2043,
        "local_files_only": True,
        "output_dir": os.path.join("experiment"),
        # sweep grids
        "sweep_n_steps": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "sweep_eps": [8 / 255.0, 16 / 255.0, 32 / 255.0, 64 / 255.0, 128 / 255],
        "sweep_lambda_noise": [1.0, 10.0, 50.0, 100.0, 150.0, 200.0],
    }


if __name__ == "__main__":
    config = get_default_config()

    checkpoint_paths = [
        os.path.join("checkpoints", "unet_best_o23oqvbx.pth"),
        #os.path.join("checkpoints", "unet_best_uz0247gg.pth"),
        os.path.join("checkpoints", "unet_best_2ji8vjn3.pth"),
        os.path.join("checkpoints", "unet_best_fk2utznx.pth"),
        os.path.join("checkpoints", "unet_best_4g2mrzt5.pth"),

        # aggiungi qui le altre reti...
    ]

    # opzionale: un molt_filter per ogni checkpoint (stessa lunghezza della lista sopra)
    # molt_filters = [2, 1]
    # run_experiments_multi_checkpoint(config, checkpoint_paths, molt_filters)

    run_experiments_multi_checkpoint(config, checkpoint_paths)