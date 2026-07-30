import os
os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"
import warnings
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import ImageOps
from datasets import load_from_disk
import torch
import random

from utils import (
    load_sample_from_hf,
    prepare_mask_and_masked_image,
    recover_image,
    set_seed_lib,
)
from data import COCOLocal, OxfordPetLocal, MagicBrushHF, TEdBenchDataset
from model import Attack, AttackSD, AttackInstructPix2Pix, Immunization, AttackSDXL, PhotoGuard
from metrics import create_metric, MetricType

warnings.filterwarnings("ignore", message="QuickGELU mismatch", category=UserWarning, module="open_clip")




# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def get_attack_subfolder(model_attack: str) -> str:
    mapping = {
        "sd_inpainting": "SD_Inpainting",
        "sd_pix2pix": "InstructionPix2Pix",
        "sd_img2img": "SD_Img2Img",
        "sd_xl_img2img": "SD_XL_Img2Img"
    }
    return mapping.get(model_attack, model_attack)


def get_attack_model_label(model_attack: str) -> str:
    labels = {
        "sd_inpainting":  "SD Inpainting",
        "sd_pix2pix":     "Instruction Pix2Pix",
        "sd_img2img":     "SD Img2Img",
        "sd_xl_img2img":  "SD XL Img2Img",  # ←
    }
    return labels.get(model_attack, model_attack)


def get_output_dir(base_output_dir, model_attack, run_wandb, sample_idx):
    subfolder = get_attack_subfolder(model_attack)
    if sample_idx == "full_dataset":
        output_dir = os.path.join(base_output_dir, subfolder, "full_dataset", run_wandb)
    else:
        output_dir = os.path.join(base_output_dir, subfolder, f"img_{sample_idx}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_models(config):
    print("Loading diffusion model and immunization checkpoint...")
    model_attack = config["model_attack"]
    if model_attack == "sd_pix2pix":
        attack_model = AttackInstructPix2Pix()
    elif model_attack == "sd_inpainting":
        attack_model = Attack()
    elif model_attack == "sd_img2img":
        attack_model = AttackSD()
    elif model_attack == "sd_xl_img2img":
        attack_model = AttackSDXL()
    else:
        raise ValueError(
            f"Unknown attack model type '{model_attack}'. Expected 'sd_inpainting', 'sd_pix2pix', 'sd_img2img', 'sd_xl_img2img'."
        )

    # Use SD 1.5 VAE for immunization regardless of attack model to ensure consistent loss scale
    if config['photo guard'] == True:
        immunization_mdl = PhotoGuard(tg=config["target"])
    else:
        immunization_mdl = Immunization(
        load_existing=config["load_existing"],
        load_path=config["checkpoint_path"],
        molt_filter=config["molt_filter"],
        tg=config["target"],
        )

    print("Done.")

    return attack_model, immunization_mdl


def load_sample(config):
    # Support multiple dataset sources: DiffVax (HF on-disk), COCO local, Oxford-Pet local
    ds_type = config.get("dataset_type", "DiffVax")
    if ds_type == "DiffVax":
        dataset = load_from_disk("./data/DiffVaxDataset_local")
        dataset_split = dataset[config["dataset_split"]]
        sample = dataset_split[config["sample_idx"]]
        image, image_mask = load_sample_from_hf(sample, split=config["dataset_split"])
        # prompt comes from sample for DiffVax
        if config["run_full_dataset"]:
            edit_prompt = sample.get("prompts", [config["edit_prompt"]])[0]
        else:
            edit_prompt = config["edit_prompt"]
    elif ds_type == "COCO":
        dataset = COCOLocal(split=config["dataset_split"])
        image, image_mask = dataset[config["sample_idx"]]
        edit_prompt = config["edit_prompt"]
        print(f"Using COCO local dataset, sample {config['sample_idx']}")
    elif ds_type == "Oxford-Pet":
        dataset = OxfordPetLocal(root="./data/Oxford-Pet", split=config["dataset_split"])
        image, image_mask = dataset[config["sample_idx"]]
        edit_prompt = config["edit_prompt"]
        print(f"Using Oxford-Pet local dataset, sample {config['sample_idx']}")
    elif ds_type == "MagicBrush":
        dataset = MagicBrushHF(split=config["dataset_split"])
        image, image_mask = dataset[config["sample_idx"]]
        if config["run_full_dataset"]:
            edit_prompt = dataset.dataset[config["sample_idx"]]["instruction"]
        else:
            edit_prompt = config["edit_prompt"]
        print(f"Using MagicBrush dataset, sample {config['sample_idx']}")
    elif ds_type == "TEdBench":
        dataset = TEdBenchDataset()
        sample_idx = config["sample_idx"]
        image, image_mask = dataset[sample_idx]
        edit_prompt = dataset.dataset[sample_idx]["caption"]
        print(f"Using TEdBench dataset, sample {sample_idx}")
    else:
        raise ValueError(f"Unsupported dataset_type: {ds_type}")

    print(f"Prompt: {edit_prompt}")
    return image, image_mask, edit_prompt


# ─────────────────────────────────────────────
# IMMUNIZATION
# ─────────────────────────────────────────────

def immunize(image, image_mask, immunization_mdl, config):
    to_pil = T.ToPILImage()
    mask_torch, masked_image_torch, image_torch = prepare_mask_and_masked_image(image, image_mask)
    image_torch        = image_torch.half().cuda()          # immagine originale pulita, per PhotoGuard
    masked_image_torch  = masked_image_torch.half().cuda()   # immagine con buco grigio, per la tua pipeline
    mask_torch          = mask_torch.half().cuda()

    set_seed_lib(config["seed"])
    if config['photo guard'] == False:
        if config['noise_mode']=="mask":
            immunized_img, l_vae, l_noise = immunization_mdl.immunize_img_targeted(
                        masked_image_torch, mask_torch,
                        noise_mode=config["noise_mode"],
                        is_2_stage=config["is_2_stage"],
                        lr=config["lr"],
                        n_steps=config["n_steps"],
                        eps=config["eps"],
                        lambda_vae=config["lambda_vae"],
                        lambda_noise=config["lambda_noise"],
                        targeted=config["targeted"],
                    )
        else:
            immunized_img, l_vae, l_noise = immunization_mdl.immunize_img_targeted(
                        image_torch, mask_torch,
                        noise_mode=config["noise_mode"],
                        is_2_stage=config["is_2_stage"],
                        lr=config["lr"],
                        n_steps=config["n_steps"],
                        eps=config["eps"],
                        lambda_vae=config["lambda_vae"],
                        lambda_noise=config["lambda_noise"],
                        targeted=config["targeted"],
                    )
    else:
        immunized_img = immunization_mdl.encoder_attack(
            image_torch, eps=config["eps"], n_steps=config["n_steps"], step_size=config['alpha']
        )

    adv_X = (immunized_img / 2 + 0.5).clamp(0, 1)
    adv_image_png = to_pil(adv_X[0]).convert("RGB")
    if config['photo guard'] == False and config['noise_mode']=="mask":
        adv_image_png = recover_image(adv_image_png, image, image_mask, background=True)

    return adv_image_png


# ─────────────────────────────────────────────
# EDITING
# ─────────────────────────────────────────────

def edit_images(attack_model, image, adv_image_png, image_mask, edit_prompt, model_attack):
    if model_attack == "sd_pix2pix":
        edited_orig = attack_model.edit_image(edit_prompt, image)[0]
        edited_adv  = attack_model.edit_image(edit_prompt, adv_image_png)[0]
        edited_orig_recovered = edited_orig
        edited_adv_recovered  = edited_adv
    elif model_attack == "sd_inpainting":
        edited_orig = attack_model.edit_image(edit_prompt, image, image_mask)[0]
        edited_adv  = attack_model.edit_image(edit_prompt, adv_image_png, image_mask)[0]
        edited_orig_recovered = recover_image(edited_orig, image, image_mask, background=False)
        edited_adv_recovered  = recover_image(edited_adv,  adv_image_png, image_mask, background=False)
    elif model_attack == "sd_img2img":
        edited_orig = attack_model.edit_image(edit_prompt, image)[0]
        edited_adv  = attack_model.edit_image(edit_prompt, adv_image_png)[0]
        edited_orig_recovered = edited_orig
        edited_adv_recovered  = edited_adv
    elif model_attack == "sd_xl_img2img":
        edited_orig = attack_model.edit_image(edit_prompt, image, None)[0]
        edited_adv  = attack_model.edit_image(edit_prompt, adv_image_png, None)[0]
        edited_orig_recovered = edited_orig #recover_image(edited_orig, image, image_mask, background=False)
        edited_adv_recovered  = edited_adv #recover_image(edited_adv,  adv_image_png, image_mask, background=False)

    else:
        raise ValueError(
            f"Unknown attack model type '{model_attack}'. Expected 'sd_inpainting', 'sd_pix2pix', or 'sd_img2img'."
        )

    print("Immunization and edits completed.")
    return edited_orig_recovered, edited_adv_recovered


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_images(output_dir, image, adv_image_png, edited_orig_recovered, edited_adv_recovered, image_mask):
    image.save(os.path.join(output_dir, "original_image.png"))
    adv_image_png.save(os.path.join(output_dir, "immunized_image.png"))
    edited_orig_recovered.save(os.path.join(output_dir, "edited_original.png"))
    edited_adv_recovered.save(os.path.join(output_dir, "edited_immunized.png"))
    image_mask.save(os.path.join(output_dir,"mask.png"))
    print(f"Images saved in {output_dir}")


def save_metrics(output_dir, edit_prompt, metrics: dict):
    prompt_file = os.path.join(output_dir, "prompt_and_metrics.txt")
    with open(prompt_file, "w") as f:
        f.write("--- Prompt ---\n")
        f.write(f"{edit_prompt}\n\n")

        f.write("--- Metrics ---\n\n")

        f.write("Image Quality (Original vs Immunized)\n")
        f.write(f"PSNR: {metrics['psnr_orig_adv']:.4f}\n")
        f.write(f"SSIM: {metrics['ssim_orig_adv']:.4f}\n")
        f.write(f"FSIM: {metrics['fsim_orig_adv']:.4f}\n\n")

        f.write("Protection Effectiveness (Edited Original vs Edited Immunized)\n")
        f.write(f"PSNR: {metrics['psnr_edit']:.4f}\n")
        f.write(f"SSIM: {metrics['ssim_edit']:.4f}\n")
        f.write(f"FSIM: {metrics['fsim_edit']:.4f}\n\n")

        f.write("Masked Editing Score\n")
        f.write(f"Global LPIPS                           : {metrics['global_lpips']:.4f}\n")
        f.write(f"Subject LPIPS                          : {metrics['masked_subject_lpips']:.4f}\n\n")

        f.write("\nQwen Editing Score\n")
        f.write(f"Attack success score (1-7 avg): {metrics['attack_success_score']:.4f}\n")
        f.write("Per-factor scores:\n")
        for factor, score in metrics['attack_success_factors'].items():
            f.write(f"  {factor:30s}: {score}\n")
        f.write(f"Raw LLM response: {metrics['response']}\n")

    print(f"Metrics saved in {prompt_file}")

# ---------------------------------------------
# METRICS
# ---------------------------------------------

def load_metrics_models():
    print("Loading metric models...")
    metrics_models = {
        "psnr":    create_metric(MetricType.PSNR),
        "ssim":    create_metric(MetricType.SSIM),
        "fsim":    create_metric(MetricType.FSIM),
        "masked":  create_metric(MetricType.MASKED, lpips_net="alex"),
        "editing_score": create_metric(MetricType.QWEN),
    }
    print("Metric models loaded.")
    return metrics_models


def compute_metrics(image, mask, adv_image_png, edited_orig_recovered, edited_adv_recovered, edit_prompt, metrics_models):
    psnr_metric    = metrics_models["psnr"]
    ssim_metric    = metrics_models["ssim"]
    fsim_metric    = metrics_models["fsim"]
    masked_metric  = metrics_models["masked"]
    qwen_metric    = metrics_models["editing_score"]

    masked_score = masked_metric.compute(image, adv_image_png, mask)
    qwen_score   = qwen_metric.compute(edited_orig_recovered, edited_adv_recovered, edit_prompt)


    metrics = {
        # Image quality
        "psnr_orig_adv": psnr_metric.calculate_metric_between_images(image, adv_image_png),
        "ssim_orig_adv": ssim_metric.calculate_metric_between_images(image, adv_image_png),
        "fsim_orig_adv": fsim_metric.calculate_metric_between_images(image, adv_image_png),
        # Protection effectiveness
        "psnr_edit":     psnr_metric.calculate_metric_between_images(edited_orig_recovered, edited_adv_recovered),
        "ssim_edit":     ssim_metric.calculate_metric_between_images(edited_orig_recovered, edited_adv_recovered),
        "fsim_edit":     fsim_metric.calculate_metric_between_images(edited_orig_recovered, edited_adv_recovered),

        "attack_success_score": qwen_score["attack_success_score"],
        "attack_success_factors": qwen_score["factor_scores"],
        "attack_success_details": qwen_score["factor_details"],
        "response": qwen_score["raw_output"],

        "global_lpips": masked_score["global_lpips"],
        "masked_subject_lpips": masked_score["subject_lpips"],

    }

    print("\n--- Image Quality (Original vs Immunized) ---")
    print(f"PSNR : {metrics['psnr_orig_adv']:.4f}")
    print(f"SSIM : {metrics['ssim_orig_adv']:.4f}")
    print(f"FSIM : {metrics['fsim_orig_adv']:.4f}") 

    print("\n--- Protection Effectiveness (Edited Original vs Edited Immunized) ---")
    print(f"PSNR : {metrics['psnr_edit']:.4f}")
    print(f"SSIM : {metrics['ssim_edit']:.4f}")
    print(f"FSIM : {metrics['fsim_edit']:.4f}")

    print("\n--- Masked Editing Score ---")
    print(f"Global LPIPS                           : {metrics['global_lpips']:.4f}")
    print(f"Subject LPIPS                          : {metrics['masked_subject_lpips']:.4f}")

    print("\n--- Qwen Editing Score ---")
    print(f"Attack success score (1-7 avg): {metrics['attack_success_score']:.4f}")
    for factor, score in metrics['attack_success_factors'].items():
        print(f"  {factor:30s}: {score}")
    print(f"Raw LLM response: {metrics['response']}")

    return metrics
# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def plot_results(image, adv_image_png, edited_orig_recovered, edited_adv_recovered,
                 edit_prompt, model_attack):
    model_label = get_attack_model_label(model_attack)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(image);                  axes[0].set_title("Original image");           axes[0].axis("off")
    axes[1].imshow(adv_image_png);          axes[1].set_title("Immunized image");          axes[1].axis("off")
    axes[2].imshow(edited_orig_recovered);  axes[2].set_title(f"Edited original [{model_label}]\n\"{edit_prompt}\""); axes[2].axis("off")
    axes[3].imshow(edited_adv_recovered);   axes[3].set_title(f"Edited immunized [{model_label}]\n\"{edit_prompt}\""); axes[3].axis("off")

    plt.suptitle(f"DiffVax: original vs immunized — {model_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()



def run_on_full_dataset(config):
    """Esegue la pipeline completa su tutto il dataset di validation con tutti e 3 i modelli di editing."""

    # Carica tutti e 3 i modelli di attacco
    print("Loading all diffusion models and immunization checkpoint...")
    attack_model_inpaint = Attack()
    attack_model_img2img = AttackSD()
    attack_model_pix2pix = AttackInstructPix2Pix()

    if config['photo guard'] == True:
        immunization_mdl = PhotoGuard(tg=config["target"])
    else:
        # Usa il VAE dal primo modello (sono uguali)
        immunization_mdl = Immunization(
            load_existing=config["load_existing"],
            load_path=config["checkpoint_path"],
            vae=attack_model_inpaint.model.vae,
            molt_filter=config["molt_filter"],
            tg=config['target']
        )
    print("Done.")

    # Load dataset depending on configured dataset_type (solo per conoscere la lunghezza
    # e, per DiffVax/MagicBrush, per accedere ai prompt/instruction nativi del dataset)
    ds_type = config.get("dataset_type", "DiffVax")
    if ds_type == "DiffVax":
        dataset = load_from_disk("./data/DiffVaxDataset_local")[config["dataset_split"]]
    elif ds_type == "COCO":
        dataset = COCOLocal(split=config["dataset_split"])
    elif ds_type == "Oxford-Pet":
        dataset = OxfordPetLocal(root="./data/Oxford-Pet", split=config["dataset_split"])
    elif ds_type == "MagicBrush":
        dataset = MagicBrushHF(split=config["dataset_split"])
    elif ds_type == "TEdBench":
        dataset = TEdBenchDataset()
    else:
        raise ValueError(f"Unsupported dataset_type: {ds_type}")

    print(f"Dataset size: {len(dataset)} samples")

    # Lista dei modelli di attacco con i loro nomi
    attack_models = [
        ("sd_inpainting", attack_model_inpaint),
        ("sd_img2img", attack_model_img2img),
        ("sd_pix2pix", attack_model_pix2pix),
    ]

    prompts = [
                    "add a cap to the person",
                    "add sunglasses to the person",
                    "add a bouquet of flower in the person's hand",
                    "add a backpack to the person",
                    "make the person smile",
                    "change the person's hair color to blonde",
                    "add a bouquet of flowers in person's hand",
                    "add earrings to the person",
                    "change the person's hair color to red",
                    "add a wristwatch to the person",
                    "make the person older",
                    "make the person younger",
                    "add a beard to the person",
                    "change the person's hairstyle to curly hair",
                    "add a tattoo on the person's arm",
                    "change the person's outfit color to blue",
                    "make the person wear a hoodie",
                    "add freckles to the person's face",
                    "change the person's expression to surprised"
    ]

    for sample_idx in range(len(dataset)):
        print(f"\n{'='*50}")
        print(f"Processing sample {sample_idx + 1}/{len(dataset)}")
        print(f"{'='*50}")

        try:
            # --- Carica sample tramite load_sample (gestisce tutti i dataset_type) ---
            sample_config = dict(config)
            sample_config["sample_idx"] = sample_idx
            sample_config["run_full_dataset"] = False  # gestiamo noi l'edit_prompt qui sotto
            image, image_mask, _ = load_sample(sample_config)

            # --- Immunizzazione (una sola volta) ---
            adv_image_png = immunize(image, image_mask, immunization_mdl, config)

            # --- Editing con tutti e 3 i modelli ---
            for model_name, attack_model in attack_models:
                try:
                    output_dir_base = get_output_dir(
                        config["base_output_dir"],
                        model_name,
                        config["run_wandb"],
                        sample_idx="full_dataset"
                    )
                    sample_output_dir = os.path.join(output_dir_base, f"img_{sample_idx}")
                    os.makedirs(sample_output_dir, exist_ok=True)

                    if ds_type == "MagicBrush":
                        edit_prompt = dataset.dataset[sample_idx]["instruction"]
                    elif ds_type == "TEdBench":
                        edit_prompt = dataset.dataset[sample_idx]["caption"]
                    elif model_name == "sd_inpainting" and ds_type == "DiffVax":
                        raw_sample = dataset[sample_idx]
                        edit_prompt = raw_sample.get("prompts", [config["edit_prompt"]] * 2)[1]
                    else:
                        edit_prompt = prompts[sample_idx % len(prompts)]
                    print(f"Edit prompt: {edit_prompt}\n")

                    edited_orig_recovered, edited_adv_recovered = edit_images(
                        attack_model, image, adv_image_png, image_mask,
                        edit_prompt, model_name)
                    

                    # --- Salvataggio immagini ---
                    save_images(sample_output_dir, image, adv_image_png,
                                edited_orig_recovered, edited_adv_recovered, image_mask)

                    # --- Salvataggio prompt ---
                    prompt_path = os.path.join(sample_output_dir, "prompt_and_metrics.txt")
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(edit_prompt)

                except Exception as e:
                    print(f"[ERROR] Sample {sample_idx} with model {model_name} failed: {e}")
                    continue

        except Exception as e:
            print(f"[ERROR] Sample {sample_idx} failed: {e}")
            continue

    return print("Full dataset processing completed.")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def get_config():
    return {
        "dataset_type":         "TEdBench",  # DiffVax | COCO | Oxford-Pet | MagicBrush | TEdBench
        "dataset_split":        "validation",
        "sample_idx":           2,
        "model_attack":         "sd_inpainting", # "sd_pix2pix", "sd_inpainting", o "sd_img2img", "sd_xl_img2img"
        "edit_prompt":          "A circus horse", 

        "photo guard":          False,
        "alpha":                2 / 255,
        
        "is_2_stage":           True,
        "targeted":             True,
        "target":               "gray", # gray | white |black | mean | opt_diffvax | opt_magicbrush
        "noise_mode":           "mask", # mask | all
        "lr":                   1e-4,
        "eps":                  64/255,
        "n_steps":              300,
        "lambda_vae":           1,
        "lambda_noise":         150,

        "seed":                 2043,
        "load_existing":        True,
        "checkpoint_path":      os.path.join("checkpoints", "unet_best_nv5dqvvb.pth"), #  diffVax_trained: unet_best_nv5dqvvb.pth DiffVax: diffvax_trained.pth magicbrush: unet_best_nkrxr2ji.pth
        "molt_filter":          2,

        "base_output_dir":      "output",
        "dataset_path":         "./data/DiffVaxDataset_local",
        "run_full_dataset":     True,
        "run_wandb":            "TedBench_diff_noise_mask_invert"
    }

def main():
    config = get_config()

    if config["run_full_dataset"]:
        run_on_full_dataset(config)
    else:
        output_dir = get_output_dir(
            config["base_output_dir"],
            config["model_attack"],
            config["run_wandb"],
            config["sample_idx"]
        )
        attack_model, immunization_mdl = load_models(config)

        image, image_mask, _ = load_sample(config)

        adv_image_png = immunize(image, image_mask, immunization_mdl, config)

        edited_orig_recovered, edited_adv_recovered = edit_images(
            attack_model, image, adv_image_png, image_mask,
            config["edit_prompt"], config["model_attack"]
        )

        save_images(output_dir, image, adv_image_png, edited_orig_recovered, edited_adv_recovered, image_mask)

        metrics_model = load_metrics_models()
        metrics = compute_metrics(image, image_mask, adv_image_png, edited_orig_recovered, edited_adv_recovered,
                                  config["edit_prompt"], metrics_model)
        
        save_metrics(output_dir, config["edit_prompt"], metrics)

        plot_results(image, adv_image_png, edited_orig_recovered, edited_adv_recovered,
                     config["edit_prompt"], config["model_attack"])

        


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()


