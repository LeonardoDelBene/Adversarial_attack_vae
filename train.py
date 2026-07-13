from collections import defaultdict
from lpips import lpips
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
import random
import warnings
import numpy as np
from diffusers import AutoencoderKL
from loss import total_loss, DynamicWeighter, LossNormalizer
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from data import ImmunizationDataset
from model import NestedUNet
from utils import set_seed_lib
from tqdm import tqdm
import wandb
from PIL import Image, ImageOps

warnings.filterwarnings("ignore", message="QuickGELU mismatch", category=UserWarning, module="open_clip")

def save_training_checkpoint(
    checkpoint_dir, unet, optimizer, dyn_weighter, loss_normalizer,
    epoch, global_step, best_val_loss, patience_count,
    best_monitored: float = float("inf"),    # ← nuovo
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":           epoch,
        "global_step":     global_step,
        "best_val_loss":   best_val_loss,
        "patience_count":  patience_count,
        "best_monitored":  best_monitored,   # ← nuovo
        "unet_state":      unet.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dyn_weighter": {
            "n_surrogates": dyn_weighter.n_surrogates,
            "W_init":       dyn_weighter.W_init,
            "T_temp":       dyn_weighter.T_temp,
            "window":       dyn_weighter.window,
            "s_clip":       dyn_weighter.s_clip,
            **dyn_weighter.state_dict(),
        },
        "loss_normalizer": loss_normalizer.state_dict(),
    }

    path = checkpoint_dir / "training_checkpoint.pth"
    torch.save(checkpoint, path)
    print(f"  ✓ Training checkpoint saved at epoch {epoch + 1} → {path}")


def load_training_checkpoint(
    checkpoint_dir, unet, optimizer, dyn_weighter, loss_normalizer, patience,
    device="cuda",
) -> dict:
    path = Path(checkpoint_dir) / "training_checkpoint.pth"

    if not path.exists():
        print("No training checkpoint found, starting from scratch.")
        return {
            "start_epoch":    0,
            "global_step":    0,
            "best_val_loss":  float("inf"),
            "patience_count": 0,
            "best_monitored": float("inf"),   # ← nuovo
        }

    checkpoint = torch.load(path, map_location=device)

    unet.load_state_dict(checkpoint["unet_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    dw = checkpoint["dyn_weighter"]
    dyn_weighter.n_surrogates = dw["n_surrogates"]
    dyn_weighter.W_init       = dw["W_init"]
    dyn_weighter.T_temp       = dw["T_temp"]
    dyn_weighter.window       = dw["window"]
    dyn_weighter.s_clip       = dw["s_clip"]
    dyn_weighter.load_state_dict({
        "loss_history": dw["loss_history"],
        "prev_weights": dw["prev_weights"],
    })

    if "loss_normalizer" in checkpoint:
        loss_normalizer.load_state_dict(checkpoint["loss_normalizer"])

    print(
        f"Training checkpoint loaded: "
        f"epoch={checkpoint['epoch'] + 1}, "
        f"global_step={checkpoint['global_step']}, "
        f"best_val_loss={checkpoint['best_val_loss']:.4f}, "
        f"best_monitored={checkpoint.get('best_monitored', float('inf')):.6f}"
    )

    return {
        "start_epoch":    checkpoint["epoch"] + 1,
        "global_step":    checkpoint["global_step"],
        "best_val_loss":  checkpoint["best_val_loss"],
        "patience_count": 0, # non usato
        "best_monitored": checkpoint.get("best_monitored", float("inf")),  # ← nuovo
    }



def training_loop(
        unet,
        nb_filter,
        dataloader,
        val_dataloader,
        dataset: str,
        n_epochs: int = 10,
        lr: float = 1e-4,
        batch: int = 2,
        weight_decay: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_vae: float = 0.1,
        eps: float = 32/255 * 2,
        val_every: int = 1,
        best_checkpoint_path: str = "checkpoints/unet_best.pth",
        training_checkpoint_dir: str = "checkpoints/training",
        device: str = "cuda",
        resume_from_checkpoint: bool = True,
        resume_only_weights=False,
        noise_on_mask: bool = False,
        dyn_weight_window: int   = 20,
        dyn_weight_T_temp: float = 0.1,
        dyn_weight_s_clip: float = 2.0,
        target: str = "gray",
        untargeted: bool = False,        # ← nuovo
        margin: float | None = 10.0,     # ← nuovo, usato solo se untargeted=True
):

    surrogate_vae_models = []

    vae1 = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-inpainting", subfolder="vae"
    ).to(device).eval()
    for param in vae1.parameters():
        param.requires_grad = False
    surrogate_vae_models.append(vae1)

    '''vae2 = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder = "vae"
    ).to(device).eval()
    for param in vae2.parameters():
        param.requires_grad = False
    surrogate_vae_models.append(vae2)'''

    n_surrogates = len(surrogate_vae_models)

    # ── Ottimizzatore, weighter e normalizer ──
    optimizer       = optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay)
    dyn_weighter    = DynamicWeighter(
        n_surrogates=n_surrogates,
        T_temp=dyn_weight_T_temp,
        window=dyn_weight_window,
        s_clip=dyn_weight_s_clip,
    )
    loss_normalizer = LossNormalizer(n_surrogates=n_surrogates)

    # ── Carica checkpoint ──
    if resume_from_checkpoint:
        state = load_training_checkpoint(
            training_checkpoint_dir, unet, optimizer,
            dyn_weighter, loss_normalizer, device,
        )
    elif resume_only_weights:
        unet.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
        print("resume_only_weights=True, starting fine-tuning...")
        state = {"start_epoch": 0, "global_step": 0,
                 "best_val_loss": float("inf"),
                 "best_monitored": float("inf")}
    else:
        print("resume_from_checkpoint=False, starting from scratch.")
        state = {"start_epoch": 0, "global_step": 0,
                 "best_val_loss": float("inf"),
                 "best_monitored": float("inf")}

    start_epoch    = state["start_epoch"]
    global_step    = state["global_step"]
    best_val_loss  = state["best_val_loss"]
    best_monitored = state.get("best_monitored", float("inf"))

    wandb.init(
        project="immunization",
        config={
            "dataset": dataset, "n_epochs": n_epochs, "lr": lr,
            "batch size": batch, "weight_decay": weight_decay,
            "alpha": alpha, "beta": beta,
            "lambda_vae": lambda_vae, "eps": eps,
            "noise_on_mask": noise_on_mask, "nb_filter": nb_filter,
            "dyn_weight_window": dyn_weight_window,
            "dyn_weight_T_temp": dyn_weight_T_temp,
            "dyn_weight_s_clip": dyn_weight_s_clip,
            "early_stopping": "disabled",
            "target": target if not untargeted else "untargeted (original image)",
            "untargeted": untargeted,
            "margin": margin,
        }
    )

    run_id = wandb.run.id
    best_checkpoint_path    = str(Path(best_checkpoint_path).parent / f"unet_best_{run_id}.pth")
    training_checkpoint_dir = str(Path(training_checkpoint_dir) / run_id)

    if start_epoch >= n_epochs:
        print(f"Training già completato ({start_epoch}/{n_epochs} epoche).")
        return unet

    unet.train()
    if not resume_from_checkpoint:
        dyn_weighter.reset()
        loss_normalizer.reset()

    # ── Target fisso, usato SOLO in modalità targeted ──
    # In modalità untargeted il "target" è l'immagine originale di ogni
    # batch, calcolato dinamicamente nel loop: non serve costruire I_target.
    I_target = None
    posterior_target_list = None

    if not untargeted:
        target_size = getattr(dataloader.dataset, "image_size", None)
        if target_size is None:
            # Fallback to the default training size if the dataset does not expose image_size
            target_size = 224

        if target == "gray":
            I_target = Image.new("RGB", (target_size, target_size), (128, 128, 128))
        elif target == "black":
            I_target = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        elif target == "white":
            I_target = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        elif target == "mean":
            I_target = Image.open("data/diffvax_mean_posterior.png").resize((target_size, target_size), Image.BICUBIC)
        else:
            raise ValueError(f"target '{target}' non valido. Scegli tra: gray, black, white")

        I_target = transforms.ToTensor()(I_target)   # [0, 1]
        I_target = (I_target * 2.0 - 1.0)             # [-1, 1]
        I_target = I_target.unsqueeze(0).to(device)   # [1, 3, target_size, target_size]

        posterior_target_list = []
        for vae in surrogate_vae_models:
            posterior_target = vae.encode(I_target).latent_dist
            posterior_target_list.append(posterior_target)

    for epoch in range(start_epoch, n_epochs):

        train_metrics = defaultdict(float)
        train_weights = []

        for batch_idx, (I, M) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)):
            I = I.to(device)
            M = M.to(device)

            # ── In modalità untargeted, il target di questo batch è
            #    il posterior dell'immagine originale stessa (no_grad: è
            #    solo un riferimento, non un parametro da ottimizzare) ──
            if untargeted:
                with torch.no_grad():
                    posterior_target_list = [
                        vae_model.encode(I).latent_dist
                        for vae_model in surrogate_vae_models
                    ]

            optimizer.zero_grad()
            unet_out = unet(I)
            unet_out = torch.clamp(unet_out, -eps, eps)
            if noise_on_mask:
                unet_out = unet_out * (1 - M)

            I_im = torch.clamp(I + unet_out, -1.0, 1.0)

            posterior_im_list = []
            for vae_model in surrogate_vae_models:
                posterior_im = vae_model.encode(I_im).latent_dist
                posterior_im_list.append(posterior_im)

            loss, log = total_loss(
                I_im=I_im, I=I, M=1 - M, I_target=I_target,
                posterior_im_list=posterior_im_list, posterior_target_list=posterior_target_list,
                dyn_weighter=dyn_weighter,
                alpha=alpha, beta=beta, lambda_vae=lambda_vae,
                noise_on_mask=noise_on_mask,
                untargeted=untargeted, margin=margin,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in log.items():
                if k == "weights":
                    train_weights.append(v)
                else:
                    train_metrics[k] += v
            global_step += 1

        # ── Medie training ──
        n_train = len(dataloader)
        train_metrics = {k: v / n_train for k, v in train_metrics.items()}

        n_surrogates = sum(1 for k in train_metrics if k.startswith("l_surrogate_") and not k.startswith("l_surrogate_mag_"))
        surrogate_str = "  ".join(
            f"l_surrogate_{i}={train_metrics[f'l_surrogate_{i}']:.4f}"
            for i in range(n_surrogates)
        )

        # ── Aggiornamento DynamicWeighter ──
        # Usa SEMPRE la magnitudine positiva della distanza vae (non il
        # termine con segno che entra nella loss totale): il weighter
        # confronta "quanto è grande la loss" rispetto alla sua baseline,
        # e questo deve restare un concetto positivo sia in targeted che
        # in untargeted.
        if untargeted:
            current_loss_surrogates = [
                train_metrics[f"l_surrogate_mag_{i}"] for i in range(n_surrogates)
            ]
        else:
            current_loss_surrogates = [
                train_metrics[f"l_surrogate_{i}"] for i in range(n_surrogates)
            ]
        normalized_losses = loss_normalizer.normalize(current_loss_surrogates)
        dyn_weighter.step(normalized_losses)
        weights = dyn_weighter.get_weights()

        # ── print ──
        print(
            f"\n── Epoch {epoch + 1} Train ──  "
            f"loss={train_metrics['l_tot']:.4f}  "
            f"l_noise={train_metrics['l_noise']:.4f}  "
            f"l_surrogates={train_metrics['l_surrogates']:.4f}  "
            f"{surrogate_str}  "
            f"\nnorm={[f'{l:.4f}' for l in normalized_losses]}  "
            f"weights={[f'{w:.4f}' for w in weights]}"
         )

        # ── Validation ──
        if (epoch + 1) % val_every == 0:
            val_metrics = validation_loop(
                unet=unet,
                val_dataloader=val_dataloader,
                surrogate_vae_models=surrogate_vae_models,
                posterior_target_list=posterior_target_list,
                dyn_weighter=dyn_weighter,
                alpha=alpha,
                beta=beta,
                lambda_vae=lambda_vae,
                noise_on_mask=noise_on_mask,
                untargeted=untargeted,
                margin=margin,
            )

            n_surrogates_val = sum(
                1 for k in val_metrics if k.startswith("l_surrogate_") and not k.startswith("l_surrogate_mag_")
            )

            # ── surrogate total losses ──
            surrogate_str_val = "  ".join(
                f"l_surrogate_{i}={val_metrics[f'l_surrogate_{i}']:.4f}"
                for i in range(n_surrogates_val)
            )

            # ── surrogate losses for weighting ──
            if untargeted:
                val_surrogates = [
                    val_metrics[f"l_surrogate_mag_{i}"]
                    for i in range(n_surrogates_val)
                ]
            else:
                val_surrogates = [
                    val_metrics[f"l_surrogate_{i}"]
                    for i in range(n_surrogates_val)
                ]

            # normalizzazione coerente con train
            val_normalized = loss_normalizer.normalize(val_surrogates)

            # usa gli stessi pesi del train (importante!)
            weights = dyn_weighter.get_weights()
            weights_sum = sum(weights)

            monitored = sum(
                w * l for w, l in zip(weights, val_normalized)
            ) / weights_sum

            # ── print ──
            print(
                f"── Epoch {epoch + 1} Val ──    "
                f"loss={val_metrics['l_tot']:.4f}  "
                f"l_noise={val_metrics['l_noise']:.4f}  "
                f"l_surr={val_metrics['l_surrogates']:.4f}  "
                f"{surrogate_str_val}  "
                f"\nmonitored={monitored:.6f}\n"
            )

            # ── Best model: salva il modello quando migliora `monitored`
            #    oppure quando migliora la validation loss (val_metrics['l_tot']).
            saved = False
            if monitored < best_monitored:
                best_monitored = monitored
                best_val_loss  = val_metrics["l_tot"]
                Path(best_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(unet.state_dict(), best_checkpoint_path)
                print(
                    f"  ✓ Best model saved "
                    f"(monitored={best_monitored:.6f}, val_loss={best_val_loss:.4f})"
                    f" → {best_checkpoint_path}\n"
                )
                saved = True

            # salva anche se la validation loss migliora (caso in cui `monitored` non è usato
            # come criterio principale ma vogliamo comunque mantenere il checkpoint migliore
            # rispetto alla val loss)
            if (not saved) and (val_metrics["l_tot"] < best_val_loss):
                best_val_loss = val_metrics["l_tot"]
                Path(best_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(unet.state_dict(), best_checkpoint_path)
                print(
                    f"  ✓ Best model saved (val_loss improved={best_val_loss:.4f}) "
                    f"→ {best_checkpoint_path}\n"
                )

        wandb.log({
          "train/loss":         train_metrics["l_tot"],
          "train/l_noise":      train_metrics["l_noise"],
          "train/l_surrogates": train_metrics["l_surrogates"],

          **{f"train/l_surrogate_{i}": train_metrics[f"l_surrogate_{i}"]
           for i in range(n_surrogates)},

          **{f"train/l_surrogate_{i}_norm": normalized_losses[i]
           for i in range(n_surrogates)},

          **{f"train/weight_{i}": w for i, w in enumerate(weights)},

          # ───────── VAL ──────────────
          "val/loss":           val_metrics["l_tot"],
          "val/l_noise":        val_metrics["l_noise"],
          "val/l_surrogates":   val_metrics["l_surrogates"],

          **{f"val/l_surrogate_{i}": val_metrics[f"l_surrogate_{i}"]
            for i in range(n_surrogates_val)},

          **{f"val/l_surrogate_{i}_norm": val_normalized[i]
           for i in range(n_surrogates_val)},

          "val/monitored":      monitored,
          "epoch": epoch + 1,
         }, step=epoch + 1)

        # ── Checkpoint ──
        save_training_checkpoint(
            checkpoint_dir=training_checkpoint_dir,
            unet=unet, optimizer=optimizer,
            dyn_weighter=dyn_weighter,
            loss_normalizer=loss_normalizer,
            epoch=epoch, global_step=global_step,
            best_val_loss=best_val_loss,
            patience_count=0,              # non usato ma mantenuto per compatibilità checkpoint
            best_monitored=best_monitored,
        )

    wandb.finish()
    return unet



def validation_loop(
    unet,
    val_dataloader,
    surrogate_vae_models,
    posterior_target_list,
    dyn_weighter,
    alpha:       float = 1.0,
    beta:        float = 1.0,
    lambda_vae: float = 0.03,
    noise_on_mask: bool = False,
    untargeted: bool = False,
    margin: float | None = 10.0,
) -> dict:

    unet.eval()
    val_metrics = {}

    with torch.no_grad():
        for I, M in tqdm(val_dataloader, desc="Validation", leave=False):
            I = I.to(device)
            M = M.to(device)

            # ── In modalità untargeted, il target di questo batch è il
            #    posterior dell'immagine originale stessa ──
            if untargeted:
                posterior_target_list = [
                    vae_model.encode(I).latent_dist
                    for vae_model in surrogate_vae_models
                ]

            # ── Forward ──
            unet_out = unet(I)
            if noise_on_mask:
                unet_out = unet_out * (1 - M)
            I_im = torch.clamp(I + unet_out, -1.0, 1.0)

            # ── VAE ──
            posterior_im_list = []
            for vae_model in surrogate_vae_models:
                posterior_im = vae_model.encode(I_im).latent_dist
                posterior_im_list.append(posterior_im)

            # ── Loss — total_loss gestisce pesi e dyn_weighter internamente ──
            _, log = total_loss(I_im=I_im, I=I, M=1 - M, I_target=None,
                                 posterior_im_list=posterior_im_list,
                                 posterior_target_list=posterior_target_list, dyn_weighter=dyn_weighter, alpha=alpha, beta=beta,
                                 lambda_vae=lambda_vae, noise_on_mask=noise_on_mask,
                                 untargeted=untargeted, margin=margin)

            for k, v in log.items():
                if k == "weights":
                    continue
                val_metrics[k] = val_metrics.get(k, 0.0) + v
    # ── Medie ──
    n = len(val_dataloader)
    val_metrics = {k: v / n for k, v in val_metrics.items()}

    unet.train()
    return val_metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def seed_worker(worker_id):
    """Propaga il seed globale ai worker del DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_gpu_memory(label: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{label}] GPU Memory — allocated: {allocated:.2f}GB | reserved: {reserved:.2f}GB | total: {total:.2f}GB")

if __name__ == "__main__":

    SEED = 2023
    set_seed_lib(SEED)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    DEBUG = False
    N_DEBUG = 100

    dataset = "MagicBrush" #DiffVax | Oxford-Pet | COCO | MagicBrush
    target = "gray" # "gray", "black", "white" o nome file in data/

    train_dataset = ImmunizationDataset(dataset= dataset, split="train")
    val_dataset = ImmunizationDataset(dataset= dataset, split="validation")

    if DEBUG:
        train_dataset = Subset(train_dataset, range(N_DEBUG))
        val_dataset = Subset(val_dataset, range(N_DEBUG))
    # Generator con lo stesso seed per lo shuffle
    g = torch.Generator()
    g.manual_seed(SEED)

    batch=16


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
    )

    nb_filter = [x * 2 for x in [32, 64, 128, 256, 512]]

    unet = NestedUNet(num_classes=3, nb_filter=nb_filter).to(device)

    trained_unet = training_loop(
        unet=unet,
        nb_filter= nb_filter,
        dataloader=train_loader,
        val_dataloader=val_loader,
        dataset= dataset,
        n_epochs=10000,
        lr=1e-4,
        batch=batch,
        weight_decay=1e-2,
        alpha=100.0,
        beta=1.0,
        lambda_vae=1,
        eps= (64 / 255),
        val_every=1,
        best_checkpoint_path="checkpoints/unet_best_z6iqwxu5.pth",
        training_checkpoint_dir="checkpoints/training/",
        device=device,
        resume_from_checkpoint=False,
        resume_only_weights = True,
        noise_on_mask=False,
        dyn_weight_window= 30,
        dyn_weight_T_temp = 0.1,
        dyn_weight_s_clip= 2.0,
        target=target,
        untargeted=False,      # ← attiva la modalità untargeted
        margin=50.0,          # ← tetto sulla distanza vae da massimizzare
    )