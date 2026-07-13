import os
os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"
import copy
import csv
import glob
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ImmunizationDataset
from loss import vae_mse, noise_loss
from model import Attack, Immunization
from utils import set_seed_lib


# ─────────────────────────────────────────────────────────────────────────────
# Inner loop: PGD-style refinement con mu_target esterno e parzialmente
# differenziabile (ultimi k step a grafo aperto verso mu_target).
# ─────────────────────────────────────────────────────────────────────────────

def inner_loop_pgd_for_target_search(
    immunization_model: Immunization,
    img: torch.Tensor,
    img_mask: torch.Tensor,
    mu_target: torch.Tensor,
    noise_mode: str = "mask",
    eps: float = 16 / 255,
    lr: float = 1e-4,
    n_steps: int = 300,
    lambda_vae: float = 1.0,
    lambda_noise: float = 100.0,
    k_grad_steps: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Variante di Immunization.targeted_unet_refinement in cui il target non e'
    self.posterior_target.mean (fisso) ma mu_target, un tensore esterno che
    DEVE rimanere collegato al grafo autograd per gli ultimi `k_grad_steps`
    step, in modo da poter calcolare d(loss_finale)/d(mu_target).

    I primi (n_steps - k_grad_steps) step vengono eseguiti in torch.no_grad()
    rispetto a mu_target (esattamente come il training normale della local_unet,
    nessun costo aggiuntivo di memoria). Solo gli ultimi k_grad_steps step
    mantengono il grafo aperto verso mu_target, cosi' la VRAM usata per il
    backward esterno e' proporzionale a k_grad_steps e non a n_steps.

    Ritorna: (img_final, l_vae_finale, l_noise_finale)
    l_vae_finale e' un tensore CON grad_fn collegato a mu_target — e' quello
    che il chiamante user'a per il backward esterno.

    NOTA SU VRAM: il chiamante e' responsabile di fare il backward su
    l_vae_finale (e di liberarne il grafo, es. con .backward() seguito da
    eliminazione dei riferimenti) PRIMA di passare al sample successivo.
    Questa funzione non accumula nulla tra una chiamata e l'altra: local_unet
    e il suo optimizer sono locali e vengono distrutti all'uscita della
    funzione (vedi `del` espliciti a fine funzione), cosi' la VRAM occupata
    da essi non si accumula con n_samples.
    """
    device = immunization_model.device
    vae = immunization_model.vae
    vae.eval()

    local_unet = copy.deepcopy(immunization_model.model).to(device)
    local_unet.eval()

    optimizer = torch.optim.Adam(local_unet.parameters(), lr=lr)

    img_orig = img.float().to(device)
    mask = img_mask.float().to(device)

    # Stage 1 (offline NestedUNet), identico a immunize_img — nessun grad verso mu_target qui.
    with torch.no_grad():
        unet_out = immunization_model.model(img_orig)
        if noise_mode == "mask":
            unet_out = unet_out * (1 - mask)
    img_adv = torch.clamp(img_orig + unet_out, immunization_model.clamp_min, immunization_model.clamp_max)

    n_no_grad_steps = max(0, n_steps - k_grad_steps)

    # ── Fase A: step "ciechi" rispetto a mu_target (no_grad sul target) ──────
    # mu_target viene comunque usato nella loss per aggiornare local_unet,
    # ma lo trattiamo come costante per questi step (.detach()), cosi' non
    # si accumula grafo inutile.
    mu_target_const = mu_target.detach()

    for step in range(n_no_grad_steps):
        noise = local_unet(img_adv)
        if noise_mode == "mask":
            noise = noise * (1 - mask)
        noise = torch.clamp(noise, -eps, eps)

        img_final = img_adv + noise
        total_delta = torch.clamp(img_final - img_orig, -eps, eps)
        img_final = torch.clamp(img_orig + total_delta, -1.0, 1.0)

        posterior_im = vae.encode(img_final).latent_dist
        l_vae = F.mse_loss(posterior_im.mean, mu_target_const.expand_as(posterior_im.mean))

        if noise_mode == "mask":
            l_noise = noise_loss(img_orig, img_final, 1 - mask, noise_on_mask=True)
        else:
            l_noise = noise_loss(img_orig, img_final, 1 - mask, noise_on_mask=False)

        loss = lambda_vae * l_vae + lambda_noise * l_noise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step "cieco": il grafo non serve oltre questa iterazione.
        del noise, img_final, total_delta, posterior_im, l_vae, l_noise, loss

    # ── Fase B: ultimi k_grad_steps step, grafo aperto verso mu_target ────────
    # Qui local_unet continua ad allenarsi normalmente (suo optimizer, suo
    # grafo), ma la loss che restituiamo al chiamante esterno e' quella
    # dell'ULTIMO step, calcolata SENZA detach su mu_target, cosi' il
    # backward esterno puo' propagare fino a mu_target.
    l_vae_final = None
    l_noise_final = None
    img_final = None

    for step in range(k_grad_steps):
        noise = local_unet(img_adv)
        if noise_mode == "mask":
            noise = noise * (1 - mask)
        noise = torch.clamp(noise, -eps, eps)

        img_final = img_adv + noise
        total_delta = torch.clamp(img_final - img_orig, -eps, eps)
        img_final = torch.clamp(img_orig + total_delta, -1.0, 1.0)

        posterior_im = vae.encode(img_final).latent_dist

        is_last_step = (step == k_grad_steps - 1)

        if is_last_step:
            # Ultimo step: NON facciamo detach di mu_target. Questa loss verra'
            # usata sia per l'update interno di local_unet sia, dal chiamante,
            # per il backward esterno su mu_target.
            l_vae = F.mse_loss(posterior_im.mean, mu_target.expand_as(posterior_im.mean))
        else:
            l_vae = F.mse_loss(posterior_im.mean, mu_target.detach().expand_as(posterior_im.mean))

        if noise_mode == "mask":
            l_noise = noise_loss(img_orig, img_final, 1 - mask, noise_on_mask=True)
        else:
            l_noise = noise_loss(img_orig, img_final, 1 - mask, noise_on_mask=False)

        loss = lambda_vae * l_vae + lambda_noise * l_noise

        if is_last_step:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if is_last_step:
            l_vae_final = l_vae
            l_noise_final = l_noise
        else:
            del noise, img_final, total_delta, posterior_im, l_vae, l_noise, loss

    img_final_detached = img_final.detach()

    # local_unet/optimizer sono locali a questa chiamata: liberiamo
    # esplicitamente i loro tensori (in particolare gli stati di Adam, che
    # occupano VRAM quanto i parametri del modello) cosi' non si accumulano
    # da una chiamata all'altra. l_vae_final/l_noise_final restano vivi
    # perche' il chiamante deve ancora farne il backward verso mu_target.
    del local_unet, optimizer
    torch.cuda.empty_cache()

    return img_final_detached, l_vae_final, l_noise_final


# ─────────────────────────────────────────────────────────────────────────────
# Config hashing — per verificare che il resume avvenga con una config
# "compatibile" con quella del checkpoint (stesso init_mode, dataset, ecc.)
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_KEYS_FOR_HASH = [
    "dataset_type", "dataset_split", "image_size", "noise_mode", "eps",
    "inner_lr", "n_steps", "lambda_vae", "lambda_noise", "k_grad_steps",
    "outer_lr", "init_mode", "checkpoint_path", "molt_filter", "seed",
]


def _config_fingerprint(config: Dict[str, Any]) -> str:
    relevant = {k: config.get(k) for k in _CONFIG_KEYS_FOR_HASH}
    raw = repr(sorted(relevant.items())).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Resume: trova e carica l'ultimo checkpoint completo (mu_target + stato
# dell'outer_optimizer + epoca + history), non il solo tensore mu_target.
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Cerca in output_dir i file "checkpoint_epoch_{N}.pt" (checkpoint completi
    di resume, diversi dai vecchi "mu_target_epoch_{N}.pt" che contengono
    solo il tensore) e ritorna il path di quello con N piu' alto, oppure
    None se non ne esiste nessuno.
    """
    pattern = os.path.join(output_dir, "checkpoint_epoch_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _epoch_of(path: str) -> int:
        m = re.search(r"checkpoint_epoch_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    candidates.sort(key=_epoch_of)
    return candidates[-1]


def _save_full_checkpoint(
    output_dir: str,
    epoch: int,
    mu_target: torch.Tensor,
    outer_optimizer: torch.optim.Optimizer,
    history: List[Dict[str, float]],
    config: Dict[str, Any],
) -> str:
    ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "mu_target": mu_target.detach().cpu(),
            "outer_optimizer_state": outer_optimizer.state_dict(),
            "history": history,
            "config_fingerprint": _config_fingerprint(config),
        },
        ckpt_path,
    )
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Outer loop: ottimizzazione di mu_target su tutto il dataset
# ─────────────────────────────────────────────────────────────────────────────

def optimize_universal_target(config: Dict[str, Any]) -> torch.Tensor:
    """
    Trova mu_target* che minimizza la l_vae FINALE dell'attacco (dopo il PGD
    interno completo), mediata su tutto il dataset.

    Ad ogni epoca esterna, per ogni immagine del dataset:
      1. esegue l'inner loop PGD completo (n_steps totali, di cui solo gli
         ultimi k_grad_steps differenziabili verso mu_target), ottiene la
         l_vae finale
      2. fa IMMEDIATAMENTE il backward di l_vae/n_samples verso mu_target
         (gradient accumulation: la somma dei gradienti per-sample equivale
         al gradiente della loss media, quindi il risultato matematico e'
         identico a un singolo backward sulla somma) e libera il grafo
      3. ogni `grad_accum_steps` sample (o a fine epoca) fa uno step di Adam
         su mu_target e azzera i gradienti accumulati

    Questo evita di accumulare in memoria i grafi autograd di tutti gli
    n_samples campioni dell'epoca (che prima causava un OOM crescente con
    n_samples): ogni grafo viene creato, usato per il backward e scartato
    prima di passare al sample successivo.

    RESUME
    ------
    Se config["resume"] è True, prima di inizializzare mu_target da zero
    viene cercato un checkpoint da cui ripartire:
      - config["resume_from"], se specificato esplicitamente, altrimenti
      - l'ultimo "checkpoint_epoch_{N}.pt" trovato in config["output_dir"]
        (via find_latest_checkpoint).
    Dal checkpoint vengono ripristinati: valori di mu_target, stato interno
    dell'outer_optimizer (i momenti di Adam — importante per non "ripartire
    a freddo" e riottenere update bruschi appena dopo il resume), la history
    delle epoche precedenti, e l'epoca da cui ripartire (start_epoch).
    Se non viene trovato nessun checkpoint, si parte normalmente da zero
    (con un avviso stampato a schermo).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed_lib(config.get("seed", 42))

    attack_model = Attack()
    immunization_model = Immunization(
        device=str(device),
        load_existing=config.get("load_existing", True),
        load_path=config.get("checkpoint_path"),
        vae=attack_model.model.vae,
        molt_filter=config.get("molt_filter", 1),
    )

    dataset = ImmunizationDataset(
        dataset=config.get("dataset_type", "DiffVax"),
        split=config.get("dataset_split", "validation"),
        image_size=config.get("image_size", 512),
    )

    n_samples = config.get("n_samples", len(dataset))
    n_samples = min(n_samples, len(dataset))

    # Numero di sample dopo i quali viene fatto uno step di Adam su
    # mu_target. 1 = uno step per sample (massimo risparmio VRAM, piu' step
    # totali per epoca). Un valore piu' alto si comporta in modo piu' simile
    # all'originale (media su piu' sample) ma tiene piu' grafi in memoria
    # tra uno step e l'altro.
    grad_accum_steps = max(1, config.get("grad_accum_steps", 1))

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    n_epochs = config.get("n_epochs", 20)
    k_grad_steps = config.get("k_grad_steps", 5)
    inner_n_steps = config.get("n_steps", 300)

    # ── Tentativo di resume ───────────────────────────────────────────────
    resume_ckpt_path: Optional[str] = None
    if config.get("resume", False):
        resume_ckpt_path = config.get("resume_from") or find_latest_checkpoint(output_dir)

    history: List[Dict[str, float]] = []
    start_epoch = 0  # numero di epoche gia' completate

    if resume_ckpt_path is not None and os.path.isfile(resume_ckpt_path):
        print(f"\n[resume] Carico checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)

        current_fp = _config_fingerprint(config)
        ckpt_fp = ckpt.get("config_fingerprint")
        if ckpt_fp is not None and ckpt_fp != current_fp:
            print(
                "[resume] ATTENZIONE: la config attuale non coincide con quella "
                "usata per salvare questo checkpoint (fingerprint diverso). "
                "Procedo comunque, ma verifica init_mode/eps/lr/ecc."
            )

        mu_target = ckpt["mu_target"].detach().clone().to(device)
        mu_target.requires_grad_(True)

        outer_optimizer = torch.optim.Adam([mu_target], lr=config.get("outer_lr", 1e-2))
        try:
            outer_optimizer.load_state_dict(ckpt["outer_optimizer_state"])
        except Exception as e:
            print(f"[resume] Impossibile ripristinare lo stato dell'optimizer ({e}); riparto con Adam 'a freddo'.")

        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0)

        print(f"[resume] Riparto dall'epoca {start_epoch + 1}/{n_epochs} "
              f"(gia' completate: {start_epoch})")
    else:
        if config.get("resume", False):
            print("\n[resume] Nessun checkpoint trovato: parto da zero.")

        # ── Inizializzazione di mu_target ─────────────────────────────────
        # Warm start: latente medio del gray target gia' usato
        # (self.posterior_target), oppure rumore gaussiano, secondo
        # config["init_mode"].
        init_mode = config.get("init_mode", "gray")
        with torch.no_grad():
            latent_shape = immunization_model.posterior_target.mean.shape  # [1, 4, 64, 64] tipicamente

            if init_mode == "gray":
                mu_target = immunization_model.posterior_target.mean.clone()
            elif init_mode == "noise":
                mu_target = torch.randn(latent_shape, device=device) * immunization_model.posterior_target.mean.std()
            elif init_mode == "zeros":
                mu_target = torch.zeros(latent_shape, device=device)
            else:
                raise ValueError(f"init_mode non riconosciuto: {init_mode}")

        mu_target = mu_target.detach().clone().to(device)
        mu_target.requires_grad_(True)

        outer_optimizer = torch.optim.Adam([mu_target], lr=config.get("outer_lr", 1e-2))

    if start_epoch >= n_epochs:
        print(
            f"[resume] Il checkpoint ha gia' completato {start_epoch} epoche "
            f">= n_epochs={n_epochs}: niente da fare. Aumenta n_epochs per continuare."
        )
        del immunization_model
        torch.cuda.empty_cache()
        return mu_target.detach()

    for epoch in range(start_epoch, n_epochs):
        print("\n" + "=" * 60)
        print(f"Outer epoch {epoch + 1}/{n_epochs}")

        outer_optimizer.zero_grad()

        epoch_l_vae_sum = 0.0
        epoch_l_noise_sum = 0.0

        for idx in range(n_samples):
            image_t, mask_t = dataset[idx]
            image_t = image_t.unsqueeze(0).to(device).float()
            mask_t = mask_t.unsqueeze(0).to(device).float()

            img_final, l_vae, l_noise = inner_loop_pgd_for_target_search(
                immunization_model=immunization_model,
                img=image_t,
                img_mask=mask_t,
                mu_target=mu_target,
                noise_mode=config.get("noise_mode", "mask"),
                eps=config.get("eps", 16 / 255),
                lr=config.get("inner_lr", 1e-4),
                n_steps=inner_n_steps,
                lambda_vae=config.get("lambda_vae", 1.0),
                lambda_noise=config.get("lambda_noise", 100.0),
                k_grad_steps=k_grad_steps,
            )

            epoch_l_vae_sum += l_vae.item()
            epoch_l_noise_sum += l_noise.item()

            # Backward immediato (gradient accumulation): equivalente
            # matematicamente a "mean_loss.backward()" fatto sulla somma di
            # tutti gli l_vae, ma il grafo di questo sample viene liberato
            # subito dopo, invece di restare vivo insieme a quello di tutti
            # gli altri sample fino a fine epoca.
            (l_vae / n_samples).backward()

            is_accum_boundary = ((idx + 1) % grad_accum_steps == 0) or (idx == n_samples - 1)
            if is_accum_boundary:
                outer_optimizer.step()
                outer_optimizer.zero_grad()

            # Il grafo di l_vae/l_noise/img_final non serve piu': li
            # eliminiamo esplicitamente prima del prossimo sample.
            del img_final, l_vae, l_noise, image_t, mask_t
            torch.cuda.empty_cache()

            if (idx + 1) % config.get("log_every_sample", 5) == 0 or idx == n_samples - 1:
                print(
                    f"  [sample {idx + 1:3d}/{n_samples}] "
                    f"l_vae={epoch_l_vae_sum / (idx + 1):.6f} (media cumulata)  "
                    f"l_noise={epoch_l_noise_sum / (idx + 1):.6f} (media cumulata)"
                )

        mean_l_vae = epoch_l_vae_sum / n_samples
        mean_l_noise = epoch_l_noise_sum / n_samples

        print(f"[Epoch {epoch + 1}] mean_l_vae={mean_l_vae:.6f}  mean_l_noise={mean_l_noise:.6f}")

        history.append({"epoch": epoch + 1, "mean_l_vae": mean_l_vae, "mean_l_noise": mean_l_noise})

        # ── Checkpoint periodico ────────────────────────────────────────────
        # Salviamo sia il checkpoint COMPLETO (per il resume: mu_target +
        # stato Adam + history + epoca) sia il vecchio formato "solo tensore"
        # (per compatibilita' con decode_target_to_image e altri script che
        # si aspettano torch.load(...) -> tensore puro).
        if (epoch + 1) % config.get("save_every", 5) == 0 or epoch == n_epochs - 1:
            full_ckpt_path = _save_full_checkpoint(
                output_dir=output_dir,
                epoch=epoch + 1,
                mu_target=mu_target,
                outer_optimizer=outer_optimizer,
                history=history,
                config=config,
            )
            print(f"Saved resume checkpoint: {full_ckpt_path}")

            tensor_ckpt_path = os.path.join(output_dir, f"mu_target_epoch_{epoch + 1}.pt")
            torch.save(mu_target.detach().cpu(), tensor_ckpt_path)
            print(f"Saved mu_target checkpoint: {tensor_ckpt_path}")

    _save_history_csv(history, output_dir, "target_search_history.csv")
    _plot_history(history, output_dir)

    final_path = os.path.join(output_dir, "mu_target_final.pt")
    torch.save(mu_target.detach().cpu(), final_path)
    print(f"\nSaved final optimized target: {final_path}")

    # Cleanup VRAM
    del immunization_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return mu_target.detach()


def _save_history_csv(history: List[Dict[str, float]], output_dir: str, filename: str) -> None:
    if not history:
        return
    path = os.path.join(output_dir, filename)
    with open(path, mode="w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved history CSV: {path}")


def _plot_history(history: List[Dict[str, float]], output_dir: str) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["mean_l_vae"] for h in history], marker="o", label="mean l_vae (finale)")
    ax.set_title("Target search — convergenza l_vae finale per epoca")
    ax.set_xlabel("Outer epoch")
    ax.set_ylabel("l_vae")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    path = os.path.join(output_dir, "target_search_l_vae.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: visualizzare il target ottimizzato decodificandolo come immagine
# ─────────────────────────────────────────────────────────────────────────────

def decode_target_to_image(mu_target_path: str, checkpoint_path: str, molt_filter: int, output_path: str) -> None:
    """
    Decodifica mu_target (tensore latente) tramite il decoder VAE, solo per
    ispezione visiva. Il risultato NON e' il target "vero" usato nella loss
    (che resta il tensore latente): e' solo una visualizzazione approssimata,
    perche' il decoder lavora sulla mean della posterior, non sull'intera
    distribuzione, e introduce una propria perdita di informazione.

    NOTA: se mu_target_path punta a un checkpoint completo di resume
    ("checkpoint_epoch_*.pt", un dict con chiave "mu_target") invece che a
    un vecchio "mu_target_epoch_*.pt" (tensore puro), viene gestito comunque
    correttamente estraendo il tensore dal dict.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attack_model = Attack()
    immunization_model = Immunization(
        device=str(device),
        load_existing=True,
        load_path=checkpoint_path,
        vae=attack_model.model.vae,
        molt_filter=molt_filter,
    )

    loaded = torch.load(mu_target_path, map_location=device)
    if isinstance(loaded, dict) and "mu_target" in loaded:
        mu_target = loaded["mu_target"].to(device)
    else:
        mu_target = loaded.to(device)

    with torch.no_grad():
        decoded = immunization_model.vae.decode(mu_target).sample
        decoded = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)

    from torchvision.utils import save_image
    save_image(decoded, output_path)
    print(f"Saved decoded target preview: {output_path}")

    del immunization_model
    torch.cuda.empty_cache()


def get_default_config() -> Dict[str, Any]:
    return {
        "dataset_type": "DiffVax",
        "dataset_split": "train",
        "image_size": 512,
        "n_samples": 100,  # None -> usa tutto il dataset; o un int per limitare
        "noise_mode": "mask",
        "eps": 64 / 255,
        "inner_lr": 1e-4,
        "n_steps": 100,
        "lambda_vae": 1.0,
        "lambda_noise": 100.0,
        "k_grad_steps": 1,
        "outer_lr": 1e-3,
        "n_epochs": 100,
        "init_mode": "gray",  # "gray" | "noise" | "zeros"
        "load_existing": True,
        "checkpoint_path": os.path.join("checkpoints", "unet_best_nv5dqvvb.pth"),
        "molt_filter": 2,
        "seed": 2043,
        "save_every": 1,
        "log_every_sample": 10,
        # Quanti sample tra un Adam step e l'altro su mu_target. 1 = step
        # per sample (consigliato per VRAM limitata). Aumenta solo se hai
        # VRAM di scorta e vuoi un comportamento piu' vicino all'originale.
        "grad_accum_steps": 2,
        "output_dir": os.path.join("experiment", "target_search"),
        # Resume: True per riprendere automaticamente dall'ultimo
        # "checkpoint_epoch_*.pt" trovato in output_dir. Puoi anche puntare
        # esplicitamente a un checkpoint con "resume_from": "<path>".
        "resume": False,
        "resume_from": None,
    }


if __name__ == "__main__":
    config = get_default_config()
    if config["n_samples"] is None:
        config.pop("n_samples")  # optimize_universal_target usera' len(dataset)
    optimize_universal_target(config)


    #PID:3991106