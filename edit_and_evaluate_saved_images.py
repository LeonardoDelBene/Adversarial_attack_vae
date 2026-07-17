"""
Edita le immagini immunizzate salvate in una o più cartelle `saved_images`
usando 3 modelli di editing (Attack, AttackInstructPix2Pix, AttackSD),
ognuno con il proprio prompt.

Ogni cartella `saved_images` contiene ora 5 sample distinti (indice sample
arbitrario, es. 40), ciascuno con:
  - `original_{idx}.*`
  - `immunized_sample{idx}_eps_*` (una per ogni eps della sweep)

Per ogni cartella `saved_images`:
  1. Trova, per ciascuno dei 5 sample, `original_{idx}.*` e le sue
     immagini `immunized_sample{idx}_eps_*`.
  2. Per ogni modello: edita ogni `original_{idx}` (-> reference del
     sample) e ogni sua immagine immunizzata, salvando i risultati
     nella stessa cartella con nome `{Model}_{nome_file_originale}`.
  3. Calcola l'EditingScore (media dei 12 fattori) confrontando
     l'edit della reference con l'edit dell'immagine immunizzata,
     per ogni sample e ogni eps.
  4. Fa la MEDIA degli score sui 5 sample per ogni eps, e aggiorna
     `sweep_eps.csv` (nella cartella padre di saved_images) scrivendo
     la media nella colonna `editing_score_<model>`, sulla riga la cui
     colonna `eps` è più vicina a quella nel nome file.

Configurazione: modifica le costanti FOLDERS / PROMPT_* qui sotto e poi
lancia semplicemente:
    python edit_immunized.py

La maschera (necessaria solo per Attack, il modello inpainting) è
diversa per ciascun sample: viene ottenuta da ImmunizationDataset()[idx],
usando lo stesso indice sample estratto dal nome file.
"""

import os
os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"
import re
import sys

import pandas as pd
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image, ImageOps

from model import Attack, AttackInstructPix2Pix, AttackSD
from data import ImmunizationDataset  # adatta il path di import se necessario
from metrics.editing_score import EditingScore  # adatta il path di import se necessario


# ============================================================
# CONFIGURAZIONE - modifica qui, niente parametri da linea di comando
# ============================================================

# Una o più cartelle saved_images da editare
FOLDERS = [
   "experiment/eps_8/saved_images", #8
    "experiment/eps_16/saved_images", #16
    "experiment/eps_32/saved_images", #32
    "experiment/vae_mse/saved_images", #64
    "experiment/eps_128/saved_images", #128
]

# Prompt per ciascun modello di editing
PROMPT_ATTACK = "A person in a garden"            # Attack (inpainting)
PROMPT_INSTRUCTPIX2PIX = "Change the color's hair to blonde"                # AttackInstructPix2Pix
PROMPT_SD = "Change the color's hair to blonde"                 # AttackSD (img2img)


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MODEL_TAGS = ("Attack", "InstructPix2Pix", "AttackSD")

# matcha "eps_0.12549" o "eps0.12549" dentro al nome file
EPS_RE = re.compile(r"eps_?([0-9]*\.?[0-9]+)")

# matcha "original_40.png" -> gruppo 1 = "40"
ORIGINAL_RE = re.compile(r"^original_(\d+)\.", re.IGNORECASE)
# matcha "immunized_sample40_eps_..." -> gruppo 1 = "40"
IMMUNIZED_SAMPLE_RE = re.compile(r"^immunized_sample(\d+)_", re.IGNORECASE)


def is_already_edited(fname):
    return any(fname.startswith(f"{tag}_") for tag in MODEL_TAGS)


def find_samples(folder):
    """
    Ritorna dict: sample_idx -> {"original": path, "immunized": [paths]}
    scansionando la cartella saved_images (che ora contiene 5 sample).
    """
    samples = {}

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(IMG_EXTENSIONS):
            continue
        if is_already_edited(fname):
            continue  # output di run precedenti

        full_path = os.path.join(folder, fname)

        m_orig = ORIGINAL_RE.match(fname)
        if m_orig:
            idx = int(m_orig.group(1))
            samples.setdefault(idx, {"original": None, "immunized": []})
            samples[idx]["original"] = full_path
            continue

        m_imm = IMMUNIZED_SAMPLE_RE.match(fname)
        if m_imm:
            idx = int(m_imm.group(1))
            samples.setdefault(idx, {"original": None, "immunized": []})
            samples[idx]["immunized"].append(full_path)

    return samples


def extract_eps_from_name(fname):
    """Estrae il valore eps (float) dal nome file, es. immunized_sample40_eps_0.501961_....png"""
    match = EPS_RE.search(os.path.basename(fname))
    if not match:
        return None
    return float(match.group(1))


def get_mask_for_sample(dataset, sample_idx):
    """Maschera specifica per il sample_idx, dallo stesso indice del dataset."""
    _, mask_tensor = dataset[sample_idx]
    return to_pil_image(mask_tensor).convert('L')


def run_edit(mod_name, mod, prompt, img, mask):
    if mod_name == 'Attack':  # inpainting: serve la maschera
        out = mod.edit_image(prompt, img, mask)
    else:
        out = mod.edit_image(prompt, img)

    if isinstance(out, (list, tuple)):
        out = out[0]
    if not isinstance(out, Image.Image):
        out = Image.fromarray(out)
    return out


def update_csv_score(csv_path, eps_value, column_name, score, tol=1e-3):
    """Scrive `score` nella colonna `column_name` della riga con eps più vicino a eps_value."""
    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        df[column_name] = pd.NA

    diffs = (df['eps'] - eps_value).abs()
    closest_idx = diffs.idxmin()

    if diffs.loc[closest_idx] > tol:
        print(
            f"[WARN] Nessuna riga con eps vicino a {eps_value} in {csv_path} "
            f"(differenza minima: {diffs.loc[closest_idx]:.6f}). Salto l'aggiornamento."
        )
        return

    df.loc[closest_idx, column_name] = score
    df.to_csv(csv_path, index=False)


def process_folder(folder, models_and_prompts, dataset, judge):
    samples = find_samples(folder)

    if not samples:
        print(f"[{folder}] nessun sample trovato, salto la cartella.")
        return

    csv_path = os.path.join(os.path.dirname(folder), 'sweep_eps.csv')
    if not os.path.exists(csv_path):
        print(f"[{folder}] sweep_eps.csv non trovato in {os.path.dirname(folder)}, salto gli score.")
        csv_path = None

    for mod_name, mod, prompt in models_and_prompts:
        # eps_value (dal nome file) -> lista di score, uno per ogni sample
        scores_by_eps = {}

        for sample_idx, paths in sorted(samples.items()):
            original_path = paths["original"]
            immunized_paths = paths["immunized"]

            if original_path is None:
                print(f"[{folder}] sample {sample_idx}: original_image non trovata, salto.")
                continue
            if not immunized_paths:
                print(f"[{folder}] sample {sample_idx}: nessuna immagine immunizzata trovata, salto.")
                continue

            mask = get_mask_for_sample(dataset, sample_idx)
            original_img = Image.open(original_path).convert('RGB')
            original_name = os.path.splitext(os.path.basename(original_path))[0]

            # 1. Edita la reference (original_image) una sola volta per modello/sample
            try:
                ref_edit = run_edit(mod_name, mod, prompt, original_img, mask)
            except Exception as e:
                print(f"[{folder}] Errore con {mod_name} su {original_path}: {e}")
                continue

            ref_save_path = os.path.join(folder, f"{mod_name}_{original_name}.png")
            ref_edit.save(ref_save_path)

            # 2. Edita ogni immagine immunizzata del sample e calcola lo score vs la reference
            desc = f"{os.path.basename(folder)} [{mod_name}] sample{sample_idx}"
            for img_path in tqdm(immunized_paths, desc=desc):
                name = os.path.splitext(os.path.basename(img_path))[0]
                img = Image.open(img_path).convert('RGB')

                try:
                    imm_edit = run_edit(mod_name, mod, prompt, img, mask)
                except Exception as e:
                    print(f"[{folder}] Errore con {mod_name} su {img_path}: {e}")
                    continue

                save_path = os.path.join(folder, f"{mod_name}_{name}.png")
                imm_edit.save(save_path)

                # editing score: media dei 12 fattori
                try:
                    result = judge(ref_edit, imm_edit, prompt)
                    avg_score = result.get('attack_success_score')
                except Exception as e:
                    print(f"[{folder}] Errore nel calcolo editing score per {save_path}: {e}")
                    continue

                if avg_score is None:
                    continue

                eps_value = extract_eps_from_name(img_path)
                if eps_value is None:
                    print(f"[{folder}] eps non trovato nel nome file {img_path}, salto.")
                    continue

                scores_by_eps.setdefault(eps_value, []).append(avg_score)

        if csv_path is None:
            continue

        column_name = f"editing_score_{mod_name.lower()}"
        for eps_value, scores in scores_by_eps.items():
            mean_score = sum(scores) / len(scores)
            update_csv_score(csv_path, eps_value, column_name, mean_score)
            print(
                f"[{folder}] {column_name} @ eps~{eps_value}: "
                f"media su {len(scores)} sample = {mean_score:.4f}"
            )


def main():
    print('Carico i modelli di editing...')
    models_and_prompts = [
        ('Attack', Attack(), PROMPT_ATTACK),
        ('InstructPix2Pix', AttackInstructPix2Pix(), PROMPT_INSTRUCTPIX2PIX),
        ('AttackSD', AttackSD(), PROMPT_SD),
    ]

    print('Carico il judge per l\'editing score...')
    judge = EditingScore()

    print('Carico il dataset per le maschere (una per ogni sample_idx)...')
    dataset = ImmunizationDataset()

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            print(f"Cartella non trovata, salto: {folder}")
            continue
        process_folder(folder, models_and_prompts, dataset, judge)

    print('Fatto.')


if __name__ == '__main__':
    main()