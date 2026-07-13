"""
Edita le immagini immunizzate salvate in una o più cartelle `saved_images`
usando 3 modelli di editing (Attack, AttackInstructPix2Pix, AttackSD),
ognuno con il proprio prompt.

Per ogni cartella `saved_images`:
  1. Trova `original_image.*` e le immagini `immunized_eps_*`.
  2. Per ogni modello: edita `original_image` (-> reference) e ogni
     immagine immunizzata, salvando i risultati nella stessa cartella
     con nome `{Model}_{nome_file_originale}`.
  3. Calcola l'EditingScore (media dei 12 fattori) confrontando
     l'edit della reference con l'edit dell'immagine immunizzata,
     per ogni eps.
  4. Aggiorna `sweep_eps.csv` (nella cartella padre di saved_images)
     scrivendo lo score nella colonna `editing_score_<model>`,
     sulla riga la cui colonna `eps` è più vicina a quella nel nome file.

Configurazione: modifica le costanti FOLDERS / PROMPT_* qui sotto e poi
lancia semplicemente:
    python edit_immunized.py

La maschera (necessaria solo per Attack, il modello inpainting) è la
stessa per tutte le immagini/cartelle: viene ottenuta una sola volta da
ImmunizationDataset()[0] (sample 0), da cui derivano tutte le immagini.
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
   "experiment/unet_best_4g2mrzt5/saved_images", #8
    "experiment/unet_best_fk2utznx/saved_images", #16
    "experiment/unet_best_2ji8vjn3/saved_images", #32
    "experiment/unet_best_uz0247gg/saved_images", #64
    "experiment/unet_best_o23oqvbx/saved_images", #128
]

# Prompt per ciascun modello di editing
PROMPT_ATTACK = "A person in a garden"            # Attack (inpainting)
PROMPT_INSTRUCTPIX2PIX = "Change the color's hair to blonde"                # AttackInstructPix2Pix
PROMPT_SD = "Change the color's hair to blonde"                 # AttackSD (img2img)


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MODEL_TAGS = ("Attack", "InstructPix2Pix", "AttackSD")

# matcha "eps_0.12549" o "eps0.12549" dentro al nome file
EPS_RE = re.compile(r"eps_?([0-9]*\.?[0-9]+)")


def is_already_edited(fname):
    return any(fname.startswith(f"{tag}_") for tag in MODEL_TAGS)


def find_images(folder):
    """Ritorna (original_path, [lista immagini immunizzate]) nella cartella saved_images."""
    original_path = None
    immunized_paths = []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(IMG_EXTENSIONS):
            continue
        if is_already_edited(fname):
            continue  # output di run precedenti

        full_path = os.path.join(folder, fname)
        if fname.lower().startswith('original_0'):
            original_path = full_path
        elif fname.lower().startswith('immunized'):
            immunized_paths.append(full_path)

    return original_path, immunized_paths


def extract_eps_from_name(fname):
    """Estrae il valore eps (float) dal nome file, es. immunized_eps_0.12549_....png"""
    match = EPS_RE.search(os.path.basename(fname))
    if not match:
        return None
    return float(match.group(1))


def get_fixed_mask():
    """Maschera del sample 0, comune a tutte le immagini/cartelle."""
    dataset = ImmunizationDataset()
    _, mask_tensor = dataset[0]
    #mask_tensor = 1 - mask_tensor
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


def process_folder(folder, models_and_prompts, mask, judge):
    original_path, immunized_paths = find_images(folder)

    if original_path is None:
        print(f"[{folder}] original_image non trovata, salto la cartella.")
        return
    if not immunized_paths:
        print(f"[{folder}] nessuna immagine immunizzata trovata, salto la cartella.")
        return

    csv_path = os.path.join(os.path.dirname(folder), 'sweep_eps.csv')
    if not os.path.exists(csv_path):
        print(f"[{folder}] sweep_eps.csv non trovato in {os.path.dirname(folder)}, salto gli score.")
        csv_path = None

    original_img = Image.open(original_path).convert('RGB')
    original_name = os.path.splitext(os.path.basename(original_path))[0]

    for mod_name, mod, prompt in models_and_prompts:
        # 1. Edita la reference (original_image) una sola volta per modello
        try:
            ref_edit = run_edit(mod_name, mod, prompt, original_img, mask)
        except Exception as e:
            print(f"[{folder}] Errore con {mod_name} su {original_path}: {e}")
            continue

        ref_save_path = os.path.join(folder, f"{mod_name}_{original_name}.png")
        ref_edit.save(ref_save_path)

        # 2. Edita ogni immagine immunizzata e calcola lo score vs la reference
        for img_path in tqdm(immunized_paths, desc=f"{os.path.basename(folder)} [{mod_name}]"):
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

            if csv_path is None or avg_score is None:
                continue

            eps_value = extract_eps_from_name(img_path)
            if eps_value is None:
                print(f"[{folder}] eps non trovato nel nome file {img_path}, salto l'aggiornamento csv.")
                continue

            column_name = f"editing_score_{mod_name.lower()}"
            update_csv_score(csv_path, eps_value, column_name, avg_score)


def main():
    print('Carico i modelli di editing...')
    models_and_prompts = [
        ('Attack', Attack(), PROMPT_ATTACK),
        ('InstructPix2Pix', AttackInstructPix2Pix(), PROMPT_INSTRUCTPIX2PIX),
        ('AttackSD', AttackSD(), PROMPT_SD),
    ]

    print('Carico il judge per l\'editing score...')
    judge = EditingScore()

    print('Calcolo la maschera (sample 0 del dataset)...')
    mask = get_fixed_mask()

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            print(f"Cartella non trovata, salto: {folder}")
            continue
        process_folder(folder, models_and_prompts, mask, judge)

    print('Fatto.')


if __name__ == '__main__':
    main()