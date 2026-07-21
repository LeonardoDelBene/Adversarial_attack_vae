"""
Script per contare quante maschere, all'interno di una cartella, hanno
una frazione di pixel "bianchi" superiore a una soglia (default: 5%).

Uso:
    python count_white_masks.py /percorso/alla/cartella
    python count_white_masks.py /percorso/alla/cartella --threshold 0.10
    python count_white_masks.py /percorso/alla/cartella --white-value 200 --ext .png .jpg

Una maschera è considerata "bianca" in un pixel se il valore (dopo
conversione in scala di grigi, 0-255) è >= --white-value (default 128).
La frazione bianca di un'immagine è: (# pixel bianchi) / (# pixel totali).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compute_white_fraction(mask_path: Path, white_value: int) -> float:
    """
    Calcola la frazione di pixel "bianchi" (>= white_value) in una maschera.
    La maschera viene aperta e convertita in scala di grigi (mode "L").
    """
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    white_fraction = (arr >= white_value).mean()
    return float(white_fraction)


def main():
    parser = argparse.ArgumentParser(
        description="Conta quante maschere in una cartella hanno una parte bianca > soglia."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Cartella contenente le maschere",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Soglia di frazione bianca (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--white-value",
        type=int,
        default=128,
        help="Valore minimo di grigio (0-255) per considerare un pixel 'bianco' (default: 128)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Estensioni dei file da considerare (default: .png .jpg .jpeg)",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    assert folder.is_dir(), f"Cartella non trovata: {folder}"

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}
    mask_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)

    assert len(mask_paths) > 0, f"Nessuna maschera trovata in {folder} con estensioni {sorted(exts)}"

    print(f"[INFO] Cartella: {folder}")
    print(f"[INFO] Maschere trovate: {len(mask_paths)}")
    print(f"[INFO] Soglia frazione bianca: {args.threshold:.2%}")
    print(f"[INFO] Valore soglia bianco (grayscale): {args.white_value}\n")

    count_above = 0
    fractions = []

    for mask_path in mask_paths:
        try:
            frac = compute_white_fraction(mask_path, args.white_value)
        except Exception as e:
            print(f"⚠️  Errore nel leggere {mask_path.name}: {e}")
            continue

        fractions.append(frac)
        above = frac > args.threshold
        if above:
            count_above += 1

        marker = "✓" if above else " "
        print(f"{marker} {mask_path.name:40s}  bianco: {frac:.2%}")

    fractions = np.array(fractions)

    print("\n" + "=" * 60)
    print("RISULTATO")
    print("=" * 60)
    print(f"Maschere totali analizzate:         {len(fractions)}")
    print(f"Maschere con bianco > {args.threshold:.0%}:        {count_above}")
    print(f"Percentuale sul totale:             {count_above / len(fractions):.2%}")
    print(f"\nFrazione bianca - min:    {fractions.min():.2%}")
    print(f"Frazione bianca - max:    {fractions.max():.2%}")
    print(f"Frazione bianca - media:  {fractions.mean():.2%}")
    print(f"Frazione bianca - mediana:{np.median(fractions):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()