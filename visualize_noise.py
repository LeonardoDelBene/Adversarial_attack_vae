"""Save the pixel-wise noise between an original and an immunized image, rescaled to 0-255."""

from pathlib import Path

import numpy as np
from PIL import Image


# Modifica questi valori prima di eseguire lo script.
ORIGINAL_IMAGE = Path("output/SD_Inpainting/full_dataset/MagicBrush_diff_opt_mask/img_124/original_image.png")
IMMUNIZED_IMAGE = Path("output/SD_Inpainting/full_dataset/MagicBrush_diff_opt_mask/img_124/immunized_image.png")
OUTPUT_IMAGE = Path("Img/noise_img_124.png")


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 RGB values in the range [0, 255]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def rescale_to_255(array: np.ndarray) -> np.ndarray:
    """Rescale an array to the full [0, 255] range via min-max normalization."""
    min_value = float(np.min(array))
    max_value = float(np.max(array))

    if max_value - min_value == 0:
        return np.zeros_like(array, dtype=np.uint8)

    rescaled = (array - min_value) / (max_value - min_value) * 255.0
    return rescaled.astype(np.uint8)


def save_noise(original_path: Path, immunized_path: Path, output_path: Path) -> None:
    original = load_rgb(original_path)
    immunized = load_rgb(immunized_path)

    if original.shape != immunized.shape:
        raise ValueError(
            "Le immagini devono avere la stessa dimensione e lo stesso numero di canali: "
            f"originale={original.shape}, immunizzata={immunized.shape}"
        )

    # Differenza pixel-wise in scala 0-255, riscalata sull'intero range 0-255.
    noise = immunized - original
    absolute_noise = np.abs(noise)
    rescaled_noise = rescale_to_255(absolute_noise)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rescaled_noise).save(output_path)
    print(f"Immagine del rumore salvata in: {output_path}")


if __name__ == "__main__":
    save_noise(ORIGINAL_IMAGE, IMMUNIZED_IMAGE, OUTPUT_IMAGE)