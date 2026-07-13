"""PSNR metric for evaluating the quality of an image."""

import json
import os

from PIL import Image
from tqdm import tqdm
from .base import Metric
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

class PSNR(Metric):
    """Peak Signal to Noise Ratio (PSNR) metric for evaluating the quality of an image."""

    def __call__(self, original_images, adversarial_images):
        """Calculate the PSNR between original and adversarial images."""
        psnr_values = []
        for img_orig, img_adv in zip(original_images, adversarial_images):
            psnr = self.calculate_metric_between_images(img_orig, img_adv)
            psnr_values.append(psnr)
        return psnr_values
    
    def calculate_metric_between_images(self, img_orig, img_adv):
        """Calculate the PSNR between two images."""
        img_orig = np.array(img_orig)
        img_adv = np.array(img_adv)
        psnr = peak_signal_noise_ratio(img_orig, img_adv)
        return psnr
    
    def evaluate_folder(self, root_dir):
        """Evaluate PSNR across a dataset organized in image folders."""
        results_summary = {}

        total = 0
        psnr_orig_sum = 0.0
        psnr_edited_sum = 0.0
        folders = [
            f for f in sorted(os.listdir(root_dir))
            if f.startswith("img_")
        ]

        for folder in tqdm(folders, desc="Evaluating PSNR", unit="folder"):
            img_dir = os.path.join(root_dir, folder)

            orig_path = os.path.join(img_dir, "original_image.png")
            immunized_path = os.path.join(img_dir, "immunized_image.png")
            edited_orig_path = os.path.join(img_dir, "edited_original.png")
            edited_immunized_path = os.path.join(img_dir, "edited_immunized.png")
            txt_path = os.path.join(img_dir, "prompt_and_metrics.txt")

            if not (os.path.exists(orig_path) and os.path.exists(immunized_path) and
                    os.path.exists(edited_orig_path) and os.path.exists(edited_immunized_path) and
                    os.path.exists(txt_path)):
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            if not prompt:
                print(f"[WARN] No prompt found in {folder}")
                continue

            original = Image.open(orig_path).convert("RGB")
            immunized = Image.open(immunized_path).convert("RGB")
            edited_original = Image.open(edited_orig_path).convert("RGB")
            edited_immunized = Image.open(edited_immunized_path).convert("RGB")

            psnr_orig = self.calculate_metric_between_images(original, immunized)
            psnr_edited = self.calculate_metric_between_images(edited_original, edited_immunized)
            psnr_edited = min(psnr_edited, 100.0)  # o altro valore ragionevole

            result = {
                "psnr_original_vs_immunized": float(psnr_orig),
                "psnr_edited_original_vs_edited_immunized": float(psnr_edited),
            }

            results_summary[folder] = result
            total += 1
            psnr_orig_sum += psnr_orig
            psnr_edited_sum += psnr_edited

            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n\n--- PSNR Evaluation ---\n")
                f.write(json.dumps(result, indent=2))
                f.write("\n")

        avg_psnr_orig = psnr_orig_sum / total if total > 0 else 0.0
        avg_psnr_edited = psnr_edited_sum / total if total > 0 else 0.0

        summary_path = os.path.join(root_dir, "global_summary.txt")
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("=== PSNR Evaluation Summary ===\n\n")
            f.write(f"Total samples evaluated: {total}\n")
            f.write(f"Average original vs immunized PSNR: {avg_psnr_orig:.4f}\n")
            f.write(f"Average edited_original vs edited_immunized PSNR: {avg_psnr_edited:.4f}\n")

        return {
            "total": total,
            "avg_psnr_original_vs_immunized": avg_psnr_orig,
            "avg_psnr_edited_original_vs_edited_immunized": avg_psnr_edited,
            "details": results_summary,
        }
