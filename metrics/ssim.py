"""SSIM metric for evaluating the quality of an image."""

import json
import os

from PIL import Image
from tqdm import tqdm
from .base import Metric
from skimage.metrics import structural_similarity
import numpy as np

class SSIM(Metric):

    def __call__(self, original_images, adversarial_images):
        """Calculate the SSIM between original and adversarial images."""
        ssim_values = []
        for img_orig, img_adv in zip(original_images, adversarial_images):
            ssim = self.calculate_metric_between_images(img_orig, img_adv)
            ssim_values.append(ssim)
        return ssim_values
    
    def calculate_metric_between_images(self, img_orig, img_adv):
        """Calculate the SSIM between two images."""
        img_orig = np.array(img_orig)
        img_adv = np.array(img_adv)
        ssim = structural_similarity(img_orig, img_adv, channel_axis=2)
        return ssim

    def evaluate_folder(self, root_dir):
        """Evaluate SSIM across a dataset organized in image folders."""
        results_summary = {}

        total = 0
        ssim_orig_sum = 0.0
        ssim_edited_sum = 0.0
        folders = [
            f for f in sorted(os.listdir(root_dir))
            if f.startswith("img_")
        ]

        for folder in tqdm(folders, desc="Evaluating SSIM", unit="folder"):
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

            ssim_orig = self.calculate_metric_between_images(original, immunized)
            ssim_edited = self.calculate_metric_between_images(edited_original, edited_immunized)

            result = {
                "ssim_original_vs_immunized": float(ssim_orig),
                "ssim_edited_original_vs_edited_immunized": float(ssim_edited),
            }

            results_summary[folder] = result
            total += 1
            ssim_orig_sum += ssim_orig
            ssim_edited_sum += ssim_edited

            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n\n--- SSIM Evaluation ---\n")
                f.write(json.dumps(result, indent=2))
                f.write("\n")

        avg_ssim_orig = ssim_orig_sum / total if total > 0 else 0.0
        avg_ssim_edited = ssim_edited_sum / total if total > 0 else 0.0

        summary_path = os.path.join(root_dir, "global_summary.txt")
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("=== SSIM Evaluation Summary ===\n\n")
            f.write(f"Total samples evaluated: {total}\n")
            f.write(f"Average original vs immunized SSIM: {avg_ssim_orig:.4f}\n")
            f.write(f"Average edited_original vs edited_immunized SSIM: {avg_ssim_edited:.4f}\n")

        return {
            "total": total,
            "avg_ssim_original_vs_immunized": avg_ssim_orig,
            "avg_ssim_edited_original_vs_edited_immunized": avg_ssim_edited,
            "details": results_summary,
        }
