import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .base import Metric

DEFAULT_SEGMENTATION_MODEL = "facebook/maskformer-swin-base-coco"


class SegmentationScore(Metric):
    """Accordo semantico tra due immagini via MaskFormer (mIoU ottimistico/pessimistico).

    ottimistico  -> mIoU calcolato solo sulle classi comuni a entrambe le immagini
    pessimistico -> mIoU calcolato su tutte le classi rilevate in almeno una delle due
                    (le classi mancanti in un'immagine contano 0 di IoU)
    """

    def __init__(
        self,
        *args,
        model_name: str = DEFAULT_SEGMENTATION_MODEL,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor

        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.processor = MaskFormerImageProcessor.from_pretrained(model_name)
        self.model = (
            MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(self.device).eval()
        )

    # ── Segmentazione ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _segment(self, image: Image.Image) -> dict[int, np.ndarray]:
        image = image.convert("RGB")
        inputs = {
            key: value.to(self.device)
            for key, value in self.processor(images=image, return_tensors="pt").items()
        }
        outputs = self.model(**inputs)
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        return {
            int(label): (segmentation == label).cpu().numpy().astype(np.uint8)
            for label in torch.unique(segmentation).tolist()
        }

    @staticmethod
    def _mean_iou(left: dict, right: dict, labels) -> float:
        if not labels:
            return 0.0
        shape = next(iter(left.values())).shape
        zeros = np.zeros(shape, dtype=np.uint8)
        values = []
        for label in labels:
            l_mask = left.get(label, zeros)
            r_mask = right.get(label, zeros)
            intersection = np.logical_and(l_mask, r_mask).sum()
            union = np.logical_or(l_mask, r_mask).sum()
            values.append(float(intersection / union) if union else 0.0)
        return float(np.mean(values))

    # ── Metrica principale ───────────────────────────────────────────────────

    def compute(self, image_orig: Image.Image, image_adv: Image.Image) -> dict:
        clean_masks = self._segment(image_orig)
        attacked_masks = self._segment(image_adv)
        common = sorted(set(clean_masks) & set(attacked_masks))
        all_labels = sorted(set(clean_masks) | set(attacked_masks))
        return {
            "optimistic_iou": self._mean_iou(clean_masks, attacked_masks, common),
            "pessimistic_iou": self._mean_iou(clean_masks, attacked_masks, all_labels),
        }

    def __call__(self, image_orig: Image.Image, image_adv: Image.Image) -> dict:
        return self.compute(image_orig, image_adv)

    # ── Valutazione su cartella dataset ──────────────────────────────────────

    def evaluate_folder(self, root_dir):
        results_summary = {}
        results_summary_edit = {}

        folders = [f for f in sorted(os.listdir(root_dir)) if f.startswith("img_")]
        for folder in tqdm(folders, desc="Evaluating Segmentation mIoU", unit="folder"):
            img_dir = os.path.join(root_dir, folder)

            orig_path = os.path.join(img_dir, "original_image.png")
            adv_path = os.path.join(img_dir, "immunized_image.png")
            edit_orig_path = os.path.join(img_dir, "edited_original.png")
            edit_adv_path = os.path.join(img_dir, "edited_immunized.png")
            txt_path = os.path.join(img_dir, "prompt_and_metrics.txt")

            if not (
                os.path.exists(orig_path)
                and os.path.exists(adv_path)
                and os.path.exists(edit_orig_path)
                and os.path.exists(edit_adv_path)
            ):
                continue

            img_orig = Image.open(orig_path).convert("RGB")
            img_adv = Image.open(adv_path).convert("RGB")
            edit_orig = Image.open(edit_orig_path).convert("RGB")
            edit_adv = Image.open(edit_adv_path).convert("RGB")

            result = self.compute(img_orig, img_adv)
            results_summary[folder] = result

            result_edit = self.compute(edit_orig, edit_adv)
            results_summary_edit[folder] = result_edit

            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n\n---- Original vs Immunized Segmentation mIoU ----\n")
                f.write(json.dumps(result, indent=2))
                f.write("\n\n---- Edited vs Adversarial Segmentation mIoU ----\n")
                f.write(json.dumps(result_edit, indent=2))
                f.write("\n")

        def _avg(summary, key):
            values = [summary[f][key] for f in summary]
            return float(np.mean(values)) if values else 0.0

        avg_opt = _avg(results_summary, "optimistic_iou")
        avg_pes = _avg(results_summary, "pessimistic_iou")
        avg_opt_edit = _avg(results_summary_edit, "optimistic_iou")
        avg_pes_edit = _avg(results_summary_edit, "pessimistic_iou")

        summary_path = os.path.join(root_dir, "global_summary.txt")
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n\n=== Segmentation mIoU Evaluation Summary ===\n")
            f.write("---- Original vs Immunized ----\n")
            f.write(f"Optimistic mIoU: {avg_opt}\n")
            f.write(f"Pessimistic mIoU: {avg_pes}\n")
            f.write("---- Edited vs Adversarial ----\n")
            f.write(f"Optimistic mIoU: {avg_opt_edit}\n")
            f.write(f"Pessimistic mIoU: {avg_pes_edit}\n")

        return {
            "avg_optimistic_iou": avg_opt,
            "avg_pessimistic_iou": avg_pes,
            "avg_optimistic_iou_edit": avg_opt_edit,
            "avg_pessimistic_iou_edit": avg_pes_edit,
            "details": results_summary,
            "details_edit": results_summary_edit,
        }