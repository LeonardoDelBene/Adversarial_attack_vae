# masked_editing_score.py

import torch
import torch.nn.functional as F
import numpy as np
import lpips
from PIL import Image
from .base import Metric
import os
from tqdm import tqdm


class LipisScore(Metric):

    def __init__(self, *args, lpips_net: str = "alex", **kwargs):
        super().__init__(*args, **kwargs)
        self.device   = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.device)
        self.lpips_fn.eval()

    # ── Conversioni ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(image: Image.Image) -> np.ndarray:
        """PIL → numpy uint8 (H, W, 3)."""
        return np.array(image.convert("RGB"))

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        """PIL → tensor [-1, 1] (1, 3, H, W)."""
        arr = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def _prepare_mask(mask: Image.Image | np.ndarray, size: tuple) -> np.ndarray:
        if isinstance(mask, Image.Image):
            # gestisce sia L che RGB (nel tuo pipeline arriva RGB)
            mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        return (mask > 0.5).astype(np.float32)

    # ── LPIPS mascherato ─────────────────────────────────────────────────────

    def _background_change_lpips(
            self,
            img1: Image.Image,
            img2: Image.Image,
            mask: np.ndarray,  # (H, W) binario float {0,1}: 1=background
    ) -> float:
        """
        LPIPS solo sul background (mask==1).

        Alto  → background molto diverso tra orig e adv (immunizzazione riuscita)
        Basso → background simile (immunizzazione fallita)
        """
        t1 = self._to_tensor(img1).to(self.device)
        t2 = self._to_tensor(img2).to(self.device)

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        t1_bg = t1 * mask_t
        t2_bg = t2 * mask_t

        with torch.no_grad():
            score = self.lpips_fn(t1_bg, t2_bg).item()

        # normalizza per la proporzione di background
        bg_ratio = mask.mean()
        return score / (bg_ratio + 1e-8) if bg_ratio > 0 else 0.0

    # ── Metrica principale ───────────────────────────────────────────────────
    def compute(
        self,
        image_orig: Image.Image,
        image_adv: Image.Image,
        mask: Image.Image | np.ndarray,
    ) -> dict:

     H, W = np.array(image_orig).shape[:2]
     mask_np = self._prepare_mask(mask, size=(H, W))

     # ── Soggetto ────────────────────────────────────────────────────────────
     subject_mask  = 1.0 - mask_np
     subject_lpips = self._background_change_lpips(image_orig, image_adv, subject_mask)

     # ── LPIPS globale ────────────────────────────────────────────────────────
     t_orig   = self._to_tensor(image_orig).to(self.device)
     t_edited = self._to_tensor(image_adv).to(self.device)
     with torch.no_grad():
        global_lpips = self.lpips_fn(t_orig, t_edited).item()

     return {
        "subject_lpips": subject_lpips,
        "global_lpips":  global_lpips
     }

    def __call__(
        self,
        image_orig:   Image.Image,
        image_edited: Image.Image,
        mask:         Image.Image | np.ndarray,
    ) -> dict:
        return self.compute(image_orig, image_edited, mask)
    def evaluate_folder(self, root_dir):
        result_summary = {}  # fix 2
        result_summary_edit={}

        folders = [f for f in sorted(os.listdir(root_dir)) if f.startswith("img_")]
        for folder in tqdm(folders):
            img_dir = os.path.join(root_dir, folder)

            img_orig = os.path.join(img_dir, "original_image.png")
            img_adv = os.path.join(img_dir, "immunized_image.png")
            edit_orig = os.path.join(img_dir, "edited_original.png")
            edit_adv = os.path.join(img_dir, "edited_immunized.png")
            mask = os.path.join(img_dir, "mask.png")
            txt_path = os.path.join(img_dir, "prompt_and_metrics.txt")

            img_orig = Image.open(img_orig).convert("RGB")
            img_adv = Image.open(img_adv).convert("RGB")
            edit_orig = Image.open(edit_orig).convert("RGB")
            edit_adv = Image.open(edit_adv).convert("RGB")
            mask = Image.open(mask)  # fix 1

            result = self.compute(img_orig, img_adv, mask)
            result_summary[folder] = result  # ora funziona con dict

            result_edit = self.compute(edit_orig, edit_adv, mask)
            result_summary_edit[folder] = result_edit

            with open(txt_path, "a") as f:
                f.write("\n\n ----- Original vs Immunized LPIPS---- \n")
                f.write(f"Subject LPIPS : {result['subject_lpips']}\n")  # fix 3
                f.write(f"Global LPIPS : {result['global_lpips']}\n")
                f.write("\n\n ----- Edited vs Adversarial LPIPS---- \n")
                f.write(f"Subject LPIPS : {result_edit['subject_lpips']}\n")  # fix 3
                f.write(f"Global LPIPS : {result_edit['global_lpips']}\n")

        subject, gl = 0, 0
        for folder in result_summary:  # fix 4
            subject = subject + result_summary[folder]['subject_lpips']
            gl = gl + result_summary[folder]['global_lpips']
        subject = subject / len(result_summary)
        gl = gl / len(result_summary)

        subject_edit, gl_edit = 0, 0
        for folder in result_summary_edit:
            subject_edit += result_summary_edit[folder]['subject_lpips']
            gl_edit += result_summary_edit[folder]['global_lpips']
        subject_edit = subject_edit / len(result_summary_edit)
        gl_edit = gl_edit / len(result_summary_edit)

        summary_path = os.path.join(root_dir, "global_summary.txt")
        with open(summary_path, "a") as f:
            f.write("\n\n---- Original vs Immunized LPIPS ----\n")
            f.write(f"Subject LPIPS: {subject}\n")
            f.write(f"Global LPIPS: {gl}\n")
            f.write("\n\n---- Edited vs Adversarial LPIPS ----\n")
            f.write(f"Subject LPIPS: {subject_edit}\n")
            f.write(f"Global LPIPS: {gl_edit}\n")


if __name__ == "__main__":
    judge = LipisScore()

  
    roots = [
        "../output/SD_Inpainting/full_dataset/VAE_noise_mask_MSE_2_STAGE",
        "../output/SD_Img2Img/full_dataset/VAE_noise_mask_MSE_2_STAGE",
        "../output/InstructionPix2Pix/full_dataset/VAE_noise_mask_MSE_2_STAGE"
        ]
    for root in roots:
        results = judge.evaluate_folder(root)
        print(f"Done {root}")

    print("Done all evaluations.")