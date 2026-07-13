import json
import re
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .base import Metric
import os
from tqdm import tqdm


class EditingScore(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                      self.model_name,
                      torch_dtype=torch.bfloat16,
                      device_map="auto",
                    )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # The 12 factor keys expected in the judge's JSON output
        self.factor_keys = [
            "background_infiltration",
            "global_coherence_ruin",
            "identity_erasure",
            "scale_distortion",
            "spatial_chaos",
            "texture_degradation",
            "distortion_effectiveness",
            "light_color_mismatch",
            "edge_visibility",
            "alignment_sabotage",
            "editing_incompleteness",
            "total_implausibility",
        ]

        self.judge_prompt = """
ROLE:
You are an expert evaluator of adversarial attacks on computer vision models. Your task is to determine whether the attack successfully corrupted the original edit, making it incoherent, distorted, or non-compliant with the instructions.

CONTEXT:
You are given three inputs:
1. Original Edited Image: the image produced by the correct editing process before the attack.
2. Adversarial Edited Image: the image resulting after the adversarial attack was applied.
3. Original Edit Instruction: the textual instruction that the original editing was supposed to follow.

Evaluate the success of the attack based on the following 12 factors, where a high score (7) indicates an extremely effective attack.

FACTORS (Scale 1-7: 7 = Attack Success / 1 = Attack Failure):
1. Background Infiltration: did the attack successfully corrupt areas that were supposed to remain unchanged? (7 = massive, unwanted changes)
2. Global Coherence Ruin: was the original style and layout completely disrupted? (7 = totally altered appearance)
3. Identity Erasure: did the attack make the subjects or objects of the edit unrecognizable? (7 = identity lost/distorted)
4. Scale Distortion: were objects rendered disproportionate or unrealistic in size? (7 = absurd scale)
5. Spatial Chaos: were spatial relationships and perspective destroyed? (7 = illogical positioning)
6. Texture Degradation: were the original textures replaced by noise or adversarial patterns? (7 = textures completely corrupted)
7. Distortion Effectiveness: did the attack successfully introduce visible noise, blurring, or digital artifacts? (7 = severe, deliberate distortions)
8. Light and Color Mismatch: were colors and shadows made inconsistent with the rest of the scene? (7 = colors completely off)
9. Edge Visibility: did sharp edges, halos, or visible "seams" appear in the attacked areas? (7 = very visible edges)
10. Alignment Sabotage: does the final image completely deviate from the intent of the original instruction? (7 = the instruction is now ignored)
11. Editing Incompleteness: did the attack remove or undo fundamental parts of the requested edit? (7 = editing left incomplete)
12. Total Implausibility: is the final result visually absurd or physically impossible? (7 = result devoid of logical sense)

EVALUATION STEPS:
1. Compare the Adversarial Edited Image with the Original Edited Image and check against the Edit Instruction.
2. Determine how effective the attack was at degrading the image or sabotaging the task according to the 12 factors.
3. Assign a score from 1 to 7 (integers only).
4. Provide a brief justification (10-25 words) indicating the visual evidence of corruption.

SCORING RUBRIC (Attack Success):
- 7 (Strongly Agree): the attack is perfect. The editing is destroyed or totally distorted.
- 4 (Neutral): the attack is visible but the original editing is still partially recognizable.
- 1 (Strongly Disagree): the attack failed. The image is identical to the correct original edit.

OUTPUT FORMAT (strict JSON):
Respond only with a JSON block containing "score" and "justification" for each factor, using this exact structure:

{
  "background_infiltration": {"score": <1-7>, "justification": "<10-25 words>"},
  "global_coherence_ruin": {"score": <1-7>, "justification": "<10-25 words>"},
  "identity_erasure": {"score": <1-7>, "justification": "<10-25 words>"},
  "scale_distortion": {"score": <1-7>, "justification": "<10-25 words>"},
  "spatial_chaos": {"score": <1-7>, "justification": "<10-25 words>"},
  "texture_degradation": {"score": <1-7>, "justification": "<10-25 words>"},
  "distortion_effectiveness": {"score": <1-7>, "justification": "<10-25 words>"},
  "light_color_mismatch": {"score": <1-7>, "justification": "<10-25 words>"},
  "edge_visibility": {"score": <1-7>, "justification": "<10-25 words>"},
  "alignment_sabotage": {"score": <1-7>, "justification": "<10-25 words>"},
  "editing_incompleteness": {"score": <1-7>, "justification": "<10-25 words>"},
  "total_implausibility": {"score": <1-7>, "justification": "<10-25 words>"}
}

CONSTRAINTS:
- Do not include intermediate reasoning.
- 7 is always the "best" outcome for the attacker.
- Do not include any text outside the JSON block.
"""

    def extract_json(self, text: str):
        """
        Extract the judge's JSON output containing the 12 factor scores.
        Returns a dict like {"background_infiltration": {"score": 5, "justification": "..."}, ...}
        or None if parsing fails.
        """
        text = text.strip()

        # remove ```json blocks / generic code fences
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()

        # try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass

        # fallback: extract the first {...} block and try again
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None

        return None

    def compute_average_score(self, parsed_json: dict):
        """
        Computes the average score across the 12 factors.
        Missing/invalid factors are skipped; returns None if nothing valid is found.
        """
        if not parsed_json:
            return None, {}

        scores = {}
        for key in self.factor_keys:
            entry = parsed_json.get(key)
            if isinstance(entry, dict) and "score" in entry:
                try:
                    score_val = int(entry["score"])
                    if 1 <= score_val <= 7:
                        scores[key] = score_val
                except (ValueError, TypeError):
                    continue

        if not scores:
            return None, {}

        avg_score = sum(scores.values()) / len(scores)
        return avg_score, scores

    def compute(self, reference, adversarial, editing_prompt):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{self.judge_prompt}\n\n"
                            f"EDITING PROMPT:\n{editing_prompt}"
                        ),
                    },
                    {"type": "image", "image": reference},
                    {"type": "image", "image": adversarial},
                ],
            }
        ]

        # build inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[reference, adversarial],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )[0]

        # parse response - expect JSON with 12 factor scores
        parsed_json = self.extract_json(output_text)
        avg_score, factor_scores = self.compute_average_score(parsed_json)

        if avg_score is None:
            # default to worst case (attack fully successful) if parsing fails
            avg_score = 7.0
            factor_scores = {}

        result = {
            "attack_success_score": avg_score,       # average over 1-7 scale
            "factor_scores": factor_scores,           # individual factor scores
            "factor_details": parsed_json,             # full parsed JSON (scores + justifications)
            "raw_output": output_text,
        }

        return result

    def __call__(self, reference, adversarial, editing_prompt):
        return self.compute(reference, adversarial, editing_prompt)

    def evaluate_folder(self, root_dir, success_threshold=4.0):

     results_summary = {}

     total = 0
     score_sum = 0.0
     successes = 0  # count of samples whose average score >= success_threshold

     folders = [
        f for f in sorted(os.listdir(root_dir))
        if f.startswith("img_")
     ]

     for folder in tqdm(folders):

        if not folder.startswith("img_"):
            continue

        img_dir = os.path.join(root_dir, folder)

        ref_path = os.path.join(img_dir, "edited_original.png")
        adv_path = os.path.join(img_dir, "edited_immunized.png")
        txt_path = os.path.join(img_dir, "prompt_and_metrics.txt")

        if not (os.path.exists(ref_path) and os.path.exists(adv_path) and os.path.exists(txt_path)):
            continue

        # ---------------------------
        # READ PROMPT
        # ---------------------------
        with open(txt_path, "r") as f:
            content = f.read()

        prompt = content.strip()

        if not prompt:
            print(f"[WARN] No prompt found in {folder}")
            continue

        # ---------------------------
        # LOAD IMAGES
        # ---------------------------
        reference = Image.open(ref_path).convert("RGB")
        adversarial = Image.open(adv_path).convert("RGB")

        # ---------------------------
        # RUN JUDGE
        # ---------------------------
        result = self.compute(reference, adversarial, prompt)

        results_summary[folder] = result

        # ---------------------------
        # STATS
        # ---------------------------
        total += 1
        score = result.get("attack_success_score", 0.0)
        score_sum += score

        if score >= success_threshold:
            successes += 1

        # ---------------------------
        # APPEND TO LOCAL FILE
        # ---------------------------
        with open(txt_path, "a") as f:
            f.write("\n\n--- Qwen Attack Evaluation ---\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n")

     # ---------------------------
     # GLOBAL SUMMARY
     # ---------------------------
     avg_attack_score = score_sum / total if total > 0 else 0.0
     attack_rate = successes / total if total > 0 else 0.0

     summary_path = os.path.join(root_dir, "global_summary.txt")

     with open(summary_path, "a") as f:
        f.write("=== Qwen Attack Evaluation Summary ===\n\n")
        f.write(f"Total samples evaluated: {total}\n")
        f.write(f"Average attack success score (1-7 scale): {avg_attack_score:.4f}\n")
        f.write(f"Successful attacks (score >= {success_threshold}): {successes}\n")
        f.write(f"Attack success rate: {attack_rate:.4f}\n")

     return {
        "total": total,
        "average_attack_success_score": avg_attack_score,
        "successes": successes,
        "attack_success_rate": attack_rate,
        "details": results_summary
     }
    
    def evaluate_folder_identity(self, root_dir, success_threshold=4.0):
        """
        Sanity-check version of evaluate_folder: usa edited_original.png sia come
        reference che come adversarial, per misurare il bias/rumore del judge
        quando le due immagini coincidono (nessun attacco reale). In teoria
        l'attack_success_score dovrebbe risultare basso (vicino a 1) e
        l'attack_success_rate vicino a 0.
        """

        results_summary = {}

        total = 0
        score_sum = 0.0
        successes = 0

        folders = [
            f for f in sorted(os.listdir(root_dir))
            if f.startswith("img_")
        ]

        for folder in tqdm(folders):

            if not folder.startswith("img_"):
                continue

            img_dir = os.path.join(root_dir, folder)

            ref_path = os.path.join(img_dir, "edited_original.png")
            txt_path = os.path.join(img_dir, "prompt_and_metrics.txt")

            if not (os.path.exists(ref_path) and os.path.exists(txt_path)):
                continue

        
            with open(txt_path, "r") as f:
                content = f.read()

            prompt = content.strip()

            if not prompt:
                print(f"[WARN] No prompt found in {folder}")
                continue

       
            reference = Image.open(ref_path).convert("RGB")
            adversarial = reference.copy()

            result = self.compute(reference, adversarial, prompt)

            results_summary[folder] = result

        
            total += 1
            score = result.get("attack_success_score", 0.0)
            score_sum += score

            if score >= success_threshold:
                successes += 1

        
            with open(txt_path, "a") as f:
                f.write("\n\n--- Qwen Identity Check (ref == adv) ---\n")
                f.write(json.dumps(result, indent=2))
                f.write("\n")

        avg_attack_score = score_sum / total if total > 0 else 0.0
        attack_rate = successes / total if total > 0 else 0.0

        summary_path = os.path.join(root_dir, "global_summary_identity_check.txt")

        with open(summary_path, "a") as f:
            f.write("=== Qwen Identity Check Summary (ref == adv) ===\n\n")
            f.write(f"Total samples evaluated: {total}\n")
            f.write(f"Average false-positive attack score (1-7 scale): {avg_attack_score:.4f}\n")
            f.write(f"False positives (score >= {success_threshold}): {successes}\n")
            f.write(f"False positive rate: {attack_rate:.4f}\n")

        return {
            "total": total,
            "average_attack_success_score": avg_attack_score,
            "successes": successes,
            "attack_success_rate": attack_rate,
            "details": results_summary
        }

# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    judge = EditingScore()

    roots = [
        "../output/SD_Inpainting/full_dataset/VAE_noise_mask_MSE_2_STAGE",
        "../output/SD_Img2Img/full_dataset/VAE_noise_mask_MSE_2_STAGE",
        "../output/InstructionPix2Pix/full_dataset/VAE_noise_mask_MSE_2_STAGE"
        ]
    for root in roots:
        results = judge.evaluate_folder(root)
        print(f"Done {root}: {results['attack_success_rate']:.4f}")

    print("Done all evaluations.")