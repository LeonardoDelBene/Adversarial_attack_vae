import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from metrics.base import Metric
import numpy as np
import re


class CaptionSimilarity (Metric):
    """
    Implementazione della metrica Caption Similarity dal paper:
    'Immunizing Images from Text to Image Editing via Adversarial Cross-Attention'

    CaptionSimilarity = cosine_similarity(Φ(t1), Φ(t2))

    dove t1, t2 sono le caption generate da LLaVA 1.5 e
    Φ è il sentence encoder SentenceBERT (all-MiniLM-L6-v2).
    """

    CAPTION_PROMPT = "USER: <image>\nGive me a caption for this image.\nASSISTANT:"
    LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    SBERT_MODEL_ID = "all-MiniLM-L6-v2"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        device      : 'cuda', 'cpu' o None (auto-detect).
        load_in_4bit: se True carica LLaVA quantizzato a 4-bit (risparmia VRAM).
        """
        super().__init__(*args, **kwargs)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._load_sbert()
        self._load_llava(kwargs["load_in_4bit"])

    # ── Caricamento modelli ──────────────────────────────────────────────────

    def _load_sbert(self) -> None:
        self.sbert = SentenceTransformer(self.SBERT_MODEL_ID, device=self.device)

    def _load_llava(self, load_in_4bit: bool) -> None:
        self.processor = LlavaProcessor.from_pretrained(self.LLAVA_MODEL_ID, use_fast= False,)

        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda:0" else torch.float32
        }

        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = {"": "cuda:0"}

        self.llava = LlavaForConditionalGeneration.from_pretrained(
            self.LLAVA_MODEL_ID,
            **model_kwargs
        )

        if not load_in_4bit:
            self.llava = self.llava.to(self.device)
    # ── Captioning ───────────────────────────────────────────────────────────

    def get_caption(self, image: Image.Image) -> str:
        """Genera una caption testuale a partire da un'immagine PIL."""
        inputs = self.processor(
            text=self.CAPTION_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output_ids = self.llava.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache = True
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    # ── Embedding ────────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> torch.Tensor:
        """Restituisce l'embedding SentenceBERT (384-dim) di un testo."""
        return self.sbert.encode(text, convert_to_tensor=True)

    # ── Metrica principale ───────────────────────────────────────────────────

    def compute(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> dict:
        """
        Calcola la Caption Similarity tra due immagini (Eq. 3 del paper).

        Parameters
        ----------
        image1 : immagine originale (PIL).
        image2 : immagine editata / avversariale (PIL).

        Returns
        -------
        dict con:
            - caption_1          : caption di image1
            - caption_2          : caption di image2
            - caption_similarity : valore cosine similarity in [-1, 1]
        """
        cap1 = self.get_caption(image1)
        cap2 = self.get_caption(image2)

        emb1, emb2 = self.sbert.encode(
            [cap1, cap2],
            convert_to_tensor=True
        )

        score = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

        return {
            "caption_1": cap1,
            "caption_2": cap2,
            "caption_similarity": score,
        }

    def __call__(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> dict:
        """Alias di compute(), permette di usare l'istanza come funzione."""
        return self.compute(image1, image2)

class AccuracyRate(CaptionSimilarity):
    """
    Metrica Accuracy Rate basata su CaptionSimilarity.

    Riusa LLaVA (già caricato dal padre) sia per generare le caption
    che come giudice semantico tramite prompt strutturato.

    Flusso per ogni coppia:
        image1, image2
            → caption1, caption2   (LLaVA, ereditato da CaptionSimilarity)
            → llm_score            (LLaVA giudice + prompt strutturato)
            → success              (llm_score >= threshold)
    """

    JUDGE_PROMPT = """Rate the semantic similarity between the following two texts on a scale from 0 to 1.

**Criteria for similarity measurement:**
1. **Main Subject Consistency:** If both descriptions refer to the same key subject or object (e.g., a person, food, an event), they should receive a higher similarity score.
2. **Relevant Description**: If the descriptions are related to the same context or topic, they should also contribute to a higher similarity score.
3. **Ignore Fine-Grained Details:** Do not penalize differences in **phrasing, sentence structure, or minor variations in detail**. Focus on **whether both descriptions fundamentally describe the same thing.**
4. **Partial Matches:** If one description contains extra information but does not contradict the other, they should still have a high similarity score.
5. **Similarity Score Range:** 
    - **1.0**: Nearly identical in meaning.
    - **0.8-0.9**: Same subject, with highly related descriptions.
    - **0.7-0.8**: Same subject, core meaning aligned, even if some details differ.
    - **0.5-0.7**: Same subject but different perspectives or missing details.
    - **0.3-0.5**: Related but not highly similar (same general theme but different descriptions).
    - **0.0-0.2**: Completely different subjects or unrelated meanings.

Text 1: {text1}
Text 2: {text2}
Output only a single number between 0 and 1. Do not include any explanation or additional text."""

    def __init__(self, *args, threshold: float = 0.7, **kwargs):
        """
        Parameters
        ----------
        threshold    : soglia oltre la quale si considera successo. Default = 0.7.
        load_in_4bit : ereditato da CaptionSimilarity.
        """
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    # ── LLM judge (usa LLaVA senza immagine) ────────────────────────────────

    def _judge_similarity(self, text1: str, text2: str) -> float:
        """
        Usa LLaVA già caricato come giudice testuale puro (senza immagine).

        Returns
        -------
        float in [0, 1]
        """
        prompt = f"USER:\n{self.JUDGE_PROMPT.format(text1=text1, text2=text2)}\nASSISTANT:"

        # Nessuna immagine: passiamo solo il testo al processor
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output_ids = self.llava.generate(
                **inputs,
                max_new_tokens=8,   # serve solo un numero
                do_sample=False,
                use_cache=True,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.processor.decode(generated, skip_special_tokens=True).strip()

        match = re.search(r"\d+(?:\.\d+)?", raw)
        if match is None:
            raise ValueError(f"LLaVA non ha restituito un numero valido: {raw!r}")

        return float(np.clip(float(match.group()), 0.0, 1.0))

    # ── Metrica principale ───────────────────────────────────────────────────

    def compute(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> dict:
        """
        Calcola l'Accuracy Rate tra due immagini.

        Returns
        -------
        dict con:
            - caption_1      : caption di image1
            - caption_2      : caption di image2
            - llm_score      : punteggio in [0, 1] assegnato da LLaVA
            - threshold      : soglia usata
            - success        : True se llm_score >= threshold
            - accuracy_rate  : 1.0 o 0.0 per una singola coppia
        """
        cap1 = self.get_caption(image1)
        cap2 = self.get_caption(image2)

        llm_score = self._judge_similarity(cap1, cap2)
        success = llm_score < self.threshold  # successo = editing fallito sull'immunizzata

        return {
            "caption_1": cap1,
            "caption_2": cap2,
            "llm_score": llm_score,
            "threshold": self.threshold,
            "success": success,
            "accuracy_rate": float(success),
        }


    def __call__(
        self,
        image1: Image.Image | None = None,
        image2: Image.Image | None = None,
    ) -> dict:
        """
        Interfaccia unificata:
          - compute()         se vengono passate image1 e image2
          - compute_dataset() se viene passata una lista di pairs
        """
        if image1 is not None and image2 is not None:
            return self.compute(image1, image2)
        raise ValueError("Passa (image1, image2)")

from pathlib import Path
from utils import load_image_from_path


def compute_caption_similarity_on_saved(base_output_dir: str, load_in_4bit: bool = True):
    """
    Calcola la caption similarity su tutte le cartelle img_* già salvate
    e aggiunge i risultati al relativo prompt_and_metrics.txt.
    Alla fine scrive un summary globale.
    """

    base = Path(base_output_dir)
    img_dirs = sorted(base.glob("img_*"), key=lambda p: int(p.name.split("_")[1]))

    if not img_dirs:
        print(f"Nessuna cartella img_* trovata in {base}")
        return

    print(f"Trovate {len(img_dirs)} cartelle. Carico CaptionSimilarity...")
    caption_similarity = CaptionSimilarity(load_in_4bit=load_in_4bit)

    all_scores = []

    for img_dir in img_dirs:
        orig_path  = img_dir / "edited_original.png"
        adv_path   = img_dir / "edited_immunized.png"
        metrics_file = img_dir / "prompt_and_metrics.txt"

        # Salta cartelle incomplete
        if not orig_path.exists() or not adv_path.exists():
            print(f"[SKIP] {img_dir.name}: immagini mancanti")
            continue

        print(f"Processing {img_dir.name}...")

        try:
            img_orig = load_image_from_path(orig_path)
            img_edit = load_image_from_path(adv_path)

            score = caption_similarity(img_orig, img_edit)

            caption_1         = score["caption_1"]
            caption_2         = score["caption_2"]
            similarity_value  = score["caption_similarity"]

            print(f"  Caption orig     : {caption_1}")
            print(f"  Caption immunized: {caption_2}")
            print(f"  Similarity       : {similarity_value:.4f}")

            # Appendi al file di metriche esistente
            with open(metrics_file, "a") as f:
                f.write("\nCaption Similarity (Edited Original vs Edited Immunized)\n")
                f.write(f"Caption original : {caption_1}\n")
                f.write(f"Caption immunized: {caption_2}\n")
                f.write(f"Caption similarity: {similarity_value:.4f}\n")

            all_scores.append({
                "sample":     img_dir.name,
                "caption_1":  caption_1,
                "caption_2":  caption_2,
                "similarity": similarity_value,
            })

        except Exception as e:
            print(f"[ERROR] {img_dir.name}: {e}")
            continue

    # --- Summary globale ---
    if not all_scores:
        print("Nessun risultato da aggregare.")
        return

    avg_similarity = sum(s["similarity"] for s in all_scores) / len(all_scores)

    summary_path = base / "global_summary.txt"

    # Aggiunge la sezione caption similarity al summary globale esistente
    with open(summary_path, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Caption Similarity — per sample\n\n")
        for s in all_scores:
            f.write(f"[{s['sample']}]\n")
            f.write(f"  Caption orig     : {s['caption_1']}\n")
            f.write(f"  Caption immunized: {s['caption_2']}\n")
            f.write(f"  Similarity       : {s['similarity']:.4f}\n\n")

        f.write(f"Average Caption Similarity: {avg_similarity:.4f}\n")

    print(f"\n{'='*50}")
    print(f"CAPTION SIMILARITY — {len(all_scores)} samples")
    print(f"Average: {avg_similarity:.4f}")
    print(f"Summary aggiornato in {summary_path}")

def compute_caption_similarity_between_2_image(img_dir):
    caption_similarity = CaptionSimilarity(load_in_4bit=True)

    base = Path("C:/Users/leona/PycharmProjects/Immunization/output") / img_dir

    img_orig = load_image_from_path(
        base / "edited_original.png"
    )

    img_edit = load_image_from_path(
        base / "edited_immunized.png"
    )
    score = caption_similarity(img_orig, img_edit)
    print(score)
    file = base / "prompt_and_metrics.txt"
    with open(file, "a") as f:
        f.write("\nCaption similarity (Edited Original vs Edited Immunized)\n")
        f.write(f"Caption original: {score['caption_1']}\n")
        f.write(f"Caption immunized: {score['caption_2']}\n")
        f.write(f"Caption similarity: {score['caption_similarity']}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    compute_caption_similarity_on_saved(
        base_output_dir="C:/Users/leona/PycharmProjects/Immunization/output/SD_Inpainting/img_full_dataset",
        load_in_4bit=True
    )
    #compute_caption_similarity_between_2_image("InstructionPix2Pix/img_0")

