"""
robustness.py

Valutazione della robustezza della pipeline di immunizzazione a
compressione JPEG, crop e blur.

Idea: applicare la trasformazione all'immagine immunizzata *già
generata e salvata su disco* prima di darla in pasto al modello di
editing target, poi confrontare gli output con le metriche esistenti
(PSNR, SSIM, FSIM, masked LPIPS, Qwen judge) rispetto all'editing pulito.

Per ciascuna trasformazione/intensità viene ora eseguito anche un
secondo branch parallelo sull'immagine ORIGINALE (non immunizzata):
la trasformazione viene applicata a `original_image.png`, il risultato
viene editato dal modello target e i risultati (immagine trasformata,
immagine editata, metriche) vengono salvati separatamente. Questo
branch serve come baseline per capire quanto degrado sia dovuto alla
trasformazione + re-editing di per sé, indipendentemente
dall'immunizzazione. I due branch sono distinti nei record tramite il
campo "source" ("immunized" oppure "original").

Struttura attesa su disco (per ciascuna root_dir):

    root_dir/
        img_1/
            original_image.png       # immagine originale (non editata)
            immunized_image.png      # immagine immunizzata
            edited_original.png      # immagine originale editata dal modello target (target pulito)
            edited_immunized.png     # immagine immunizzata editata dal modello target
            mask.png                 # maschera di inpainting/editing
            prompt_and_metrics.txt   # prompt e log usati per l'editing
        img_2/
            ...

Ora è possibile passare più root_dir: vengono elaborate in serie, una
alla volta, ciascuna con la propria sotto-cartella di output sotto
`output_dir/<tipo_modello>/<nome_root_dir>` (il tipo di modello è
dedotto dal path della root_dir, vedi MODEL_TYPE_KEYWORDS). È inoltre
possibile passare a `evaluate_robustness` un modello di editing diverso
per ciascun tipo (SD_Inpainting / SD_Img2Img / InstructPix2Pix),
selezionato automaticamente in base al tipo dedotto dalla root_dir.
I record di tutte le root_dir vengono comunque raccolti e restituiti
in un'unica lista da `evaluate_robustness`, ma checkpoint e summary
vengono salvati solo per singola root_dir (nessun file aggregato).
"""

import io
import os
os.environ["HF_HOME"] = "/equilibrium/ldelbene/cache/hf"
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter

from model import Attack, AttackInstructPix2Pix, AttackSD
from metrics.editing_score import EditingScore
from metrics.segmentation_score import SegmentationScore

from collections import defaultdict


ImageOrTensor = Union[Image.Image, torch.Tensor]


# ---------------------------------------------------------------------------
# Trasformazioni "purificanti"
# ---------------------------------------------------------------------------

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def center_crop_resize(img: Image.Image, frac: float) -> Image.Image:
    assert 0 < frac <= 1.0
    w, h = img.size
    new_w, new_h = int(w * frac ** 0.5), int(h * frac ** 0.5)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


def random_crop_resize(img: Image.Image, frac: float, rng: Optional[np.random.Generator] = None) -> Image.Image:
    assert 0 < frac <= 1.0
    rng = rng or np.random.default_rng()
    w, h = img.size
    new_w, new_h = int(w * frac ** 0.5), int(h * frac ** 0.5)
    left = int(rng.integers(0, max(w - new_w, 1)))
    top = int(rng.integers(0, max(h - new_h, 1)))
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def identity(img: Image.Image, intensity: Any = None, *args, **kwargs) -> Image.Image:
    """Trasformazione identità per il caso baseline 'clean'."""
    return img


TRANSFORMS: dict[str, tuple[Callable[..., Image.Image], list]] = {
    "clean": (identity, [None]),
    "jpeg": (jpeg_compress, [85, 60, 30]),
    "crop_center": (center_crop_resize, [0.9, 0.75, 0.5]),
    "crop_random": (random_crop_resize, [0.9, 0.75, 0.5]),
    "blur": (gaussian_blur, [0.5, 1.0, 2.0, 4.0]),
}


# ---------------------------------------------------------------------------
# Conversioni tensor <-> PIL
# ---------------------------------------------------------------------------

def to_pil(tensor: torch.Tensor, value_range: Tuple[float, float] = (0.0, 1.0)) -> Image.Image:
    lo, hi = value_range
    x = (tensor.detach().cpu().clamp(lo, hi) - lo) / (hi - lo)
    x = (x * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def from_pil(img: Image.Image, value_range: Tuple[float, float] = (0.0, 1.0), device: str = "cpu") -> torch.Tensor:
    lo, hi = value_range
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x * (hi - lo) + lo
    return x.permute(2, 0, 1).to(device)


def _ensure_pil(img: Optional[ImageOrTensor]) -> Optional[Image.Image]:
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        return to_pil(img)
    return img


def _normalize_edit_output(output: Any) -> Any:
    if isinstance(output, (list, tuple)) and len(output) > 0:
        return output[0]
    return output


def _extract_prompt_from_text(content: str) -> str:
    lines = [line.rstrip() for line in content.splitlines()]
    clean_lines = []
    for line in lines:
        if not line.strip():
            if clean_lines:
                clean_lines.append("")
            continue

        low = line.strip().lower()
        if low.startswith("---") or low.startswith("==="):
            break
        if "qwen" in low or "attack evaluation" in low:
            break
        if low.startswith("{") or low.startswith("["):
            break
        if any(keyword in low for keyword in ["subject lpips", "global lpips", "psnr", "ssim", "fsim", "miou", "iou"]):
            break
        clean_lines.append(line)

    prompt = "\n".join(clean_lines).strip()
    if prompt:
        return prompt

    for line in lines:
        if line.strip():
            return line.strip()

    return ""


def _resolve_edit_callable(edit_target: Any) -> Callable[[ImageOrTensor, Optional[ImageOrTensor], str], Any]:
    if hasattr(edit_target, "edit_image") and callable(getattr(edit_target, "edit_image")):
        method = edit_target.edit_image
    elif callable(edit_target):
        method = edit_target
    else:
        raise TypeError(
            f"edit_target must be callable or expose an edit_image method, got {type(edit_target).__name__}"
        )

    signature = inspect.signature(method)
    params = [p.name for p in signature.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if params and params[0] in ("self", "cls"):
        params = params[1:]

    def edit_fn(image: ImageOrTensor, mask: Optional[ImageOrTensor], prompt: str, **kwargs) -> Any:
        image = _ensure_pil(image)
        mask = _ensure_pil(mask)

        if len(params) >= 3 and params[0] in ("prompt", "text"):
            result = method(prompt, image, mask, **kwargs)
        elif len(params) >= 2 and params[0] in ("image", "img") and params[1] in ("prompt", "text"):
            result = method(image, prompt, mask, **kwargs)
        else:
            try:
                result = method(prompt, image, mask, **kwargs)
            except TypeError:
                result = method(image, prompt, mask, **kwargs)

        return _normalize_edit_output(result)

    return edit_fn


def _resolve_metrics_callable(metrics_target: Any) -> Callable[[Any, Any, Optional[Any], Optional[str]], dict]:
    if hasattr(metrics_target, "compute") and callable(getattr(metrics_target, "compute")):
        compute_method = metrics_target.compute

        def metrics_fn(adversarial: Any, reference: Any, mask: Optional[Any] = None, editing_prompt: Optional[str] = None) -> dict:
            if editing_prompt is None:
                raise ValueError("editing_prompt is required for EditingScore-style metrics")
            # compute(reference, adversarial, prompt) -> valuta adversarial (l'edit sotto test)
            # rispetto al reference (l'edit target pulito)
            return compute_method(reference, adversarial, editing_prompt)

        return metrics_fn

    if callable(metrics_target):
        def metrics_fn(adversarial: Any, reference: Any, mask: Optional[Any] = None, editing_prompt: Optional[str] = None) -> dict:
            try:
                return metrics_target(adversarial=adversarial, reference=reference, mask=mask, editing_prompt=editing_prompt)
            except TypeError:
                try:
                    return metrics_target(reference=reference, adversarial=adversarial, mask=mask)
                except TypeError:
                    return metrics_target(adversarial, reference, mask)

        return metrics_fn

    raise TypeError(
        f"metrics_target must be callable or expose a compute method, got {type(metrics_target).__name__}"
    )


# ---------------------------------------------------------------------------
# Caricamento sample da cartelle img_N/ già generate
# ---------------------------------------------------------------------------

EXPECTED_FILES = {
    "original": "original_image.png",
    "immunized": "immunized_image.png",
    "edited_original": "edited_original.png",
    "edited_immunized": "edited_immunized.png",
    "mask": "mask.png",
    "prompt": "prompt_and_metrics.txt",
}


def _get_required_file(folder: Path, filename: str) -> Path:
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"File '{filename}' non trovato in {folder}")
    return path


@dataclass
class ImgFolderSample:
    sample_id: str
    folder: Path
    original_pil: Image.Image
    immunized_pil: Image.Image
    edited_original_pil: Image.Image
    edited_immunized_pil: Image.Image
    mask_pil: Optional[Image.Image]
    prompt: str

    @classmethod
    def from_dir(cls, folder: Path) -> "ImgFolderSample":
        original_path = _get_required_file(folder, EXPECTED_FILES["original"])
        immunized_path = _get_required_file(folder, EXPECTED_FILES["immunized"])
        edited_original_path = _get_required_file(folder, EXPECTED_FILES["edited_original"])
        edited_immunized_path = _get_required_file(folder, EXPECTED_FILES["edited_immunized"])
        mask_path = _get_required_file(folder, EXPECTED_FILES["mask"])
        prompt_path = _get_required_file(folder, EXPECTED_FILES["prompt"])

        prompt_text = prompt_path.read_text(encoding="utf-8")
        prompt = _extract_prompt_from_text(prompt_text)

        return cls(
            sample_id=folder.name,
            folder=folder,
            original_pil=Image.open(original_path).convert("RGB"),
            immunized_pil=Image.open(immunized_path).convert("RGB"),
            edited_original_pil=Image.open(edited_original_path).convert("RGB"),
            edited_immunized_pil=Image.open(edited_immunized_path).convert("RGB"),
            mask_pil=Image.open(mask_path).convert("L"),
            prompt=prompt,
        )


def iter_img_folders(root_dir: Path, pattern: str = "img_*") -> list[Path]:
    folders = [p for p in root_dir.glob(pattern) if p.is_dir()]

    def _idx(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except ValueError:
            return 0

    return sorted(folders, key=_idx)


# ---------------------------------------------------------------------------
# Runner principale e Config
# ---------------------------------------------------------------------------

@dataclass
class RobustnessConfig:
    # Una o più root_dir da elaborare in serie. Si può passare sia un
    # singolo Path/str sia una lista/tupla di Path/str: viene sempre
    # normalizzato internamente in una lista di Path in __post_init__.
    root_dirs: Union[str, Path, Sequence[Union[str, Path]]]
    transforms: dict = field(default_factory=lambda: TRANSFORMS)
    value_range: Tuple[float, float] = (0.0, 1.0)
    n_samples: Optional[int] = None  # Se None, elabora l'intera directory per ciascuna root_dir
    seed: int = 2043
    output_dir: Path = Path("robustness_results")
    mask: Optional[torch.Tensor] = None
    save_images: bool = True
    # Se True (default), oltre al branch sull'immagine immunizzata viene
    # eseguito anche un branch sull'immagine originale: la trasformazione
    # viene applicata a original_image.png, il risultato viene editato e
    # i risultati (input trasformato, edit, metriche) vengono salvati
    # separatamente, con "source"="original" nei record.
    evaluate_original_branch: bool = True

    def __post_init__(self):
        if isinstance(self.root_dirs, (str, Path)):
            raw_dirs: list = [self.root_dirs]
        else:
            raw_dirs = list(self.root_dirs)

        if not raw_dirs:
            raise ValueError("root_dirs non può essere vuoto")

        self.root_dirs = [Path(d) for d in raw_dirs]
        for d in self.root_dirs:
            if not d.is_dir():
                raise NotADirectoryError(f"root_dir non trovata: {d}")

        self.output_dir = Path(self.output_dir)


# Mappa keyword (case-insensitive, cercata nel path della root_dir) -> nome
# "canonico" del tipo di modello. Questo nome viene usato sia come
# sotto-cartella di output sia come chiave per selezionare
# automaticamente il modello di editing corretto (vedi
# `evaluate_robustness` / EDIT_TARGET_LABELS). L'ordine conta: viene
# usata la prima keyword trovata, quindi le keyword più specifiche vanno
# prima.
MODEL_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("instructionpix2pix", "InstructPix2Pix"),
    ("instructpix2pix", "InstructPix2Pix"),
    ("img2img", "SD_Img2Img"),
    ("inpainting", "SD_Inpainting"),
]


def _infer_model_output_subdir(root_dir: Path) -> str:
    """Deduce il nome della sotto-cartella di output (es. 'SD_Inpainting')
    cercando le keyword note nel path della root_dir. Se nessuna keyword
    viene trovata, ritorna 'unknown_model'."""
    path_str = str(root_dir).lower()
    for keyword, label in MODEL_TYPE_KEYWORDS:
        if keyword in path_str:
            return label
    return "unknown_model"


def _process_variant(
    *,
    source_pil: Image.Image,
    source_label: str,
    t_name: str,
    t_fn: Callable,
    intensity: Any,
    rng: np.random.Generator,
    sample: "ImgFolderSample",
    edit_fn: Callable,
    metrics_fn: Callable,
    sample_out_dir: Path,
    root_dir: Path,
    config: "RobustnessConfig",
) -> Optional[dict]:
    """Applica la trasformazione (t_name, intensity) a `source_pil`, edita
    il risultato con `edit_fn`, salva le immagini su disco (se richiesto)
    e calcola le metriche rispetto a `sample.edited_original_pil`.
    Ritorna il record del risultato, oppure None se il calcolo delle
    metriche fallisce.

    `source_label` distingue il branch ("immunized" oppure "original") e
    viene usato sia per il campo "source" nel record sia per generare nomi
    di file distinti su disco.
    """
    # 1. Applicazione della trasformazione/degradazione
    if t_name == "crop_random":
        transformed_pil = t_fn(source_pil, intensity, rng)
    elif t_name == "clean":
        transformed_pil = t_fn(source_pil)
    else:
        transformed_pil = t_fn(source_pil, intensity)

    # 2. Re-editing dell'immagine trasformata mediante il modello target
    transformed_edit = edit_fn(transformed_pil, sample.mask_pil, sample.prompt)

    # 3. Salvataggio immagini trasformate ed editate su disco
    tag = f"{t_name}_{intensity}" if intensity is not None else t_name
    suffix = "" if source_label == "immunized" else f"_{source_label}"

    input_path = sample_out_dir / f"{tag}_input{suffix}.png"
    edited_path = sample_out_dir / f"{tag}_edited{suffix}.png"

    if config.save_images:
        transformed_pil.save(input_path)
        if isinstance(transformed_edit, Image.Image):
            transformed_edit.save(edited_path)

    # 4. Calcolo delle metriche di confronto (sempre rispetto all'edit
    #    "pulito" dell'originale, per entrambi i branch)
    try:
        metrics = metrics_fn(
            adversarial=transformed_edit,
            reference=sample.edited_original_pil,
            mask=sample.mask_pil,
            editing_prompt=sample.prompt,
        )
    except Exception as exc:
        print(f"[error] metriche per {sample.folder.name} ({source_label}) trasformazione {t_name} intensita {intensity}: {exc}")
        return None

    return {
        "root_dir": str(root_dir),
        "sample_id": sample.sample_id,
        "source": source_label,
        "transform": t_name,
        "intensity": intensity,
        "saved_input_path": str(input_path) if config.save_images else None,
        "saved_edited_path": str(edited_path) if config.save_images else None,
        **metrics,
    }


def _run_single_root(
    root_dir: Path,
    edit_fn: Callable,
    metrics_fn: Callable,
    config: RobustnessConfig,
    sample_out_base: Path,
) -> list[dict]:
    """Esegue la valutazione di robustezza su una singola root_dir e
    salva i risultati (checkpoint + summary) sotto sample_out_base."""

    rng = np.random.default_rng(config.seed)
    records: list[dict] = []

    folders = iter_img_folders(root_dir)
    if config.n_samples is not None:
        folders = folders[: config.n_samples]

    print(f"[{root_dir.name}] Inizio valutazione su {len(folders)} campioni...")

    for folder in folders:
        try:
            sample = ImgFolderSample.from_dir(folder)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue

        # Cartella di output per la specifica immagine (es. <output_dir>/<root_dir>/img_1)
        sample_out_dir = sample_out_base / sample.sample_id
        if config.save_images:
            sample_out_dir.mkdir(parents=True, exist_ok=True)

            # Salvataggio / Copia delle immagini baseline principali
            sample.original_pil.save(sample_out_dir / "original_image.png")
            sample.edited_original_pil.save(sample_out_dir / "edited_original.png")
            sample.immunized_pil.save(sample_out_dir / "immunized_image.png")
            sample.edited_immunized_pil.save(sample_out_dir / "edited_immunized.png")
            if sample.mask_pil is not None:
                sample.mask_pil.save(sample_out_dir / "mask.png")

        for t_name, (t_fn, intensities) in config.transforms.items():
            for intensity in intensities:
                # Branch principale: trasformazione + editing sull'immagine immunizzata
                record = _process_variant(
                    source_pil=sample.immunized_pil,
                    source_label="immunized",
                    t_name=t_name,
                    t_fn=t_fn,
                    intensity=intensity,
                    rng=rng,
                    sample=sample,
                    edit_fn=edit_fn,
                    metrics_fn=metrics_fn,
                    sample_out_dir=sample_out_dir,
                    root_dir=root_dir,
                    config=config,
                )
                if record is not None:
                    records.append(record)

                # Branch baseline: stessa trasformazione + editing sull'immagine originale
                if config.evaluate_original_branch:
                    record_orig = _process_variant(
                        source_pil=sample.original_pil,
                        source_label="original",
                        t_name=t_name,
                        t_fn=t_fn,
                        intensity=intensity,
                        rng=rng,
                        sample=sample,
                        edit_fn=edit_fn,
                        metrics_fn=metrics_fn,
                        sample_out_dir=sample_out_dir,
                        root_dir=root_dir,
                        config=config,
                    )
                    if record_orig is not None:
                        records.append(record_orig)

        _save_checkpoint(records, sample_out_base / "records.json")

    # Salvataggio delle medie per questa singola root_dir
    _save_summary(records, sample_out_base)

    return records


def evaluate_robustness(
    edit_target: Union[Any, dict[str, Any]],
    metrics_target: Any,
    config: RobustnessConfig,
) -> list[dict]:
    """Esegue la valutazione di robustezza in serie su tutte le
    root_dir indicate in config.root_dirs.

    Per ciascuna root_dir viene creata una sotto-cartella
    `output_dir/<tipo_modello>/<nome_root_dir>` con i propri checkpoint
    e summary (nessun file aggregato viene salvato). `<tipo_modello>`
    (es. 'SD_Inpainting', 'SD_Img2Img', 'InstructPix2Pix') viene dedotto
    automaticamente dal path della root_dir, vedi MODEL_TYPE_KEYWORDS.

    Per ciascuna combinazione trasformazione/intensità viene eseguito un
    branch sull'immagine immunizzata ("source"="immunized") e, se
    `config.evaluate_original_branch` è True (default), anche un branch
    sull'immagine originale ("source"="original"), usato come baseline.

    `edit_target` può essere:
    - un singolo modello/callable di editing, usato per tutte le root_dir
      (comportamento precedente); oppure
    - un dict che mappa il tipo di modello dedotto (le stesse etichette
      di MODEL_TYPE_KEYWORDS, es. 'SD_Inpainting', 'SD_Img2Img',
      'InstructPix2Pix') al modello/callable di editing da usare per
      quel tipo. In questo caso, per ogni root_dir viene selezionato
      automaticamente il modello corretto in base al tipo dedotto dal
      path.
    """
    metrics_fn = _resolve_metrics_callable(metrics_target)

    # Se edit_target è un dict, risolviamo un edit_fn per ciascuna
    # etichetta presente, con caching (una root_dir con lo stesso tipo
    # riusa lo stesso edit_fn già risolto).
    is_per_type = isinstance(edit_target, dict)
    single_edit_fn = None if is_per_type else _resolve_edit_callable(edit_target)
    resolved_edit_fns: dict[str, Callable] = {}

    config.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for root_dir in config.root_dirs:
        model_subdir = _infer_model_output_subdir(root_dir)

        if is_per_type:
            if model_subdir not in edit_target:
                raise KeyError(
                    f"Nessun modello di editing fornito per il tipo '{model_subdir}' "
                    f"(dedotto da {root_dir}). Chiavi disponibili: {list(edit_target.keys())}"
                )
            if model_subdir not in resolved_edit_fns:
                resolved_edit_fns[model_subdir] = _resolve_edit_callable(edit_target[model_subdir])
            edit_fn = resolved_edit_fns[model_subdir]
        else:
            edit_fn = single_edit_fn

        sample_out_base = config.output_dir / model_subdir / root_dir.name
        if config.save_images:
            sample_out_base.mkdir(parents=True, exist_ok=True)

        root_records = _run_single_root(
            root_dir=root_dir,
            edit_fn=edit_fn,
            metrics_fn=metrics_fn,
            config=config,
            sample_out_base=sample_out_base,
        )
        all_records.extend(root_records)

    return all_records


# ---------------------------------------------------------------------------
# Helper per il calcolo e il salvataggio delle medie
# ---------------------------------------------------------------------------

def _save_summary(records: list[dict], output_dir: Path) -> None:
    """Calcola la media di ciascuna metrica per ogni combinazione
    (transform, intensity, source) e salva i risultati sia in formato
    JSON che TXT.
    """
    if not records:
        print(f"[warning] Nessun record trovato per calcolare le medie in {output_dir}.")
        return

    # Struttura: metrics_accum[(transform, intensity, source)][metric_key] = [val1, val2, ...]
    metrics_accum = defaultdict(lambda: defaultdict(list))

    # Chiavi non numeriche da ignorare nel calcolo della media
    ignore_keys = {"root_dir", "sample_id", "source", "transform", "intensity", "saved_input_path", "saved_edited_path"}

    for r in records:
        key = (r["transform"], str(r["intensity"]), r.get("source", "immunized"))
        for k, v in r.items():
            if k not in ignore_keys and isinstance(v, (int, float, np.number)):
                metrics_accum[key][k].append(float(v))

    summary_dict = {}

    for (t_name, intensity, source), metrics in metrics_accum.items():
        comb_key = f"{t_name}@{intensity}@{source}"
        summary_dict[comb_key] = {}
        for m_key, vals in metrics.items():
            summary_dict[comb_key][m_key] = float(np.mean(vals)) if vals else 0.0

    # 1. Salvataggio in summary.json
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # 2. Salvataggio in summary.txt (leggibile)
    txt_path = output_dir / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== AVERAGE RESULTS SUMMARY ===\n\n")
        for comb_key, metrics in summary_dict.items():
            f.write(f"--- {comb_key} ---\n")
            for m_key, m_val in metrics.items():
                f.write(f"  {m_key}: {m_val:.6f}\n")
            f.write("\n")

    print(f"\n[OK] Medie salvate con successo in:\n - {json_path}\n - {txt_path}")


def _save_checkpoint(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)


def summarize(records: list[dict], metric_key: str = "masked_lpips", source: Optional[str] = "immunized") -> dict:
    """Riepiloga `metric_key` per (transform, intensity).

    Se `source` è specificato (default "immunized"), filtra i record per
    quel branch; passare `source=None` per includere entrambi i branch
    insieme (in tal caso il branch viene incluso nella chiave del risultato).
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        if metric_key not in r:
            continue
        r_source = r.get("source", "immunized")
        if source is not None and r_source != source:
            continue
        key = (r["transform"], r["intensity"]) if source is not None else (r["transform"], r["intensity"], r_source)
        groups[key].append(float(r[metric_key]))

    return {"@".join(str(part) for part in key): float(np.mean(vals)) for key, vals in groups.items() if vals}


class CombinedMetrics:
    def __init__(self, editing_score: EditingScore, segmentation_score: SegmentationScore):
        self.editing_score = editing_score
        self.segmentation_score = segmentation_score

    def compute(
        self,
        adversarial: Image.Image,
        reference: Image.Image,
        mask: Optional[Image.Image] = None,
        editing_prompt: Optional[str] = None
    ) -> dict:
        # 1. Calcolo EditingScore (PSNR, SSIM, LPIPS, Qwen, ecc.)
        metrics_dict = self.editing_score.compute(
            reference=reference,
            adversarial=adversarial,
            editing_prompt=editing_prompt
        )

        # 2. Calcolo SegmentationScore (optimistic_iou, pessimistic_iou)
        seg_dict = self.segmentation_score.compute(
            image_orig=reference,
            image_adv=adversarial
        )

        # 3. Unione dei risultati
        metrics_dict.update(seg_dict)
        return metrics_dict


if __name__ == "__main__":
    cfg = RobustnessConfig(
        root_dirs=[
            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE_FT_2_STAGE"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE_FT_2_STAGE"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE_FT_2_STAGE"),
           
            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/VAE_MSE_TARGET_OPT"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/VAE_MSE_TARGET_OPT"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/VAE_MSE_TARGET_OPT"),

            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/DiffVax"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/DiffVax"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/DiffVax"),

            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/PhotoGuard"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/PhotoGuard"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/PhotoGuard"),

            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/MagicBrush_gray_FT"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/MagicBrush_gray_FT"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/MagicBrush_gray_FT"),

            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/MagicBrush_target_opt"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/MagicBrush_target_opt"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/MagicBrush_target_opt"),

            Path("/equilibrium/ldelbene/Immunization/output/SD_Inpainting/full_dataset/MagicBrush_photoguard"),
            Path("/equilibrium/ldelbene/Immunization/output/SD_Img2Img/full_dataset/MagicBrush_photoguard"),
            Path("/equilibrium/ldelbene/Immunization/output/InstructionPix2Pix/full_dataset/MagicBrush_photoguard"),
        ],
        n_samples=25,
        # output_dir è la base comune: la sottocartella per tipo di
        # modello (SD_Inpainting / SD_Img2Img / InstructPix2Pix) viene
        # dedotta automaticamente dal path di ciascuna root_dir.
        output_dir=Path("robustness_results"),
        # Esegue anche il branch di baseline sull'immagine originale
        # (trasformazione + editing + salvataggio + metriche).
        evaluate_original_branch=True,
    )

    # Inizializzazione modelli di attacco/editing e metriche
    attack = Attack()                    # usato per root_dir con "inpainting" -> SD_Inpainting
    attackSD = AttackSD()                # usato per root_dir con "img2img" -> SD_Img2Img
    pix2pix = AttackInstructPix2Pix()    # usato per root_dir con "instructionpix2pix"/"instructpix2pix" -> InstructPix2Pix

    editing_score = EditingScore()
    segmentation_score = SegmentationScore()

    # Combinazione delle metriche
    combined_metrics = CombinedMetrics(
        editing_score=editing_score,
        segmentation_score=segmentation_score
    )

    # Un modello di editing diverso per tipo, selezionato automaticamente
    # in base al tipo dedotto dal path di ciascuna root_dir (vedi
    # MODEL_TYPE_KEYWORDS / _infer_model_output_subdir).
    edit_targets = {
        "SD_Inpainting": attack,
        "SD_Img2Img": attackSD,
        "InstructPix2Pix": pix2pix,
    }

    # Valutazione della robustezza su tutte le root_dir, in serie
    records = evaluate_robustness(edit_targets, combined_metrics, cfg)

    # Riepilogo per la metrica di segmentazione (es. pessimistic_iou o optimistic_iou)
    # Di default sul branch "immunized"; passare source="original" o
    # source=None (entrambi insieme) per gli altri riepiloghi.
    summary_seg = summarize(records, metric_key="pessimistic_iou")
    summary_qwen = summarize(records, metric_key="qwen_score")
    summary_seg_original = summarize(records, metric_key="pessimistic_iou", source="original")

    print("=== Summary Pessimistic mIoU (immunized, tutte le root_dir) ===")
    print(json.dumps(summary_seg, indent=2))

    print("\n=== Summary Qwen Score (immunized, tutte le root_dir) ===")
    print(json.dumps(summary_qwen, indent=2))

    print("\n=== Summary Pessimistic mIoU (original baseline, tutte le root_dir) ===")
    print(json.dumps(summary_seg_original, indent=2))