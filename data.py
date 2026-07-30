import os

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datasets import load_from_disk, load_dataset
from PIL import Image, ImageOps
from pathlib import Path
from utils import load_sample_from_hf, prepare_mask_and_masked_image


class COCOLocal(Dataset):
    """
    Carica COCO da cartella locale con struttura:

    Immagini:
        /andromeda/datasets/COCO/COCO2017_train/train2017/
        /andromeda/datasets/COCO/COCO2017_val/val2017/
    Maschere:
        /equilibrium/ldelbene/Immunization/data/COCO_mask/train/
        /equilibrium/ldelbene/Immunization/data/COCO_mask/val/
    """

    IMAGES_DIRS = {
        "train": "/andromeda/datasets/COCO/COCO2017_train/train2017",
        "val": "/andromeda/datasets/COCO/COCO2017_val/val2017",
    }
    MASKS_DIRS = {
        "train": "/equilibrium/ldelbene/Immunization/data/COCO_mask/train",
        "val": "/equilibrium/ldelbene/Immunization/data/COCO_mask/val",
    }

    def __init__(self, split: str = "train"):
        assert split in ("train", "val"), f"split deve essere 'train' o 'val', ricevuto: {split}"

        img_dir = Path(self.IMAGES_DIRS[split])
        mask_dir = Path(self.MASKS_DIRS[split])

        # Indicizza le maschere per stem (nome senza estensione) per match rapido
        mask_by_stem = {p.stem: p for p in mask_dir.glob("*.png")}

        # Tieni solo le immagini per cui esiste la maschera corrispondente
        self.pairs = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            if img_path.stem in mask_by_stem:
                self.pairs.append((img_path, mask_by_stem[img_path.stem]))

        print(f"[COCOLocal] split={split} | coppie immagine-maschera: {len(self.pairs)}")

        assert len(self.pairs) > 0, \
            f"Nessuna coppia immagine-maschera trovata per split='{split}'"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

class OxfordPetLocal(Dataset):
    """
    Carica Oxford-Pet da cartella locale con struttura:
    Oxford-Pet/
        train/
            img/
            mask/
        validation/
            img/
            mask/
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 224):
        self.image_size = image_size

        split_folder = "validation" if split == "val" else split
        img_dir = Path(root) / split_folder / "img"
        mask_dir = Path(root) / split_folder / "mask"

        self.img_paths = sorted(img_dir.glob("*"))
        self.mask_paths = sorted(mask_dir.glob("*"))

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.img_paths)} immagini vs {len(self.mask_paths)} maschere"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        return image, mask


class TEdBenchDataset(Dataset):
    """
    Carica TEdBench da HuggingFace Hub (bahjat-kawar/tedbench).

    Split disponibile: "val" (100 esempi). Il dataset HF contiene solo
    "original_image", "caption" ed "edited_image": non fornisce maschere.
    Le maschere vengono quindi caricate da cartella locale, abbinate
    all'esempio per indice posizionale (non per nome/stem):

        data/tedbench_mask/mask_fine/000.png
        data/tedbench_mask/mask_fine/001.png
        ...

    cioè l'esempio idx=0 dello split "val" usa la maschera "000.png",
    l'esempio idx=1 usa "001.png", e così via.
    """

    MASK_DIR = "./data/tedbench_masks"

    def __init__(self, split: str = "val", cache_dir: str = "/equilibrium/ldelbene/cache/hf", target_size=(512,512)):
        assert split == "val", f"TEdBench ha solo lo split 'val', ricevuto: {split}"

        self.dataset = load_dataset("bahjat-kawar/tedbench", split="val", cache_dir=cache_dir)
        self.mask_dir = Path(self.MASK_DIR)
        self.target_size = target_size

        print(f"[TEdBenchDataset] split={split} | esempi: {len(self.dataset)}")

        assert len(self.dataset) > 0, "Nessun esempio trovato per TEdBench"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample["original_image"].convert("RGB")

        mask_path = self.mask_dir / f"img_{idx}.png"
        assert mask_path.exists(), f"Maschera non trovata: {mask_path}"
        mask = Image.open(mask_path).convert("L")
        #mask = ImageOps.invert(mask)
        image = image.resize(self.target_size, Image.BICUBIC)
        mask  = mask.resize(self.target_size, Image.NEAREST)

        return image, mask


class ImmunizationDataset(Dataset):
    def __init__(
        self,
        dataset:        str = "DiffVax",  # DiffVax | Oxford-Pet | COCO | MagicBrush | TEdBench
        split:          str = "train",
        image_size:     int = 224,
    ):
        self.split      = split
        self.image_size = image_size

        if dataset == "DiffVax":
            dataset = load_from_disk("data/DiffVaxDataset_local")
            self.dataset = dataset[split]
        elif dataset == "Oxford-Pet":
            self.dataset = OxfordPetLocal(root="./data/Oxford-Pet", split=split, image_size=image_size)
        elif dataset == "COCO":
            self.dataset = COCOLocal(split=split)
        elif dataset == "MagicBrush":
            self.dataset = MagicBrushHF(split=split)
        elif dataset == "TEdBench":
            self.dataset = TEdBenchDataset(split=split)
        else:
            raise ValueError(f"dataset non supportato: {dataset}")


        self.image_transform = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR
        )

        self.mask_transform = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
     if isinstance(self.dataset, OxfordPetLocal):
        image, mask = self.dataset[idx]
     elif isinstance(self.dataset, COCOLocal):        
        image, mask = self.dataset[idx]
     elif isinstance(self.dataset, MagicBrushHF):
        image, mask = self.dataset[idx]
     elif isinstance(self.dataset, TEdBenchDataset):
        image, mask = self.dataset[idx]
     else:
        sample = self.dataset[idx]
        image, mask = load_sample_from_hf(sample, split=self.split)

     image = self.image_transform(image)
     mask = self.mask_transform(mask)

     M, _, I = prepare_mask_and_masked_image(image, mask)
     I = I.squeeze(0)
     M = M.squeeze(0)
     return I, M
    




class MagicBrushHF(Dataset):
    """
    Carica MagicBrush direttamente da HuggingFace Hub (osunlp/MagicBrush).

    Split disponibili: "train" (8807 esempi), "dev" (528 esempi).
    Non esiste uno split "val"/"test" pubblico: il test set è distribuito
    separatamente con password per evitare data contamination.

    Ogni esempio contiene anche "instruction" (l'edit testuale) e
    "img_id"/"turn_index" (per gestire le sequenze multi-turn), che qui
    non vengono usati per restare compatibili con l'interfaccia
    (image, mask) del resto della pipeline — ma sono recuperabili se serve.
    """

    SPLIT_MAP = {
        "train": "train",
        "val":   "dev",
        "validation":   "dev",
    }

    def __init__(self, split: str = "train", single_turn_only: bool = True, cache_dir: str = "/equilibrium/ldelbene/cache/hf", target_size=(512,512)):
        hf_split = self.SPLIT_MAP.get(split)
        if hf_split is None:
            raise ValueError(f"split deve essere 'train'/'val'/'dev', ricevuto: {split}")

        self.dataset = load_dataset("osunlp/MagicBrush", split=hf_split, cache_dir=cache_dir)
        self.target_size = target_size

        if single_turn_only:
            # tiene solo turn_index == 1: source_img è l'immagine originale
            # reale, non un'immagine già editata da un turno precedente
            self.dataset = self.dataset.filter(lambda ex: ex["turn_index"] == 1)

        print(f"[MagicBrushHF] split={hf_split} | single_turn_only={single_turn_only} | esempi: {len(self.dataset)}")

        assert len(self.dataset) > 0, f"Nessun esempio trovato per split='{hf_split}'"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample["source_img"].convert("RGB")
        mask  = sample["mask_img"].convert("L")

        # resize a size fissa, divisibile per il fattore di downsampling della NestedUNet
        image = image.resize(self.target_size, Image.BICUBIC)
        mask  = mask.resize(self.target_size, Image.NEAREST)

        mask_arr = np.array(mask)
        mask_arr = np.where(mask_arr < 1, 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode="L")

        mask = ImageOps.invert(mask)
        return image, mask