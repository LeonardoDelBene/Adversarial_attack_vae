from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import random
from pathlib import Path
from transformers import set_seed
from huggingface_hub import snapshot_download, hf_hub_download

totensor = T.ToTensor()
topil = T.ToPILImage()


def recover_image(image, init_image, mask, background=False):
    # Forza tutto in RGB/L prima della conversione in tensore
    image = image.convert("RGB")
    init_image = init_image.convert("RGB")
    mask = mask.convert("L")  # singolo canale grayscale

    image = totensor(image)        # (3, H, W)
    init_image = totensor(init_image)  # (3, H, W)
    mask = totensor(mask)          # (1, H, W)

    # Espandi la maschera a 3 canali per il broadcast
    mask = mask.expand_as(image)   # (3, H, W), valori in [0, 1]

    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image

    return topil(result)


def prepare_mask_and_masked_image(image, mask):
    """Prepare image and mask tensors for inpainting."""
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image, image


def prepare_image_return_3d(image):
    """Prepare single image for model input."""
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


def set_seed_lib(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)


def load_image(image_name, data_dir, is_mask=False, images_subdir="images", masks_subdir="masks"):
    """Load image or mask from data directory."""
    data_path = Path(data_dir)
    if is_mask:
        image = (
            Image.open(data_path / masks_subdir / f"mask_{image_name}.png")
            .convert("RGB")
            .resize((512, 512))
        )
    else:
        image = (
            Image.open(data_path / images_subdir / f"{image_name}.png")
            .convert("RGB")
            .resize((512, 512))
        )
    return image


def load_image_from_path(image_path, size=(512, 512)):
    """Load image from file path."""
    image = Image.open(image_path).convert("RGB").resize(size)
    return image


def save_image(img, img_path):
    """Save image to file."""
    img.save(img_path, "PNG")

def load_sample_from_hf(sample, split="train"):
    """Load image and mask from HuggingFace dataset sample."""
    image = sample["image"].convert("RGB").resize((512, 512))

    # Download mask with correct path prefix
    mask_path = split + "/" + sample["mask"]  # e.g. "train/masks/mask_image_0.png"
    mask_local = hf_hub_download(
        repo_id="ozdentarikcan/DiffVaxDataset",
        filename=mask_path,
        repo_type="dataset"
    )
    mask = Image.open(mask_local).convert("RGB").resize((512, 512))

    return image, mask
