from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_nnunet_model(model_folder: str, checkpoint_name: str, device: str):
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise ImportError("nnUNet v2 is required. Install it with `pip install nnunetv2`.") from exc

    dev = resolve_device(device)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=dev,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name=checkpoint_name)
    model = predictor.network.to(dev)
    model.eval()
    return model, dev


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: Optional[str] = None) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.files = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not self.files:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        image = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        sample: dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(image)[None],
            "case_id": path.stem,
            "filename": path.name,
        }
        if self.mask_dir is not None:
            mask_path = find_mask(path.stem, self.mask_dir)
            if mask_path is not None:
                sample["mask"] = torch.from_numpy(load_label(mask_path))
        return sample


class PairDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, filenames: list[str]) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        name = self.filenames[index]
        image = np.array(Image.open(self.image_dir / name).convert("L"), dtype=np.float32) / 255.0
        mask = load_label(self.mask_dir / name)
        return {
            "image": torch.from_numpy(image)[None],
            "mask": torch.from_numpy(mask),
        }


def model_input_channels(model: torch.nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    return 1


def adapt_channels(image: torch.Tensor, channels: int) -> torch.Tensor:
    if image.shape[1] == channels:
        return image
    if image.shape[1] == 1 and channels == 3:
        return image.repeat(1, 3, 1, 1)
    if image.shape[1] == 3 and channels == 1:
        return image.mean(dim=1, keepdim=True)
    raise RuntimeError(f"Cannot adapt input with {image.shape[1]} channels to model with {channels} channels.")


def load_label(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.int64)
    return np.array(Image.open(path), dtype=np.int64)


def find_mask(case_id: str, mask_dir: Path) -> Optional[Path]:
    candidates = [case_id]
    stripped = re.sub(r"_0+$", "", case_id)
    if stripped != case_id:
        candidates.append(stripped)
    stripped = re.sub(r"_[0-9]+$", "", case_id)
    if stripped != case_id:
        candidates.append(stripped)

    for cid in candidates:
        for ext in [".png", ".jpg", ".jpeg", ".npy"]:
            path = mask_dir / f"{cid}{ext}"
            if path.exists():
                return path
    for path in mask_dir.iterdir():
        if any(cid in path.stem for cid in candidates):
            return path
    return None


def dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    scores = np.zeros(num_classes, dtype=np.float32)
    for class_id in range(num_classes):
        pred_c = pred == class_id
        target_c = target == class_id
        denom = pred_c.sum() + target_c.sum()
        scores[class_id] = 1.0 if denom == 0 else float(2.0 * np.logical_and(pred_c, target_c).sum() / denom)
    return scores


def synthetic_matches(case_id: str, filenames: list[str]) -> list[str]:
    return [name for name in filenames if case_id in Path(name).stem]

