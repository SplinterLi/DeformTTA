from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deformtta.models import MultiClassSpatialTransformer, ShapeDeformNet


def labels_to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.clip(labels.astype(np.int64), 0, num_classes)
    return np.eye(num_classes + 1, dtype=np.float32)[labels].transpose(2, 0, 1)


def mask_array_to_onehot(array: np.ndarray, num_classes: int) -> np.ndarray:
    if array.ndim == 2:
        return labels_to_onehot(array, num_classes)
    if array.ndim == 3 and array.shape[0] <= 32:
        chw = array.astype(np.float32)[: num_classes + 1]
    elif array.ndim == 3 and array.shape[-1] <= 32:
        chw = array.transpose(2, 0, 1).astype(np.float32)[: num_classes + 1]
    else:
        raise ValueError(f"Unsupported mask shape: {array.shape}")

    if chw.shape[0] < num_classes + 1:
        pad = np.zeros((num_classes + 1 - chw.shape[0], *chw.shape[1:]), dtype=np.float32)
        chw = np.concatenate([chw, pad], axis=0)
    denom = np.maximum(chw.sum(axis=0, keepdims=True), 1e-6)
    return (chw / denom).astype(np.float32)


def smooth_onehot(onehot: np.ndarray, kernel_size: int = 31, sigma: float = 5.0) -> torch.Tensor:
    tensor = torch.from_numpy(onehot).float().unsqueeze(0)
    grid = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gaussian = torch.exp(-(grid**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian.view(1, 1, -1, 1) * gaussian.view(1, 1, 1, -1)).to(tensor)

    channels = []
    for channel in range(tensor.shape[1]):
        blurred = F.conv2d(tensor[:, channel : channel + 1], kernel, padding=kernel_size // 2)
        min_v, max_v = blurred.min(), blurred.max()
        if max_v > min_v:
            blurred = (blurred - min_v) / (max_v - min_v)
        channels.append(blurred)
    return torch.cat(channels, dim=1)


def list_mask_files(mask_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".npy"}
    return sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in exts)


def load_mask(path: Path, num_classes: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return mask_array_to_onehot(np.load(path), num_classes)
    labels = np.array(Image.open(path).convert("L"), dtype=np.int64)
    labels[labels > num_classes] = 0
    return labels_to_onehot(labels, num_classes)


def _looks_like_ground_truth(path: Path) -> bool:
    parts = {p.lower() for p in path.resolve().parts}
    return "labelsts" in parts or "labelstr" in parts


@torch.no_grad()
def generate_deformation_masks(
    model_path: str,
    input_mask_dir: str,
    output_dir: str,
    scales: Iterable[float],
    num_classes: int,
    image_size: int = 384,
    max_samples: int = -1,
    device: str = "cuda",
    allow_ground_truth_masks: bool = False,
) -> None:
    input_dir = Path(input_mask_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if _looks_like_ground_truth(input_dir) and not allow_ground_truth_masks:
        raise RuntimeError("input_mask_dir appears to point to ground-truth labels. Use model predictions or pseudo masks for TTA.")

    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = ShapeDeformNet(in_channels=num_classes + 1, out_channels=2 * num_classes).to(dev)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    transformer = MultiClassSpatialTransformer(size=(image_size, image_size), num_classes=num_classes).to(dev)
    files = list_mask_files(input_dir)
    if max_samples > 0:
        files = files[:max_samples]

    for path in files:
        dist = smooth_onehot(load_mask(path, num_classes)).to(dev)
        if dist.shape[-2:] != (image_size, image_size):
            dist = F.interpolate(dist, size=(image_size, image_size), mode="bilinear", align_corners=False)

        flow = model(dist)
        original = torch.argmax(dist[0, : num_classes + 1], dim=0).cpu().numpy().astype(np.uint8)
        Image.fromarray(original).save(output / f"{path.stem}_scale0.00.png")

        for scale in scales:
            warped = transformer(dist, flow * float(scale))
            labels = torch.argmax(warped[0, : num_classes + 1], dim=0).cpu().numpy().astype(np.uint8)
            Image.fromarray(labels).save(output / f"{path.stem}_scale{float(scale):.2f}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate motion-conditioned masks with ShapeDeformNet.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_ground_truth_masks", action="store_true")
    args = parser.parse_args()
    generate_deformation_masks(**vars(args))


if __name__ == "__main__":
    main()

