from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from deformtta.segmentation import (
    IMAGE_EXTS,
    ImageDataset,
    PairDataset,
    adapt_channels,
    dice_per_class,
    find_mask,
    load_label,
    load_nnunet_model,
    model_input_channels,
    synthetic_matches,
)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    return (-prob * torch.log(prob + 1e-8)).sum(dim=1).mean()


def collect_norm_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    return params


def run_tent(
    model_folder: str,
    checkpoint_name: str,
    images_dir: str,
    output_dir: str,
    masks_dir: str | None,
    num_classes: int,
    tta_steps: int,
    lr: float = 1e-4,
    batch_size: int = 1,
    online: bool = False,
    device: str = "cuda",
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, dev = load_nnunet_model(model_folder, checkpoint_name, device)
    channels = model_input_channels(model)
    initial_state = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(collect_norm_parameters(model), lr=lr)
    loader = DataLoader(ImageDataset(images_dir, masks_dir), batch_size=batch_size, shuffle=False)
    rows = []

    for batch in loader:
        if not online:
            model.load_state_dict(initial_state)
            optimizer = torch.optim.Adam(collect_norm_parameters(model), lr=lr)

        image = adapt_channels(batch["image"].to(dev), channels)
        model.train()
        for _ in range(tta_steps):
            loss = entropy_loss(model(image))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(image), dim=1).cpu().numpy()

        for i, case_id in enumerate(batch["case_id"]):
            pred_i = pred[i].astype(np.uint8)
            Image.fromarray(pred_i).save(output / f"{case_id}.png")
            row = {"case_id": case_id}
            if "mask" in batch:
                gt = batch["mask"][i].numpy()
                dice = dice_per_class(pred_i, gt, num_classes)
                row.update({f"class_{j}": float(dice[j]) for j in range(num_classes)})
                row["mean_ignore_bg"] = float(dice[1:].mean()) if num_classes > 1 else float(dice.mean())
            rows.append(row)

    csv_path = output / "dice_results.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return str(csv_path)


class DiceCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long())
        classes = logits.shape[1]
        target_onehot = F.one_hot(target.long(), num_classes=classes).permute(0, 3, 1, 2).float()
        prob = F.softmax(logits, dim=1)
        dims = (0, 2, 3)
        dice = (2 * (prob * target_onehot).sum(dims) + 1e-5) / ((prob + target_onehot).sum(dims) + 1e-5)
        return ce + (1.0 - dice.mean())


def run_synthetic_adaptation(
    model_folder: str,
    checkpoint_name: str,
    real_images_dir: str,
    real_masks_dir: str,
    synthetic_images_dir: str,
    synthetic_masks_dir: str,
    output_dir: str,
    num_classes: int,
    epochs: int,
    lr: float,
    batch_size: int,
    max_synthetic: int,
    weight_decay: float = 0.0,
    device: str = "cuda",
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, dev = load_nnunet_model(model_folder, checkpoint_name, device)
    channels = model_input_channels(model)
    initial_state = deepcopy(model.state_dict())
    criterion = DiceCELoss()
    real_files = sorted(p for p in Path(real_images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    synth_names = sorted(p.name for p in Path(synthetic_images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = []

    for real_path in real_files:
        matches = synthetic_matches(real_path.stem, synth_names)[:max_synthetic]
        if not matches:
            continue

        model.load_state_dict(initial_state)
        for param in model.parameters():
            param.requires_grad = False
        params = collect_norm_parameters(model)
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loader = DataLoader(PairDataset(synthetic_images_dir, synthetic_masks_dir, matches), batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for batch in loader:
                image = adapt_channels(batch["image"].to(dev), channels)
                mask = batch["mask"].to(dev)
                loss = criterion(model(image), mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        image_np = np.array(Image.open(real_path).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image_np)[None, None].to(dev)
        image = adapt_channels(image, channels)
        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(image), dim=1)[0].cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(output / f"{real_path.stem}.png")

        mask_path = find_mask(real_path.stem, Path(real_masks_dir))
        if mask_path is None:
            continue
        dice = dice_per_class(pred, load_label(mask_path), num_classes)
        row = {"case_id": real_path.stem, "mean_ignore_bg": float(dice[1:].mean())}
        row.update({f"class_{i}": float(dice[i]) for i in range(num_classes)})
        rows.append(row)

    csv_path = output / "per_image_adapt_results.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return str(csv_path)

