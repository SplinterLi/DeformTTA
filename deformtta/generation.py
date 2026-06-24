from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image


PALETTE = {
    1: [30, 144, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [0, 255, 255],
    5: [255, 0, 255],
    6: [255, 255, 0],
    7: [128, 0, 255],
    8: [255, 128, 0],
}


def mask_to_rgb(path: Path) -> Image.Image:
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in PALETTE.items():
        rgb[mask == label] = color
    return Image.fromarray(rgb)


def list_masks(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def load_pipeline(controlnet_path: str, diffusion_path: str, device: str) -> StableDiffusionControlNetPipeline:
    dtype = torch.float16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        diffusion_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe.to(device)


@torch.no_grad()
def generate_synthetic_images(
    controlnet_path: str,
    diffusion_path: str,
    conditioning_dir: str,
    output_dir: str,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    max_images: int = -1,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    dev = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    masks = list_masks(Path(conditioning_dir))
    if max_images > 0:
        masks = masks[:max_images]

    pipe = load_pipeline(controlnet_path, diffusion_path, dev)
    generator = torch.Generator(device=dev).manual_seed(seed)
    for mask_path in masks:
        control_image = mask_to_rgb(mask_path)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        image.save(output / mask_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ultrasound images from deformed masks.")
    parser.add_argument("--controlnet_path", required=True)
    parser.add_argument("--diffusion_path", required=True)
    parser.add_argument("--conditioning_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generate_synthetic_images(**vars(args))


if __name__ == "__main__":
    main()

