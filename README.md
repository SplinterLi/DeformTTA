# DeformTTA

Minimal release code for **DeformTTA: Morphology-Aware Controllable Deformation for Test-Time Adaptation in Fetal Cardiac Ultrasound Image Segmentation**.

This repository contains only the code needed to run the conference release pipeline. It does not include clinical data, model checkpoints, generated images, or experiment outputs.

## Method Summary

DeformTTA targets fetal four-chamber ultrasound segmentation under scanner, protocol, and cardiac-motion domain shifts. The released pipeline uses:

- a trained deformation network to create motion-conditioned fetal cardiac masks;
- a ControlNet-based generator to synthesize ultrasound images from the deformed masks;
- nnUNet test-time adaptation using the synthetic image-mask pairs;
- an optional Tent baseline for comparison.

The release is intentionally small and excludes auxiliary research scripts outside the main pipeline.

## Structure

```text
DeformTTA_open_source/
├── configs/main_config.example.json
├── deformtta/
│   ├── deformation.py
│   ├── generation.py
│   ├── models.py
│   ├── segmentation.py
│   └── tta.py
├── run_deformtta.py
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n deformtta python=3.10 -y
conda activate deformtta
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for your system if the default wheel is not appropriate.

## Required External Assets

Provide these paths in `configs/main_config.example.json` or in a copied config file:

- `model_folder`: trained nnUNet model folder.
- `real_images_dir`: target-domain test images.
- `real_masks_dir`: target labels for metric reporting.
- `initial_masks_dir`: initial prediction or pseudo-mask directory used for deformation. Do not use target ground-truth labels for TTA.
- `deformation.model_path`: trained ShapeDeformNet checkpoint.
- `controlnet.controlnet_path`: trained ControlNet checkpoint.
- `controlnet.diffusion_path`: base diffusion model path.

## Usage

Copy and edit the config:

```bash
cp configs/main_config.example.json configs/main_config.json
```

Check the configured run:

```bash
python run_deformtta.py --config configs/main_config.json --dry-run
```

Run DeformTTA full:

```bash
python run_deformtta.py --config configs/main_config.json --only deformtta_full
```

Run the Tent baseline:

```bash
python run_deformtta.py --config configs/main_config.json --only tent
```

Outputs are written under `experiments/deformtta_main_runs/` by default.
