# DeformTTA

Minimal release code for **DeformTTA: Morphology-Aware Controllable Deformation for Test-Time Adaptation in Fetal Cardiac Ultrasound Image Segmentation**.

# Abstract
Fetal cardiac segmentation frequently suffers from performance degradation under domain shifts induced by heterogeneous scanners and rapid physiological motion. Standard test-time adaptation (TTA)methods typically rely on generic consistency or entropy minimization, which lack explicit anatomical constraints and fail to preserve biological plausibility under complex cardiac deformations. We propose DeformTTA, a physiology-driven TTA framework that bolsters segmentation robustness by explicitly modeling cardiac dynamics. Unlike domainagnostic augmentations, DeformTTA leverages a deformation network to simulate realistic fetal heart movements, integrated with a generative model for authentic texture synthesis. Furthermore, a morphology-aware shape prior is introduced to regularize the adaptation process, ensuring the structural integrity of the predicted segmentations. By optimizing for motion-conditioned consistency and morphological alignment, DeformTTA effectively generalizes to target domains during inference. Multi-center experiments demonstrate that DeformTTA significantly outperforms state-of-the-art baselines, particularly in segmenting complex pathological cases. Our code will be made publicly available.

## Method Summary

DeformTTA targets fetal four-chamber ultrasound segmentation under scanner, protocol, and cardiac-motion domain shifts. The released pipeline uses:

- a trained deformation network to create motion-conditioned fetal cardiac masks;
- a ControlNet-based generator to synthesize ultrasound images from the deformed masks;
- SegNet test-time adaptation using the synthetic image-mask pairs;

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

Outputs are written under `experiments/deformtta_main_runs/` by default.
