from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_metric(csv_path: str) -> float | None:
    path = Path(csv_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = [float(row["mean_ignore_bg"]) for row in rows if row.get("mean_ignore_bg")]
    return sum(values) / len(values) if values else None


def run(cfg: dict[str, Any], dry_run: bool = False, only: set[str] | None = None) -> list[dict[str, Any]]:
    root = Path(cfg.get("workspace_root", ".")).resolve()
    outputs_root = root / cfg.get("outputs_root", "experiments/deformtta_main_runs")
    common = cfg["common"]
    results = []

    for exp in cfg["experiments"]:
        if not exp.get("enabled", True):
            continue
        if only and exp["name"] not in only:
            continue

        exp_dir = outputs_root / exp["name"]
        print(f"[Experiment] {exp['name']} ({exp['type']})")

        if dry_run:
            print(f"[Dry run] output_dir={exp_dir}")
            results.append({"name": exp["name"], "type": exp["type"], "output_dir": str(exp_dir), "mean_metric": None})
            continue

        exp_dir.mkdir(parents=True, exist_ok=True)

        if exp["type"] == "tent":
            from deformtta.tta import run_tent

            tent = cfg["tent"]
            csv_path = run_tent(
                model_folder=common["model_folder"],
                checkpoint_name=common.get("checkpoint_name", "checkpoint_final.pth"),
                images_dir=common["real_images_dir"],
                output_dir=str(exp_dir / "predictions"),
                masks_dir=common.get("real_masks_dir"),
                num_classes=int(common.get("num_classes", 7)),
                tta_steps=int(tent.get("tta_steps", 3)),
                lr=float(tent.get("lr", 1e-4)),
                batch_size=int(common.get("batch_size", 1)),
                online=bool(tent.get("online", False)),
                device=common.get("device", "cuda"),
            )
        elif exp["type"] == "deformtta":
            from deformtta.deformation import generate_deformation_masks
            from deformtta.generation import generate_synthetic_images
            from deformtta.tta import run_synthetic_adaptation

            deform = cfg["deformation"]
            control = cfg["controlnet"]
            adapt = cfg["adapt_with_synth"]
            mask_dir = exp_dir / "deformation_masks"
            synth_dir = exp_dir / "synthetic_images"
            adapt_dir = exp_dir / "adapt_results"

            generate_deformation_masks(
                model_path=deform["model_path"],
                input_mask_dir=common["initial_masks_dir"],
                output_dir=str(mask_dir),
                scales=deform.get("scales", [0.5, 0.75, 1.0, 1.25]),
                num_classes=int(deform.get("num_classes", 4)),
                image_size=int(deform.get("image_size", 384)),
                max_samples=int(deform.get("max_samples", -1)),
                device=common.get("device", "cuda"),
                allow_ground_truth_masks=bool(deform.get("allow_ground_truth_masks", False)),
            )
            generate_synthetic_images(
                controlnet_path=control["controlnet_path"],
                diffusion_path=control["diffusion_path"],
                conditioning_dir=str(mask_dir),
                output_dir=str(synth_dir),
                prompt=control["prompt"],
                negative_prompt=control.get("negative_prompt", ""),
                num_inference_steps=int(control.get("num_inference_steps", 20)),
                guidance_scale=float(control.get("guidance_scale", 7.5)),
                max_images=int(control.get("max_images", -1)),
                seed=int(control.get("seed", 42)),
                device=common.get("device", "cuda"),
            )
            csv_path = run_synthetic_adaptation(
                model_folder=common["model_folder"],
                checkpoint_name=common.get("checkpoint_name", "checkpoint_final.pth"),
                real_images_dir=common["real_images_dir"],
                real_masks_dir=common["real_masks_dir"],
                synthetic_images_dir=str(synth_dir),
                synthetic_masks_dir=str(mask_dir),
                output_dir=str(adapt_dir),
                num_classes=int(common.get("num_classes", 7)),
                epochs=int(adapt.get("epochs", 1)),
                lr=float(adapt.get("lr", 1e-4)),
                batch_size=int(adapt.get("batch_size", 2)),
                max_synthetic=int(adapt.get("max_synthetic", 5)),
                weight_decay=float(adapt.get("weight_decay", 0.0)),
                device=common.get("device", "cuda"),
            )
        else:
            raise ValueError(f"Unsupported experiment type: {exp['type']}")

        results.append({
            "name": exp["name"],
            "type": exp["type"],
            "result_csv": csv_path,
            "output_dir": str(exp_dir),
            "mean_metric": mean_metric(csv_path),
        })

    if not dry_run:
        outputs_root.mkdir(parents=True, exist_ok=True)
        with (outputs_root / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal DeformTTA release pipeline.")
    parser.add_argument("--config", default="configs/main_config.example.json")
    parser.add_argument("--only", default="", help="Comma-separated experiment names.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    only = {item.strip() for item in args.only.split(",") if item.strip()} or None
    results = run(load_config(args.config), dry_run=args.dry_run, only=only)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
