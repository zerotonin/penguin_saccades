# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/make_configs                           ║
# ║  « centroid model finds the birds, instance model poses one »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Writes centroid.json and centered_instance.json. Top-down is    ║
# ║  the right choice for two or three penguins in a 640 x 480       ║
# ║  frame: the centroid stage separates animals, the instance       ║
# ║  stage sees one bird per crop. Rotation augmentation covers the  ║
# ║  full circle because the camera looks straight up.               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Write the two SLEAP training configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from sleap.nn.config import (
    CenteredInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    TrainingJobConfig,
    UNetConfig,
)

ANCHOR: str = "neck"
CROP_SIZE_PX: int = 256          # a gentoo spans up to ~250 px in these clips
EPOCHS: int = 150
BATCH_SIZE: int = 4
SIGMA_PX: float = 2.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    centroid_config().save_json(str(args.out_dir / "centroid.json"))
    centered_instance_config().save_json(str(args.out_dir / "centered_instance.json"))
    print(f"configs written to {args.out_dir}")


def centroid_config() -> TrainingJobConfig:
    cfg = base_config("centroid")
    cfg.data.preprocessing.input_scaling = 0.5
    cfg.model.heads.centroid = CentroidsHeadConfig(anchor_part=ANCHOR, sigma=SIGMA_PX, output_stride=4)
    return cfg


def centered_instance_config() -> TrainingJobConfig:
    cfg = base_config("centered_instance")
    cfg.data.preprocessing.input_scaling = 1.0
    cfg.data.instance_cropping.center_on_part = ANCHOR
    cfg.data.instance_cropping.crop_size = CROP_SIZE_PX
    cfg.model.backbone.unet.filters = 24
    cfg.model.heads.centered_instance = CenteredInstanceConfmapsHeadConfig(
        anchor_part=ANCHOR, sigma=SIGMA_PX, output_stride=2
    )
    return cfg


def base_config(run_name: str) -> TrainingJobConfig:
    cfg = TrainingJobConfig()
    cfg.data.labels.validation_fraction = 0.15
    cfg.data.preprocessing.ensure_grayscale = True
    cfg.model.backbone.unet = UNetConfig(
        max_stride=16, filters=16, filters_rate=2.0, output_stride=4, up_interpolate=True
    )
    aug = cfg.optimization.augmentation_config
    aug.rotate = True
    aug.rotation_min_angle = -180
    aug.rotation_max_angle = 180
    aug.scale = True
    aug.scale_min = 0.8
    aug.scale_max = 1.2
    aug.brightness = True
    aug.brightness_min_val = -0.2
    aug.brightness_max_val = 0.2
    aug.contrast = True
    aug.contrast_min_gamma = 0.7
    aug.contrast_max_gamma = 1.3
    aug.random_flip = True
    aug.flip_horizontal = True
    cfg.optimization.batch_size = BATCH_SIZE
    cfg.optimization.epochs = EPOCHS
    cfg.optimization.initial_learning_rate = 1e-4
    cfg.optimization.early_stopping.plateau_patience = 15
    cfg.optimization.learning_rate_schedule.reduce_on_plateau = True
    cfg.optimization.learning_rate_schedule.plateau_patience = 6
    cfg.outputs.run_name = run_name
    cfg.outputs.runs_folder = "models"
    cfg.outputs.save_visualizations = False
    cfg.outputs.checkpointing.best_model = True
    return cfg


if __name__ == "__main__":
    main()
