# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — sleap/make_configs                           ║
# ║  « centroid model finds the birds, instance model poses one »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Writes centroid.json and centered_instance.json for one data    ║
# ║  set. Top-down is the right choice for a few penguins per        ║
# ║  frame: the centroid stage separates animals, the instance       ║
# ║  stage sees one bird per crop. Rotation augmentation covers the  ║
# ║  full circle because both cameras look straight down or up.      ║
# ║  The presets differ only in crop size, which follows the         ║
# ║  bird's extent in pixels.                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Write the two SLEAP training configurations for a data set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sleap.nn.config import (
    CenteredInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    TrainingJobConfig,
    UNetConfig,
)

ANCHOR: str = "neck"
EPOCHS: int = 200
BATCH_SIZE: int = 4
SIGMA_PX: float = 2.5             # centroid targets, at half resolution
INSTANCE_SIGMA_PX: float = 5.0    # broader targets: a 200 px bird needs no 2 px precision
INSTANCE_STRIDE: int = 32         # receptive field must cover the whole silhouette


@dataclass(frozen=True)
class Preset:
    """Per-data-set geometry; crop size is a multiple of the instance stride."""

    name: str
    crop_size_px: int
    centroid_input_scaling: float


PRESETS: dict[str, Preset] = {
    # closed pool, 640 x 480 clips: a gentoo spans up to ~250 px
    "closed": Preset("closed", crop_size_px=320, centroid_input_scaling=0.5),
    # open water, 2704 x 1520 GoPro frames: neck to tail up to ~180 px, anchored at the neck
    "open": Preset("open", crop_size_px=384, centroid_input_scaling=0.5),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="closed")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    centroid_config(preset).save_json(str(args.out_dir / "centroid.json"))
    centered_instance_config(preset).save_json(str(args.out_dir / "centered_instance.json"))
    print(f"{preset.name} configs written to {args.out_dir}")


def centroid_config(preset: Preset) -> TrainingJobConfig:
    cfg = base_config("centroid")
    cfg.data.preprocessing.input_scaling = preset.centroid_input_scaling
    cfg.model.heads.centroid = CentroidsHeadConfig(anchor_part=ANCHOR, sigma=SIGMA_PX, output_stride=4)
    return cfg


def centered_instance_config(preset: Preset) -> TrainingJobConfig:
    cfg = base_config("centered_instance")
    cfg.data.preprocessing.input_scaling = 1.0
    cfg.data.instance_cropping.center_on_part = ANCHOR
    cfg.data.instance_cropping.crop_size = preset.crop_size_px
    cfg.model.backbone.unet.filters = 32
    cfg.model.backbone.unet.max_stride = INSTANCE_STRIDE
    cfg.model.heads.centered_instance = CenteredInstanceConfmapsHeadConfig(
        anchor_part=ANCHOR, sigma=INSTANCE_SIGMA_PX, output_stride=2
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
    # No contrast augmentation: SLEAP 1.4.1 passes its gamma range (0.7-1.3) to
    # albumentations, which expects percent, so gamma ~0.01 bleaches every crop.
    aug.contrast = False
    cfg.optimization.batch_size = BATCH_SIZE
    cfg.optimization.epochs = EPOCHS
    cfg.optimization.initial_learning_rate = 1e-4
    cfg.optimization.early_stopping.plateau_patience = 20
    cfg.optimization.learning_rate_schedule.reduce_on_plateau = True
    cfg.optimization.learning_rate_schedule.plateau_patience = 10
    cfg.outputs.run_name = run_name
    cfg.outputs.runs_folder = "models"
    cfg.outputs.save_visualizations = False
    cfg.outputs.checkpointing.best_model = True
    return cfg


if __name__ == "__main__":
    main()
