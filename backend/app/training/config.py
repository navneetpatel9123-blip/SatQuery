"""
SatQuery AI — Remote Sensing Training & Dataset Configuration
==============================================================

Provides strongly-typed configuration dataclasses for:
- BigEarthNet VLM adaptation training
- Change detection model training
- Checkpoint validation schemas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BigEarthNetConfig:
    """Configuration for BigEarthNet Remote-Sensing VLM fine-tuning."""

    dataset_dir: str = "./datasets/bigearthnet"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "rs_adapted_vlm.pth"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    train_split: float = 0.8
    val_split: float = 0.2
    image_size: tuple[int, int] = (120, 120)
    in_channels: int = 4  # RGB + NIR
    embed_dim: int = 512
    num_classes: int = 19
    seed: int = 42
    device: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if self.epochs <= 0:
            errors.append("epochs must be > 0")
        if self.batch_size <= 0:
            errors.append("batch_size must be > 0")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if not (0.0 < self.train_split < 1.0):
            errors.append("train_split must be between 0 and 1")
        return errors

    @property
    def target_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.checkpoint_name


@dataclass
class ChangeDetectionTrainingConfig:
    """Configuration for Bi-Temporal Change Detection training (e.g. LEVIR-CD, WHU-CD, OSCD)."""

    dataset_name: str = "LEVIR-CD"
    dataset_dir: str = "./datasets/levir_cd"
    checkpoint_dir: str = "./checkpoints"
    checkpoint_name: str = "siamese_change_net.pth"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patch_size: tuple[int, int] = (256, 256)
    in_channels: int = 4
    base_filters: int = 32
    loss_function: str = "bce_dice"
    seed: int = 42
    device: Optional[str] = None

    def validate(self) -> List[str]:
        errors = []
        if self.epochs <= 0:
            errors.append("epochs must be > 0")
        if self.batch_size <= 0:
            errors.append("batch_size must be > 0")
        return errors

    @property
    def target_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.checkpoint_name


def validate_checkpoint_file(checkpoint_path: str, expected_model_type: str = "rs_vlm") -> Dict[str, Any]:
    """
    Validates a PyTorch checkpoint file without fabricating data.
    Checks file existence, header metadata, state dict keys, and integrity.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return {
            "valid": False,
            "status": "NOT_FOUND",
            "message": f"Checkpoint file does not exist at '{checkpoint_path}'.",
            "model_type": expected_model_type,
            "keys_present": [],
        }

    try:
        import torch
    except ImportError:
        return {
            "valid": False,
            "status": "TORCH_UNAVAILABLE",
            "message": "PyTorch is required to inspect .pth checkpoint files.",
            "model_type": expected_model_type,
            "keys_present": [],
        }

    try:
        ckpt = torch.load(str(path), map_location="cpu")
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            has_state = "model_state_dict" in ckpt or any(k.startswith("enc") or k.startswith("conv") for k in keys)
            return {
                "valid": has_state,
                "status": "VALID" if has_state else "INVALID_SCHEMA",
                "message": "Valid checkpoint dictionary." if has_state else "Missing expected state_dict keys.",
                "model_type": ckpt.get("model_type", expected_model_type),
                "epoch": ckpt.get("epoch", None),
                "keys_present": keys[:10],
            }
        else:
            return {
                "valid": True,
                "status": "VALID_RAW_STATE",
                "message": "Valid raw PyTorch state object.",
                "model_type": expected_model_type,
                "keys_present": [],
            }
    except Exception as e:
        return {
            "valid": False,
            "status": "CORRUPT",
            "message": f"Failed to load checkpoint file: {str(e)}",
            "model_type": expected_model_type,
            "keys_present": [],
        }
