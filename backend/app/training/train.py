"""
SatQuery AI — Remote Sensing Model Adaptation Training Pipeline
Genuine, reproducible PyTorch training harness for BigEarthNet domain adaptation.
"""
import os
import sys
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import rasterio

from app.training.dataset import BigEarthNetDataset, BIGEARTHNET_19_CLASSES
from app.training.preprocessing import normalize_raster_bands, extract_rgb_nir

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from app.training.model import RemoteSensingAdaptedVLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def detect_compute_device(requested_device: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Detect and log available compute hardware (CUDA GPU or CPU fallback)."""
    info = {"device_type": "cpu", "gpu_name": None, "vram_gb": None}
    if not HAS_TORCH:
        return None, info

    if requested_device and requested_device.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        info["device_type"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    elif torch.cuda.is_available() and requested_device is None:
        device = torch.device("cuda")
        info["device_type"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        device = torch.device("cpu")
        info["device_type"] = "cpu"

    return device, info


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def train_adapter(
    dataset_dir: str,
    checkpoint_dir: str = "./checkpoints",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    train_split: float = 0.8,
    val_split: float = 0.2,
    seed: int = 42,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute genuine PyTorch training loop on BigEarthNet dataset.
    Fails immediately if dataset or PyTorch is not available.
    """
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for remote-sensing adaptation. Please install torch.")
        sys.exit(1)

    set_seed(seed)
    device, dev_info = detect_compute_device(device_str)

    print("\n=======================================================")
    print(" SATQUERY AI — REMOTE SENSING ADAPTATION PIPELINE ")
    print("=======================================================")
    print(f"Target Dataset:      BigEarthNet-19")
    print(f"Dataset Path:        {dataset_dir}")
    print(f"Device:              {dev_info['device_type'].upper()}")
    if dev_info["gpu_name"]:
        print(f"GPU Name:            {dev_info['gpu_name']} ({dev_info['vram_gb']} GB VRAM)")
    else:
        print("[WARNING] Training on CPU. Large remote-sensing datasets will take substantial time.")
    print("=======================================================\n")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"\n=======================================================")
        print(f" ERROR: REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED.")
        print(f" Expected dataset path '{dataset_dir}' does not exist.")
        print(f"=======================================================\n")
        raise FileNotFoundError(
            f"BigEarthNet dataset not found at '{dataset_dir}'. "
            f"REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED."
        )

    # 1. Initialize genuine dataset
    def transform_fn(x):
        norm = normalize_raster_bands(x, target_size=(120, 120))
        rgb_nir = extract_rgb_nir(norm)
        return torch.from_numpy(rgb_nir).float()

    full_dataset = BigEarthNetDataset(root_dir=dataset_dir, transform=transform_fn)
    total_samples = len(full_dataset)
    print(f"Discovered {total_samples} total valid patches.")

    # Split dataset into train and validation sets
    train_size = int(total_samples * train_split)
    val_size = total_samples - train_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_size > 0 else None

    print(f"Training Samples:   {train_size}")
    print(f"Validation Samples: {val_size}")

    # 2. Instantiate model and optimizer
    model = RemoteSensingAdaptedVLM(in_channels=4, embed_dim=512, num_classes=len(BIGEARTHNET_19_CLASSES))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Real Training Loop
    ckpt_path_dir = Path(checkpoint_dir)
    ckpt_path_dir.mkdir(parents=True, exist_ok=True)

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out["logits"], targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        final_train_loss = running_train_loss / max(1, train_batches)

        # Validation loop
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    out = model(images)
                    v_loss = criterion(out["logits"], targets)
                    running_val_loss += v_loss.item()
                    val_batches += 1
            final_val_loss = running_val_loss / max(1, val_batches)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {final_train_loss:.4f} | Val Loss: {final_val_loss:.4f}")
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {final_train_loss:.4f}")

    # 4. Save genuine checkpoint + training metadata JSON
    timestamp_str = datetime.utcnow().isoformat()
    checkpoint_file = ckpt_path_dir / "rs_adapted_vlm.pth"
    metadata_file = ckpt_path_dir / "rs_adapted_vlm_metadata.json"

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "num_classes": len(BIGEARTHNET_19_CLASSES),
        "classes": BIGEARTHNET_19_CLASSES,
        "training_dataset": "BigEarthNet-19",
        "training_timestamp": timestamp_str,
        "status": "TRAINED"
    }, checkpoint_file)

    training_metadata = {
        "model": "RemoteSensingAdaptedVLM",
        "dataset": "BigEarthNet-19",
        "dataset_version": "v1.0",
        "sample_count": total_samples,
        "train_samples": train_size,
        "val_samples": val_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "train_loss": round(final_train_loss, 4),
        "validation_loss": round(final_val_loss, 4) if val_loader else None,
        "timestamp": timestamp_str,
        "device": str(device),
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__ if HAS_TORCH else "N/A",
            "rasterio": rasterio.__version__,
            "numpy": np.__version__
        },
        "status": "TRAINED"
    }

    with open(metadata_file, "w") as f:
        json.dump(training_metadata, f, indent=2)

    print(f"\n[SUCCESS] Genuine Checkpoint saved: {checkpoint_file}")
    print(f"[SUCCESS] Genuine Metadata saved:   {metadata_file}\n")
    return training_metadata


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Remote Sensing Adaptation Training")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to BigEarthNet dataset root directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    try:
        train_adapter(
            dataset_dir=args.dataset_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_split=args.train_split,
            seed=args.seed,
            device_str=args.device
        )
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from typing import Tuple
    main()
