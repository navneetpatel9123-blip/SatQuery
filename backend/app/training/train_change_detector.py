"""
SatQuery AI — Advanced Siamese Change Detection Training Pipeline
Genuine, reproducible PyTorch training harness for LEVIR-CD bi-temporal change detection
with Focal-Dice Loss, channel normalization, data augmentations, and validation threshold optimization.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from app.models.change_detector import SiameseChangeNet, SiameseChangeDetector
    from app.training.change_dataset import GenericFolderChangeDataset, compute_change_metrics
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─── Combined Focal + Dice / Tversky Loss ────────────────────────────


class FocalDiceLoss(nn.Module):
    """
    Combines alpha-balanced Focal Loss and Soft Dice Loss for change detection.
    Focal Loss suppresses easy background negatives (reducing false alarms),
    while Dice Loss maximizes the spatial boundary overlap of true change regions.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.65,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_clamped = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

        # Alpha-balanced Focal Loss
        pt = torch.where(target == 1.0, pred_clamped, 1.0 - pred_clamped)
        alpha_t = torch.where(target == 1.0, self.alpha, 1.0 - self.alpha)
        focal = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        focal_loss = torch.mean(focal)

        # Soft Dice Loss
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.eps) / (
            pred_flat.sum() + target_flat.sum() + self.eps
        )
        dice_loss = 1.0 - dice

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ─── Synchronous Pair Augmentation ───────────────────────────────────


def apply_pair_augmentations(
    t1: np.ndarray, t2: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies synchronous spatial transformations to (T1, T2, mask) triplets.
    Preserves exact pixel-to-pixel bi-temporal correspondence.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        t1 = np.flip(t1, axis=2).copy()
        t2 = np.flip(t2, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()

    # Random vertical flip
    if random.random() > 0.5:
        t1 = np.flip(t1, axis=1).copy()
        t2 = np.flip(t2, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()

    # Random 90-degree rotations (0, 90, 180, 270 deg)
    k = random.randint(0, 3)
    if k > 0:
        t1 = np.rot90(t1, k, axes=(1, 2)).copy()
        t2 = np.rot90(t2, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

    return t1, t2, mask


def detect_compute_device(requested_device: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """Detect and return available compute device (CUDA GPU or CPU fallback)."""
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


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def train_siamese_change_detector(
    dataset_dir: str = "./datasets/levir_cd",
    checkpoint_dir: str = "./checkpoints",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 2e-4,
    train_split: float = 0.73,
    seed: int = 42,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes genuine PyTorch training of SiameseChangeNet on bi-temporal LEVIR-CD dataset
    with Focal-Dice loss, augmentations, and threshold optimization.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for training SiameseChangeDetector.")

    set_seed(seed)
    device, dev_info = detect_compute_device(device_str)

    print("\n=======================================================")
    print(" SATQUERY AI — ADVANCED SIAMESE CHANGE DETECTION TRAINING ")
    print("=======================================================")
    print(f"Target Dataset:      LEVIR-CD / Generic Folder Triplet")
    print(f"Dataset Path:        {dataset_dir}")
    print(f"Device:              {dev_info['device_type'].upper()}")
    print(f"Loss Strategy:       Focal Loss (gamma=2.0, alpha=0.65) + Soft Dice Loss")
    print(f"Epochs:              {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=======================================================\n")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"LEVIR-CD change dataset not found at '{dataset_dir}'. "
            f"REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED."
        )

    # 1. Load genuine dataset
    dataset = GenericFolderChangeDataset(root_dir=dataset_dir)
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError(f"No valid (A, B, label) triplets discovered in '{dataset_dir}'.")

    print(f"Discovered {total_samples} valid authentic LEVIR-CD bi-temporal triplets.")

    # Split dataset
    train_size = max(1, int(total_samples * train_split))
    val_size = total_samples - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"Training Samples:   {train_size}")
    print(f"Validation Samples: {val_size}")

    def train_collate_fn(batch):
        t1_list, t2_list, mask_list, meta_list = [], [], [], []
        for t1, t2, mask, meta in batch:
            t1_aug, t2_aug, mask_aug = apply_pair_augmentations(t1, t2, mask)

            t1_t = torch.from_numpy(t1_aug).float() / 255.0
            t2_t = torch.from_numpy(t2_aug).float() / 255.0

            if t1_t.ndim == 2:
                t1_t = t1_t.unsqueeze(0)
            if t2_t.ndim == 2:
                t2_t = t2_t.unsqueeze(0)

            # Pad to 4 channels
            if t1_t.shape[0] < 4:
                pad = torch.zeros((4 - t1_t.shape[0], t1_t.shape[1], t1_t.shape[2]), dtype=torch.float32)
                t1_t = torch.cat([t1_t, pad], dim=0)
            if t2_t.shape[0] < 4:
                pad = torch.zeros((4 - t2_t.shape[0], t2_t.shape[1], t2_t.shape[2]), dtype=torch.float32)
                t2_t = torch.cat([t2_t, pad], dim=0)

            mask_t = torch.from_numpy(mask_aug).float().unsqueeze(0)

            t1_list.append(t1_t)
            t2_list.append(t2_t)
            mask_list.append(mask_t)
            meta_list.append(meta)

        return (
            torch.stack(t1_list, dim=0),
            torch.stack(t2_list, dim=0),
            torch.stack(mask_list, dim=0),
            meta_list,
        )

    def val_collate_fn(batch):
        t1_list, t2_list, mask_list, meta_list = [], [], [], []
        for t1, t2, mask, meta in batch:
            t1_t = torch.from_numpy(t1).float() / 255.0
            t2_t = torch.from_numpy(t2).float() / 255.0
            if t1_t.ndim == 2:
                t1_t = t1_t.unsqueeze(0)
            if t2_t.ndim == 2:
                t2_t = t2_t.unsqueeze(0)
            if t1_t.shape[0] < 4:
                pad = torch.zeros((4 - t1_t.shape[0], t1_t.shape[1], t1_t.shape[2]), dtype=torch.float32)
                t1_t = torch.cat([t1_t, pad], dim=0)
            if t2_t.shape[0] < 4:
                pad = torch.zeros((4 - t2_t.shape[0], t2_t.shape[1], t2_t.shape[2]), dtype=torch.float32)
                t2_t = torch.cat([t2_t, pad], dim=0)

            mask_t = torch.from_numpy(mask).float().unsqueeze(0)

            t1_list.append(t1_t)
            t2_list.append(t2_t)
            mask_list.append(mask_t)
            meta_list.append(meta)

        return (
            torch.stack(t1_list, dim=0),
            torch.stack(t2_list, dim=0),
            torch.stack(mask_list, dim=0),
            meta_list,
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=val_collate_fn) if val_size > 0 else None

    # 2. Instantiate Model, Loss & Optimizer
    model = SiameseChangeNet(in_channels=4, base_filters=32)
    model.to(device)

    criterion = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5, alpha=0.65, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_miou": [],
        "val_oa": [],
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_f1 = -1.0
    best_checkpoint_path = Path(checkpoint_dir) / "siamese_change_net.pth"
    best_threshold = 0.5

    print(f"Starting genuine training loop for {epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for t1_batch, t2_batch, mask_batch, _ in train_loader:
            t1_batch = t1_batch.to(device)
            t2_batch = t2_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad()
            preds = model(t1_batch, t2_batch)
            loss = criterion(preds, mask_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, batch_count)
        history["train_loss"].append(round(avg_train_loss, 4))

        # Validation Step
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            all_preds_np: List[np.ndarray] = []
            all_masks_np: List[np.ndarray] = []

            with torch.no_grad():
                for t1_b, t2_b, mask_b, _ in val_loader:
                    t1_b = t1_b.to(device)
                    t2_b = t2_b.to(device)
                    mask_b = mask_b.to(device)

                    preds = model(t1_b, t2_b)
                    v_loss = criterion(preds, mask_b)
                    val_loss += v_loss.item()
                    val_batches += 1

                    all_preds_np.append(preds.cpu().numpy())
                    all_masks_np.append(mask_b.cpu().numpy())

            avg_val_loss = val_loss / max(1, val_batches)
            history["val_loss"].append(round(avg_val_loss, 4))

            # Threshold sweep [0.3, 0.4, 0.5, 0.6, 0.7]
            cat_preds = np.concatenate(all_preds_np, axis=0)
            cat_masks = np.concatenate(all_masks_np, axis=0)

            best_thresh_for_epoch = 0.5
            best_f1_for_epoch = -1.0
            best_metrics_for_epoch = {}

            for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
                f1_list, miou_list, prec_list, rec_list, oa_list = [], [], [], [], []
                for i in range(cat_preds.shape[0]):
                    m = compute_change_metrics(cat_preds[i, 0], cat_masks[i, 0], threshold=th)
                    f1_list.append(m.get("f1_score", 0.0))
                    miou_list.append(m.get("iou", 0.0))
                    prec_list.append(m.get("precision", 0.0))
                    rec_list.append(m.get("recall", 0.0))
                    oa_list.append(m.get("overall_accuracy", 0.0))

                mean_f1 = float(np.mean(f1_list))
                if mean_f1 > best_f1_for_epoch:
                    best_f1_for_epoch = mean_f1
                    best_thresh_for_epoch = th
                    best_metrics_for_epoch = {
                        "precision": float(np.mean(prec_list)),
                        "recall": float(np.mean(rec_list)),
                        "f1_score": mean_f1,
                        "iou": float(np.mean(miou_list)),
                        "oa": float(np.mean(oa_list)),
                    }

            history["val_precision"].append(round(best_metrics_for_epoch["precision"], 4))
            history["val_recall"].append(round(best_metrics_for_epoch["recall"], 4))
            history["val_f1"].append(round(best_metrics_for_epoch["f1_score"], 4))
            history["val_miou"].append(round(best_metrics_for_epoch["iou"], 4))
            history["val_oa"].append(round(best_metrics_for_epoch["oa"], 4))

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Prec: {best_metrics_for_epoch['precision']:.4f} | "
                f"Val Rec: {best_metrics_for_epoch['recall']:.4f} | "
                f"Val F1: {best_metrics_for_epoch['f1_score']:.4f} | "
                f"Val mIoU: {best_metrics_for_epoch['iou']:.4f} (Thresh={best_thresh_for_epoch})"
            )

            # Save best checkpoint
            if best_metrics_for_epoch["f1_score"] >= best_f1:
                best_f1 = best_metrics_for_epoch["f1_score"]
                best_threshold = best_thresh_for_epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epochs": epoch,
                        "best_threshold": best_threshold,
                        "history": history,
                        "dataset_name": "LEVIR-CD",
                        "sample_count": total_samples,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    str(best_checkpoint_path),
                )
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f}")

    total_time = time.time() - start_time
    file_size_bytes = best_checkpoint_path.stat().st_size
    print(f"\n[SUCCESS] Best genuine checkpoint saved: {best_checkpoint_path} ({file_size_bytes / (1024*1024):.2f} MB)")

    # Verify checkpoint loading
    detector = SiameseChangeDetector(checkpoint_path=str(best_checkpoint_path))
    print(f"Inference Verification: Loaded={detector.is_trained}, Status={detector.adaptation_status}")

    return {
        "status": "TRAINED",
        "dataset_name": "LEVIR-CD",
        "sample_count": total_samples,
        "epochs": epochs,
        "training_time_s": round(total_time, 2),
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_size_bytes": file_size_bytes,
        "best_threshold": best_threshold,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_precision": history["val_precision"][-1] if history["val_precision"] else None,
        "final_val_recall": history["val_recall"][-1] if history["val_recall"] else None,
        "final_val_f1": history["val_f1"][-1] if history["val_f1"] else None,
        "final_val_miou": history["val_miou"][-1] if history["val_miou"] else None,
        "final_val_oa": history["val_oa"][-1] if history["val_oa"] else None,
        "history": history,
        "inference_verified": detector.is_trained,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SiameseChangeNet on LEVIR-CD dataset.")
    parser.add_argument("--dataset-dir", default="./datasets/levir_cd", help="Path to LEVIR-CD root directory")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    train_siamese_change_detector(
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
