"""
SatQuery AI — Change Detection Dataset & CDVQA Adapters
========================================================

Provides dataset-independent interfaces for bi-temporal change detection and CDVQA:
- `BaseChangeDetectionDataset`: Abstract interface for LEVIR-CD, WHU-CD, OSCD, etc.
- `CDVQADatasetAdapter`: Interface for Change Detection Visual Question Answering.
- `compute_change_metrics`: Standardized evaluation calculator (mIoU, Precision, Recall, F1, OA).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class BaseChangeDetectionDataset(ABC):
    """Abstract dataset loader for bi-temporal change detection."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.root_path = Path(root_dir)
        self.split = split
        self.transform = transform
        self.available = self.root_path.exists() and self.root_path.is_dir()

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
            t1_raster: (C, H, W) numpy array for Pre-change image.
            t2_raster: (C, H, W) numpy array for Post-change image.
            change_mask: (H, W) binary numpy array (0 or 1).
            metadata: dictionary with sample details (sample_id, coords, dates).
        """
        ...


class GenericFolderChangeDataset(BaseChangeDetectionDataset):
    """
    Standard triplet folder dataset loader:
        root_dir/
            A/ (or T1/) -> Pre-change rasters (.tif / .png)
            B/ (or T2/) -> Post-change rasters (.tif / .png)
            label/ (or mask/) -> Binary change masks (.png / .tif)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        super().__init__(root_dir, split, transform)
        self.samples: List[Tuple[Path, Path, Optional[Path]]] = []

        if self.available:
            self._discover_samples()

    def _discover_samples(self):
        # Look for A and B directories
        t1_dir = self.root_path / "A"
        if not t1_dir.exists():
            t1_dir = self.root_path / "T1"
        t2_dir = self.root_path / "B"
        if not t2_dir.exists():
            t2_dir = self.root_path / "T2"
        mask_dir = self.root_path / "label"
        if not mask_dir.exists():
            mask_dir = self.root_path / "mask"

        if t1_dir.exists() and t2_dir.exists():
            t1_files = sorted(list(t1_dir.glob("*.tif")) + list(t1_dir.glob("*.png")))
            for f1 in t1_files:
                f2 = t2_dir / f1.name
                f_mask = mask_dir / f1.name if mask_dir.exists() else None
                if f2.exists():
                    self.samples.append((f1, f2, f_mask if (f_mask and f_mask.exists()) else None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if not self.available or len(self.samples) == 0:
            raise FileNotFoundError(
                f"Change dataset at '{self.root_path}' is not available or contains 0 samples."
            )

        f1, f2, f_mask = self.samples[idx]
        import rasterio
        with rasterio.open(f1) as s1:
            t1 = s1.read()
        with rasterio.open(f2) as s2:
            t2 = s2.read()

        if f_mask:
            with rasterio.open(f_mask) as sm:
                mask = sm.read(1)
                mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((t1.shape[1], t1.shape[2]), dtype=np.uint8)

        meta = {"sample_id": f1.stem, "t1_file": str(f1.name), "t2_file": str(f2.name)}
        return t1, t2, mask, meta


class CDVQADatasetAdapter:
    """
    Dataset adapter for Change Detection Visual Question Answering (CDVQA).
    Pairs bi-temporal rasters with natural language question-answer pairs
    describing semantic change transitions.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.available = self.data_path.exists() and self.data_path.is_dir()
        self.status_message = "READY" if self.available else "NOT RUN — DATASET NOT AVAILABLE"

    def load_qa_pairs(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        qa_file = self.data_path / "cdvqa_annotations.json"
        if not qa_file.exists():
            return []
        with open(qa_file, "r") as f:
            return json.load(f)


def compute_change_metrics(
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard remote-sensing change detection evaluation metrics:
    - Overall Accuracy (OA)
    - Precision
    - Recall
    - F1-Score
    - Intersection over Union (IoU) / mIoU
    """
    pred_bin = (prediction_mask > threshold).astype(bool)
    gt_bin = (ground_truth_mask > 0).astype(bool)

    tp = float(np.sum(pred_bin & gt_bin))
    fp = float(np.sum(pred_bin & ~gt_bin))
    fn = float(np.sum(~pred_bin & gt_bin))
    tn = float(np.sum(~pred_bin & ~gt_bin))

    total = tp + fp + fn + tn
    oa = (tp + tn) / max(total, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)

    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    iou = tp / max(tp + fp + fn, 1.0)

    return {
        "overall_accuracy": round(oa, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
