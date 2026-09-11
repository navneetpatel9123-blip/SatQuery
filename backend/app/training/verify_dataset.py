"""
SatQuery AI — Dataset Verification Tool
Validates BigEarthNet / Remote Sensing dataset directory structure and integrity.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from app.training.dataset import BIGEARTHNET_19_CLASSES


def verify_dataset(
    dataset_dir: Optional[str] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Dict[str, Any]:
    """
    Inspects and validates the BigEarthNet dataset directory.
    Returns structured metrics or cleanly reports DATASET_NOT_AVAILABLE.
    """
    if not dataset_dir:
        dataset_dir = "./datasets/bigearthnet"

    data_path = Path(dataset_dir)
    
    if not data_path.exists() or not data_path.is_dir():
        return {
            "dataset_found": False,
            "status": "DATASET_NOT_AVAILABLE",
            "message": f"Dataset directory '{dataset_dir}' does not exist.",
            "image_count": 0,
            "metadata_count": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "missing_files": [],
            "class_distribution": {},
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0
        }

    # Discover sample subdirectories
    patch_dirs = [p for p in data_path.iterdir() if p.is_dir()]
    if not patch_dirs:
        return {
            "dataset_found": False,
            "status": "DATASET_NOT_AVAILABLE",
            "message": f"Dataset directory '{dataset_dir}' contains 0 sample patch folders.",
            "image_count": 0,
            "metadata_count": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "missing_files": [],
            "class_distribution": {},
            "train_count": 0,
            "validation_count": 0,
            "test_count": 0
        }

    valid_samples = 0
    invalid_samples = 0
    missing_files = []
    class_distribution = {c: 0 for c in BIGEARTHNET_19_CLASSES}
    total_images = 0
    total_metadata = 0

    for patch_dir in patch_dirs:
        # Check metadata JSON
        json_files = list(patch_dir.glob("*_labels_metadata.json")) + list(patch_dir.glob("*.json"))
        tif_files = list(patch_dir.glob("*.tif")) + list(patch_dir.glob("*.tiff"))

        has_meta = len(json_files) > 0
        has_tif = len(tif_files) > 0

        total_metadata += len(json_files)
        total_images += len(tif_files)

        if not has_meta or not has_tif:
            invalid_samples += 1
            missing_files.append(str(patch_dir.name))
            continue

        try:
            with open(json_files[0], "r") as f:
                meta = json.load(f)
            labels = meta.get("labels", [])
            for lbl in labels:
                if lbl in class_distribution:
                    class_distribution[lbl] += 1
            valid_samples += 1
        except Exception:
            invalid_samples += 1
            missing_files.append(f"{patch_dir.name} (corrupt metadata)")

    # Calculate splits
    total_valid = valid_samples
    train_count = int(total_valid * train_split)
    val_count = int(total_valid * val_split)
    test_count = total_valid - (train_count + val_count)

    return {
        "dataset_found": True,
        "status": "READY" if valid_samples > 0 else "DATASET_NOT_AVAILABLE",
        "message": f"Dataset verified: {valid_samples} valid patches found.",
        "image_count": total_images,
        "metadata_count": total_metadata,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "missing_files": missing_files[:20],
        "class_distribution": class_distribution,
        "train_count": train_count,
        "validation_count": val_count,
        "test_count": test_count
    }


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Dataset Verification Tool")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/bigearthnet", help="Path to dataset root")
    args = parser.parse_args()

    print("\n=======================================================")
    print(" SATQUERY AI — REMOTE SENSING DATASET VERIFICATION ")
    print("=======================================================")
    print(f"Target Directory: {args.dataset_dir}")
    
    report = verify_dataset(args.dataset_dir)
    print(f"Dataset Found:        {report['dataset_found']}")
    print(f"Status:               {report['status']}")
    print(f"Message:              {report['message']}")
    print(f"Valid Samples:        {report['valid_samples']}")
    print(f"Invalid Samples:      {report['invalid_samples']}")
    print(f"Total GeoTIFF Images: {report['image_count']}")
    print(f"Total Metadata Files: {report['metadata_count']}")
    print(f"Train Split Count:    {report['train_count']}")
    print(f"Val Split Count:      {report['validation_count']}")
    print(f"Test Split Count:     {report['test_count']}")
    print("=======================================================\n")

    if not report["dataset_found"] or report["valid_samples"] == 0:
        print("RESULT: DATASET_NOT_AVAILABLE")
        sys.exit(0)
    else:
        print("RESULT: DATASET_VERIFIED_AND_READY")
        sys.exit(0)


if __name__ == "__main__":
    main()
