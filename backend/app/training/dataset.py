"""
SatQuery AI — Remote Sensing Dataset Adapters
Official dataset interfaces for BigEarthNet, RSVQA, and VRSBench benchmarks.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# BigEarthNet 19-class standard nomenclature (CORINE Land Cover mapping)
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]

class BigEarthNetDataset:
    """
    Genuine PyTorch/NumPy dataset loader for BigEarthNet Sentinel-2 / Sentinel-1.
    
    Expected Structure:
        root_dir/
            patch_name_1/
                patch_name_1_labels_metadata.json
                patch_name_1_B01.tif (or single multi-band .tif)
                ...
            patch_name_2/
                ...
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Any] = None,
        classes: List[str] = BIGEARTHNET_19_CLASSES
    ):
        self.root_path = Path(root_dir)
        self.split = split
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Strictly check dataset presence — never fabricate
        if not self.root_path.exists() or not self.root_path.is_dir():
            raise FileNotFoundError(
                f"BigEarthNet dataset directory not found at '{self.root_path}'. "
                f"REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED."
            )
            
        # Discover patch directories
        self.patches = [p for p in self.root_path.iterdir() if p.is_dir()]
        if len(self.patches) == 0:
            raise FileNotFoundError(
                f"No patch directories found in BigEarthNet directory '{self.root_path}'. "
                f"REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED."
            )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        patch_dir = self.patches[idx]
        json_files = list(patch_dir.glob("*_labels_metadata.json")) + list(patch_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"Missing metadata JSON in patch {patch_dir}")
            
        with open(json_files[0], "r") as f:
            metadata = json.load(f)
            
        labels = metadata.get("labels", [])
        # Create multi-hot label vector
        target = np.zeros(len(self.classes), dtype=np.float32)
        for lbl in labels:
            if lbl in self.class_to_idx:
                target[self.class_to_idx[lbl]] = 1.0
                
        # Load image bands
        import rasterio
        tif_files = sorted(list(patch_dir.glob("*.tif")) + list(patch_dir.glob("*.tiff")))
        if not tif_files:
            raise FileNotFoundError(f"Missing GeoTIFF band files in patch {patch_dir}")
            
        if len(tif_files) == 1:
            with rasterio.open(tif_files[0]) as src:
                img_data = src.read()
        else:
            bands = []
            for tf in tif_files:
                with rasterio.open(tf) as src:
                    bands.append(src.read(1))
            img_data = np.stack(bands, axis=0)
            
        if self.transform:
            img_data = self.transform(img_data)
            
        return img_data, target


class RSVQADatasetAdapter:
    """
    Dataset adapter for Remote Sensing Visual Question Answering (RSVQA / RSVQAxBEN).
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.available = False
            self.status_message = "NOT RUN — DATASET NOT AVAILABLE"
        else:
            self.available = True
            self.status_message = "READY"

    def load_questions(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        with open(self.data_path / "questions.json", "r") as f:
            return json.load(f)


class VRSBenchDatasetAdapter:
    """
    Dataset adapter for VRSBench (Vision-Language Remote Sensing Benchmark).
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.available = False
            self.status_message = "NOT RUN — DATASET NOT AVAILABLE"
        else:
            self.available = True
            self.status_message = "READY"

    def load_annotations(self) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        with open(self.data_path / "annotations.json", "r") as f:
            return json.load(f)
