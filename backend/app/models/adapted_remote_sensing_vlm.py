"""
SatQuery AI — Adapted Remote Sensing Vision-Language Model
Implementation of the domain-adapted Remote Sensing VLM component.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import rasterio

from app.schemas import RasterAsset, VQAOutput, CaptionOutput
from app.models.base_vlm import BaseRemoteSensingVisionModel
from app.training.preprocessing import normalize_raster_bands, extract_rgb_nir
from app.training.dataset import BIGEARTHNET_19_CLASSES

try:
    import torch
    from app.training.model import RemoteSensingAdaptedVLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AdaptedRemoteSensingModel(BaseRemoteSensingVisionModel):
    """
    Genuine Remote-Sensing Adapted Vision-Language Model.
    
    Status Lifecycle:
    - 'TRAINING_READY': Architecture and dataset pipeline exist, but no trained checkpoint is loaded.
    - 'TRAINED': A genuine PyTorch checkpoint has been trained on BigEarthNet and loaded.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self._checkpoint_path = checkpoint_path
        self._is_loaded = False
        self._model = None
        self._classes = BIGEARTHNET_19_CLASSES

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

    @property
    def model_name(self) -> str:
        return "AdaptedRemoteSensingVLM"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def is_trained(self) -> bool:
        """True only if an actual trained checkpoint has been loaded."""
        return self._is_loaded

    @property
    def adaptation_status(self) -> str:
        return "TRAINED" if self._is_loaded else "TRAINING_READY"

    def load_checkpoint(self, checkpoint_path: str):
        """Load genuine weights from a PyTorch checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path}'. "
                f"REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED."
            )

        if not HAS_TORCH:
            raise ImportError("PyTorch is required to load AdaptedRemoteSensingModel weights.")

        checkpoint = torch.load(str(path), map_location="cpu")
        self._model = RemoteSensingAdaptedVLM(in_channels=4, embed_dim=512, num_classes=len(self._classes))
        if "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self._model.load_state_dict(checkpoint)
        self._model.eval()
        self._is_loaded = True
        self._checkpoint_path = str(path)

    def _read_raster_tensor(self, asset: RasterAsset):
        with rasterio.open(asset.path) as src:
            pixels = src.read()
        norm = normalize_raster_bands(pixels, target_size=(120, 120))
        rgb_nir = extract_rgb_nir(norm)
        if HAS_TORCH:
            tensor = torch.from_numpy(rgb_nir).float().unsqueeze(0)
            return tensor, pixels
        return rgb_nir, pixels

    def encode_image(self, asset: RasterAsset) -> Any:
        """Extract adapted visual features."""
        if not self._is_loaded or not HAS_TORCH:
            raise RuntimeError(
                "AdaptedRemoteSensingModel is in 'TRAINING_READY' state. "
                "No trained checkpoint is loaded. Use RuleBasedRemoteSensingVQA/Captioner fallback."
            )
        tensor, _ = self._read_raster_tensor(asset)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        return features.squeeze(0).numpy()

    def generate_caption(self, asset: RasterAsset) -> CaptionOutput:
        """Generate scene caption grounded in model predictions."""
        if not self._is_loaded or not HAS_TORCH:
            raise RuntimeError(
                "AdaptedRemoteSensingModel is in 'TRAINING_READY' state. "
                "No trained checkpoint is loaded. Use RuleBasedRemoteSensingCaptioner fallback."
            )

        tensor, raw_pixels = self._read_raster_tensor(asset)
        with torch.no_grad():
            out = self._model(tensor)
            probs = out["probabilities"].squeeze(0).numpy()

        # Find top predicted classes
        top_indices = np.where(probs > 0.3)[0]
        if len(top_indices) == 0:
            top_indices = [np.argmax(probs)]

        detected_labels = [self._classes[idx] for idx in top_indices]
        top_prob = float(np.max(probs))

        caption = f"Satellite image depicting {', '.join(detected_labels)} with adapted remote-sensing visual features."
        return CaptionOutput(
            caption=caption,
            confidence=top_prob,
            confidence_type="MODEL_SCORE",
            adaptation_status="TRAINED",
            evidence={
                "detected_classes": detected_labels,
                "model_probabilities": {self._classes[i]: float(probs[i]) for i in top_indices},
                "modality": asset.modality
            },
            model=self.model_name,
            warnings=[]
        )

    def answer_question(self, asset: RasterAsset, question: str) -> VQAOutput:
        """Answer question using model predictions."""
        if not self._is_loaded or not HAS_TORCH:
            raise RuntimeError(
                "AdaptedRemoteSensingModel is in 'TRAINING_READY' state. "
                "No trained checkpoint is loaded. Use RuleBasedRemoteSensingVQA fallback."
            )

        tensor, raw_pixels = self._read_raster_tensor(asset)
        with torch.no_grad():
            out = self._model(tensor)
            probs = out["probabilities"].squeeze(0).numpy()

        q = question.lower()
        # Query class match
        matched_class = None
        matched_prob = 0.0
        for i, cls_name in enumerate(self._classes):
            if any(term in q for term in cls_name.lower().split()):
                matched_class = cls_name
                matched_prob = float(probs[i])
                break

        if matched_class:
            present = matched_prob > 0.5
            ans = f"Yes, {matched_class} is present (score: {matched_prob:.2f})." if present else f"No, {matched_class} is unlikely (score: {matched_prob:.2f})."
            conf = matched_prob if present else (1.0 - matched_prob)
        else:
            top_idx = int(np.argmax(probs))
            ans = f"The primary land cover detected is {self._classes[top_idx]}."
            conf = float(probs[top_idx])

        return VQAOutput(
            answer=ans,
            confidence=conf,
            confidence_type="MODEL_SCORE",
            adaptation_status="TRAINED",
            evidence={
                "class_probabilities": {self._classes[i]: float(probs[i]) for i in range(len(self._classes)) if probs[i] > 0.2},
                "modality": asset.modality
            },
            model=self.model_name,
            warnings=[]
        )
