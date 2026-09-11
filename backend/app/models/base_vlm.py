from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.schemas import RasterAsset, VQAOutput, CaptionOutput

class BaseRemoteSensingVisionModel(ABC):
    """
    Unified pluggable interface for Remote-Sensing Vision & Vision-Language Models.
    
    Supports:
    - Feature encoding / visual representation
    - Remote-sensing scene captioning
    - Remote-sensing visual question answering
    
    Both adapted deep models (e.g. AdaptedRemoteSensingModel) and rule-based baselines
    (e.g. RuleBasedRemoteSensingVQA/Captioner) adhere to or interoperate with this contract.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the vision model architecture."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string of the model/weights."""
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """True only if an actual trained/fine-tuned checkpoint is loaded."""
        pass

    @property
    @abstractmethod
    def adaptation_status(self) -> str:
        """One of: 'TRAINED', 'TRAINING_READY', 'RULE_BASED_BASELINE', 'NOT_TRAINED'."""
        pass

    @abstractmethod
    def encode_image(self, asset: RasterAsset) -> Any:
        """Extract visual representation/features from the remote-sensing asset."""
        pass

    @abstractmethod
    def generate_caption(self, asset: RasterAsset) -> CaptionOutput:
        """Generate a grounded, descriptive caption of the remote sensing asset."""
        pass

    @abstractmethod
    def answer_question(self, asset: RasterAsset, question: str) -> VQAOutput:
        """Answer a natural language question grounded in the remote sensing asset."""
        pass
