from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BenchmarkAdapter(ABC):
    """Abstract Base Class for all SatQuery evaluation benchmark adapters."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load annotation details, questions/images pairs, and ground truths from dataset files."""
        pass

    @abstractmethod
    def prepare_sample(self, sample: Dict[str, Any]) -> Any:
        """Prepare individual evaluation sample before passing it to inference."""
        pass

    @abstractmethod
    def run_inference(self, sample: Any, model: Any) -> Any:
        """Execute model VQA/Caption/Grounding inference against a sample."""
        pass

    @abstractmethod
    def format_prediction(self, prediction: Any) -> Any:
        """Format raw prediction into standard metric-compatible types."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Execute overall evaluation loop across dataset and report aggregated metric scores."""
        pass
