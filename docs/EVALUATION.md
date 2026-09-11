# Remote Sensing Adaptation & Evaluation Pipeline

SatQuery AI provides a reproducible pipeline to adapt vision models for remote sensing classification and evaluate them against open-source benchmarks.

## 1. Remote Sensing Adaptation Pipeline

Located in the `training/` directory:
- **`configs/bigearthnet_config.yaml`**: Holds configurations for ResNet/ViT baselines, learning rates, epochs, and BigEarthNet dataset channels.
- **`train.py`**: reproducible PyTorch training harness.
  - Loads a custom ResNet-50 structure.
  - Modifies the initial convolution layer to accept 4 spectral bands (RGB + NIR) instead of standard 3-band RGB.
  - Fine-tunes multilabel outputs targeting BigEarthNet classification tags.
  - Saves model checkpoints.

Run training simulation:
```bash
python training/train.py --epochs 5 --demo
```

Run PyTorch training on CUDA (requires torch/torchvision):
```bash
python training/train.py --config ./training/configs/bigearthnet_config.yaml --epochs 10
```

Run single GeoTIFF classification inference:
```bash
python training/infer.py --image path/to/your/image.tif
```

## 2. Evaluation Benchmark Adapters

To test model VQA and Captioning performance, SatQuery supports adapters under `benchmarks/` which inherit from the common interface **`BenchmarkAdapter`**:

```python
class BenchmarkAdapter(ABC):
    def load_dataset(self) -> List[Dict[str, Any]]: ...
    def prepare_sample(self, sample) -> Any: ...
    def run_inference(self, sample, model) -> Any: ...
    def format_prediction(self, prediction) -> Any: ...
    def evaluate(self) -> Dict[str, Any]: ...
```

### RSVQA Benchmark Evaluation
Located under `benchmarks/rsvqa/evaluate.py`:
*   Inherits from `BenchmarkAdapter` and overrides VQA sample inference.
*   Evaluates class-wise accuracy metrics (Presence questions, Counting questions).

Run RSVQA evaluation:
```bash
python benchmarks/rsvqa/evaluate.py
```

### VRSBench Benchmark Evaluation
Located under `benchmarks/vrsbench/evaluate.py`:
*   Inherits from `BenchmarkAdapter` and overrides scene-caption evaluation.
*   Computes BLEU and accuracy metrics by comparing VQA answers against ground-truth descriptions.

Run VRSBench evaluation:
```bash
python benchmarks/vrsbench/evaluate.py
```
