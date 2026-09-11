# Model Registry & Adapters

The SatQuery AI Model Registry provides a unified interface to load, query, and manage remote-sensing vision models.

## Registered Models

| Model ID | Model Name | Checkpoint | Framework | Modalities | RS Adapted | Dataset | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rs_vqa_adapted` | SatQuery VQA Adapted ResNet50 | `resnet50_bigearthnet.pth` | PyTorch | `["optical", "multispectral"]` | True | BigEarthNet-19 | STANDBY |
| `base_vlm` | Generic VLM Baseline | `vlm_base_llama.bin` | HuggingFace | `["optical", "multispectral", "sar"]` | False | LAION-5B / COCO | STANDBY |
| `demo_fallback` | SatQuery Demo Adapter | `demo_weights_v1.bin` | PyTorch | `["optical", "multispectral", "sar", "unknown"]` | True | SatQuery Demo Set | LOADED |

## Model Adapters Interface

To integrate new models into the platform, inherit from the corresponding abstract adapter base class in `backend/app/services/model_registry.py`:

### `BaseModelAdapter`
Declares lazy loading and standard lifecycle methods:
*   `load(self)`: Lazily load weights/checkpoints.
*   `unload(self)`: Unload weights to clear VRAM/RAM.
*   `health_check(self) -> bool`: Validate operational status.
*   `validate_input(self, asset: RasterAsset) -> bool`: Validate asset modality constraints.
*   `predict(self, asset: RasterAsset, query: str) -> Any`: Run raw inference prediction.
*   `postprocess(self, prediction: Any) -> Any`: Standardize outputs.

### `VQAModelAdapter`
Responsible for visual question answering.
```python
def answer_question(self, asset: RasterAsset, question: str) -> Tuple[str, float, List[EvidenceResult], List[str]]:
    # Custom model execution logic
    pass
```

### `CaptionModelAdapter`
Responsible for structured scene description. Output must map to the keys:
- `SCENE OVERVIEW`
- `LAND COVER`
- `MAJOR OBJECTS`
- `SPATIAL OBSERVATIONS`
- `UNCERTAINTIES`

### `GroundingModelAdapter`
Responsible for spatial bounding box or segmentation localization matching text query inputs.
