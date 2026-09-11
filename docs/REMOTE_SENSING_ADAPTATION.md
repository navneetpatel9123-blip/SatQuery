# Remote-Sensing VLM Model Adaptation

**Status**: 🟢 `TRAINING_READY` (Dataset & Pipeline Complete, Real Weights Pending Dataset Download)

## 1. Requirement & Rationale
Under the **SIH26167** problem statement, remote-sensing vision components must provide a genuine domain adaptation path on open satellite datasets (such as **BigEarthNet**) rather than relying solely on generic vision models or un-adapted prompts.

To ensure **strict scientific honesty**:
- Rule-based baselines (`RuleBasedRemoteSensingVQA`, `RuleBasedRemoteSensingCaptioner`) are maintained as transparent, deterministic fallbacks.
- The adapted architecture (`AdaptedRemoteSensingModel`) is implemented using Parameter-Efficient Fine-Tuning (PEFT).
- The training pipeline (`backend/app/training/train.py`) and dataset adapter (`backend/app/training/dataset.py`) expect official BigEarthNet directory structures and fail loudly (`REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED`) if data is absent.
- **No fake training metrics, fake loss curves, or fake `.pth` weight files are fabricated.**

---

## 2. Model Architecture & PEFT Adaptation
The `RemoteSensingAdaptedVLM` architecture in `backend/app/training/model.py` comprises:
1. **Multispectral Input Layer**: Accepts 4-channel (RGB + NIR) or 12-channel Sentinel-2 arrays.
2. **Visual Encoder Trunk**: Deep convolutional / residual trunk extracting high-level spatial and spectral feature maps.
3. **PEFT Domain Adapter (`RSAdapterLayer`)**:
   - Bottleneck projection down to dimension 128 with GELU activation and LayerNorm.
   - Scaled residual connection: $x_{\text{out}} = \text{LayerNorm}(x + \alpha \cdot \text{Adapter}(x))$.
   - Enables fine-tuning specifically on satellite spectral patterns without catastrophic forgetting.
4. **Task Heads**:
   - Multi-label classification head for **BigEarthNet-19** classes (CORINE Land Cover nomenclature).
   - VQA and text representation embedding projection head.

---

## 3. Dataset Adapter
`backend/app/training/dataset.py` provides:
- **`BigEarthNetDataset`**:
  - Reads individual patch folders containing `*_labels_metadata.json` and band GeoTIFFs (`*.tif`).
  - Converts labels into 19-class multi-hot vectors.
  - Automatically normalizes raster reflectance and resizes to standard patch dimensions (120x120).
- **`RSVQADatasetAdapter` & `VRSBenchDatasetAdapter`**:
  - Structured adapters for Remote Sensing VQA and Vision-Language benchmarks.
  - If benchmark files are not mounted locally, reports `NOT RUN — DATASET NOT AVAILABLE`.

---

## 4. Reproducible Training Execution

### Prerequisites
Install PyTorch in your Python environment:
```bash
pip install torch torchvision
```

### Running Domain Adaptation
To train the adapted model on a local BigEarthNet Sentinel-2 directory:
```bash
python -m app.training.train \
    --dataset_dir /path/to/BigEarthNet-v1.0 \
    --checkpoint_dir ./checkpoints \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4
```

If the dataset path does not exist, the training harness immediately halts with:
```
ERROR: REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED.
BigEarthNet dataset not found at '/path/to/BigEarthNet-v1.0'.
```

---

## 5. Checkpoint Format & Loading
The trained checkpoint is saved as a PyTorch state dictionary containing:
```python
{
    "epoch": 10,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "num_classes": 19,
    "classes": BIGEARTHNET_19_CLASSES,
    "training_dataset": "BigEarthNet-19",
    "status": "TRAINED"
}
```
When instantiated with a valid checkpoint path:
```python
model = AdaptedRemoteSensingModel(checkpoint_path="./checkpoints/rs_adapted_vlm.pth")
assert model.is_trained == True
assert model.adaptation_status == "TRAINED"
```

---

## 6. Orchestrator Integration & Transparent Fallback
When a user asks `SINGLE_IMAGE_VQA` or `SINGLE_IMAGE_CAPTION`:
1. `PipelineOrchestrator` checks `AdaptedRemoteSensingModel.is_trained`.
2. **If Trained Weights Exist**: Executes `AdaptedRemoteSensingModel`, sets `confidence_type = "MODEL_SCORE"`, and records `REMOTE_SENSING_MODEL_SELECTED` in the execution trace.
3. **If Checkpoint Unavailable (Current State)**:
   - Emits trace event: `ADAPTED_MODEL_UNAVAILABLE` (details: `status="TRAINING_READY"`).
   - Emits trace event: `RULE_BASED_FALLBACK_SELECTED` (details: `fallback_model="remote_sensing_vqa"`).
   - Seamlessly delegates to `RuleBasedRemoteSensingVQA` / `RuleBasedRemoteSensingCaptioner`.

---

## 7. Current Training & Benchmark Status Summary

| Item | Current Status | Notes |
|---|---|---|
| **Pipeline & Adapters** | 🟢 `IMPLEMENTED` | Full PyTorch training loop, dataset parser, and PEFT adapter ready. |
| **Model Registry** | 🟢 `TRAINING_READY` | Registered under `adapted_remote_sensing_vlm`. |
| **Actual Weights** | 🟡 `TRAINING_READY` | Real training not yet executed; BigEarthNet dataset download required. |
| **VQA / Captioning** | 🟢 `RULE_BASED_FALLBACK` | Active and fully operational with transparent trace logging. |
| **VRSBench / RSVQA** | 🟡 `NOT RUN` | Dataset not available locally; adapters implemented. |
