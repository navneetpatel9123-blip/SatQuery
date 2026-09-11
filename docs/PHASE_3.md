# Phase 3: Remote-Sensing VQA + Scene Captioning

This document outlines the design, implementation, and capabilities of the Remote-Sensing Visual Question Answering & Scene Description/Captioning modules (Phase 3) for the SatQuery AI platform.

---

## 1. Unified Router Architecture

The core routing agent operates as a multi-stage parser and controller:

```
                  USER QUERY
                       |
                       v
              Query Normalization (AgentService.normalize_query)
                       |
                       v
             Pronoun Context Resolution (Using Turns History)
                       |
                       v
              Task Classification (AgentService.classify_task)
                       |
             +---------+---------+
             |         |         |
             v         v         v
         SINGLE_VQA CAPTIONING GROUNDING
             |         |         |
             +---------+---------+
                       |
                       v
         Model Registry Selection (Route constraints validation)
                       |
                       v
         BaseModelAdapter execution (load / validate_input / predict)
                       |
                       v
                Evidence Engine (Spatial regions or pixel ratios)
                       |
                       v
         Confidence Categorization (HIGH / MEDIUM / LOW)
                       |
                       v
        Trace DAG Timeline synthesis (TraceNode steps runtime logging)
                       |
                       v
                Standardized API Response (AnalysisResponse)
```

---

## 2. Model Registry & Interfaces

The Model Registry ([`model_registry.py`](file:///d:/satQUREY%20AI%20PROJECT/backend/app/services/model_registry.py)) lazy-loads and exposes remote-sensing specialized networks through clean abstract adapter boundaries.

### BaseModelAdapter Lifecycle Interface
All model wrappers must inherit from `BaseModelAdapter` and implement:
*   `load()`: Load weights into GPU/RAM.
*   `unload()`: Release resources and set status to STANDBY.
*   `health_check()`: Verify readiness.
*   `validate_input(asset)`: Check modality constraints.
*   `predict(asset, query)`: Run raw inference.
*   `postprocess(prediction)`: Format to output schemas.
*   `get_metadata()`: Retrieve registry descriptor.

### Specialty Model Implementations
1.  **`rs_vqa_adapted`**: Adapted ResNet50 baseline trained on BigEarthNet-19 channels. Performs windowed spectral scans to identify green canopy ratios or water body absorption signatures on raw GeoTIFF grids.
2.  **`demo_fallback`**: Emits spatial bounding regions, structured multi-key scene captions, and context validation logs for offline demo scenarios.
3.  **`base_vlm`**: General baseline VLM providing standard optical checkups.

---

## 3. Conversational Multi-turn Chat Experience

Multi-turn context is managed safely by keeping conversation memory states mapped by keys: `(image_id, session_id)`.
*   **Context Safety:** Because maps are indexed by the unique UUID of the ingested `RasterAsset`, context from one image will never leak into another.
*   **Resolution:** Heuristic resolving parses relative follow-up queries (e.g. `"Where is it?"`, `"Highlight it"`) against previous turns to translate semantic targets (e.g. `"water"`, `"road"`) into grounding actions contextually.

---

## 4. Benchmark Adapter Interfaces

The common abstract interface `BenchmarkAdapter` ([`benchmark_adapter.py`](file:///d:/satQUREY%20AI%20PROJECT/benchmarks/benchmark_adapter.py)) guarantees decoupling evaluations from frontend state.

```python
class BenchmarkAdapter(ABC):
    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def prepare_sample(self, sample) -> Any: ...
    @abstractmethod
    def run_inference(self, sample, model) -> Any: ...
    @abstractmethod
    def format_prediction(self, prediction) -> Any: ...
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]: ...
```

---

## 5. Verification

Run automated test coverage:
```bash
python -m pytest backend/tests/test_analysis.py
```
Run benchmark evaluators:
```bash
python benchmarks/rsvqa/evaluate.py
python benchmarks/vrsbench/evaluate.py
```
