# SatQuery AI

> Bi-Temporal Satellite Image Change Detection & VQA Platform

A full-stack geospatial AI system that ingests co-registered GeoTIFF image pairs, runs a multi-model pipeline (Change Detection → Change Understanding → Evidence Extraction), and presents results through a premium dashboard with full execution-trace explainability.

## Project Structure

- **Phase 2**: GeoTIFF Ingestion & Validation Engine
- **Phase 3**: Remote-Sensing VQA + Scene Captioning

## Quick Start

### 1. Install Backend Dependencies
Navigate to the `backend` folder and install packages in editable mode:
```bash
cd backend
python -m pip install -e ".[dev]"
```

### 2. Start Uvicorn Backend Server
```bash
python -m uvicorn app.main:app --reload --port 8000
```
FastAPI documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Start Next.js Frontend Server
Navigate to the `frontend` folder and run the Next.js dev server:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000/upload](http://localhost:3000/upload) in your browser to launch the mission control workspace.

## Remote-Sensing Training & Benchmarks

### Adaptation Pipeline (BigEarthNet)
Train a ResNet-50 baseline adapted for multispectral remote sensing bands:
```bash
python training/train.py --epochs 5 --demo
```

Run GeoTIFF inference:
```bash
python training/infer.py --image path/to/your/image.tif
```

### Benchmark Evaluations
Run evaluation metrics against RSVQA and VRSBench benchmark adapters:
```bash
python benchmarks/rsvqa/evaluate.py
python benchmarks/vrsbench/evaluate.py
```

## Documentation

Comprehensive architecture details, agent schemas, and workflows are available under the `/docs` folder:
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Data Pipeline Details](docs/DATA_PIPELINE.md)
- [Model Registry Adapters](docs/MODEL_REGISTRY.md)
- [Agent Workflows](docs/AGENT_WORKFLOW.md)
- [Remote Sensing Adaptation & Metrics](docs/EVALUATION.md)
