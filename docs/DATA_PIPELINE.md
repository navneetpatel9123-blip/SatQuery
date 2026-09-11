# Data Pipeline Overview

The SatQuery AI geospatial data pipeline moves imagery from raw user uploads to standardized raster assets, ready to feed downstream remote sensing AI agents.

```
User Upload (GeoTIFF, PNG, JPEG)
       ↓
`RasterIngestionService` Secure Storage
       ↓
`rasterio` Metadata Parser
       ↓
Ingestion Validation Checks (PASS / WARNING / FAIL)
       ↓
`generate_preview` (Aspect-ratio downsampled PNG)
       ↓
Standardized `RasterAsset` State & `asset.json` Persistence
       ↓
`AgentService` Router (VQA / Caption / Grounding Specialist Models)
```

## Data Stages

### 1. Upload & Secure Storage
Uploaded files are stored at `backend/data/uploads/{asset_id}/{filename}` using UUIDs to prevent file collisions. Files are written in chunks of 1MB to prevent server RAM saturation.

### 2. Raster Parsing & Metadata Extraction
Using `rasterio`, the pipeline extracts:
- CRS (Coordinate Reference System) and EPSG codes.
- Geospatial bounds (West, South, East, North).
- Pixel resolution and Affine coordinates transformation matrix.
- Datetime timestamp from `TIFFTAG_DATETIME`.
- Number of spectral bands and datatype (e.g. `uint8`, `float32`).

### 3. Ingestion Validation
Checks for CRS availability, coordinate bounds validity, file size limits (500MB), and readable headers. The engine returns one of three validation states:
- `PASS`: Geospatial data fully populated.
- `WARNING`: Readable, but lacking CRS/georeferencing coordinates (e.g., standard PNG/JPEG benchmark uploads).
- `FAIL`: File corrupted or unreadable.

### 4. Preview Extraction
Generates a contrast-stretched PNG preview limited to a max dimension of 512px, protecting the web client from memory crashes when viewing massive satellite scenes.

### 5. AI Ingestion Endpoint Consumption
The standard `RasterAsset` model is passed to downstream specialist agents for question-answering, scene description, and grounding localization.
