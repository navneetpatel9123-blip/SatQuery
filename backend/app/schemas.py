"""
Shared Pydantic schemas for SatQuery AI.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional


# ─── Enums ─────────────────────────────────────────────

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    DETECTING = "detecting_changes"
    UNDERSTANDING = "understanding_changes"
    EXTRACTING = "extracting_evidence"
    COMPLETED = "completed"
    FAILED = "failed"


class ChangeType(str, Enum):
    BUILT_UP_EXPANSION = "built_up_expansion"
    DEMOLITION = "demolition"
    VEGETATION_GAIN = "vegetation_gain"
    DEFORESTATION = "deforestation"
    WATER_CHANGE = "water_body_change"
    AGRICULTURAL = "agricultural_conversion"
    INFRASTRUCTURE = "infrastructure_development"
    OTHER = "other"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgreementScore(str, Enum):
    STRONG_AGREEMENT = "STRONG_AGREEMENT"
    MODERATE_AGREEMENT = "MODERATE_AGREEMENT"
    OPTICAL_DOMINANT = "OPTICAL_DOMINANT"
    SAR_DOMINANT = "SAR_DOMINANT"
    CONFLICT = "CONFLICT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ─── Image Metadata ───────────────────────────────────

class GeoTIFFMetadata(BaseModel):
    """Metadata extracted from a GeoTIFF file."""
    filename: str
    crs: str
    bounds: list[float] = Field(description="[west, south, east, north]")
    resolution: list[float] = Field(description="[x_res, y_res] in CRS units")
    width: int
    height: int
    band_count: int
    dtype: str
    capture_date: Optional[datetime] = None
    file_size_mb: float


class ImagePairValidation(BaseModel):
    """Validation results for a GeoTIFF image pair."""
    is_valid: bool
    crs_match: bool
    bounds_overlap_pct: float
    same_aoi: bool
    temporal_order_valid: bool
    resolution_compatible: bool
    errors: list[str] = []
    warnings: list[str] = []


# ─── Analysis Results ──────────────────────────────────

class ChangeRegion(BaseModel):
    """A detected change region."""
    region_id: str
    change_type: ChangeType
    area_sq_m: float
    centroid: list[float] = Field(description="[lon, lat]")
    bbox: list[float] = Field(description="[west, south, east, north]")
    confidence: float
    description: str


class EvidenceItem(BaseModel):
    """A piece of evidence supporting a conclusion."""
    evidence_id: str
    type: str  # spatial, statistical, spectral
    description: str
    value: Optional[str] = None
    confidence: float
    related_regions: list[str] = []


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    analysis_id: str
    status: AnalysisStatus
    task_type: str = "bi_temporal_change_vqa"
    summary: str
    overall_confidence: Confidence
    change_regions: list[ChangeRegion] = []
    evidence: list[EvidenceItem] = []
    statistics: dict = {}
    created_at: datetime
    completed_at: Optional[datetime] = None


# ─── Execution Trace ───────────────────────────────────

# ─── Capability Outputs (VQA/Captioning) ───────────────

class VQAOutput(BaseModel):
    """Output schema for Remote-Sensing VQA compatible with VRSBench/RSVQA."""
    answer: str
    confidence: float
    confidence_type: str = Field(default="HEURISTIC", description="'HEURISTIC' or 'MODEL_SCORE'")
    adaptation_status: Optional[str] = Field(default="RULE_BASED_BASELINE", description="e.g. 'RULE_BASED_BASELINE', 'TRAINING_READY', 'TRAINED'")
    evidence: dict = Field(default_factory=dict, description="Must include 'spectral_features', 'regions', 'metadata' keys.")
    model: str
    warnings: list[str] = []


class CaptionOutput(BaseModel):
    """Output schema for Scene Captioning."""
    caption: str
    confidence: float
    confidence_type: str = Field(default="HEURISTIC", description="'HEURISTIC' or 'MODEL_SCORE'")
    adaptation_status: Optional[str] = Field(default="RULE_BASED_BASELINE", description="e.g. 'RULE_BASED_BASELINE', 'TRAINING_READY', 'TRAINED'")
    evidence: dict = Field(default_factory=dict, description="Grounding and statistical evidence used to form the caption.")
    model: str
    warnings: list[str] = []


class OpticalSARRegion(BaseModel):
    """A region analyzed using both Optical and SAR modalities."""
    region_id: str
    classification: str
    optical_evidence: dict = {}
    sar_evidence: dict = {}
    agreement: dict = Field(default_factory=lambda: {"label": AgreementScore.INSUFFICIENT_DATA, "score": 0.0})
    confidence: float
    explanation: str
    warnings: list[str] = []


class OpticalSAROutput(BaseModel):
    """Output schema for Optical-SAR Cross-Modal Analysis."""
    analysis_type: str = "OPTICAL_SAR"
    status: str
    regions: list[OpticalSARRegion] = []
    summary: dict = {}
    execution_trace: list[dict] = []


class TraceNode(BaseModel):
    """A single node in the execution trace DAG."""
    step_id: str
    model_name: str
    display_name: str
    status: str  # pending, running, success, error
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    inputs_summary: dict = {}
    outputs_summary: dict = {}
    confidence: Optional[float] = None
    artifacts: list[str] = []
    children: list[str] = []  # step_ids of child nodes
    error: Optional[str] = None


class ExecutionTrace(BaseModel):
    """Full execution trace for an analysis run."""
    analysis_id: str
    nodes: list[TraceNode]
    edges: list[dict] = Field(
        default=[],
        description="List of {source, target} step_id pairs"
    )
    total_duration_ms: Optional[float] = None


class RasterAsset(BaseModel):
    """Standardized metadata and validation for a single raster asset."""
    id: str
    filename: str
    path: str
    modality: str  # "optical", "multispectral", "sar", "unknown"
    modality_certain: bool
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    epsg: Optional[int] = None
    bounds: list[float]  # [left, bottom, right, top]
    resolution: list[float]  # [x_res, y_res]
    transform: list[float]  # 6-element affine transform matrix
    nodata: Optional[float] = None
    timestamp: Optional[datetime] = None
    validation_status: str  # "PASS", "WARNING", "FAIL"
    warnings: list[str] = []
    errors: list[str] = []


class PairValidationResult(BaseModel):
    """Validation results for a geospatial image pair."""
    pair_type: str  # "optical_sar", "t1_t2", "unknown"
    compatible: bool
    spatial_overlap: float  # percentage of overlap
    crs_compatible: bool
    resolution_compatible: bool
    temporal_valid: bool
    coregistration_status: str  # "PASS", "WARNING", "FAIL", "NOT_VERIFIED"
    warnings: list[str] = []
    errors: list[str] = []


class ModelInfo(BaseModel):
    """Registry metadata for a remote sensing model."""
    model_id: str
    name: str  # Kept for backward compatibility
    model_name: str
    version: str
    task: str  # "REMOTE_SENSING_VQA", "REMOTE_SENSING_CAPTIONER", "GROUNDING_MODEL", etc.
    supported_modalities: list[str]  # ["optical", "sar", etc.]
    input_format: str
    checkpoint: Optional[str] = None
    framework: Optional[str] = None
    device_requirements: str
    remote_sensing_adapted: bool
    training_dataset: Optional[str] = None
    status: str  # "LOADED", "STANDBY", "UNAVAILABLE"


class EvidenceResult(BaseModel):
    """Quantified visual or metadata evidence for an AI response."""
    type: str  # "spatial", "textual", "statistical", "metadata"
    content: str
    bbox: Optional[list[float]] = None  # [left, bottom, right, top] in CRS or normalized units
    mask: Optional[list[list[float]]] = None  # polygons or bounding box paths
    certainty: str  # "OBSERVED", "INFERRED", "UNCERTAIN"
    score: float


class AnalysisResponse(BaseModel):
    """Standardized response payload for all SatQuery AI analysis queries."""
    analysis_id: str
    task: str  # "SINGLE_VQA", "CAPTIONING", "GROUNDING"
    answer: str
    evidence: list[EvidenceResult] = []
    confidence: dict = {}  # e.g., {"level": "HIGH", "score": 0.95, "type": "Model confidence"}
    model: ModelInfo
    warnings: list[str] = []
    execution_trace: list[TraceNode] = []


