"use client";

import { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import styles from "./page.module.css";

const GeospatialMap = dynamic(() => import("./GeospatialMap"), { ssr: false });

interface RasterAsset {
  id: string;
  filename: string;
  path: string;
  modality: string;
  modality_certain: boolean;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  crs: string;
  epsg: number | null;
  bounds: number[];
  resolution: number[];
  transform: number[];
  nodata: number | null;
  timestamp: string | null;
  validation_status: string;
  warnings: string[];
  errors: string[];
}

interface PairValidationResult {
  pair_type: string;
  compatible: boolean;
  spatial_overlap: number;
  crs_compatible: boolean;
  resolution_compatible: boolean;
  temporal_valid: boolean;
  coregistration_status: string;
  warnings: string[];
  errors: string[];
}

interface SlotState {
  file: File | null;
  asset: RasterAsset | null;
  progress: number;
  isUploading: boolean;
  error: string | null;
  xhr: XMLHttpRequest | null;
}

interface ModelInfo {
  model_id: string;
  name: string;
  model_name: string;
  version: string;
  task: string;
  supported_modalities: string[];
  input_format: string;
  checkpoint: string | null;
  framework: string | null;
  device_requirements: string;
  remote_sensing_adapted: boolean;
  training_dataset: string | null;
  status: string;
}

interface ChatMessage {
  sender: "user" | "assistant";
  text: string;
  task?: string;
  confidence?: {
    level: string;
    score: number;
    type: string;
  };
  evidence?: EvidenceResult[];
  execution_trace?: TraceNode[];
  warnings?: string[];
  model?: ModelInfo;
}

interface EvidenceResult {
  type: string;
  content: string;
  bbox: number[] | null; // [left, bottom, right, top]
  certainty: string;
  score: number;
}

interface TraceNode {
  step_id: string;
  model_name: string;
  display_name: string;
  status: string; // "pending" | "running" | "success" | "error" | "warning"
  duration_ms: number | null;
  inputs_summary: Record<string, any>;
  outputs_summary: Record<string, any>;
  error?: string | null;
}

interface AnalysisResponse {
  analysis_id: string;
  task: string;
  answer: string;
  evidence: EvidenceResult[];
  confidence: {
    level: string;
    score: number;
    type: string;
  };
  model: ModelInfo;
  warnings: string[];
  execution_trace: TraceNode[];
}

const initialSlotState = (): SlotState => ({
  file: null,
  asset: null,
  progress: 0,
  isUploading: false,
  error: null,
  xhr: null,
});

export default function UploadWorkspace() {
  const [slots, setSlots] = useState<Record<string, SlotState>>({
    optical: initialSlotState(),
    sar: initialSlotState(),
    t1: initialSlotState(),
    t2: initialSlotState(),
  });

  const [draggingSlot, setDraggingSlot] = useState<string | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<string | null>(null);
  const [pairValidation, setPairValidation] = useState<PairValidationResult | null>(null);
  const [isValidatingPair, setIsValidatingPair] = useState(false);

  // Analysis Mode State
  const [isAnalysisMode, setIsAnalysisMode] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [chatQuery, setChatQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [showEvidence, setShowEvidence] = useState(true);
  const [evidenceOpacity, setEvidenceOpacity] = useState(0.6);

  // Geospatial Map state
  const [activeViewportTab, setActiveViewportTab] = useState<"map" | "canvas">("map");
  const [detectedRegions, setDetectedRegions] = useState<any[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);

  // Conversational follow-ups
  const [chatHistory, setChatHistory] = useState<Record<string, ChatMessage[]>>({});
  const [sessionId, setSessionId] = useState<string>("");

  useEffect(() => {
    // Generate a conversation session ID
    setSessionId(Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15));
  }, []);

  // Zoom/Pan state for the viewer
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const viewerRef = useRef<HTMLDivElement>(null);

  const API_BASE = "http://localhost:8000/api/v1";

  // Fetch models list on startup
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(data);
        if (data.length > 0) {
          setSelectedModelId(data[0].model_id);
        }
      }
    } catch (e) {
      console.error("Failed to load model registry from backend:", e);
      // Fallback local list if server is starting
      const fallbackModels: ModelInfo[] = [
        {
          model_id: "rs_vqa_adapted",
          name: "SatQuery VQA (Adapted)",
          model_name: "SatQuery VQA Adapted ResNet50",
          version: "1.0.0",
          task: "REMOTE_SENSING_VQA",
          supported_modalities: ["optical", "multispectral"],
          input_format: "Multispectral GeoTIFF (RGB+NIR)",
          checkpoint: "resnet50_bigearthnet_sih26167.pth",
          framework: "PyTorch",
          device_requirements: "CPU/GPU",
          remote_sensing_adapted: true,
          training_dataset: "BigEarthNet-19",
          status: "STANDBY"
        },
        {
          model_id: "demo_fallback",
          name: "SatQuery Demo Adapter",
          model_name: "SatQuery Demo Adapter",
          version: "1.0.0",
          task: "MULTI_TASK_DEMO",
          supported_modalities: ["optical", "sar", "multispectral", "unknown"],
          input_format: "GeoTIFF/PNG/JPEG",
          checkpoint: "demo_weights_v1.bin",
          framework: "PyTorch",
          device_requirements: "CPU",
          remote_sensing_adapted: true,
          training_dataset: "SatQuery Curated Demo Set",
          status: "LOADED"
        },
        {
          model_id: "base_vlm",
          name: "Generic VLM Baseline",
          model_name: "Generic Vision-Language Model Baseline",
          version: "0.8.0",
          task: "REMOTE_SENSING_VQA",
          supported_modalities: ["optical", "sar", "multispectral"],
          input_format: "Any 3-Channel Image Format",
          checkpoint: "vlm_base_llama_vision.bin",
          framework: "HuggingFace Transformers",
          device_requirements: "CPU/GPU",
          remote_sensing_adapted: false,
          training_dataset: "LAION-5B / COCO",
          status: "STANDBY"
        }
      ];
      setModels(fallbackModels);
      setSelectedModelId("rs_vqa_adapted");
    }
  };

  // Demo Mode and Comparison States
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [comparisonMode, setComparisonMode] = useState<"slider" | "t1" | "t2">("slider");
  const [t2Opacity, setT2Opacity] = useState(0.5);

  const loadDemoDataset = async () => {
    setIsDemoLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/v1/ingestion/demo-pair`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_name: "random" }),
      });
      if (res.ok) {
        const data = await res.json();
        setSlots((prev) => ({
          ...prev,
          t1: { ...prev.t1, asset: data.t1_asset, isUploading: false, error: null },
          t2: { ...prev.t2, asset: data.t2_asset, isUploading: false, error: null },
        }));
        setSelectedSlot("t1");
        if (data.pair_validation) {
          setPairValidation(data.pair_validation);
        }
      }
    } catch (err) {
      console.error("Failed to load demo pair:", err);
    } finally {
      setIsDemoLoading(false);
    }
  };

  // Check and automatically run pair validation when appropriate slots are filled
  useEffect(() => {
    const hasOpticalAndSar = !!(slots.optical.asset && slots.sar.asset);
    const hasT1AndT2 = !!(slots.t1.asset && slots.t2.asset);

    if (hasOpticalAndSar || hasT1AndT2) {
      const asset1Id = hasOpticalAndSar ? slots.optical.asset!.id : slots.t1.asset!.id;
      const asset2Id = hasOpticalAndSar ? slots.sar.asset!.id : slots.t2.asset!.id;
      
      triggerPairValidation(asset1Id, asset2Id);
    } else {
      setPairValidation(null);
    }
  }, [slots.optical.asset, slots.sar.asset, slots.t1.asset, slots.t2.asset]);

  const triggerPairValidation = async (id1: string, id2: string) => {
    setIsValidatingPair(true);
    try {
      const res = await fetch(`${API_BASE}/ingestion/validate-pair`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ asset1_id: id1, asset2_id: id2 }),
      });
      if (res.ok) {
        const data = await res.json();
        setPairValidation(data);
      }
    } catch (e) {
      console.error("Error connecting to pair validation API:", e);
    } finally {
      setIsValidatingPair(false);
    }
  };

  const uploadFile = (slotKey: string, file: File) => {
    if (slots[slotKey].xhr) {
      slots[slotKey].xhr?.abort();
    }

    setSlots((prev) => ({
      ...prev,
      [slotKey]: {
        ...prev[slotKey],
        file,
        progress: 0,
        isUploading: true,
        error: null,
        asset: null,
      },
    }));

    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        setSlots((prev) => ({
          ...prev,
          [slotKey]: { ...prev[slotKey], progress: percent },
        }));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const asset: RasterAsset = JSON.parse(xhr.responseText);
          setSlots((prev) => ({
            ...prev,
            [slotKey]: {
              ...prev[slotKey],
              isUploading: false,
              asset,
              xhr: null,
            },
          }));
          setSelectedSlot((curr) => curr || slotKey);
        } catch (e) {
          handleUploadError(slotKey, "Failed to parse upload data.");
        }
      } else {
        try {
          const errData = JSON.parse(xhr.responseText);
          handleUploadError(slotKey, errData.detail || "Validation or parsing failed.");
        } catch (e) {
          handleUploadError(slotKey, `Ingestion failed (${xhr.statusText})`);
        }
      }
    });

    xhr.addEventListener("error", () => {
      handleUploadError(slotKey, "Network communication error.");
    });

    xhr.open("POST", `${API_BASE}/ingestion/upload`);
    xhr.send(formData);

    setSlots((prev) => ({
      ...prev,
      [slotKey]: { ...prev[slotKey], xhr },
    }));
  };

  const handleUploadError = (slotKey: string, message: string) => {
    setSlots((prev) => ({
      ...prev,
      [slotKey]: {
        ...prev[slotKey],
        isUploading: false,
        error: message,
        xhr: null,
      },
    }));
  };

  const removeFile = async (slotKey: string) => {
    const slot = slots[slotKey];
    if (slot.xhr) {
      slot.xhr.abort();
    }
    
    const assetId = slot.asset?.id;
    
    setSlots((prev) => ({
      ...prev,
      [slotKey]: initialSlotState(),
    }));

    if (selectedSlot === slotKey) {
      setSelectedSlot(null);
      resetZoomAndPan();
    }

    if (assetId) {
      try {
        await fetch(`${API_BASE}/ingestion/${assetId}`, { method: "DELETE" });
      } catch (e) {
        console.error("Failed to delete asset on backend:", e);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent, slotKey: string) => {
    e.preventDefault();
    setDraggingSlot(slotKey);
  };

  const handleDragLeave = () => {
    setDraggingSlot(null);
  };

  const handleDrop = (e: React.DragEvent, slotKey: string) => {
    e.preventDefault();
    setDraggingSlot(null);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      uploadFile(slotKey, file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, slotKey: string) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      uploadFile(slotKey, file);
    }
  };

  // Zoom/Pan Handlers
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const scale = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.min(Math.max(0.5, z * scale), 20));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    setPan({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const resetZoomAndPan = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Run AI Query Analysis
  const runAiAnalysis = async (queryText: string) => {
    if (!selectedSlot || !slots[selectedSlot]?.asset) return;
    
    const assetId = slots[selectedSlot].asset!.id;
    const t1Id = slots.t1?.asset?.id || slots.optical?.asset?.id || assetId;
    const t2Id = slots.t2?.asset?.id || slots.sar?.asset?.id || null;
    
    // Add user message to history
    const userMsg: ChatMessage = { sender: "user", text: queryText };
    setChatHistory((prev) => ({
      ...prev,
      [assetId]: [...(prev[prev.hasOwnProperty(assetId) ? assetId : ""] || prev[assetId] || []), userMsg]
    }));
    
    setIsAnalyzing(true);
    setChatQuery("");
    
    try {
      const isChangeQuery = /change|why|evidence|difference|compare|built-up|detect|increase/i.test(queryText);
      
      let response;
      if (isChangeQuery && t1Id && t2Id) {
        response = await fetch(`http://localhost:8000/api/v1/orchestrate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: queryText,
            t1_image_id: t1Id,
            t2_image_id: t2Id,
          })
        });
      } else {
        response = await fetch(`${API_BASE}/analysis/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_id: assetId,
            query: queryText,
            model_id: selectedModelId || null,
            session_id: sessionId
          })
        });
      }
      
      if (response.ok) {
        const data = await response.json();
        
        // Extract change regions if present
        if (data.results?.regions) {
          setDetectedRegions(data.results.regions);
        }

        const answerText = data.answer || data.results?.answer || data.results?.caption || (
          data.results?.regions ? `Identified ${data.results.regions.length} detected change region(s) across the bi-temporal satellite pair.` : "Analysis complete."
        );
        
        const resObj: AnalysisResponse = {
          analysis_id: data.analysis_id || "orch-" + Date.now(),
          task: data.task || "BI_TEMPORAL_ANALYSIS",
          answer: answerText,
          evidence: Array.isArray(data.evidence) ? data.evidence : (
            data.results?.regions ? data.results.regions.map((r: any) => ({
              type: "spatial",
              content: `${r.change_type || "Change"}: Area ${(r.area_sq_m || r.geospatial_area || 0).toFixed(1)} m²`,
              bbox: r.bbox,
              certainty: "HIGH",
              score: r.confidence || 0.85
            })) : []
          ),
          confidence: data.confidence || {
            level: "HIGH",
            score: data.results?.confidence || 0.88,
            type: "EVIDENCE_SCORE"
          },
          model: data.model || {
            model_id: data.models_used?.[0]?.model_id || "pipeline_orchestrator",
            name: data.models_used?.[0]?.model_id || "Pipeline Orchestrator",
            model_name: "SatQuery Pipeline",
            version: "1.0.0",
            task: data.task || "ORCHESTRATION",
            supported_modalities: ["optical", "multispectral", "sar"],
            input_format: "GeoTIFF",
            checkpoint: null,
            framework: null,
            device_requirements: "CPU",
            remote_sensing_adapted: true,
            training_dataset: null,
            status: "LOADED"
          },
          warnings: data.warnings || [],
          execution_trace: data.execution_trace || []
        };

        setAnalysisResult(resObj);
        
        // Add assistant message to history
        const assistantMsg: ChatMessage = {
          sender: "assistant",
          text: resObj.answer,
          task: resObj.task,
          confidence: resObj.confidence,
          evidence: resObj.evidence,
          execution_trace: resObj.execution_trace,
          warnings: resObj.warnings,
          model: resObj.model
        };
        setChatHistory((prev) => ({
          ...prev,
          [assetId]: [...(prev[assetId] || []), assistantMsg]
        }));
      } else {
        const err = await response.json();
        alert(err.detail || "AI analysis call failed.");
      }
    } catch (e) {
      alert("Error contacting the AI analysis router.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const selectedAsset = selectedSlot ? slots[selectedSlot]?.asset : null;

  // Calculate relative coordinates for Bounding Box rendering
  // bounds: [left, bottom, right, top]
  const getBBoxOverlayStyle = (bbox: number[]) => {
    if (!selectedAsset) return {};
    const [left, bottom, right, top] = selectedAsset.bounds;
    const [bLeft, bBottom, bRight, bTop] = bbox;
    
    const widthGeo = right - left;
    const heightGeo = top - bottom;
    
    if (widthGeo === 0 || heightGeo === 0) return {};
    
    // Relative percentages
    const x = ((bLeft - left) / widthGeo) * 100;
    const y = ((top - bTop) / heightGeo) * 100;
    const w = ((bRight - bLeft) / widthGeo) * 100;
    const h = ((bTop - bBottom) / heightGeo) * 100;
    
    return {
      left: `${x}%`,
      top: `${y}%`,
      width: `${w}%`,
      height: `${h}%`,
      position: "absolute" as const,
      border: "2px solid #ef4444",
      backgroundColor: "rgba(239, 68, 68, 0.15)",
      boxShadow: "0 0 8px #ef4444",
      pointerEvents: "none" as const,
      opacity: showEvidence ? evidenceOpacity : 0
    };
  };

  return (
    <main className={styles.workspace}>
      {/* ─── Navigation ─────────────────────────── */}
      <nav className={styles.nav}>
        <div className={styles.nav__brand}>
          <div className={styles.nav__logo}>S</div>
          <div className={styles.nav__name}>
            Sat<span>Query</span> AI
          </div>
        </div>
        <div className={styles.nav__status} style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={loadDemoDataset}
            disabled={isDemoLoading}
            style={{
              background: "linear-gradient(135deg, #06b6d4, #3b82f6)",
              border: "none",
              fontWeight: "bold",
              fontSize: "11px",
              padding: "4px 12px",
              display: "flex",
              alignItems: "center",
              gap: "5px",
            }}
          >
            {isDemoLoading ? "⏳ Loading Demo..." : "⚡ Load LEVIR-CD Demo Pair (T1 + T2)"}
          </button>

          {isAnalysisMode ? (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setIsAnalysisMode(false);
                setAnalysisResult(null);
              }}
            >
              ← Back to Ingestion
            </button>
          ) : (
            <span className="badge badge--accent">
              <span className="badge-dot" />
              IngestionFOUNDATION Active
            </span>
          )}
          <span style={{ marginLeft: "6px", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
            {isAnalysisMode ? "ANALYSIS MODULE" : "MISSION CONTROL WORKSPACE"}
          </span>
        </div>
      </nav>

      {/* ─── Workspace Ingestion Mode ────────────── */}
      {!isAnalysisMode && (
        <div className={styles.grid}>
          {/* Left Side: Upload Slots */}
          <div className={styles.panel__left}>
            <div className="glass-card--static" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
              <div className={styles.card__header} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h2 className={styles.card__title}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  Data Workspace
                </h2>

                <button
                  type="button"
                  onClick={loadDemoDataset}
                  disabled={isDemoLoading}
                  style={{
                    background: "rgba(6, 182, 212, 0.15)",
                    border: "1px solid rgba(6, 182, 212, 0.4)",
                    color: "#06b6d4",
                    borderRadius: "4px",
                    padding: "2px 8px",
                    fontSize: "10px",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {isDemoLoading ? "Loading..." : "⚡ Quick Demo"}
                </button>
              </div>
              
              <div style={{ padding: "12px", overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: "10px" }}>
                <p style={{ fontSize: "11px", color: "var(--color-text-secondary)", marginBottom: "4px" }}>
                  Upload files to start ingestion. Supported: GeoTIFF (.tif, .tiff), PNG, JPEG (no CRS).
                </p>

                <div className={styles.upload__slots}>
                  {Object.keys(slots).map((key) => {
                    const slot = slots[key];
                    const isDragging = draggingSlot === key;
                    const hasAsset = !!slot.asset;
                    const isSelected = selectedSlot === key;

                    let slotClass = styles.upload__slot;
                    if (isSelected) slotClass += ` ${styles["upload__slot--active"]}`;
                    if (isDragging) slotClass += ` ${styles["upload__slot--dragging"]}`;
                    
                    if (hasAsset) {
                      if (slot.asset?.validation_status === "PASS") slotClass += ` ${styles["upload__slot--success"]}`;
                      else if (slot.asset?.validation_status === "WARNING") slotClass += ` ${styles["upload__slot--warning"]}`;
                      else slotClass += ` ${styles["upload__slot--fail"]}`;
                    } else if (slot.error) {
                      slotClass += ` ${styles["upload__slot--fail"]}`;
                    }

                    const slotLabels: Record<string, string> = {
                      optical: "Optical Image (RGB)",
                      sar: "SAR Image (Radar)",
                      t1: "Bi-Temporal: T1 Image",
                      t2: "Bi-Temporal: T2 Image",
                    };

                    return (
                      <div
                        key={key}
                        className={slotClass}
                        onDragOver={(e) => handleDragOver(e, key)}
                        onDragLeave={handleDragLeave}
                        onDrop={(e) => handleDrop(e, key)}
                        onClick={() => hasAsset && setSelectedSlot(key)}
                      >
                        <div className={styles.slot__file_info}>
                          <span className={styles.slot__title}>{slotLabels[key]}</span>
                          {hasAsset && (
                            <span className={`badge badge--${slot.asset?.validation_status === "PASS" ? "success" : "warning"}`} style={{ fontSize: "9px", padding: "1px 6px" }}>
                              {slot.asset?.validation_status}
                            </span>
                          )}
                          {slot.error && (
                            <span className="badge badge--error" style={{ fontSize: "9px", padding: "1px 6px" }}>
                              FAIL
                            </span>
                          )}
                        </div>

                        {!hasAsset && !slot.isUploading && !slot.error && (
                          <div className={styles.slot__placeholder}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                              <polyline points="17 8 12 3 7 8" />
                              <line x1="12" y1="3" x2="12" y2="15" />
                            </svg>
                            <span>Drag & drop or <label style={{ color: "var(--color-accent-cyan)", cursor: "pointer", textDecoration: "underline" }}>
                              browse
                              <input
                                type="file"
                                style={{ display: "none" }}
                                onChange={(e) => handleFileChange(e, key)}
                                accept=".tif,.tiff,.png,.jpg,.jpeg"
                              />
                            </label></span>
                          </div>
                        )}

                        {slot.isUploading && (
                          <div style={{ display: "flex", flexDirection: "column", gap: "4px", padding: "10px 0" }}>
                            <div style={{ display: "flex", justifyContent: "between", fontSize: "11px", color: "var(--color-text-secondary)" }}>
                              <span>Uploading...</span>
                              <span style={{ marginLeft: "auto" }}>{slot.progress}%</span>
                            </div>
                            <div className={styles.progress__bar}>
                              <div className={styles.progress__fill} style={{ width: `${slot.progress}%` }} />
                            </div>
                          </div>
                        )}

                        {slot.error && !slot.isUploading && (
                          <div style={{ display: "flex", flexDirection: "column", gap: "8px", padding: "4px 0" }}>
                            <span style={{ fontSize: "11px", color: "var(--color-error)", wordBreak: "break-word" }}>
                              {slot.error}
                            </span>
                            <div style={{ display: "flex", gap: "10px", alignSelf: "flex-end" }}>
                              <label className="btn btn-secondary btn-sm" style={{ cursor: "pointer" }}>
                                Retry
                                <input
                                  type="file"
                                  style={{ display: "none" }}
                                  onChange={(e) => handleFileChange(e, key)}
                                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                                />
                              </label>
                              <button type="button" className="btn btn-ghost btn-sm" onClick={(e) => { e.stopPropagation(); removeFile(key); }}>
                                Clear
                              </button>
                            </div>
                          </div>
                        )}

                        {hasAsset && !slot.isUploading && (
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "4px" }}>
                            <div className={styles.file__details}>
                              <div className={styles.file__name} title={slot.asset?.filename}>
                                {slot.asset?.filename}
                              </div>
                              <div className={styles.file__meta}>
                                {slot.asset?.width}x{slot.asset?.height} • {slot.asset?.bands}b • {slot.asset?.crs === "UNKNOWN" ? "No CRS" : slot.asset?.crs.split(":").pop()}
                              </div>
                            </div>
                            <button
                              type="button"
                              className={`${styles.btn_icon} ${styles["btn_icon--danger"]}`}
                              onClick={(e) => { e.stopPropagation(); removeFile(key); }}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <polyline points="3 6 5 6 21 6" />
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                              </svg>
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Right Side: Viewport & Ingestion Summary */}
          <div className={styles.panel__right}>
            <div className={`${styles.viewer__card} glass-card--static`}>
              <div className={styles.card__header} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h2 className={styles.card__title}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                  GIS Inspector Viewport
                </h2>

                {/* Viewport Mode Switcher */}
                <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                  <button
                    type="button"
                    onClick={() => setActiveViewportTab("map")}
                    style={{
                      padding: "4px 10px",
                      borderRadius: "6px",
                      fontSize: "11px",
                      fontWeight: 600,
                      cursor: "pointer",
                      border: activeViewportTab === "map" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                      background: activeViewportTab === "map" ? "rgba(6, 182, 212, 0.2)" : "rgba(255,255,255,0.05)",
                      color: activeViewportTab === "map" ? "#06b6d4" : "#94a3b8",
                      transition: "all 0.15s ease"
                    }}
                  >
                    🗺️ Satellite Map
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveViewportTab("canvas")}
                    style={{
                      padding: "4px 10px",
                      borderRadius: "6px",
                      fontSize: "11px",
                      fontWeight: 600,
                      cursor: "pointer",
                      border: activeViewportTab === "canvas" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                      background: activeViewportTab === "canvas" ? "rgba(6, 182, 212, 0.2)" : "rgba(255,255,255,0.05)",
                      color: activeViewportTab === "canvas" ? "#06b6d4" : "#94a3b8",
                      transition: "all 0.15s ease"
                    }}
                  >
                    🔬 Raster Canvas
                  </button>
                </div>
              </div>

              {activeViewportTab === "map" ? (
                <div style={{ flex: 1, width: "100%", height: "100%", minHeight: "380px", position: "relative" }}>
                  <GeospatialMap
                    t1Asset={slots.t1?.asset || slots.optical?.asset || selectedAsset}
                    t2Asset={slots.t2?.asset || slots.sar?.asset || null}
                    regions={detectedRegions}
                    selectedRegionId={selectedRegionId}
                    onSelectRegion={(id) => setSelectedRegionId(id)}
                  />
                </div>
              ) : (
                <div
                  ref={viewerRef}
                  className={styles.viewer__content}
                  onWheel={handleWheel}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                >
                  {selectedAsset ? (
                    <>
                      <img
                        src={`http://localhost:8000/api/v1/ingestion/${selectedAsset.id}/preview?t=${Date.now()}`}
                        alt="Geospatial Preview"
                        className={styles.viewer__img}
                        draggable={false}
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                        }}
                      />
                      <div className={styles.viewer__overlay}>
                        <div className={styles.viewer__overlay_title}>Geospatial Metadata</div>
                        <div>Modality: <span style={{ color: "white" }}>{selectedAsset.modality.toUpperCase()}</span></div>
                        <div>CRS: <span style={{ color: "white" }}>{selectedAsset.crs.length > 30 ? `${selectedAsset.crs.substring(0, 30)}...` : selectedAsset.crs}</span></div>
                        <div>Bands: <span style={{ color: "white" }}>{selectedAsset.bands} ({selectedAsset.dtype})</span></div>
                        <div>Bounds: <span style={{ color: "white", fontSize: "10px" }}>
                          W: {selectedAsset.bounds[0]?.toFixed(4)} • E: {selectedAsset.bounds[2]?.toFixed(4)}
                          <br />
                          S: {selectedAsset.bounds[1]?.toFixed(4)} • N: {selectedAsset.bounds[3]?.toFixed(4)}
                        </span></div>
                        <div>Resolution: <span style={{ color: "white" }}>{selectedAsset.resolution.map(r => r.toFixed(5)).join(", ")}</span></div>
                      </div>

                      <div className={styles.viewer__controls}>
                        <button type="button" className={styles.viewer__control_btn} onClick={() => setZoom((z) => Math.min(20, z * 1.2))}>+</button>
                        <button type="button" className={styles.viewer__control_btn} onClick={() => setZoom((z) => Math.max(0.5, z / 1.2))}>-</button>
                        <button type="button" className={styles.viewer__control_btn} onClick={resetZoomAndPan}>👁</button>
                      </div>
                    </>
                  ) : (
                    <div className={styles.viewer__no_selection}>
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ opacity: 0.3 }}>
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <circle cx="9" cy="9" r="2" />
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                      </svg>
                      <span>Select an uploaded image to load geospatial preview</span>
                    </div>
                  )}
                </div>
              )}

              {selectedAsset && (
                <div className={styles.metadata__inspector}>
                  {selectedAsset.validation_status !== "PASS" && (
                    <div className={`${styles.validation__status_box} ${selectedAsset.validation_status === "WARNING" ? styles["validation__status_box--warning"] : styles["validation__status_box--fail"]}`}>
                      <strong>Ingestion {selectedAsset.validation_status}:</strong>
                      <ul style={{ paddingLeft: "16px", margin: "2px 0 0 0" }}>
                        {selectedAsset.errors.map((e, idx) => <li key={idx}>{e}</li>)}
                        {selectedAsset.warnings.map((w, idx) => <li key={idx}>{w}</li>)}
                      </ul>
                    </div>
                  )}
                  
                  <div className={styles.metadata__grid}>
                    <div className={styles.metadata__item}>
                      <span className={styles.metadata__label}>Filename</span>
                      <span className={styles.metadata__value} title={selectedAsset.filename}>{selectedAsset.filename}</span>
                    </div>
                    <div className={styles.metadata__item}>
                      <span className={styles.metadata__label}>Modality</span>
                      <span className={styles.metadata__value} style={{ textTransform: "capitalize" }}>
                        {selectedAsset.modality} {selectedAsset.modality_certain ? "✓" : "(uncertain)"}
                      </span>
                    </div>
                    <div className={styles.metadata__item}>
                      <span className={styles.metadata__label}>EPSG Code</span>
                      <span className={styles.metadata__value}>{selectedAsset.epsg ? `EPSG:${selectedAsset.epsg}` : "N/A"}</span>
                    </div>
                    <div className={styles.metadata__item}>
                      <span className={styles.metadata__label}>Dimensions</span>
                      <span className={styles.metadata__value}>{selectedAsset.width} × {selectedAsset.height} px</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Pair Validation Panel */}
            <div className={styles.pair__card}>
              <div style={{ display: "flex", justifyContent: "between", alignItems: "center" }}>
                <h3 className={styles.card__title} style={{ fontSize: "11px" }}>
                  Pair Validation Analysis
                </h3>
                {isValidatingPair && (
                  <span className="badge badge--info" style={{ fontSize: "9px" }}>
                    <span className="badge-dot" /> Validating Pair...
                  </span>
                )}
              </div>

              <div className={styles.pair__content}>
                {pairValidation ? (
                  <>
                    <div className={styles.pair__checks}>
                      <div className={`${styles.pair__check} ${pairValidation.crs_compatible ? styles["pair__check--success"] : styles["pair__check--fail"]}`}>
                        <span className={`${styles.pair__check_icon} ${pairValidation.crs_compatible ? styles["pair__check_icon--pass"] : styles["pair__check_icon--fail"]}`}>
                          {pairValidation.crs_compatible ? "✓" : "✗"}
                        </span>
                        <span>CRS Compatible</span>
                      </div>

                      <div className={`${styles.pair__check} ${pairValidation.spatial_overlap >= 90 ? styles["pair__check--success"] : pairValidation.spatial_overlap > 0 ? styles["pair__check--warning"] : styles["pair__check--fail"]}`}>
                        <span className={`${styles.pair__check_icon} ${pairValidation.spatial_overlap >= 90 ? styles["pair__check_icon--pass"] : pairValidation.spatial_overlap > 0 ? styles["pair__check_icon--warning"] : styles["pair__check_icon--fail"]}`}>
                          {pairValidation.spatial_overlap > 0 ? "✓" : "✗"}
                        </span>
                        <span>Spatial Overlap ({pairValidation?.spatial_overlap != null ? pairValidation.spatial_overlap.toFixed(1) : "0.0"}%)</span>
                      </div>

                      <div className={`${styles.pair__check} ${pairValidation.resolution_compatible ? styles["pair__check--success"] : styles["pair__check--warning"]}`}>
                        <span className={`${styles.pair__check_icon} ${pairValidation.resolution_compatible ? styles["pair__check_icon--pass"] : styles["pair__check_icon--warning"]}`}>
                          {pairValidation.resolution_compatible ? "✓" : "⚠"}
                        </span>
                        <span>Resolution Compatible</span>
                      </div>

                      <div className={`${styles.pair__check} ${pairValidation.coregistration_status === "NOT_VERIFIED" ? styles["pair__check--warning"] : pairValidation.coregistration_status === "PASS" ? styles["pair__check--success"] : styles["pair__check--fail"]}`}>
                        <span className={`${styles.pair__check_icon} ${pairValidation.coregistration_status === "PASS" ? styles["pair__check_icon--pass"] : styles["pair__check_icon--warning"]}`}>
                          {pairValidation.coregistration_status === "PASS" ? "✓" : "⚠"}
                        </span>
                        <span>Co-registration ({pairValidation.coregistration_status.replace("_", " ")})</span>
                      </div>
                    </div>

                    <div className={styles.analysis__actions}>
                      <div className={styles.pair__selection_hint}>
                        <div className={styles.pair__selection_indicator}>
                          <span>Pair Type:</span>
                          <span className={styles.pair__type_badge}>{pairValidation.pair_type}</span>
                        </div>
                      </div>
                      
                      <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => setIsAnalysisMode(true)}
                      >
                        Continue to Analysis
                      </button>
                    </div>
                  </>
                ) : (
                  <div style={{ display: "flex", width: "100%", justifyContent: "between", alignItems: "center" }}>
                    <span style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
                      Upload either [Optical + SAR] pair or [T1 + T2] pair to continue. Or select a single image and click analysis below.
                    </span>
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={!selectedAsset}
                      onClick={() => setIsAnalysisMode(true)}
                    >
                      Analyze Single Image
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── Workspace AI Analysis Mode ───────────── */}
      {isAnalysisMode && (
        <div className={styles.grid}>
          {/* Left Side: Ask SatQuery Chatbot Panel */}
          <div className={styles.panel__left}>
            <div className="glass-card--static" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
              <div className={styles.card__header}>
                <h2 className={styles.card__title}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  Ask SatQuery
                </h2>
              </div>
              
              <div style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "16px", flex: 1, overflowY: "auto" }}>
                {/* Model selection dropdown */}
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <label style={{ fontSize: "10px", textTransform: "uppercase", color: "var(--color-text-secondary)", fontWeight: 700 }}>
                    Select Specialty Model
                  </label>
                  <select
                    value={selectedModelId}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    style={{
                      background: "var(--color-bg-secondary)",
                      color: "white",
                      border: "1px solid var(--color-border)",
                      padding: "8px 12px",
                      borderRadius: "6px",
                      fontSize: "12px",
                      width: "100%"
                    }}
                  >
                    {models.map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.name} ({m.version})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Text query input */}
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <label style={{ fontSize: "10px", textTransform: "uppercase", color: "var(--color-text-secondary)", fontWeight: 700 }}>
                    Enter Vision-Language Query
                  </label>
                  <textarea
                    rows={4}
                    value={chatQuery}
                    onChange={(e) => setChatQuery(e.target.value)}
                    placeholder="e.g. Is there a water body? Or Describe this scene..."
                    style={{
                      background: "var(--color-bg-secondary)",
                      color: "white",
                      border: "1px solid var(--color-border)",
                      padding: "10px 12px",
                      borderRadius: "6px",
                      fontSize: "13px",
                      resize: "none",
                      width: "100%"
                    }}
                  />
                </div>

                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={isAnalyzing || !chatQuery.trim()}
                  onClick={() => runAiAnalysis(chatQuery)}
                  style={{ width: "100%" }}
                >
                  {isAnalyzing ? "Analyzing Imagery..." : "Analyze"}
                </button>

                {/* Suggestions list */}
                <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "10px" }}>
                  <span style={{ fontSize: "10px", textTransform: "uppercase", color: "var(--color-text-muted)", fontWeight: 700 }}>
                    Suggested Queries
                  </span>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    {[
                      "Describe this scene",
                      "What land cover is visible?",
                      "Is there any water body?",
                      "Highlight the water body",
                      "Highlight the major road",
                      "Identify agricultural regions"
                    ].map((sug) => (
                      <button
                        key={sug}
                        type="button"
                        onClick={() => {
                          setChatQuery(sug);
                          runAiAnalysis(sug);
                        }}
                        style={{
                          background: "rgba(255, 255, 255, 0.02)",
                          border: "1px solid var(--color-border)",
                          color: "var(--color-text-secondary)",
                          textAlign: "left",
                          padding: "8px 12px",
                          borderRadius: "6px",
                          fontSize: "12px",
                          cursor: "pointer",
                          transition: "all var(--transition-fast)"
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = "var(--color-accent-cyan)";
                          e.currentTarget.style.color = "white";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = "var(--color-border)";
                          e.currentTarget.style.color = "var(--color-text-secondary)";
                        }}
                      >
                        {sug}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side: Map Viewer with Bounding Box Overlay & Trace Result */}
          <div className={styles.panel__right}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 400px", gap: "16px", height: "100%", overflow: "hidden" }}>
              
              {/* Mid Viewport */}
              <div className="glass-card--static" style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
                <div className={styles.card__header} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h2 className={styles.card__title}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="2" y1="12" x2="22" y2="12" />
                      <line x1="12" y1="2" x2="12" y2="22" />
                    </svg>
                    GIS Analysis Viewport
                  </h2>

                  {/* Viewport Mode Switcher & Overlays */}
                  <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                    <button
                      type="button"
                      onClick={() => setActiveViewportTab("map")}
                      style={{
                        padding: "4px 10px",
                        borderRadius: "6px",
                        fontSize: "11px",
                        fontWeight: 600,
                        cursor: "pointer",
                        border: activeViewportTab === "map" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                        background: activeViewportTab === "map" ? "rgba(6, 182, 212, 0.2)" : "rgba(255,255,255,0.05)",
                        color: activeViewportTab === "map" ? "#06b6d4" : "#94a3b8",
                        transition: "all 0.15s ease"
                      }}
                    >
                      🗺️ Satellite Map ({detectedRegions.length > 0 ? `${detectedRegions.length} Regions` : "Live"})
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveViewportTab("canvas")}
                      style={{
                        padding: "4px 10px",
                        borderRadius: "6px",
                        fontSize: "11px",
                        fontWeight: 600,
                        cursor: "pointer",
                        border: activeViewportTab === "canvas" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                        background: activeViewportTab === "canvas" ? "rgba(6, 182, 212, 0.2)" : "rgba(255,255,255,0.05)",
                        color: activeViewportTab === "canvas" ? "#06b6d4" : "#94a3b8",
                        transition: "all 0.15s ease"
                      }}
                    >
                      🔬 Raster Canvas
                    </button>

                    {analysisResult && analysisResult.evidence.length > 0 && analysisResult.evidence[0].bbox && activeViewportTab === "canvas" && (
                      <div style={{ display: "flex", alignItems: "center", gap: "6px", marginLeft: "6px" }}>
                        <label style={{ fontSize: "10px", color: "var(--color-text-secondary)", display: "flex", alignItems: "center", gap: "4px" }}>
                          <input
                            type="checkbox"
                            checked={showEvidence}
                            onChange={(e) => setShowEvidence(e.target.checked)}
                          />
                          Overlay
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="1.0"
                          step="0.1"
                          value={evidenceOpacity}
                          onChange={(e) => setEvidenceOpacity(parseFloat(e.target.value))}
                          style={{ width: "50px" }}
                          title="Overlay Opacity"
                        />
                      </div>
                    )}
                  </div>
                </div>

                {activeViewportTab === "map" ? (
                  <div style={{ flex: 1, width: "100%", height: "100%", minHeight: "380px", position: "relative" }}>
                    <GeospatialMap
                      t1Asset={slots.t1?.asset || slots.optical?.asset || selectedAsset}
                      t2Asset={slots.t2?.asset || slots.sar?.asset || null}
                      regions={detectedRegions}
                      selectedRegionId={selectedRegionId}
                      onSelectRegion={(id) => setSelectedRegionId(id)}
                    />
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", height: "100%", flex: 1 }}>
                    {/* Visual Comparison Toolbar for Bi-temporal Pairs */}
                    {slots.t1?.asset && slots.t2?.asset && (
                      <div style={{
                        padding: "8px 14px",
                        background: "rgba(0,0,0,0.4)",
                        borderBottom: "1px solid var(--color-border)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: "10px",
                        zIndex: 10
                      }}>
                        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                          <span style={{ fontSize: "11px", fontWeight: "bold", color: "var(--color-accent-cyan)" }}>
                            Bi-Temporal Compare:
                          </span>
                          <button
                            type="button"
                            onClick={() => { setComparisonMode("t1"); setT2Opacity(0); }}
                            style={{
                              padding: "2px 8px",
                              borderRadius: "4px",
                              fontSize: "10px",
                              fontWeight: 600,
                              cursor: "pointer",
                              border: comparisonMode === "t1" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                              background: comparisonMode === "t1" ? "rgba(6, 182, 212, 0.2)" : "transparent",
                              color: comparisonMode === "t1" ? "#06b6d4" : "#94a3b8"
                            }}
                          >
                            T1 (Pre-Change)
                          </button>
                          <button
                            type="button"
                            onClick={() => { setComparisonMode("slider"); }}
                            style={{
                              padding: "2px 8px",
                              borderRadius: "4px",
                              fontSize: "10px",
                              fontWeight: 600,
                              cursor: "pointer",
                              border: comparisonMode === "slider" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                              background: comparisonMode === "slider" ? "rgba(6, 182, 212, 0.2)" : "transparent",
                              color: comparisonMode === "slider" ? "#06b6d4" : "#94a3b8"
                            }}
                          >
                            T1 ↔ T2 Blend
                          </button>
                          <button
                            type="button"
                            onClick={() => { setComparisonMode("t2"); setT2Opacity(1); }}
                            style={{
                              padding: "2px 8px",
                              borderRadius: "4px",
                              fontSize: "10px",
                              fontWeight: 600,
                              cursor: "pointer",
                              border: comparisonMode === "t2" ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                              background: comparisonMode === "t2" ? "rgba(6, 182, 212, 0.2)" : "transparent",
                              color: comparisonMode === "t2" ? "#06b6d4" : "#94a3b8"
                            }}
                          >
                            T2 (Post-Change)
                          </button>
                        </div>

                        {comparisonMode === "slider" && (
                          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <span style={{ fontSize: "10px", color: "var(--color-text-muted)" }}>T1</span>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={t2Opacity}
                              onChange={(e) => setT2Opacity(parseFloat(e.target.value))}
                              style={{ width: "100px", cursor: "pointer" }}
                              title={`T2 Opacity: ${Math.round(t2Opacity * 100)}%`}
                            />
                            <span style={{ fontSize: "10px", color: "var(--color-text-muted)" }}>T2 ({Math.round(t2Opacity * 100)}%)</span>
                          </div>
                        )}
                      </div>
                    )}

                    <div
                      ref={viewerRef}
                      className={styles.viewer__content}
                      onWheel={handleWheel}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      style={{ flex: 1, position: "relative" }}
                    >
                      {/* Non-georeferenced asset banner */}
                      {selectedAsset && selectedAsset.crs === "UNKNOWN" && (
                        <div style={{
                          position: "absolute",
                          top: "10px",
                          left: "50%",
                          transform: "translateX(-50%)",
                          background: "rgba(15, 23, 42, 0.85)",
                          backdropFilter: "blur(6px)",
                          border: "1px solid rgba(245, 158, 11, 0.4)",
                          color: "#f59e0b",
                          padding: "4px 12px",
                          borderRadius: "20px",
                          fontSize: "10px",
                          fontWeight: 600,
                          zIndex: 20,
                          pointerEvents: "none",
                          display: "flex",
                          alignItems: "center",
                          gap: "5px"
                        }}>
                          <span>ℹ️</span> Non-georeferenced asset — normalized spatial visualization
                        </div>
                      )}

                      {selectedAsset && (
                        <div
                          style={{
                            position: "relative",
                            width: "100%",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                            transformOrigin: "center"
                          }}
                        >
                          {/* Base Layer (T1 or Selected Asset) */}
                          <img
                            src={`http://localhost:8000/api/v1/ingestion/${(slots.t1?.asset || selectedAsset).id}/preview?t=${Date.now()}`}
                            alt="Pre-Change Raster"
                            style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", position: "relative" }}
                            draggable={false}
                          />

                          {/* Overlaid Layer (T2 with Opacity when both T1 and T2 exist) */}
                          {slots.t1?.asset && slots.t2?.asset && (
                            <img
                              src={`http://localhost:8000/api/v1/ingestion/${slots.t2.asset.id}/preview?t=${Date.now()}`}
                              alt="Post-Change Raster"
                              style={{
                                maxWidth: "100%",
                                maxHeight: "100%",
                                objectFit: "contain",
                                position: "absolute",
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                margin: "auto",
                                opacity: t2Opacity,
                                transition: "opacity 0.05s ease"
                              }}
                              draggable={false}
                            />
                          )}
                          
                          {/* Bounding box SVG/Div overlay mapper */}
                          {analysisResult && analysisResult.evidence.map((ev, idx) => {
                            if (ev.bbox) {
                              const boxStyle = getBBoxOverlayStyle(ev.bbox);
                              return (
                                <div key={idx} style={boxStyle} title={`${ev.content} (${(ev.score * 100).toFixed(0)}%)`}>
                                  <div style={{
                                    position: "absolute",
                                    top: "-20px",
                                    left: "0",
                                    background: "#ef4444",
                                    color: "white",
                                    fontSize: "9px",
                                    padding: "2px 6px",
                                    borderRadius: "3px",
                                    fontWeight: "bold",
                                    whiteSpace: "nowrap"
                                  }}>
                                    {ev.content.substring(0, 20)} ({(ev.score * 100).toFixed(0)}%)
                                  </div>
                                </div>
                              );
                            }
                            return null;
                          })}
                        </div>
                      )}

                      <div className={styles.viewer__controls}>
                        <button type="button" className={styles.viewer__control_btn} onClick={() => setZoom((z) => Math.min(20, z * 1.2))}>+</button>
                        <button type="button" className={styles.viewer__control_btn} onClick={() => setZoom((z) => Math.max(0.5, z / 1.2))}>-</button>
                        <button type="button" className={styles.viewer__control_btn} onClick={resetZoomAndPan}>👁</button>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Side: Analysis result & Execution Trace */}
              <div style={{ display: "flex", flexDirection: "column", gap: "16px", overflowY: "auto", height: "100%" }}>
                {/* Result Card */}
                <div className="glass-card--static" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "12px", minHeight: "350px", maxHeight: "600px", overflowY: "auto" }}>
                  <h3 className={styles.card__title} style={{ fontSize: "11px" }}>
                    AI Conversation History
                  </h3>
                  
                  <div style={{ display: "flex", flexDirection: "column", gap: "12px", overflowY: "auto", maxHeight: "350px", paddingRight: "4px", marginBottom: "8px", borderBottom: (analysisResult && !isAnalyzing) ? "1px solid var(--color-border)" : "none", paddingBottom: (analysisResult && !isAnalyzing) ? "12px" : "0" }}>
                    {selectedAsset && (chatHistory[selectedAsset.id] || []).map((msg, idx) => (
                      <div
                        key={idx}
                        style={{
                          alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                          maxWidth: "85%",
                          background: msg.sender === "user" ? "rgba(6, 182, 212, 0.12)" : "rgba(255, 255, 255, 0.02)",
                          border: msg.sender === "user" ? "1px solid rgba(6, 182, 212, 0.2)" : "1px solid var(--color-border)",
                          borderRadius: msg.sender === "user" ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                          padding: "10px 12px",
                          fontSize: "12.5px",
                          lineHeight: "1.45"
                        }}
                      >
                        <div style={{
                          fontSize: "9px",
                          textTransform: "uppercase",
                          color: msg.sender === "user" ? "var(--color-accent-cyan)" : "var(--color-text-muted)",
                          marginBottom: "4px",
                          fontWeight: "bold",
                          display: "flex",
                          justifyContent: "space-between",
                          gap: "10px"
                        }}>
                          <span>{msg.sender === "user" ? "You" : (msg.model?.model_name || "SatQuery Assistant")}</span>
                          {msg.confidence && (
                            <span style={{ color: msg.confidence.level === "HIGH" ? "var(--color-success)" : "var(--color-warning)" }}>
                              {msg.confidence.level} ({Math.round(msg.confidence.score * 100)}%)
                            </span>
                          )}
                        </div>
                        <div style={{ color: "white", whiteSpace: "pre-line" }}>
                          {msg.text}
                        </div>
                      </div>
                    ))}

                    {isAnalyzing && (
                      <div style={{ display: "flex", alignSelf: "flex-start", maxWidth: "85%", background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--color-border)", borderRadius: "12px 12px 12px 2px", padding: "12px", width: "100%" }}>
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", width: "100%" }}>
                          <span style={{ fontSize: "9px", color: "var(--color-text-muted)", fontWeight: "bold", textTransform: "uppercase" }}>Thinking...</span>
                          <span style={{ fontSize: "11px", color: "var(--color-text-secondary)" }}>Running specialist model inference...</span>
                        </div>
                      </div>
                    )}

                    {(!selectedAsset || !(chatHistory[selectedAsset.id] || []).length) && !isAnalyzing && (
                      <div style={{ padding: "40px 0", textAlign: "center", color: "var(--color-text-muted)", fontSize: "12px" }}>
                        Ask a follow-up question or run a VQA/Scene Description query.
                      </div>
                    )}
                  </div>

                  {analysisResult && !isAnalyzing && (
                    <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span className={styles.pair__type_badge}>{analysisResult.task.replace("_", " ")}</span>
                        <span className={`badge badge--${analysisResult.confidence.level === "HIGH" ? "success" : "warning"}`} style={{ fontSize: "10px" }}>
                          {analysisResult.confidence.type}: {analysisResult.confidence.level} ({Math.round(analysisResult.confidence.score * 100)}%)
                        </span>
                      </div>

                      {/* Evidence block */}
                      {analysisResult.evidence.length > 0 && (
                        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                          <span style={{ fontSize: "10px", textTransform: "uppercase", color: "var(--color-text-muted)", fontWeight: 700 }}>
                            Extracted Evidence
                          </span>
                          {analysisResult.evidence.map((ev, idx) => (
                            <div
                              key={idx}
                              style={{
                                background: "rgba(10, 14, 26, 0.4)",
                                borderLeft: "3px solid var(--color-accent-cyan)",
                                padding: "8px 12px",
                                borderRadius: "4px",
                                fontSize: "11px"
                              }}
                            >
                              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "2px" }}>
                                <strong style={{ textTransform: "capitalize", color: "var(--color-accent-cyan)" }}>{ev.type} Evidence</strong>
                                <span className="badge badge--info" style={{ fontSize: "8px", padding: "0px 4px" }}>{ev.certainty}</span>
                              </div>
                              <div style={{ color: "var(--color-text-secondary)" }}>{ev.content}</div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Warnings if any */}
                      {analysisResult.warnings.length > 0 && (
                        <div className={styles.validation__status_box + " " + styles["validation__status_box--warning"]}>
                          <strong style={{ fontSize: "10px" }}>Warnings:</strong>
                          <ul style={{ paddingLeft: "14px", margin: 0, fontSize: "10px" }}>
                            {analysisResult.warnings.map((w, idx) => <li key={idx}>{w}</li>)}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Execution Trace DAG Timeline */}
                {analysisResult && !isAnalyzing && (
                  <div className="glass-card--static" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
                    <h3 className={styles.card__title} style={{ fontSize: "11px" }}>
                      Execution Trace DAG
                    </h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px", position: "relative" }}>
                      {analysisResult.execution_trace.map((node, idx) => {
                        let statusColor = "var(--color-text-muted)";
                        if (node.status === "success") statusColor = "var(--color-success)";
                        else if (node.status === "warning") statusColor = "var(--color-warning)";
                        else if (node.status === "error") statusColor = "var(--color-error)";

                        return (
                          <div key={idx} style={{ display: "flex", gap: "12px", position: "relative" }}>
                            {/* vertical timeline connector line */}
                            {idx < analysisResult.execution_trace.length - 1 && (
                              <div style={{
                                position: "absolute",
                                left: "6px",
                                top: "14px",
                                bottom: "-16px",
                                width: "2px",
                                background: "var(--color-border)",
                                zIndex: 1
                              }} />
                            )}
                            
                            {/* Timeline dot */}
                            <div style={{
                              width: "14px",
                              height: "14px",
                              borderRadius: "50%",
                              background: statusColor,
                              border: "3px solid var(--color-bg-primary)",
                              boxShadow: "0 0 6px " + statusColor,
                              zIndex: 2,
                              marginTop: "2px"
                            }} />

                            <div style={{ display: "flex", flexDirection: "column", gap: "2px", flex: 1 }}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                <span style={{ fontSize: "11px", fontWeight: "bold", color: "white" }}>{node.display_name}</span>
                                {typeof node.duration_ms === "number" && (
                                  <span style={{ fontSize: "9px", fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                                    {node.duration_ms.toFixed(1)}ms
                                  </span>
                                )}
                              </div>
                              <span style={{ fontSize: "9px", color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                                Model: {node.model_name}
                              </span>
                              {node.error && (
                                <span style={{ fontSize: "10px", color: "var(--color-error)", marginTop: "2px" }}>
                                  Error: {node.error}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>
        </div>
      )}
    </main>
  );
}
