import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress header and footer on cover page
            return
        
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Header
        self.drawString(54, 11 * 72 - 36, "SatQuery AI — Bi-Temporal Geospatial Intelligence Platform")
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
        
        # Footer
        self.line(54, 48, 8.5 * 72 - 54, 48)
        self.drawString(54, 34, "CONFIDENTIAL & PROPRIETARY — TEAM & JUDGES COMPREHENSIVE DOSSIER")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 34, page_str)
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    c_primary = colors.HexColor("#1E1B4B")     # Indigo Deep
    c_secondary = colors.HexColor("#4338CA")   # Indigo Bright
    c_accent = colors.HexColor("#0284C7")      # Sky Blue
    c_dark = colors.HexColor("#0F172A")        # Slate Dark
    c_light = colors.HexColor("#F8FAFC")       # Off-white background
    c_border = colors.HexColor("#E2E8F0")      # Border gray
    
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=32,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#E0E7FF"),
        alignment=TA_CENTER
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=c_dark,
        spaceAfter=5,
        alignment=TA_JUSTIFY
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=c_dark,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#1E293B")
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_dark
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # ---------------------------------------------------------
    # COVER PAGE
    # ---------------------------------------------------------
    story.append(Spacer(1, 15))
    
    banner_data = [
        [Paragraph("SatQuery AI", title_style)],
        [Paragraph("Bi-Temporal Satellite Image Change Detection & Geospatial VQA Platform", subtitle_style)],
        [Paragraph("COMPLETE SYSTEM ARCHITECTURE & JUDGES PRESENTATION DOSSIER", ParagraphStyle('SubSub', parent=subtitle_style, fontSize=9.5, textColor=colors.HexColor("#93C5FD")))]
    ]
    banner_table = Table(banner_data, colWidths=[504])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_primary),
        ('TOPPADDING', (0,0), (-1,-1), 20),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 25))
    
    meta_card_data = [
        [Paragraph("<b>Target Audience:</b> Project Team, Pitch Judges, Technical Evaluators", meta_style)],
        [Paragraph("<b>Document Version:</b> 2.0 (Full 6-Phase Pipeline Coverage)", meta_style)],
        [Paragraph("<b>Core Capabilities:</b> Multi-spectral GeoTIFF Ingestion, Optical-SAR Cross-Modal Fusion, Rule-Based & ML Change Understanding, Audit-Grade Evidence Extraction, Mission Control UX", meta_style)],
        [Paragraph("<b>Date:</b> September 2026 | <b>Status:</b> Production Ready (130/130 Tests Passing)", meta_style)]
    ]
    meta_table = Table(meta_card_data, colWidths=[504])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 25))
    
    exec_summary_box = [
        [Paragraph("<b>EXECUTIVE OVERVIEW FOR JUDGES & EVALUATORS</b><br/><br/>"
                   "SatQuery AI solves the critical challenge of analyzing bi-temporal satellite image pairs (T1 pre-change vs T2 post-change). "
                   "Unlike traditional black-box vision systems, SatQuery AI combines <b>multi-spectral spectral indices (NDVI, NDWI, NDBI)</b>, "
                   "<b>SAR radar backscatter fusion</b>, and <b>explainable audit trails</b> to identify <i>where</i> changes occur, "
                   "<i>what</i> land-cover transition happened (e.g., Deforestation, Urban Expansion, Demolition), and <i>why</i> the AI reached that conclusion with 100% mathematical trace transparency.", callout_style)]
    ]
    exec_table = Table(exec_summary_box, colWidths=[504])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1.5, c_accent),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(exec_table)
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY & PROBLEM STATEMENT
    # ---------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Value Proposition", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8))
    
    story.append(Paragraph(
        "Modern satellite intelligence relies heavily on Earth Observation (EO) data from optical (Sentinel-2, Landsat) and Synthetic Aperture Radar (SAR / Sentinel-1) constellations. "
        "However, defense, environmental monitoring, and urban planning teams face three massive bottlenecks when analyzing bi-temporal (T1 vs T2) imagery:",
        body_style
    ))
    
    story.append(Paragraph("• <b>The Black-Box Reliability Gap:</b> Deep learning models predict change masks but fail to provide auditability or explain why a cluster of pixels was flagged as urban expansion vs seasonal vegetation drop.", bullet_style))
    story.append(Paragraph("• <b>Multi-Modal Blind Spots:</b> Optical sensors cannot penetrate cloud cover or rain, while SAR sensors operate through all weather but suffer from complex speckle noise. Fusing them seamlessly requires deterministic cross-modal validation.", bullet_style))
    story.append(Paragraph("• <b>Heavy Geospatial Format Complexity:</b> Raw GeoTIFF rasters contain non-standard Coordinate Reference Systems (CRS), multi-band combinations (RGB, NIR, SWIR), and massive pixel dimensions that crash conventional web apps.", bullet_style))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>The SatQuery AI Solution:</b>", h2_style))
    story.append(Paragraph(
        "SatQuery AI provides an end-to-end full-stack platform featuring windowed raster ingestion, a modular multi-model registry, multi-spectral spectral index delta engines, SAR backscatter cross-validation, and an interactive Mission Control dashboard. "
        "Every analysis returns an explicit <b>Execution Trace</b>, zero-hallucination structured evidence metrics, and interactive spatial bounding boxes.",
        body_style
    ))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("Key Differentiators Summary Table", h3_style))
    
    diff_data = [
        [Paragraph("Feature / Metric", table_header_style), Paragraph("Traditional GIS / Standard AI", table_header_style), Paragraph("SatQuery AI Platform", table_header_style)],
        [Paragraph("Explainability", table_cell_style), Paragraph("Black-box probabilities; no audit trail", table_cell_style), Paragraph("Full Execution Trace + Feature Deltas (NDVI, NDWI, NDBI)", table_cell_style)],
        [Paragraph("Sensor Modalities", table_cell_style), Paragraph("Optical-only OR SAR-only in isolation", table_cell_style), Paragraph("Fused Optical-SAR Cross-Modal Engine with Agreement Scores", table_cell_style)],
        [Paragraph("Raster Processing", table_cell_style), Paragraph("Full image loading into memory (OOM crash)", table_cell_style), Paragraph("Windowed GDAL/Rasterio reading & downsampled previews", table_cell_style)],
        [Paragraph("Evidence Extraction", table_cell_style), Paragraph("LLM text generation prone to hallucination", table_cell_style), Paragraph("Deterministic Math & Rule Engine (Zero Hallucination)", table_cell_style)],
        [Paragraph("Testing & Quality", table_cell_style), Paragraph("Ad-hoc script execution", table_cell_style), Paragraph("130/130 Automated Pytest Suite covering edge cases", table_cell_style)]
    ]
    diff_table = Table(diff_data, colWidths=[110, 190, 204])
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light])
    ]))
    story.append(diff_table)
    
    story.append(Spacer(1, 12))
    
    # ---------------------------------------------------------
    # SECTION 2: SYSTEM ARCHITECTURE & 6-PHASE PIPELINE
    # ---------------------------------------------------------
    story.append(Paragraph("2. Full System Architecture & 6-Phase Pipeline", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8))
    
    story.append(Paragraph(
        "SatQuery AI follows a decoupled micro-architecture separating raster data processing, model inference adapters, agentic orchestration, and the Next.js frontend user interface.",
        body_style
    ))
    
    story.append(Paragraph("<b>End-to-End Pipeline Workflow:</b>", h2_style))
    
    pipeline_steps = [
        [Paragraph("Phase 1 & 2: Ingestion Engine", table_header_style), Paragraph("Rasterio parsing, CRS metadata validation, windowed reading, PNG thumbnail generator.", table_header_style)],
        [Paragraph("Phase 3: Change Detector", table_header_style), Paragraph("Bi-temporal pixel matrix subtraction, contour grouping, spatial bounding box extraction.", table_header_style)],
        [Paragraph("Phase 4: Change Understander", table_header_style), Paragraph("SpectralLandCoverClassifier (T1/T2), NDVI/NDWI/NDBI delta calculation, transition taxonomy.", table_header_style)],
        [Paragraph("Phase 5: Evidence Engine", table_header_style), Paragraph("Spatial, Spectral, Statistical, Temporal evidence categorization & contradiction detection.", table_header_style)],
        [Paragraph("Phase 6: Orchestrator & SAR", table_header_style), Paragraph("NLP intent routing, Optical-SAR cross-modal fusion (Agreement / Conflict analysis).", table_header_style)],
        [Paragraph("Phase 7: Adaptation & Benchmarks", table_header_style), Paragraph("BigEarthNet ResNet-50 adaptation, RSVQA & VRSBench benchmark evaluation pipeline.", table_header_style)]
    ]
    pipe_table = Table(pipeline_steps, colWidths=[150, 354])
    pipe_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), c_primary),
        ('BACKGROUND', (1,0), (1,-1), c_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pipe_table)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 3: TECHNICAL DEEP DIVE INTO EACH PHASE
    # ---------------------------------------------------------
    story.append(Paragraph("3. Technical Phase Deep Dives", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8))
    
    # Phase 1 & 2
    story.append(Paragraph("Phase 1 & 2: Ingestion, Metadata & Model Registry", h2_style))
    story.append(Paragraph(
        "The Ingestion Service processes GeoTIFF rasters with full support for multispectral bands (Red, Green, Blue, Near-Infrared / NIR, Short-Wave Infrared / SWIR). "
        "Using <code>rasterio</code>, it extracts spatial metadata: Coordinate Reference System (e.g. EPSG:4326), bounding box coordinates, pixel resolution (m/px), and band data types. "
        "Memory overhead is minimized through windowed sampling during thumbnail PNG generation.",
        body_style
    ))
    story.append(Paragraph("• <b>Lazy-Loading Model Registry:</b> Models (adapted VLM, change detectors, land-cover classifiers) are registered with lazy initialization, ensuring zero VRAM usage until an inference query arrives.", bullet_style))
    story.append(Paragraph("• <b>CPU Fallbacks:</b> Gracefully handles environments without GPU CUDA support, executing NumPy optimized matrix operations.", bullet_style))
    
    story.append(Spacer(1, 4))
    
    # Phase 3 & 4
    story.append(Paragraph("Phase 3 & 4: Change Detection & Spectral Taxonomy Understanding", h2_style))
    story.append(Paragraph(
        "Phase 3 isolates change region masks between pre-change image T1 and post-change image T2. "
        "Phase 4 then inspects each bounding box and runs the <code>SpectralLandCoverClassifier</code> on T1 and T2 pixel matrices.",
        body_style
    ))
    story.append(Paragraph("<b>Spectral Indices Formulae:</b>", h3_style))
    story.append(Paragraph("• <b>NDVI (Normalized Difference Vegetation Index):</b> <code>(NIR - Red) / (NIR + Red)</code> — Quantifies vegetation health & biomass.", bullet_style))
    story.append(Paragraph("• <b>NDWI (Normalized Difference Water Index):</b> <code>(Green - NIR) / (Green + NIR)</code> — Highlights open water bodies.", bullet_style))
    story.append(Paragraph("• <b>NDBI (Normalized Difference Built-Up Index):</b> <code>(SWIR - NIR) / (SWIR + NIR)</code> — Identifies artificial structures & concrete.", bullet_style))
    
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Transition Taxonomy Rules:</b>", h3_style))
    
    tax_data = [
        [Paragraph("Change Category", table_header_style), Paragraph("T1 Land Cover", table_header_style), Paragraph("T2 Land Cover", table_header_style), Paragraph("Spectral Trigger / Delta Signature", table_header_style)],
        [Paragraph("BUILT_UP_EXPANSION", table_cell_style), Paragraph("Agriculture / Bare Soil", table_cell_style), Paragraph("Built-Up", table_cell_style), Paragraph("NDBI Increase (> +0.05), NDVI Decrease", table_cell_style)],
        [Paragraph("DEMOLITION", table_cell_style), Paragraph("Built-Up", table_cell_style), Paragraph("Bare Soil", table_cell_style), Paragraph("NDBI Decrease (< -0.05)", table_cell_style)],
        [Paragraph("DEFORESTATION", table_cell_style), Paragraph("Forest / Vegetation", table_cell_style), Paragraph("Bare Soil", table_cell_style), Paragraph("NDVI Severe Drop (Δ < -0.30)", table_cell_style)],
        [Paragraph("VEGETATION_GAIN", table_cell_style), Paragraph("Bare Soil", table_cell_style), Paragraph("Vegetation", table_cell_style), Paragraph("NDVI Strong Surge (Δ > +0.25)", table_cell_style)],
        [Paragraph("WATER_BODY_CHANGE", table_cell_style), Paragraph("Land / Water", table_cell_style), Paragraph("Water / Land", table_cell_style), Paragraph("NDWI Delta Shift (|Δ| > +0.20)", table_cell_style)],
        [Paragraph("ROAD_DEVELOPMENT", table_cell_style), Paragraph("Agriculture", table_cell_style), Paragraph("Linear Built-Up", table_cell_style), Paragraph("NDBI Rise + High Spatial Aspect Ratio", table_cell_style)]
    ]
    tax_table = Table(tax_data, colWidths=[115, 90, 90, 209])
    tax_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light])
    ]))
    story.append(tax_table)
    
    story.append(Spacer(1, 8))
    
    # Phase 5
    story.append(Paragraph("Phase 5: Audit-Grade Evidence Extraction & Contradiction Engine", h2_style))
    story.append(Paragraph(
        "Phase 5 is the verification layer. Rather than taking Phase 4 predictions as absolute truth, Phase 5 evaluates raw pixel statistics, spatial dimensions, and spectral deltas to calculate an independent <code>evidence_score</code>.",
        body_style
    ))
    story.append(Paragraph("• <b>Contradiction Penalty:</b> If a region is classified as <i>Built-Up Expansion</i> but NDBI actually decreased, the engine penalizes the score (-0.2 per contradiction) and flags a pipeline warning.", bullet_style))
    story.append(Paragraph("• <b>Structured Categories:</b> Isolates evidence into Spatial (m² area, bbox), Spectral (NDVI/NDWI/NDBI deltas), Statistical (Mean/StdDev band ratios), Temporal (acquisition time delta), and Visual (crop coordinates).", bullet_style))
    
    story.append(Spacer(1, 4))
    
    # Phase 6
    story.append(Paragraph("Phase 6: Multi-Modal Optical-SAR Fusion & Orchestrator", h2_style))
    story.append(Paragraph(
        "SAR imagery (Sentinel-1) measures radar roughness and dielectric properties, providing cloud-penetrating surveillance. "
        "The <code>RuleBasedOpticalSARAnalyzer</code> validates spatial co-registration (CRS & transform matrices) and computes cross-modal agreement scores:",
        body_style
    ))
    story.append(Paragraph("• <b>STRONG_AGREEMENT:</b> Optical NDBI high + SAR backscatter mean high → 100% confidence Urban Structure.", bullet_style))
    story.append(Paragraph("• <b>CONFLICT:</b> Optical NDWI indicates water but SAR backscatter is high → Flagged for analyst manual review (possible double-bounce scatter from flooded structures or oil slick).", bullet_style))
    story.append(Paragraph("• <b>OPTICAL_DOMINANT / SAR_DOMINANT:</b> Selected when cloud cover degrades optical bands, relying on SAR polarizations (VV, VH).", bullet_style))
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 4: FRONTEND DASHBOARD & TECH STACK
    # ---------------------------------------------------------
    story.append(Paragraph("4. Frontend Mission Control & Full Technology Stack", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8))
    
    story.append(Paragraph(
        "The web interface is built using Next.js (React) and Vanilla CSS, featuring a sleek, dark-mode glassmorphic aesthetic tailored for military intelligence and geospatial analysts.",
        body_style
    ))
    
    story.append(Paragraph("<b>Frontend User Interface Features:</b>", h2_style))
    story.append(Paragraph("• <b>Dual-Canvas Bi-Temporal Viewer:</b> Side-by-side synchronized zoom & pan for T1 and T2 raster thumbnails with overlaid change bounding boxes.", bullet_style))
    story.append(Paragraph("• <b>Interactive Region Inspection Drawer:</b> Click any detected region box to view spectral index delta charts, land-cover transition pills, and raw band statistics.", bullet_style))
    story.append(Paragraph("• <b>Live Execution Trace Timeline:</b> Step-by-step telemetry stream displaying exact timestamped operations (e.g. <code>INGESTION_COMPLETED → CHANGE_DETECTION_COMPLETED → EVIDENCE_ATTACHED</code>).", bullet_style))
    story.append(Paragraph("• <b>Audit Evidence Breakdown Card:</b> Highlights supporting vs contradicting evidence with clear color-coded indicators.", bullet_style))
    
    story.append(Spacer(1, 8))
    story.append(Paragraph("Complete Technology Stack Overview", h3_style))
    
    tech_stack_data = [
        [Paragraph("Layer", table_header_style), Paragraph("Technologies / Libraries Used", table_header_style), Paragraph("Role & Architecture Responsibility", table_header_style)],
        [Paragraph("Frontend UI", table_cell_style), Paragraph("Next.js, React, Vanilla CSS, Lucide Icons", table_cell_style), Paragraph("Responsive Mission Control dashboard, dual raster viewer, state management", table_cell_style)],
        [Paragraph("Backend Framework", table_cell_style), Paragraph("FastAPI, Uvicorn, Pydantic, Python 3.10+", table_cell_style), Paragraph("RESTful API gateway, schema validation, async request routing", table_cell_style)],
        [Paragraph("Geospatial & Rasters", table_cell_style), Paragraph("Rasterio, GDAL, NumPy, PIL (Pillow)", table_cell_style), Paragraph("GeoTIFF ingestion, CRS validation, windowed sampling, spectral index math", table_cell_style)],
        [Paragraph("AI / ML Models", table_cell_style), Paragraph("PyTorch, ResNet-50, Scikit-Learn", table_cell_style), Paragraph("BigEarthNet multispectral adaptation, benchmark feature extractors", table_cell_style)],
        [Paragraph("Testing & Quality", table_cell_style), Paragraph("Pytest, Pytest-Asyncio, Coverage", table_cell_style), Paragraph("130 automated unit & integration tests across all 6 phases", table_cell_style)]
    ]
    tech_table = Table(tech_stack_data, colWidths=[95, 175, 234])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light])
    ]))
    story.append(tech_table)
    
    story.append(Spacer(1, 12))
    
    # ---------------------------------------------------------
    # SECTION 5: TEAM PRESENTATION & JUDGE PITCH SCRIPT
    # ---------------------------------------------------------
    story.append(Paragraph("5. Step-by-Step Team & Judge Pitch Script (5-Min Masterclass)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8))
    
    story.append(Paragraph(
        "Use this exact minute-by-minute script when presenting SatQuery AI to judges and evaluators during live demonstrations:",
        body_style
    ))
    
    pitch_script = [
        [Paragraph("Time", table_header_style), Paragraph("Speaker Focus", table_header_style), Paragraph("Exact Script / Key Phrases to Say to Judges", table_header_style)],
        [
            Paragraph("0:00 - 1:00", table_cell_style),
            Paragraph("The Hook & Problem Statement", table_cell_style),
            Paragraph("<i>\"Judges, every day millions of satellite images are captured, but defense and environmental analysts face a critical bottleneck: black-box AI models flag changes without explaining why, and cloud cover blinds optical sensors. Today, we introduce <b>SatQuery AI</b> — an audit-grade bi-temporal geospatial platform that fuses multi-spectral optical and SAR radar imagery with 100% explainable execution traces.\"</i>", table_cell_style)
        ],
        [
            Paragraph("1:00 - 2:30", table_cell_style),
            Paragraph("Live Demo & Ingestion", table_cell_style),
            Paragraph("<i>\"Watch how seamless this is. We upload a T1 pre-change GeoTIFF and a T2 post-change GeoTIFF. Our GDAL pipeline validates the CRS and streams downsampled thumbnails instantly. The Change Detection engine extracts change bounding boxes, while Phase 4 computes NDVI, NDWI, and NDBI spectral deltas to classify transitions like Deforestation or Urban Expansion.\"</i>", table_cell_style)
        ],
        [
            Paragraph("2:30 - 3:30", table_cell_style),
            Paragraph("Optical-SAR & Evidence Engine", table_cell_style),
            Paragraph("<i>\"What truly sets SatQuery AI apart is Phase 5 & 6. Notice this region: Optical NDBI indicates new construction, and our SAR engine validates it against radar backscatter, returning a <b>STRONG_AGREEMENT</b> score. If a contradiction occurs — say, spectral indices disagree — our Evidence Engine penalizes the score and alerts the analyst.\"</i>", table_cell_style)
        ],
        [
            Paragraph("3:30 - 4:30", table_cell_style),
            Paragraph("Execution Trace & Architecture", table_cell_style),
            Paragraph("<i>\"Every single step is tracked in our live Execution Trace log. No hallucinated LLM responses — strictly reproducible mathematical & spectral evidence. Our backend is backed by 130 passing Pytest unit tests, benchmarked against RSVQA and BigEarthNet datasets.\"</i>", table_cell_style)
        ],
        [
            Paragraph("4:30 - 5:00", table_cell_style),
            Paragraph("Conclusion & Call to Action", table_cell_style),
            Paragraph("<i>\"SatQuery AI bridges the gap between raw Earth Observation data and auditable, high-confidence geospatial intelligence. Thank you, and we are ready for your questions!\"</i>", table_cell_style)
        ]
    ]
    pitch_table = Table(pitch_script, colWidths=[60, 110, 334])
    pitch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light])
    ]))
    story.append(pitch_table)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 6: JUDGE Q&A DEFENSE CHEAT SHEET
    # ---------------------------------------------------------
    story.append(Paragraph("6. Judge Q&A Defense Cheat Sheet (Anticipated Technical Questions)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=8))
    
    story.append(Paragraph(
        "Be prepared to answer these exact technical questions from judges during the Q&A session:",
        body_style
    ))
    
    qa_list = [
        ("Q1: Is your system using deep learning or rule-based models?",
         "<b>Answer:</b> SatQuery AI uses a <b>hybrid modular architecture</b>. Ingestion, spectral index deltas (NDVI/NDWI/NDBI), and contradiction evidence extraction use transparent, deterministic rule engines to guarantee auditability and zero hallucination. For high-level VQA and scene captioning, we integrate PyTorch ResNet-50 adapters fine-tuned on BigEarthNet multispectral datasets. All models inherit from pluggable Abstract Base Classes (ABC), allowing seamless ML drop-in replacements."),
        
        ("Q2: How do you handle cloud cover in optical satellite imagery?",
         "<b>Answer:</b> That is why we built our <b>Phase 6 Optical-SAR Cross-Modal Fusion Engine</b>. When optical sensors are obscured by clouds (NDVI/NDWI data unavailable), the orchestrator identifies <code>SAR_DOMINANT</code> mode and uses Sentinel-1 Synthetic Aperture Radar backscatter polarizations (VV, VH), which penetrate clouds, rain, and night."),
        
        ("Q3: How do you prevent out-of-memory (OOM) crashes when uploading gigabyte-sized GeoTIFFs?",
         "<b>Answer:</b> Our Phase 1 Ingestion engine utilizes <code>rasterio</code> windowed read blocks (decimated reading). Instead of loading the full 16-bit multi-gigabyte matrix into RAM, we sample bounding windows and downsample thumbnails for UI rendering while preserving full spatial resolution metadata (CRS, transform matrix, pixel bounds) for backend processing."),
        
        ("Q4: How do you prove that your evidence engine isn't hallucinating?",
         "<b>Answer:</b> Phase 5 reuses the exact mathematical spectral index values (e.g. ΔNDVI = -0.42) calculated during Phase 4. It does not use an ungrounded LLM generator. Furthermore, if spectral indices contradict the classified transition (e.g., NDBI drops during Urban Expansion), an explicit penalty (-0.2) is applied and a warning is logged in the <b>Execution Trace</b>."),
        
        ("Q5: What benchmarks have you evaluated your platform against?",
         "<b>Answer:</b> We evaluate against <b>BigEarthNet</b> for multispectral land cover classification, as well as <b>RSVQA</b> (Remote Sensing Visual Question Answering) and <b>VRSBench</b> adapters located in our <code>benchmarks/</code> codebase.")
    ]
    
    for q_title, q_ans in qa_list:
        card_data = [
            [Paragraph(f"<b>{q_title}</b>", ParagraphStyle('QTitle', parent=table_header_style, textColor=c_primary))],
            [Paragraph(q_ans, body_style)]
        ]
        card_table = Table(card_data, colWidths=[504])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEF2FF")),
            ('BACKGROUND', (0,1), (-1,1), c_light),
            ('BOX', (0,0), (-1,-1), 1, c_secondary),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 6))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully at: {filename}")

if __name__ == "__main__":
    out_pdf = r"d:\satQUREY AI PROJECT\SatQuery_AI_Project_Summary_and_Judge_Presentation.pdf"
    build_pdf(out_pdf)
