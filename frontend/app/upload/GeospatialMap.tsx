"use client";

import { useEffect, useRef, useState } from "react";
import L from "leaflet";

export interface GeoRegion {
  region_id: string;
  change_type: string;
  area_sq_m?: number;
  pixel_area?: number;
  geospatial_area?: number;
  centroid?: [number, number]; // [lon, lat] or [x, y]
  bbox: [number, number, number, number]; // [west, south, east, north]
  confidence: number;
  description?: string;
  evidence?: any;
  t1_land_cover?: string;
  t2_land_cover?: string;
}

export interface MapAsset {
  id: string;
  filename: string;
  bounds?: number[]; // [west, south, east, north]
  crs?: string;
  resolution?: number[];
  width: number;
  height: number;
  modality?: string;
}

interface GeospatialMapProps {
  t1Asset?: MapAsset | null;
  t2Asset?: MapAsset | null;
  regions?: GeoRegion[];
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string) => void;
}

// Color map for change categories
const CHANGE_COLORS: Record<string, string> = {
  BUILT_UP_EXPANSION: "#f43f5e", // Crimson / Rose
  built_up_expansion: "#f43f5e",
  DEFORESTATION: "#ea580c",     // Orange
  deforestation: "#ea580c",
  VEGETATION_GAIN: "#10b981",   // Emerald
  vegetation_gain: "#10b981",
  WATER_BODY_CHANGE: "#06b6d4", // Cyan
  water_body_change: "#06b6d4",
  DEMOLITION: "#e11d48",        // Red
  demolition: "#e11d48",
  AGRICULTURAL_CONVERSION: "#eab308", // Amber
  agricultural_conversion: "#eab308",
  ROAD_INFRASTRUCTURE_DEVELOPMENT: "#8b5cf6", // Purple
  infrastructure_development: "#8b5cf6",
  OTHER: "#38bdf8",
  other: "#38bdf8",
};

export default function GeospatialMap({
  t1Asset,
  t2Asset,
  regions = [],
  selectedRegionId,
  onSelectRegion,
}: GeospatialMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layersGroupRef = useRef<L.FeatureGroup | null>(null);

  const [activeBasemap, setActiveBasemap] = useState<"satellite" | "dark" | "osm">("satellite");
  const [showT1Bounds, setShowT1Bounds] = useState(true);
  const [showT2Bounds, setShowT2Bounds] = useState(true);
  const [showRegions, setShowRegions] = useState(true);
  const [showCentroids, setShowCentroids] = useState(true);
  const [isGeoreferenced, setIsGeoreferenced] = useState(true);
  const [mapStats, setMapStats] = useState({ zoom: 13, center: [0, 0] });

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    // Fix default marker icon paths in Next.js
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
      iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
      shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
    });

    const defaultCenter: [number, number] = [20.5937, 78.9629]; // Default to India center
    const map = L.map(mapContainerRef.current, {
      center: defaultCenter,
      zoom: 5,
      zoomControl: false,
      attributionControl: false,
    });

    // Custom dark zoom control
    L.control.zoom({ position: "bottomright" }).addTo(map);

    // Feature group for drawing dynamic overlays
    const layersGroup = L.featureGroup().addTo(map);
    layersGroupRef.current = layersGroup;
    mapInstanceRef.current = map;

    map.on("moveend", () => {
      const c = map.getCenter();
      setMapStats({ zoom: map.getZoom(), center: [Number(c.lat.toFixed(4)), Number(c.lng.toFixed(4))] });
    });

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Basemap Tiles
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Remove existing tile layers
    map.eachLayer((layer) => {
      if (layer instanceof L.TileLayer) {
        map.removeLayer(layer);
      }
    });

    let tileUrl = "";
    let maxZoom = 19;
    if (activeBasemap === "satellite") {
      tileUrl = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
    } else if (activeBasemap === "dark") {
      tileUrl = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
    } else {
      tileUrl = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
    }

    L.tileLayer(tileUrl, {
      maxZoom,
      attribution: "&copy; Esri, OpenStreetMap, CartoDB",
    }).addTo(map);
  }, [activeBasemap]);

  // Render Overlays: Footprints, Change Regions, and Centroids
  useEffect(() => {
    const map = mapInstanceRef.current;
    const layersGroup = layersGroupRef.current;
    if (!map || !layersGroup) return;

    layersGroup.clearLayers();

    const boundsToFit: L.LatLngBounds[] = [];

    // Helper to validate and convert bounds [w, s, e, n]
    const getLatLngBounds = (b?: number[]): L.LatLngBounds | null => {
      if (!b || b.length !== 4) return null;
      const [west, south, east, north] = b;
      // Sanity check for WGS84 range
      if (south >= -90 && north <= 90 && west >= -180 && east <= 180) {
        return L.latLngBounds([south, west], [north, east]);
      }
      // Pixel coordinate fallback (0..width, 0..height) mapping to default LatLng area
      const baseLat = 20.5937;
      const baseLng = 78.9629;
      const scale = 0.0001; // ~10m scale per pixel
      return L.latLngBounds(
        [baseLat + south * scale, baseLng + west * scale],
        [baseLat + north * scale, baseLng + east * scale]
      );
    };

    let hasGeoref = false;

    // 1. Render T1 Footprint
    if (t1Asset && t1Asset.bounds && showT1Bounds) {
      const b = getLatLngBounds(t1Asset.bounds);
      if (b) {
        hasGeoref = true;
        boundsToFit.push(b);
        const rect = L.rectangle(b, {
          color: "#06b6d4",
          weight: 2,
          dashArray: "6, 6",
          fillColor: "#06b6d4",
          fillOpacity: 0.08,
        });
        rect.bindTooltip(`T1 Footprint: ${t1Asset.filename} (${t1Asset.width}×${t1Asset.height})`, {
          sticky: true,
          className: "leaflet-custom-tooltip",
        });
        layersGroup.addLayer(rect);
      }
    }

    // 2. Render T2 Footprint
    if (t2Asset && t2Asset.bounds && showT2Bounds) {
      const b = getLatLngBounds(t2Asset.bounds);
      if (b) {
        hasGeoref = true;
        boundsToFit.push(b);
        const rect = L.rectangle(b, {
          color: "#8b5cf6",
          weight: 2,
          dashArray: "4, 4",
          fillColor: "#8b5cf6",
          fillOpacity: 0.08,
        });
        rect.bindTooltip(`T2 Footprint: ${t2Asset.filename} (${t2Asset.width}×${t2Asset.height})`, {
          sticky: true,
          className: "leaflet-custom-tooltip",
        });
        layersGroup.addLayer(rect);
      }
    }

    // 3. Render Change Regions
    if (showRegions && regions.length > 0) {
      regions.forEach((reg) => {
        const color = CHANGE_COLORS[reg.change_type] || "#38bdf8";
        const isSelected = selectedRegionId === reg.region_id;
        const b = getLatLngBounds(reg.bbox);

        if (b) {
          hasGeoref = true;
          boundsToFit.push(b);
          const rect = L.rectangle(b, {
            color: isSelected ? "#ffffff" : color,
            weight: isSelected ? 3 : 2,
            fillColor: color,
            fillOpacity: isSelected ? 0.45 : 0.22,
          });

          // Click handler
          rect.on("click", () => {
            if (onSelectRegion) onSelectRegion(reg.region_id);
          });

          // Rich HTML popup
          const area = reg.area_sq_m || reg.geospatial_area || (reg.pixel_area ? reg.pixel_area * 100 : 0);
          const conf = Math.round(reg.confidence * 100);
          const popupContent = `
            <div style="font-family: inherit; font-size: 12px; color: #f1f5f9; padding: 4px;">
              <div style="font-weight: 700; color: ${color}; font-size: 13px; text-transform: uppercase; margin-bottom: 4px;">
                ${reg.change_type.replace(/_/g, " ")}
              </div>
              <div style="color: #94a3b8; margin-bottom: 6px;">Region ID: <span style="color: #cbd5e1;">${reg.region_id}</span></div>
              <div style="display: flex; gap: 8px; margin-bottom: 6px;">
                <span style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">Confidence: <strong>${conf}%</strong></span>
                <span style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">Area: <strong>${area.toFixed(1)} m²</strong></span>
              </div>
              ${reg.t1_land_cover && reg.t2_land_cover ? `
                <div style="font-size: 11px; color: #38bdf8; margin-bottom: 4px;">
                  Transition: ${reg.t1_land_cover} → ${reg.t2_land_cover}
                </div>
              ` : ""}
              <div style="font-size: 11px; color: #94a3b8; line-height: 1.3;">
                ${reg.description || "Detected spectral change region."}
              </div>
            </div>
          `;

          rect.bindPopup(popupContent, {
            className: "leaflet-custom-popup",
            closeButton: false,
          });

          layersGroup.addLayer(rect);
        }

        // 4. Render Centroid Marker
        if (showCentroids && reg.centroid && reg.centroid.length === 2) {
          const [lon, lat] = reg.centroid;
          if (lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
            hasGeoref = true;
            const circle = L.circleMarker([lat, lon], {
              radius: isSelected ? 8 : 5,
              color: "#ffffff",
              weight: isSelected ? 2 : 1.5,
              fillColor: color,
              fillOpacity: 0.9,
            });

            circle.on("click", () => {
              if (onSelectRegion) onSelectRegion(reg.region_id);
            });

            circle.bindTooltip(`Region ${reg.region_id}: ${reg.change_type}`, {
              sticky: true,
              className: "leaflet-custom-tooltip",
            });

            layersGroup.addLayer(circle);
          }
        }
      });
    }

    setIsGeoreferenced(hasGeoref || (!t1Asset && !t2Asset));

    // Auto-fit to active layers
    if (boundsToFit.length > 0) {
      let merged = boundsToFit[0];
      boundsToFit.forEach((b) => {
        merged = merged.extend(b);
      });
      map.fitBounds(merged, { padding: [40, 40], maxZoom: 17, animate: true });
    }
  }, [t1Asset, t2Asset, regions, selectedRegionId, showT1Bounds, showT2Bounds, showRegions, showCentroids]);

  const handleResetView = () => {
    const map = mapInstanceRef.current;
    const layersGroup = layersGroupRef.current;
    if (!map || !layersGroup) return;

    const bounds = layersGroup.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [30, 30], maxZoom: 17 });
    } else {
      map.setView([20.5937, 78.9629], 5);
    }
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", borderRadius: "12px", overflow: "hidden", border: "1px solid var(--color-border)" }}>
      {/* Leaflet Map DOM Node */}
      <div ref={mapContainerRef} style={{ width: "100%", height: "100%", background: "#0a0e1a" }} />

      {/* Non-Georeferenced Notice Banner */}
      {!isGeoreferenced && (
        <div style={{
          position: "absolute",
          top: 12,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 1000,
          background: "rgba(245, 158, 11, 0.9)",
          color: "#000",
          padding: "6px 14px",
          borderRadius: "20px",
          fontSize: "12px",
          fontWeight: 600,
          boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
          display: "flex",
          alignItems: "center",
          gap: "6px",
        }}>
          <span>⚠️</span> Non-Georeferenced Asset (Displaying in Normalized Spatial Canvas)
        </div>
      )}

      {/* Floating Layer & Basemap Control Panel */}
      <div style={{
        position: "absolute",
        top: 12,
        right: 12,
        zIndex: 1000,
        background: "rgba(17, 24, 39, 0.85)",
        backdropFilter: "blur(12px)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: "10px",
        padding: "10px 14px",
        color: "#f1f5f9",
        fontSize: "12px",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        boxShadow: "0 8px 24px rgba(0,0,0,0.6)",
        maxWidth: "220px",
      }}>
        <div style={{ fontWeight: 700, fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.05em", color: "#06b6d4", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "4px" }}>
          Geospatial Layers
        </div>

        {/* Basemap Selection */}
        <div style={{ display: "flex", gap: "4px", marginTop: "2px" }}>
          {(["satellite", "dark", "osm"] as const).map((bm) => (
            <button
              key={bm}
              onClick={() => setActiveBasemap(bm)}
              style={{
                flex: 1,
                padding: "3px 6px",
                borderRadius: "4px",
                border: activeBasemap === bm ? "1px solid #06b6d4" : "1px solid rgba(255,255,255,0.1)",
                background: activeBasemap === bm ? "rgba(6, 182, 212, 0.2)" : "rgba(255,255,255,0.05)",
                color: activeBasemap === bm ? "#06b6d4" : "#94a3b8",
                fontSize: "10px",
                fontWeight: 600,
                cursor: "pointer",
                textTransform: "capitalize",
              }}
            >
              {bm}
            </button>
          ))}
        </div>

        {/* Layer Checkboxes */}
        <div style={{ display: "flex", flexDirection: "column", gap: "5px", marginTop: "4px" }}>
          <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", color: "#cbd5e1" }}>
            <input
              type="checkbox"
              checked={showT1Bounds}
              onChange={(e) => setShowT1Bounds(e.target.checked)}
              style={{ accentColor: "#06b6d4" }}
            />
            <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", background: "#06b6d4" }} />
            T1 Footprint
          </label>

          <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", color: "#cbd5e1" }}>
            <input
              type="checkbox"
              checked={showT2Bounds}
              onChange={(e) => setShowT2Bounds(e.target.checked)}
              style={{ accentColor: "#8b5cf6" }}
            />
            <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", background: "#8b5cf6" }} />
            T2 Footprint
          </label>

          <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", color: "#cbd5e1" }}>
            <input
              type="checkbox"
              checked={showRegions}
              onChange={(e) => setShowRegions(e.target.checked)}
              style={{ accentColor: "#f43f5e" }}
            />
            <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", background: "#f43f5e" }} />
            Change Polygons ({regions.length})
          </label>

          <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", color: "#cbd5e1" }}>
            <input
              type="checkbox"
              checked={showCentroids}
              onChange={(e) => setShowCentroids(e.target.checked)}
              style={{ accentColor: "#10b981" }}
            />
            <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", background: "#10b981" }} />
            Region Centroids
          </label>
        </div>

        {/* Reset View Button */}
        <button
          onClick={handleResetView}
          style={{
            marginTop: "6px",
            padding: "5px 10px",
            borderRadius: "6px",
            background: "rgba(6, 182, 212, 0.15)",
            border: "1px solid rgba(6, 182, 212, 0.4)",
            color: "#06b6d4",
            fontSize: "11px",
            fontWeight: 600,
            cursor: "pointer",
            textAlign: "center",
          }}
        >
          🔍 Recenter View
        </button>
      </div>

      {/* Coordinate & Zoom Status Footer */}
      <div style={{
        position: "absolute",
        bottom: 12,
        left: 12,
        zIndex: 1000,
        background: "rgba(17, 24, 39, 0.75)",
        backdropFilter: "blur(8px)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: "6px",
        padding: "4px 10px",
        color: "#94a3b8",
        fontSize: "11px",
        fontFamily: "monospace",
      }}>
        Zoom: {mapStats.zoom} | Center: [{mapStats.center[0]}, {mapStats.center[1]}] | Regions: {regions.length}
      </div>
    </div>
  );
}
