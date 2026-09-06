/**
 * GoogleMapView — Official Google Maps JavaScript API integration for VAJRA
 * Supports dark tactical theme, satellite/hybrid view, nationwide convective grid dots across whole India,
 * dynamic 1-3km micro-grid mesh with area names, current status, and ConvGRU predictions.
 */
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { type ForecastData } from "@/hooks/useWeatherForecast";
import { type AirportMetar } from "@/hooks/useAirportMetar";
import {
  NATIONWIDE_INDIAN_GRID,
  generateNationwideConvectiveGrid,
  generateViewportGrid,
  generateLocalMicroGrid,
  findNearestIndianSector,
  isInsideIndia,
  type IndianGridSector,
  type MicroGridCell,
} from "@/lib/indiaMeteorologicalGrid";
import {
  mlNowcastingEngine,
  type AtmosphericConditioning,
  calcDewPoint,
  calcLCL,
  calcLightningFlashRate,
} from "@/lib/mlNowcastingEngine";
import { LightningCanvasOverlay } from "./LightningCanvasOverlay";
import { CloudLayerOverlay } from "./CloudLayerOverlay";

export type GoogleMapViewProps = {
  center?: { lat: number; lng?: number; lon?: number };
  zoom?: number;
  mapLayer?: "radar" | "satellite" | "admin";
  gridResolutionKm?: 1 | 1.5 | 2 | 3;
  selectedSector?: IndianGridSector | null;
  selectedMicroCell?: MicroGridCell | null;
  selectedDotCoords?: { lat: number; lon: number } | null;
  liveSectors?: IndianGridSector[];
  onSelectSector?: (sector: IndianGridSector) => void;
  onSelectMicroCell?: (cell: MicroGridCell) => void;
  onSelectDot?: (data: ForecastData | null, lat: number, lon: number) => void;
  onSelectCell?: (cell: any) => void;
  onOpen3DView?: (sector: IndianGridSector) => void;
  airports?: AirportMetar[];
  stormCells?: Array<{
    id: number;
    x: number;
    y: number;
    size: number;
    intensity: number;
    driftVx?: number;
    driftVy?: number;
    reflectivityDbz?: number;
    temperatureC?: number;
    relativeHumidity?: number;
    dewPointC?: number;
    lclCloudBaseMeters?: number;
    lightningRatePerMin?: number;
    lightningProbability?: number;
    cloudCoveragePercent?: number;
  }>;
  showStormTrails?: boolean;
  focusedLatLon?: { lat: number; lon: number } | null;
  atmosphericCond?: Partial<AtmosphericConditioning>;
  hazards?: {
    thunderstorm: boolean;
    cloudburst: boolean;
    hail: boolean;
    lightning: boolean;
    downburst: boolean;
  };
  showAttentionMap?: boolean;
  leadMinutes?: number;
  onMapReady?: (map: google.maps.Map) => void;
  onFallbackToTactical?: () => void;
  className?: string;
};

const GOOGLE_API_KEY =
  import.meta.env.VITE_GOOGLE_MAPS_API_KEY ||
  import.meta.env.VITE_FRONTEND_FORGE_API_KEY ||
  "";

function percentToLatLon(x: number, y: number) {
  const lat = 37.5 - (y / 100) * (37.5 - 6.5);
  const lon = 67 + (x / 100) * (98.5 - 67);
  return { lat, lon };
}

const X_KM = 31.5;
const Y_KM = 31.0;

function driftToVelocity(driftX: number, driftY: number) {
  const vxKmh = driftX * X_KM * 6;
  const vyKmh = -driftY * Y_KM * 6;
  const speed = Math.sqrt(vxKmh * vxKmh + vyKmh * vyKmh);
  const bearing = (Math.atan2(vxKmh, vyKmh) * 180) / Math.PI;
  return { speed, bearing: ((bearing % 360) + 360) % 360 };
}

function bearingToCardinal(b: number) {
  const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return dirs[Math.round(b / 22.5) % 16];
}

// Dark styling for Google Maps matching the VAJRA military HUD aesthetic
const darkMapStyles: google.maps.MapTypeStyle[] = [
  { elementType: "geometry", stylers: [{ color: "#07111D" }] },
  { elementType: "labels.text.stroke", stylers: [{ color: "#07111D" }] },
  { elementType: "labels.text.fill", stylers: [{ color: "#74889E" }] },
  {
    featureType: "administrative.locality",
    elementType: "labels.text.fill",
    stylers: [{ color: "#38BDF8" }],
  },
  {
    featureType: "administrative.country",
    elementType: "geometry.stroke",
    stylers: [{ color: "#1E3A5F" }, { weight: 1.5 }],
  },
  {
    featureType: "administrative.province",
    elementType: "geometry.stroke",
    stylers: [{ color: "#162B44" }, { weight: 1 }],
  },
  {
    featureType: "poi",
    stylers: [{ visibility: "off" }],
  },
  {
    featureType: "road",
    elementType: "geometry",
    stylers: [{ color: "#0E243A" }],
  },
  {
    featureType: "road",
    elementType: "geometry.stroke",
    stylers: [{ color: "#091726" }],
  },
  {
    featureType: "road.highway",
    elementType: "geometry",
    stylers: [{ color: "#153352" }],
  },
  {
    featureType: "transit",
    stylers: [{ visibility: "off" }],
  },
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#040B13" }],
  },
  {
    featureType: "water",
    elementType: "labels.text.fill",
    stylers: [{ color: "#2B4764" }],
  },
];

let googleScriptPromise: Promise<void> | null = null;

function loadGoogleMapsScript(apiKey: string): Promise<void> {
  if (googleScriptPromise) return googleScriptPromise;
  if (window.google?.maps) return Promise.resolve();

  googleScriptPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=geometry`;
    script.async = true;
    script.defer = true;
    script.onload = () => {
      if (window.google?.maps) {
        resolve();
      } else {
        reject(new Error("Google Maps object not found after load"));
      }
    };
    script.onerror = () => {
      googleScriptPromise = null;
      reject(new Error("Failed to load Google Maps script from googleapis.com"));
    };
    document.head.appendChild(script);
  });

  return googleScriptPromise;
}

export function GoogleMapView({
  center = { lat: 20.5937, lng: 78.9629 },
  zoom = 5,
  mapLayer = "radar",
  gridResolutionKm = 1.5,
  selectedSector,
  selectedMicroCell,
  selectedDotCoords,
  liveSectors = [],
  onSelectSector,
  onSelectMicroCell,
  onSelectDot,
  onSelectCell,
  onOpen3DView,
  airports = [],
  stormCells = [],
  showStormTrails = true,
  focusedLatLon = null,
  atmosphericCond = {},
  hazards = { thunderstorm: true, cloudburst: true, hail: true, lightning: true, downburst: true },
  showAttentionMap = false,
  leadMinutes = 0,
  onMapReady,
  onFallbackToTactical,
  className = "",
}: GoogleMapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<google.maps.Map | null>(null);
  const infoWindowRef = useRef<google.maps.InfoWindow | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [currentZoom, setCurrentZoom] = useState(zoom);

  // Hover telemetry state for real-time cursor placement readout
  const [hoveredSector, setHoveredSector] = useState<IndianGridSector | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);

  const liveSectorsRef = useRef(liveSectors);
  liveSectorsRef.current = liveSectors;
  const onSelectSectorRef = useRef(onSelectSector);
  onSelectSectorRef.current = onSelectSector;
  const onSelectDotRef = useRef(onSelectDot);
  onSelectDotRef.current = onSelectDot;
  const onSelectCellRef = useRef(onSelectCell);
  onSelectCellRef.current = onSelectCell;
  const onOpen3DViewRef = useRef(onOpen3DView);
  onOpen3DViewRef.current = onOpen3DView;

  // Generate dense nationwide India convective mesh dots covering all regions
  const nationwideSectors = useMemo(
    () => generateNationwideConvectiveGrid(1.2, liveSectors),
    [liveSectors]
  );
  const nationwideSectorsRef = useRef(nationwideSectors);
  nationwideSectorsRef.current = nationwideSectors;

  // Real-time lightning telemetry mapped per cell
  const lightningCells = useMemo(() => {
    return stormCells.map((sc) => {
      const { lat, lon } = percentToLatLon(sc.x, sc.y);
      const override = mlNowcastingEngine.getCellOverride(sc.id);
      const temp = override?.temperatureC ?? sc.temperatureC ?? 32;
      const rh = override?.relativeHumidity ?? sc.relativeHumidity ?? 75;
      const dbz = override?.reflectivityDbz ?? sc.reflectivityDbz ?? (36 + sc.intensity * 26);
      const cape = override?.cape ?? (atmosphericCond.cape || 2200);
      const freezing = override?.freezingLevel ?? (atmosphericCond.freezingLevelMeters || 4150);

      const metrics = calcLightningFlashRate(cape, dbz, freezing, temp, rh);

      return {
        id: sc.id,
        lat,
        lon,
        size: override?.size ?? sc.size,
        intensity: override?.intensity ?? sc.intensity,
        reflectivityDbz: dbz,
        lightningRatePerMin: sc.lightningRatePerMin ?? metrics.lightningRatePerMin,
        lightningProbability: sc.lightningProbability ?? metrics.lightningProbability,
        temperatureC: temp,
        relativeHumidity: rh,
      };
    });
  }, [stormCells, atmosphericCond]);

  // Real-time volumetric cloud clusters mapped per cell
  const cloudCells = useMemo(() => {
    return stormCells.map((sc) => {
      const { lat, lon } = percentToLatLon(sc.x, sc.y);
      const override = mlNowcastingEngine.getCellOverride(sc.id);
      const temp = override?.temperatureC ?? sc.temperatureC ?? 32;
      const rh = override?.relativeHumidity ?? sc.relativeHumidity ?? 75;
      const dp = calcDewPoint(temp, rh);
      const lcl = calcLCL(temp, dp);

      return {
        id: sc.id,
        lat,
        lon,
        size: override?.size ?? sc.size,
        intensity: override?.intensity ?? sc.intensity,
        reflectivityDbz: override?.reflectivityDbz ?? sc.reflectivityDbz ?? (36 + sc.intensity * 26),
        driftVx: override?.driftVx ?? sc.driftVx ?? 0.3,
        driftVy: override?.driftVy ?? sc.driftVy ?? -0.2,
        temperatureC: temp,
        relativeHumidity: rh,
        lclCloudBaseMeters: lcl,
        cloudCoveragePercent: Math.min(100, Math.max(25, rh * 0.9 + (4000 - lcl) * 0.008)),
      };
    });
  }, [stormCells]);

  // Pre-compute all storm track positions: past 6 steps + future 12 steps
  const allTrackPositions = useMemo(() => {
    const allLeads = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80];
    return allLeads.map((l) => ({
      lead: l,
      cells: mlNowcastingEngine.generateNowcast(Math.max(0, l), atmosphericCond).cells,
    }));
  }, [atmosphericCond]);

  // Build per-cell track (trail + projection)
  const cellTracks = useMemo(() => {
    const tracks: Record<
      number,
      {
        trail: { lat: number; lon: number; lead: number }[];
        proj: { lat: number; lon: number; lead: number }[];
        current: { lat: number; lon: number; intensity: number; size: number };
        velocity: { speed: number; bearing: number };
      }
    > = {};

    stormCells.forEach((sc) => {
      const trail: { lat: number; lon: number; lead: number }[] = [];
      const proj: { lat: number; lon: number; lead: number }[] = [];
      let current = { lat: 0, lon: 0, intensity: sc.intensity, size: sc.size };

      allTrackPositions.forEach(({ lead, cells }) => {
        const cell = cells.find((c) => c.id === sc.id);
        if (!cell) return;
        const pos = { lat: cell.lat, lon: cell.lon, lead };
        if (lead <= 0) trail.push(pos);
        else proj.push(pos);
        if (lead === 0 || (lead === 10 && trail.length === 0))
          current = { lat: cell.lat, lon: cell.lon, intensity: cell.intensity, size: cell.size };
      });

      const driftVx = sc.driftVx ?? 0;
      const driftVy = sc.driftVy ?? 0;
      const velocity = driftToVelocity(driftVx, driftVy);
      tracks[sc.id] = { trail, proj, current, velocity };
    });

    return tracks;
  }, [stormCells, allTrackPositions]);

  // Active overlay references for cleanup
  const overlaysRef = useRef<Array<google.maps.Circle | google.maps.Marker | google.maps.Polyline>>([]);

  const clearOverlays = useCallback(() => {
    overlaysRef.current.forEach((item) => item.setMap(null));
    overlaysRef.current = [];
  }, []);

  const getSeverityColor = (sev: string) => {
    switch (sev) {
      case "severe": return "#EF4444";
      case "high": return "#F97316";
      case "moderate": return "#EAB308";
      case "mild": return "#38BDF8";
      default: return "#22C55E";
    }
  };

  const formatSectorInfoWindow = (sector: IndianGridSector, clickedLat?: number, clickedLon?: number) => {
    const color = getSeverityColor(sector.severity);
    const latStr = (clickedLat ?? sector.lat).toFixed(3);
    const lonStr = (clickedLon ?? sector.lon).toFixed(3);
    return `
      <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #E2E8F0; background: #07111D; padding: 12px; border-radius: 6px; line-height: 1.5; min-width: 270px; border: 1.5px solid ${color}; box-shadow: 0 0 15px ${color}33;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <b style="color: ${color}; font-size: 13px; letter-spacing: 0.05em;">${sector.name.toUpperCase()}</b>
          <span style="background: ${color}22; color: ${color}; border: 1px solid ${color}44; padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;">
            ${sector.severity.toUpperCase()}
          </span>
        </div>
        <div style="color: #94A3B8; font-size: 10px; margin-bottom: 8px;">
          ${sector.state} (${sector.region}) · ${latStr}°N, ${lonStr}°E · Elev: ${sector.elevationMeters}M
        </div>
        <div style="margin: 6px 0; padding: 6px 0; border-top: 1px solid rgba(255,255,255,0.1); border-bottom: 1px solid rgba(255,255,255,0.1); display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px;">
          <div>STATUS: <b style="color: ${color};">${sector.statusLabel}</b></div>
          <div>RADAR: <b style="color: ${color};">${sector.reflectivityDbz} dBZ</b></div>
          <div>RAIN RATE: <b style="color: #38BDF8;">${sector.rainRateMmHr} mm/hr</b></div>
          <div>CAPE: <b style="color: #F8FAFC;">${sector.capeJkg} J/kg</b></div>
          <div>WIND GUST: <b style="color: #F8FAFC;">${sector.windGustKmh} km/h</b></div>
          <div>HAIL PROB: <b style="color: #F8FAFC;">${sector.hailProbability}%</b></div>
          <div>TEMP: <b style="color: #F8FAFC;">${sector.temperatureC}°C</b></div>
          <div>HUMIDITY: <b style="color: #F8FAFC;">${sector.humidityPercent}%</b></div>
        </div>
        <div style="font-size: 10px; color: #38BDF8; margin-top: 6px;">
          <b>+30M CONVGRU NOWCAST:</b> ${sector.predictions.t30.reflectivityDbz} dBZ (${sector.predictions.t30.trend.toUpperCase()})<br/>
          <span style="color: #CBD5E1;">${sector.predictions.summary}</span>
        </div>
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(56, 189, 248, 0.3); display: flex;">
          <button
            onclick="window.vajraOpen3DDigitalTwin && window.vajraOpen3DDigitalTwin('${sector.id}')"
            style="width: 100%; padding: 8px 12px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.3), rgba(14, 165, 233, 0.45)); border: 1.5px solid #38BDF8; border-radius: 4px; color: #FFFFFF; font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 700; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 6px; box-shadow: 0 0 14px rgba(56, 189, 248, 0.45); letter-spacing: 0.05em;"
          >
            ⚡ LAUNCH 3D STREET VIEW & AREA PERSPECTIVE
          </button>
        </div>
      </div>
    `;
  };

  // Initialize Google Maps instance
  useEffect(() => {
    let cancelled = false;

    async function init() {
      if (!GOOGLE_API_KEY) {
        setStatus("error");
        setErrorMsg("VITE_GOOGLE_MAPS_API_KEY is not set in .env");
        return;
      }

      try {
        (window as any).gm_authFailure = () => {
          if (!cancelled) {
            setStatus("error");
            setErrorMsg("Google Maps billing or API restriction error. Google Maps requires billing to be enabled on the Google Cloud project.");
          }
        };

        await loadGoogleMapsScript(GOOGLE_API_KEY);
        if (cancelled || !containerRef.current) return;

        const cLng = center.lng ?? center.lon ?? 78.9629;
        const map = new window.google.maps.Map(containerRef.current, {
          center: { lat: center.lat, lng: cLng },
          zoom: zoom,
          styles: mapLayer === "satellite" ? undefined : darkMapStyles,
          mapTypeId:
            mapLayer === "satellite"
              ? google.maps.MapTypeId.HYBRID
              : google.maps.MapTypeId.ROADMAP,
          disableDefaultUI: false,
          zoomControl: true,
          mapTypeControl: false,
          scaleControl: true,
          streetViewControl: false,
          rotateControl: false,
          fullscreenControl: false,
          backgroundColor: "#07111D",
        });

        infoWindowRef.current = new google.maps.InfoWindow();

        mapRef.current = map;
        setStatus("ready");
        onMapReady?.(map);

        map.addListener("zoom_changed", () => {
          const z = map.getZoom();
          if (typeof z === "number") setCurrentZoom(z);
        });

        // Click anywhere on map: focuses that area if inside India and shows InfoWindow
        map.addListener("click", (e: google.maps.MapMouseEvent) => {
          if (!e.latLng) return;
          const lat = e.latLng.lat();
          const lon = e.latLng.lng();

          if (!isInsideIndia(lat, lon)) return;

          const pool =
            liveSectorsRef.current && liveSectorsRef.current.length > 0
              ? liveSectorsRef.current
              : nationwideSectorsRef.current;
          const nearest = findNearestIndianSector(lat, lon, pool);

          onSelectSectorRef.current?.(nearest);
          onSelectDotRef.current?.(null, lat, lon);

          if (infoWindowRef.current) {
            infoWindowRef.current.setPosition({ lat, lng: lon });
            infoWindowRef.current.setContent(formatSectorInfoWindow(nearest, lat, lon));
            infoWindowRef.current.open(map);
          }
        });
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setErrorMsg(err instanceof Error ? err.message : "Failed to initialize Google Maps");
      }
    }

    init();

    return () => {
      cancelled = true;
      (window as any).gm_authFailure = null;
      clearOverlays();
      mapRef.current = null;
    };
  }, []);

  // Register global window bridge for InfoWindow button action
  useEffect(() => {
    (window as any).vajraOpen3DDigitalTwin = (sectorId: string) => {
      const pool = [
        ...(liveSectorsRef.current || []),
        ...(nationwideSectorsRef.current || []),
        ...NATIONWIDE_INDIAN_GRID,
      ];
      const found = pool.find((s) => s.id === sectorId) || hoveredSector;
      if (found && onOpen3DViewRef.current) {
        onOpen3DViewRef.current(found);
      }
    };
    return () => {
      delete (window as any).vajraOpen3DDigitalTwin;
    };
  }, [hoveredSector]);

  // Switch map type between dark styled roadmap and satellite
  useEffect(() => {
    const map = mapRef.current;
    if (!map || status !== "ready") return;

    if (mapLayer === "satellite") {
      map.setOptions({ styles: undefined });
      map.setMapTypeId(google.maps.MapTypeId.HYBRID);
    } else {
      map.setOptions({ styles: darkMapStyles });
      map.setMapTypeId(google.maps.MapTypeId.ROADMAP);
    }
  }, [mapLayer, status]);

  // Center pan update
  useEffect(() => {
    const map = mapRef.current;
    if (!map || status !== "ready") return;

    const cLng = center.lng ?? center.lon ?? 78.9629;
    map.panTo({ lat: center.lat, lng: cLng });
  }, [center.lat, center.lng, center.lon, status]);

  // Render whole India convective dots + 1-3km microgrid + hazards + METAR
  const renderGoogleOverlays = useCallback(() => {
    const map = mapRef.current;
    if (!map || status !== "ready") return;

    clearOverlays();

    const z = map.getZoom() ?? 5;
    const infoWindow = infoWindowRef.current;

    // 1. Storm Cells (Simulated/Predicted parent convective vortices & Movement Vectors)
    if (hazards.thunderstorm) {
      stormCells.forEach((cell) => {
        const track = cellTracks[cell.id];
        const currentPos = track?.current?.lat
          ? { lat: track.current.lat, lon: track.current.lon }
          : percentToLatLon(cell.x, cell.y);
        const velocity = track?.velocity ?? { speed: 42, bearing: 70 };
        const radiusMeters = (cell.size || 10) * 1800;

        // Motion Trails & Projections
        if (showStormTrails && track) {
          const { trail, proj } = track;

          // Historical trail (Cyan line with glow)
          if (trail.length >= 2) {
            const trailPath = trail.map((p) => ({ lat: p.lat, lng: p.lon }));
            const glow = new google.maps.Polyline({
              map,
              path: trailPath,
              strokeColor: "#38BDF8",
              strokeOpacity: 0.18,
              strokeWeight: 7,
            });
            const line = new google.maps.Polyline({
              map,
              path: trailPath,
              strokeColor: "#38BDF8",
              strokeOpacity: 0.85,
              strokeWeight: 2.5,
            });
            overlaysRef.current.push(glow, line);
          }

          // Projected route ahead with directional arrow (Orange dashed line)
          if (proj.length >= 2) {
            const projPath = [
              { lat: currentPos.lat, lng: currentPos.lon },
              ...proj.map((p) => ({ lat: p.lat, lng: p.lon })),
            ];

            const projLine = new google.maps.Polyline({
              map,
              path: projPath,
              strokeColor: "#F97316",
              strokeOpacity: 0.85,
              strokeWeight: 2,
              icons: [
                {
                  icon: {
                    path: "M 0,-1 0,1",
                    strokeOpacity: 1,
                    scale: 3,
                    strokeColor: "#F97316",
                  },
                  offset: "0",
                  repeat: "14px",
                },
                {
                  icon: {
                    path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                    scale: 3.5,
                    strokeColor: "#FFFFFF",
                    strokeWeight: 1,
                    fillColor: "#F97316",
                    fillOpacity: 1,
                  },
                  offset: "100%",
                },
              ],
            });
            overlaysRef.current.push(projLine);

            // Waypoint markers along projected path
            proj.forEach((p, i) => {
              if (i % 2 !== 0) return;
              const wp = new google.maps.Circle({
                map,
                center: { lat: p.lat, lng: p.lon },
                radius: 1400,
                fillColor: "#F97316",
                fillOpacity: Math.max(0.18, 0.65 - i * 0.08),
                strokeColor: "#F97316",
                strokeOpacity: Math.max(0.25, 0.75 - i * 0.08),
                strokeWeight: 1,
                clickable: false,
              });
              overlaysRef.current.push(wp);
            });
          }
        }

        // Storm Cell Halo & Core
        const halo = new google.maps.Circle({
          map,
          center: { lat: currentPos.lat, lng: currentPos.lon },
          radius: radiusMeters,
          strokeColor: hazards.cloudburst ? "#EF4444" : "#F97316",
          strokeOpacity: 0.8,
          strokeWeight: 1.5,
          fillColor: hazards.cloudburst ? "#EF4444" : "#F97316",
          fillOpacity: 0.14 * cell.intensity,
          clickable: false,
        });

        const core = new google.maps.Circle({
          map,
          center: { lat: currentPos.lat, lng: currentPos.lon },
          radius: radiusMeters * 0.45,
          strokeColor: "#EF4444",
          strokeOpacity: 1,
          strokeWeight: 2,
          fillColor: "#EF4444",
          fillOpacity: 0.45 * cell.intensity,
          clickable: true,
        });

        core.addListener("click", () => {
          onSelectCellRef.current?.(cell);
          if (infoWindow) {
            infoWindow.setPosition({ lat: currentPos.lat, lng: currentPos.lon });
            infoWindow.setContent(`
              <div style="font-family:'IBM Plex Mono',monospace;color:#f8fafc;padding:6px;min-width:210px;background:#07111D;">
                <div style="color:#ef4444;font-weight:bold;font-size:12px;display:flex;align-items:center;gap:4px;">
                  ⚡ CONVECTIVE EVENT #${cell.id}
                </div>
                <div style="font-size:10px;color:#94a3b8;margin:3px 0;">
                  ${currentPos.lat.toFixed(3)}°N, ${currentPos.lon.toFixed(3)}°E
                </div>
                <div style="font-size:11px;margin:5px 0;color:#38bdf8;">
                  SPEED: ${velocity.speed.toFixed(0)} KM/H · BEARING: ${bearingToCardinal(velocity.bearing)} (${velocity.bearing.toFixed(0)}°)
                </div>
                <div style="font-size:9px;color:#22c55e;border-top:1px solid rgba(255,255,255,0.1);padding-top:4px;margin-top:4px;">
                  CLICK TO TUNE IN ML STUDIO OR INSPECTOR
                </div>
              </div>
            `);
            infoWindow.open(map);
          }
        });

        // Doppler Pulse wave ring
        const dopplerWave = new google.maps.Circle({
          map,
          center: { lat: currentPos.lat, lng: currentPos.lon },
          radius: radiusMeters * 1.35,
          strokeColor: "#38BDF8",
          strokeOpacity: 0.6,
          strokeWeight: 1.5,
          fillColor: "#38BDF8",
          fillOpacity: 0.06,
          clickable: false,
        });

        // Motion vector & velocity label badge
        const labelText = `⚡ #${cell.id} · ${velocity.speed.toFixed(0)} KM/H · ${bearingToCardinal(velocity.bearing)} (${velocity.bearing.toFixed(0)}°)`;
        const badgeMarker = new google.maps.Marker({
          map,
          position: { lat: currentPos.lat, lng: currentPos.lon },
          icon: {
            path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
            rotation: velocity.bearing,
            scale: 4.5,
            fillColor: "#38BDF8",
            fillOpacity: 0.95,
            strokeColor: "#FFFFFF",
            strokeWeight: 1.2,
            anchor: new google.maps.Point(0, 3),
          },
          title: labelText,
        });

        overlaysRef.current.push(halo, core, dopplerWave, badgeMarker);
      });
    }

    // 1.5 Spatial Attention rings
    if (showAttentionMap) {
      stormCells.forEach((cell) => {
        const { lat, lon } = percentToLatLon(cell.x, cell.y);
        const attentionRadius = (cell.size || 10) * 3500;

        const attentionCircle = new google.maps.Circle({
          map,
          center: { lat, lng: lon },
          radius: attentionRadius,
          strokeColor: "#38BDF8",
          strokeOpacity: 0.9,
          strokeWeight: 2,
          fillColor: "#38BDF8",
          fillOpacity: 0.16 * cell.intensity,
          clickable: false,
        });

        overlaysRef.current.push(attentionCircle);
      });
    }

    // 2. Airport METAR markers
    airports.forEach((airport) => {
      const isSevere = airport.windGustKt && airport.windGustKt >= 30;
      const markerColor = isSevere ? "#EF4444" : airport.flightCategory === "VFR" ? "#22C55E" : "#38BDF8";

      const marker = new google.maps.Marker({
        map,
        position: { lat: airport.lat, lng: airport.lon },
        title: `${airport.icaoId} · ${airport.name}`,
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: 6,
          fillColor: markerColor,
          fillOpacity: 0.9,
          strokeColor: "#FFFFFF",
          strokeWeight: 1.5,
        },
      });

      marker.addListener("click", () => {
        const nearestSector = findNearestIndianSector(
          airport.lat,
          airport.lon,
          liveSectors.length > 0 ? liveSectors : NATIONWIDE_INDIAN_GRID
        );
        onSelectSector?.(nearestSector);
        onSelectDot?.(null, airport.lat, airport.lon);

        if (infoWindow) {
          infoWindow.setContent(`
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #E2E8F0; background: #07111D; padding: 10px; border-radius: 4px; line-height: 1.5; border: 1px solid ${markerColor};">
              <b style="color: ${markerColor};">${airport.icaoId} — ${airport.name}</b><br/>
              WIND: ${airport.windSpeedKt} KT (DIR: ${airport.windDirection}°) | GUST: ${airport.windGustKt || 0} KT<br/>
              CATEGORY: <b style="color: ${markerColor};">${airport.flightCategory}</b> · TEMP: ${airport.temp}°C
            </div>
          `);
          infoWindow.open(map, marker);
        }
      });

      overlaysRef.current.push(marker);
    });

    // 3. DENSE NATIONWIDE VIEWPORT GRID (1-3 KM CONVECTIVE AREA NODES BOUNDED TO INDIA)
    const bounds = map.getBounds();
    let viewportBounds = { north: 37.5, south: 6.5, east: 97.5, west: 68.0 };
    if (bounds) {
      const ne = bounds.getNorthEast();
      const sw = bounds.getSouthWest();
      viewportBounds = {
        north: ne.lat(),
        south: sw.lat(),
        east: ne.lng(),
        west: sw.lng(),
      };
    }

    const viewportGrid = generateViewportGrid(viewportBounds, z, gridResolutionKm, liveSectors);

    viewportGrid.forEach((sector) => {
      const isSelected =
        (selectedSector && (selectedSector.id === sector.id || (Math.abs(selectedSector.lat - sector.lat) < 0.01 && Math.abs(selectedSector.lon - sector.lon) < 0.01))) ||
        (selectedDotCoords &&
          Math.abs(selectedDotCoords.lat - sector.lat) < 0.01 &&
          Math.abs(selectedDotCoords.lon - sector.lon) < 0.01);

      const color = getSeverityColor(sector.severity);

      // Generous radius scaling to make hover and click effortless on Google Maps
      let radiusMeters = 10000;
      if (z <= 5) radiusMeters = 16000;
      else if (z <= 6) radiusMeters = 11000;
      else if (z <= 7) radiusMeters = 6000;
      else if (z <= 8) radiusMeters = 3500;
      else if (z <= 9) radiusMeters = 2000;
      else if (z <= 10) radiusMeters = 1200;
      else if (z <= 11) radiusMeters = 750;
      else if (z <= 12) radiusMeters = 450;
      else radiusMeters = Math.max(200, (gridResolutionKm * 1000) * 0.45);

      if (isSelected) radiusMeters *= 1.35;

      const circle = new google.maps.Circle({
        map,
        center: { lat: sector.lat, lng: sector.lon },
        radius: radiusMeters,
        fillColor: color,
        fillOpacity: isSelected ? 0.96 : sector.severity === "severe" ? 0.88 : 0.72,
        strokeColor: isSelected ? "#FFFFFF" : color,
        strokeWeight: isSelected ? 2.5 : 1,
      });

      // Cursor placement (hover) event listeners
      circle.addListener("mouseover", (e: google.maps.MapMouseEvent) => {
        setHoveredSector(sector);
        if (e.domEvent && e.domEvent instanceof MouseEvent) {
          setHoverPos({ x: e.domEvent.clientX, y: e.domEvent.clientY });
        }
        circle.setOptions({
          strokeColor: "#38BDF8",
          strokeWeight: 3,
          fillOpacity: 0.98,
        });
      });

      circle.addListener("mousemove", (e: google.maps.MapMouseEvent) => {
        if (e.domEvent && e.domEvent instanceof MouseEvent) {
          setHoverPos({ x: e.domEvent.clientX, y: e.domEvent.clientY });
        }
      });

      circle.addListener("mouseout", () => {
        setHoveredSector(null);
        setHoverPos(null);
        circle.setOptions({
          strokeColor: isSelected ? "#FFFFFF" : color,
          strokeWeight: isSelected ? 2.5 : 1,
          fillOpacity: isSelected ? 0.96 : sector.severity === "severe" ? 0.88 : 0.72,
        });
      });

      // Click event listener
      circle.addListener("click", () => {
        onSelectSector?.(sector);
        onSelectDot?.(null, sector.lat, sector.lon);

        if (infoWindow) {
          infoWindow.setPosition({ lat: sector.lat, lng: sector.lon });
          infoWindow.setContent(formatSectorInfoWindow(sector));
          infoWindow.open(map);
        }
      });

      overlaysRef.current.push(circle);
    });
  }, [
    nationwideSectors,
    stormCells,
    showStormTrails,
    cellTracks,
    airports,
    hazards,
    showAttentionMap,
    gridResolutionKm,
    selectedSector,
    selectedMicroCell,
    selectedDotCoords,
    liveSectors,
    status,
    onSelectSector,
    onSelectMicroCell,
    onSelectDot,
    clearOverlays,
  ]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || status !== "ready") return;

    renderGoogleOverlays();

    const listener = map.addListener("idle", renderGoogleOverlays);

    return () => {
      google.maps.event.removeListener(listener);
    };
  }, [renderGoogleOverlays, status]);

  return (
    <div
      className={`relative w-full h-full ${className}`}
      style={{
        position: "absolute",
        inset: 0,
        background: "#07111D",
        overflow: "hidden",
      }}
    >
      {/* Isolated map canvas container with zero React children */}
      <div
        ref={containerRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
        }}
      />

      {/* Floating Tactical HUD Hover Tooltip for instant cursor placement information */}
      {hoveredSector && hoverPos && (
        <div
          style={{
            position: "fixed",
            left: Math.min(window.innerWidth - 320, hoverPos.x + 18),
            top: Math.min(window.innerHeight - 250, hoverPos.y + 18),
            zIndex: 99999,
            pointerEvents: "none",
            background: "rgba(7, 17, 29, 0.96)",
            backdropFilter: "blur(8px)",
            border: `1.5px solid ${getSeverityColor(hoveredSector.severity)}`,
            boxShadow: `0 8px 32px rgba(0, 0, 0, 0.8), 0 0 16px ${getSeverityColor(hoveredSector.severity)}44`,
            borderRadius: "6px",
            padding: "10px 14px",
            fontFamily: "'IBM Plex Mono', monospace",
            color: "#E2E8F0",
            fontSize: "11px",
            lineHeight: 1.5,
            minWidth: "260px",
            maxWidth: "320px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignContent: "center", marginBottom: "4px" }}>
            <b style={{ color: getSeverityColor(hoveredSector.severity), fontSize: "12px" }}>
              {hoveredSector.name.toUpperCase()}
            </b>
            <span
              style={{
                background: `${getSeverityColor(hoveredSector.severity)}22`,
                color: getSeverityColor(hoveredSector.severity),
                border: `1px solid ${getSeverityColor(hoveredSector.severity)}44`,
                padding: "1px 5px",
                borderRadius: "3px",
                fontSize: "9px",
                fontWeight: "bold",
              }}
            >
              {hoveredSector.severity.toUpperCase()}
            </span>
          </div>
          <div style={{ color: "#94A3B8", fontSize: "10px", marginBottom: "6px" }}>
            {hoveredSector.state} ({hoveredSector.region}) · {hoveredSector.lat.toFixed(3)}°N, {hoveredSector.lon.toFixed(3)}°E · Elev: {hoveredSector.elevationMeters}M
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "4px 8px",
              padding: "5px 0",
              borderTop: "1px solid rgba(255,255,255,0.1)",
              borderBottom: "1px solid rgba(255,255,255,0.1)",
              marginBottom: "6px",
            }}
          >
            <div>STATUS: <b style={{ color: getSeverityColor(hoveredSector.severity) }}>{hoveredSector.statusLabel}</b></div>
            <div>RADAR: <b style={{ color: getSeverityColor(hoveredSector.severity) }}>{hoveredSector.reflectivityDbz} dBZ</b></div>
            <div>RAIN RATE: <b style={{ color: "#38BDF8" }}>{hoveredSector.rainRateMmHr} mm/hr</b></div>
            <div>CAPE: <b style={{ color: "#F8FAFC" }}>{hoveredSector.capeJkg} J/kg</b></div>
            <div>WIND GUST: <b style={{ color: "#F8FAFC" }}>{hoveredSector.windGustKmh} km/h</b></div>
            <div>HAIL PROB: <b style={{ color: "#F8FAFC" }}>{hoveredSector.hailProbability}%</b></div>
          </div>
          <div style={{ fontSize: "10px", color: "#38BDF8" }}>
            <b>+30M CONVGRU:</b> {hoveredSector.predictions.t30.reflectivityDbz} dBZ ({hoveredSector.predictions.t30.trend.toUpperCase()})
          </div>
        </div>
      )}

      {status === "loading" && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(7, 17, 29, 0.85)",
            zIndex: 10,
            color: "#38BDF8",
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "12px",
            letterSpacing: "0.08em",
          }}
        >
          CONNECTING TO GOOGLE MAPS API & NATIONWIDE GRID...
        </div>
      )}
      {status === "error" && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: "10px",
            background: "rgba(7, 17, 29, 0.95)",
            zIndex: 10,
            color: "#EF4444",
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "12px",
            letterSpacing: "0.08em",
            padding: "20px",
            textAlign: "center",
          }}
        >
          <span>GOOGLE MAPS NOTICE: {errorMsg}</span>
          {onFallbackToTactical && (
            <button
              onClick={onFallbackToTactical}
              style={{
                marginTop: "8px",
                padding: "8px 16px",
                background: "rgba(56, 189, 248, 0.15)",
                border: "1px solid #38BDF8",
                borderRadius: "4px",
                color: "#38BDF8",
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "11px",
                fontWeight: "bold",
                cursor: "pointer",
              }}
            >
              SWITCH TO 100% FREE TACTICAL LEAFLET MAP
            </button>
          )}
        </div>
      )}

      {/* Dynamic Animated Cloud Cover and Lightning Visual Overlays */}
      <CloudLayerOverlay
        map={mapRef.current}
        engine="google"
        cells={cloudCells}
        leadMinutes={leadMinutes}
        enabled={hazards.thunderstorm || hazards.lightning}
      />
      <LightningCanvasOverlay
        map={mapRef.current}
        engine="google"
        cells={lightningCells}
        enabled={hazards.lightning}
      />

      {/* Zoom and Grid Resolution readout */}
      <div
        style={{
          position: "absolute",
          bottom: 12,
          left: 12,
          zIndex: 500,
          background: "rgba(9, 21, 34, 0.85)",
          border: "1px solid rgba(56, 189, 248, 0.2)",
          borderRadius: 4,
          padding: "3px 8px",
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: "10px",
          color: "#94A3B8",
          letterSpacing: "0.05em",
          pointerEvents: "none",
        }}
      >
        GOOGLE MAPS · ZOOM: <span style={{ color: "#38BDF8" }}>{currentZoom}</span> · MESH:{" "}
        <span style={{ color: "#38BDF8" }}>{gridResolutionKm} KM</span> · SECTORS:{" "}
        <span style={{ color: "#38BDF8" }}>{nationwideSectors.length} NATIONWIDE</span>
      </div>
    </div>
  );
}
