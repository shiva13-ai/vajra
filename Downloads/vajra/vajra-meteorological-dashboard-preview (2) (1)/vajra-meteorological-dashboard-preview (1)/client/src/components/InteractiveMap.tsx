/**
 * InteractiveMap — VAJRA Tactical Leaflet Map
 *
 * LIVE features:
 *  • RainViewer radar tiles (auto-playing, continuous stream)
 *  • RainViewer satellite infrared tiles (cloud-top temperature)
 *  • Animated storm cell movement: airplane-style trail + projected path + phase badge
 *  • Sharp storm boundary rings with gradient halos
 *  • Rain arrival ETA rings per storm cell
 *  • Doppler scan sweep animation overlay
 *  • Storm phase detection: GENESIS / MATURE / DISSIPATING
 */
import {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { type ForecastData } from "@/hooks/useWeatherForecast";
import { type AirportMetar } from "@/hooks/useAirportMetar";
import { type RVFrame } from "@/hooks/useRainViewer";
import {
  generateNationwideConvectiveGrid,
  generateViewportGrid,
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

// India domain bounds
const INDIA = { north: 37.5, south: 6.5, west: 67.0, east: 98.5 };
const X_KM = 31.5; // km per 1% of x-domain
const Y_KM = 31.0; // km per 1% of y-domain

export type StormCellInput = {
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
};

export type InteractiveMapProps = {
  center?: { lat: number; lng?: number; lon?: number };
  zoom?: number;
  mapLayer?: "radar" | "satellite" | "admin";
  showRadar?: boolean;
  showSatelliteIR?: boolean;
  radarFrames?: RVFrame[];
  satelliteFrames?: RVFrame[];
  radarHost?: string;
  currentRadarFrameIndex?: number;
  currentSatFrameIndex?: number;
  gridResolutionKm?: 1 | 1.5 | 2 | 3;
  liveSectors?: IndianGridSector[];
  selectedSector?: IndianGridSector | null;
  selectedMicroCell?: MicroGridCell | null;
  selectedDotCoords?: { lat: number; lon: number } | null;
  onSelectSector?: (sector: IndianGridSector) => void;
  onSelectMicroCell?: (cell: MicroGridCell) => void;
  onSelectDot?: (data: ForecastData | null, lat: number, lon: number) => void;
  onOpen3DView?: (sector: IndianGridSector) => void;
  airports?: AirportMetar[];
  stormCells?: StormCellInput[];
  atmosphericCond?: Partial<AtmosphericConditioning>;
  hazards?: {
    thunderstorm: boolean;
    cloudburst: boolean;
    hail: boolean;
    lightning: boolean;
    downburst: boolean;
  };
  showAttentionMap?: boolean;
  showStormTrails?: boolean;
  focusedLatLon?: { lat: number; lon: number } | null;
  leadMinutes?: number;
  onMapReady?: (map: L.Map) => void;
  className?: string;
};

function pctToLatLon(x: number, y: number) {
  const lat = INDIA.north - (y / 100) * (INDIA.north - INDIA.south);
  const lon = INDIA.west + (x / 100) * (INDIA.east - INDIA.west);
  return { lat, lon };
}

// Compute speed (km/h) and bearing from drift vectors
function driftToVelocity(driftX: number, driftY: number) {
  const vxKmh = driftX * X_KM * 6; // per step (10 min) → km/h
  const vyKmh = -driftY * Y_KM * 6; // flip y (south is positive in domain)
  const speed = Math.sqrt(vxKmh * vxKmh + vyKmh * vyKmh);
  const bearing = (Math.atan2(vxKmh, vyKmh) * 180) / Math.PI;
  return { speed, bearing: ((bearing % 360) + 360) % 360 };
}

function bearingToCardinal(b: number) {
  const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  return dirs[Math.round(b / 22.5) % 16];
}

function stormPhase(intensity: number, leadMin: number): { label: string; color: string } {
  // Approximate phase based on intensity and forecast evolution
  if (intensity >= 0.85) return { label: "MATURE", color: "#EF4444" };
  if (intensity >= 0.65) return { label: "INTENSIFYING", color: "#F97316" };
  if (intensity >= 0.45) return { label: "ACTIVE", color: "#EAB308" };
  if (intensity >= 0.30) return { label: "DISSIPATING", color: "#38BDF8" };
  return { label: "GENESIS", color: "#22C55E" };
}

function etaMinutes(
  stormLat: number,
  stormLon: number,
  targetLat: number,
  targetLon: number,
  speedKmh: number
): number {
  const dLat = (targetLat - stormLat) * 111;
  const dLon =
    (targetLon - stormLon) *
    111 *
    Math.cos((((stormLat + targetLat) / 2) * Math.PI) / 180);
  const dist = Math.sqrt(dLat * dLat + dLon * dLon);
  if (speedKmh <= 0) return 999;
  return Math.round((dist / speedKmh) * 60);
}

export function InteractiveMap({
  center = { lat: 20.5937, lng: 78.9629 },
  zoom = 5,
  mapLayer = "radar",
  showRadar = true,
  showSatelliteIR = false,
  radarFrames = [],
  satelliteFrames = [],
  radarHost = "",
  currentRadarFrameIndex = 0,
  currentSatFrameIndex = 0,
  gridResolutionKm = 1.5,
  liveSectors = [],
  selectedSector,
  selectedMicroCell,
  selectedDotCoords,
  onSelectSector,
  onSelectMicroCell,
  onSelectDot,
  onOpen3DView,
  airports = [],
  stormCells = [],
  atmosphericCond = {},
  hazards = {
    thunderstorm: true,
    cloudburst: true,
    hail: true,
    lightning: true,
    downburst: true,
  },
  showAttentionMap = false,
  showStormTrails = true,
  focusedLatLon,
  leadMinutes = 0,
  onMapReady,
  className = "",
}: InteractiveMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const radarLayerRef = useRef<L.TileLayer | null>(null);
  const satelliteIRLayerRef = useRef<L.TileLayer | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const trailLayerRef = useRef<L.LayerGroup | null>(null);
  const scanLayerRef = useRef<L.LayerGroup | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const [currentZoom, setCurrentZoom] = useState(zoom);

  const liveSectorsRef = useRef(liveSectors);
  liveSectorsRef.current = liveSectors;
  const onSelectSectorRef = useRef(onSelectSector);
  onSelectSectorRef.current = onSelectSector;
  const onOpen3DViewRef = useRef(onOpen3DView);
  onOpen3DViewRef.current = onOpen3DView;
  const onSelectDotRef = useRef(onSelectDot);
  onSelectDotRef.current = onSelectDot;

  const nationwideSectors = useMemo(
    () => generateNationwideConvectiveGrid(1.2, liveSectors),
    [liveSectors]
  );
  const nationwideSectorsRef = useRef(nationwideSectors);
  nationwideSectorsRef.current = nationwideSectors;

  // Real-time lightning telemetry mapped per cell
  const lightningCells = useMemo(() => {
    return stormCells.map((sc) => {
      const { lat, lon } = pctToLatLon(sc.x, sc.y);
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
      const { lat, lon } = pctToLatLon(sc.x, sc.y);
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

  // ── Initialize Leaflet ──────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const cLng = center.lng ?? center.lon ?? 78.9629;
    const map = L.map(containerRef.current, {
      center: [center.lat, cLng],
      zoom,
      minZoom: 4,
      maxZoom: 18,
      zoomControl: false,
      attributionControl: false,
    });

    // Base dark tile
    const baseTile = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
      { subdomains: "abcd", maxZoom: 19 }
    ).addTo(map);
    tileLayerRef.current = baseTile;

    // Layer groups
    const trailLayer = L.layerGroup().addTo(map);
    trailLayerRef.current = trailLayer;

    const markersGroup = L.layerGroup().addTo(map);
    markersLayerRef.current = markersGroup;

    const scanLayer = L.layerGroup().addTo(map);
    scanLayerRef.current = scanLayer;

    L.control.zoom({ position: "topright" }).addTo(map);

    mapRef.current = map;
    onMapReady?.(map);

    map.on("click", (e: L.LeafletMouseEvent) => {
      const { lat, lng: lon } = e.latlng;
      if (!isInsideIndia(lat, lon)) return;
      const pool =
        liveSectorsRef.current && liveSectorsRef.current.length > 0
          ? liveSectorsRef.current
          : nationwideSectorsRef.current;
      const nearest = findNearestIndianSector(lat, lon, pool);
      onSelectSectorRef.current?.(nearest);
      onSelectDotRef.current?.(null, lat, lon);
    });

    const timer = setTimeout(() => map.invalidateSize(), 200);
    map.on("zoomend", () => setCurrentZoom(map.getZoom()));

    return () => {
      clearTimeout(timer);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // ── Base tile layer swap ────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (tileLayerRef.current) map.removeLayer(tileLayerRef.current);

    let url = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png";
    let subdomains: string = "abcd";
    let maxZoom = 19;

    if (mapLayer === "satellite") {
      url = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}";
      subdomains = "abc";
      maxZoom = 20;
    } else if (mapLayer === "admin") {
      url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
      subdomains = "abc";
    }

    const newTile = L.tileLayer(url, { subdomains, maxZoom, opacity: 1 }).addTo(map);
    tileLayerRef.current = newTile;
    map.invalidateSize();
  }, [mapLayer]);

  // ── Radar tile layer (live auto-playing RainViewer) ─────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (radarLayerRef.current) {
      map.removeLayer(radarLayerRef.current);
      radarLayerRef.current = null;
    }

    const frame = radarFrames[currentRadarFrameIndex];
    if (!showRadar || !frame || !radarHost) return;

    // Color 2 = smooth colorful radar, options 1_1 = colour + smooth
    const radarUrl = `${radarHost}${frame.path}/512/{z}/{x}/{y}/2/1_1.png`;
    const radarLayer = L.tileLayer(radarUrl, {
      opacity: frame.isNowcast ? 0.55 : 0.72,
      zIndex: 400,
      className: frame.isNowcast ? "radar-nowcast-tile" : "radar-past-tile",
    }).addTo(map);
    radarLayerRef.current = radarLayer;
  }, [showRadar, radarFrames, currentRadarFrameIndex, radarHost]);

  // ── Satellite IR tile layer ─────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (satelliteIRLayerRef.current) {
      map.removeLayer(satelliteIRLayerRef.current);
      satelliteIRLayerRef.current = null;
    }

    const frame = satelliteFrames[currentSatFrameIndex];
    if (!showSatelliteIR || !frame || !radarHost) return;

    // Color 0 = infrared grayscale cloud-top temp
    const satUrl = `${radarHost}${frame.path}/512/{z}/{x}/{y}/0/0_0.png`;
    const satLayer = L.tileLayer(satUrl, {
      opacity: 0.5,
      zIndex: 350,
      className: "satellite-ir-tile",
    }).addTo(map);
    satelliteIRLayerRef.current = satLayer;
  }, [showSatelliteIR, satelliteFrames, currentSatFrameIndex, radarHost]);

  // ── Center pan ──────────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const cLng = center.lng ?? center.lon ?? 78.9629;
    const curr = map.getCenter();
    if (Math.abs(curr.lat - center.lat) > 0.05 || Math.abs(curr.lng - cLng) > 0.05) {
      map.flyTo([center.lat, cLng], zoom, { duration: 1.2 });
    }
  }, [center.lat, center.lng, center.lon, zoom]);

  // ── Render storm trails + projected paths ───────────────────────────
  const renderTrails = useCallback(() => {
    const map = mapRef.current;
    const trailLayer = trailLayerRef.current;
    if (!map || !trailLayer || !showStormTrails) return;

    trailLayer.clearLayers();

    stormCells.forEach((sc) => {
      const track = cellTracks[sc.id];
      if (!track) return;
      const { trail, proj, current, velocity } = track;

      // ── Historical trail (solid gradient polyline) ──
      if (trail.length >= 2) {
        const trailCoords: L.LatLngExpression[] = trail.map((p) => [p.lat, p.lon]);
        // Main trail line
        L.polyline(trailCoords, {
          color: "#38BDF8",
          weight: 2.5,
          opacity: 0.75,
          dashArray: "none",
          lineCap: "round",
          lineJoin: "round",
        }).addTo(trailLayer);

        // Glow layer behind trail
        L.polyline(trailCoords, {
          color: "#38BDF8",
          weight: 7,
          opacity: 0.15,
          lineCap: "round",
        }).addTo(trailLayer);

        // Trail start fade dot
        if (trail[0]) {
          L.circleMarker([trail[0].lat, trail[0].lon], {
            radius: 3,
            fillColor: "#38BDF8",
            fillOpacity: 0.3,
            color: "#38BDF8",
            weight: 1,
            opacity: 0.3,
          }).addTo(trailLayer);
        }
      }

      // ── Projected path (dashed route line ahead) ──
      if (proj.length >= 2) {
        const allCoords: L.LatLngExpression[] = [
          [current.lat, current.lon],
          ...proj.map((p): L.LatLngExpression => [p.lat, p.lon]),
        ];

        // Projected dashed path
        L.polyline(allCoords, {
          color: "#F97316",
          weight: 1.8,
          opacity: 0.7,
          dashArray: "8 6",
          lineCap: "round",
        }).addTo(trailLayer);

        // Projected glow
        L.polyline(allCoords, {
          color: "#F97316",
          weight: 6,
          opacity: 0.08,
          lineCap: "round",
        }).addTo(trailLayer);

        // Waypoint dots at each projected step
        proj.forEach((p, i) => {
          if (i % 2 !== 0) return; // every other step
          L.circleMarker([p.lat, p.lon], {
            radius: 2.5,
            fillColor: "#F97316",
            fillOpacity: 0.6 - i * 0.06,
            color: "#F97316",
            weight: 1,
            opacity: 0.6 - i * 0.06,
          }).addTo(trailLayer);
        });

        // Arrowhead at the end of projected path
        const last = proj[proj.length - 1];
        const prev = proj[proj.length - 2] ?? current;
        if (last && prev) {
          const angle = Math.atan2(last.lon - prev.lon, last.lat - prev.lat) * (180 / Math.PI);
          const arrowIcon = L.divIcon({
            className: "storm-arrow-icon",
            html: `<div style="
              width: 0; height: 0;
              border-left: 6px solid transparent;
              border-right: 6px solid transparent;
              border-bottom: 14px solid rgba(249,115,22,0.85);
              transform: rotate(${angle}deg);
              transform-origin: center bottom;
              filter: drop-shadow(0 0 4px #F97316);
            "></div>`,
            iconSize: [12, 14],
            iconAnchor: [6, 7],
          });
          L.marker([last.lat, last.lon], { icon: arrowIcon, interactive: false }).addTo(trailLayer);
        }
      }

      // ── Rain arrival ETA ring around current storm core ──
      if (focusedLatLon && velocity.speed > 0) {
        const eta = etaMinutes(
          current.lat,
          current.lon,
          focusedLatLon.lat,
          focusedLatLon.lon,
          velocity.speed
        );
        if (eta < 180) {
          // Ring showing 30-min and 60-min rain arrival radius
          [30, 60, 90].forEach((mins) => {
            const radiusKm = velocity.speed * (mins / 60);
            const radiusM = radiusKm * 1000;
            L.circle([current.lat, current.lon], {
              radius: radiusM,
              color: mins === 30 ? "#EF4444" : mins === 60 ? "#F97316" : "#EAB308",
              weight: 1,
              opacity: 0.45 - mins * 0.003,
              fillOpacity: 0,
              dashArray: "4 8",
            }).addTo(trailLayer);
          });
        }
      }

      // ── Storm core animated marker (phase badge) ──
      const { label: phaseLabel, color: phaseColor } = stormPhase(current.intensity, 0);
      const { speed, bearing } = velocity;
      const cardinal = bearingToCardinal(bearing);
      const dbzApprox = Math.round(38 + current.intensity * 24);
      const phaseIcon = L.divIcon({
        className: "storm-phase-icon",
        html: `
          <div class="storm-core-wrapper">
            <div class="storm-pulse-ring" style="--c:${phaseColor}; --sz:${Math.round(current.size * 4)}px;"></div>
            <div class="storm-pulse-ring storm-pulse-ring-2" style="--c:${phaseColor}; --sz:${Math.round(current.size * 6)}px;"></div>
            <div class="storm-core-dot" style="background:${phaseColor}; box-shadow: 0 0 12px ${phaseColor}, 0 0 24px ${phaseColor}44;"></div>
            <div class="storm-phase-badge" style="border-color:${phaseColor}; color:${phaseColor};">
              <span class="storm-phase-label">${phaseLabel}</span>
              <span class="storm-phase-dbz">${dbzApprox} dBZ</span>
              <span class="storm-phase-vel">${Math.round(speed)} km/h ${cardinal}</span>
            </div>
          </div>
        `,
        iconSize: [0, 0],
        iconAnchor: [0, 0],
      });

      L.marker([current.lat, current.lon], { icon: phaseIcon, zIndexOffset: 1000 })
        .addTo(trailLayer);
    });
  }, [stormCells, cellTracks, showStormTrails, focusedLatLon]);

  // ── Render grid dots + airports ─────────────────────────────────────
  const renderMarkers = useCallback(() => {
    const map = mapRef.current;
    const layer = markersLayerRef.current;
    if (!map || !layer) return;

    layer.clearLayers();
    const z = map.getZoom();

    // Airport METAR
    airports.forEach((airport) => {
      const isSevere = airport.windGustKt != null && airport.windGustKt >= 30;
      const markerColor =
        isSevere ? "#EF4444" : airport.flightCategory === "VFR" ? "#22C55E" : "#38BDF8";
      const icao = airport.icaoId ?? "AP";

      const icon = L.divIcon({
        html: `<div style="
          display:flex;align-items:center;justify-content:center;
          width:28px;height:28px;border-radius:4px;
          background:rgba(7,17,29,0.9);border:1.5px solid ${markerColor};
          box-shadow:0 0 10px ${markerColor}66;
          color:${markerColor};font-family:'IBM Plex Mono',monospace;font-size:7px;font-weight:bold;
          cursor:pointer;
        ">${icao.slice(1)}</div>`,
        className: "custom-airport-marker",
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      });

      const m = L.marker([airport.lat, airport.lon], { icon });
      m.bindTooltip(`
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#E2E8F0;background:#07111D;padding:8px;border:1px solid ${markerColor};border-radius:4px;">
          <b style="color:${markerColor};">${icao} · ${airport.name}</b><br/>
          WIND: ${airport.windSpeedKt} KT @ ${airport.windDirection}°<br/>
          ${airport.windGustKt ? `<span style="color:#EF4444;">GUST: ${airport.windGustKt} KT</span><br/>` : ""}
          CATEGORY: <b style="color:${markerColor};">${airport.flightCategory}</b>
        </div>
      `, { opacity: 0.97 });
      m.on("click", () => {
        const nearest = findNearestIndianSector(airport.lat, airport.lon);
        onSelectSector?.(nearest);
        onSelectDot?.(null, airport.lat, airport.lon);
      });
      layer.addLayer(m);
    });

    // Convective grid dots
    const bounds = map.getBounds();
    const vb = {
      north: bounds.getNorth(),
      south: bounds.getSouth(),
      east: bounds.getEast(),
      west: bounds.getWest(),
    };
    const viewGrid = generateViewportGrid(vb, z, gridResolutionKm, liveSectors);

    viewGrid.forEach((sector) => {
      const isSelected =
        (selectedSector &&
          (selectedSector.id === sector.id ||
            (Math.abs(selectedSector.lat - sector.lat) < 0.01 &&
              Math.abs(selectedSector.lon - sector.lon) < 0.01))) ||
        (selectedDotCoords &&
          Math.abs(selectedDotCoords.lat - sector.lat) < 0.01 &&
          Math.abs(selectedDotCoords.lon - sector.lon) < 0.01);

      const color =
        sector.severity === "severe" ? "#EF4444" :
        sector.severity === "high" ? "#F97316" :
        sector.severity === "moderate" ? "#EAB308" :
        sector.severity === "mild" ? "#38BDF8" : "#22C55E";

      let dotRadius = z <= 5 ? 4.5 : z <= 7 ? 5.5 : z <= 9 ? 6.5 : z <= 11 ? 7 : 8;
      if (isSelected) dotRadius += 3;

      const dot = L.circleMarker([sector.lat, sector.lon], {
        radius: dotRadius,
        fillColor: color,
        color: isSelected ? "#FFFFFF" : color,
        weight: isSelected ? 2.5 : 1,
        opacity: 0.95,
        fillOpacity: isSelected ? 0.97 : sector.severity === "severe" ? 0.88 : 0.72,
      });

      dot.bindTooltip(`
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#E2E8F0;background:#07111D;padding:10px;border-radius:6px;line-height:1.6;min-width:240px;border:1px solid ${color};">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <b style="color:${color};font-size:12px;">${sector.name.toUpperCase()}</b>
            <span style="background:${color}22;color:${color};padding:2px 6px;border-radius:3px;font-size:9px;font-weight:bold;">${sector.severity.toUpperCase()}</span>
          </div>
          <span style="color:#94A3B8;font-size:10px;">${sector.state} · ${sector.lat.toFixed(3)}°N, ${sector.lon.toFixed(3)}°E · ${sector.elevationMeters}M</span><br/>
          <div style="margin:5px 0;padding:5px 0;border-top:1px solid rgba(255,255,255,0.1);display:grid;grid-template-columns:1fr 1fr;gap:4px;">
            <span>STATUS: <b style="color:${color};">${sector.statusLabel}</b></span>
            <span>RADAR: <b style="color:${color};">${sector.reflectivityDbz} dBZ</b></span>
            <span>RAIN: <b style="color:#38BDF8;">${sector.rainRateMmHr} mm/hr</b></span>
            <span>CAPE: <b>${sector.capeJkg} J/kg</b></span>
            <span>GUST: <b>${sector.windGustKmh} km/h</b></span>
            <span>HAIL: <b>${sector.hailProbability}%</b></span>
          </div>
          <div style="font-size:10px;color:#38BDF8;margin-top:4px;">
            <b>+30M CONVGRU:</b> ${sector.predictions.t30.reflectivityDbz} dBZ (${sector.predictions.t30.trend.toUpperCase()})<br/>
            <em>${sector.predictions.summary}</em>
          </div>
        </div>
      `, { opacity: 0.97 });

      dot.bindPopup(`
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#E2E8F0;background:#07111D;padding:10px;border-radius:6px;line-height:1.6;min-width:250px;border:1.5px solid ${color};">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <b style="color:${color};font-size:12px;">${sector.name.toUpperCase()}</b>
            <span style="background:${color}22;color:${color};padding:2px 6px;border-radius:3px;font-size:9px;font-weight:bold;">${sector.severity.toUpperCase()}</span>
          </div>
          <span style="color:#94A3B8;font-size:10px;">${sector.state} · ${sector.lat.toFixed(3)}°N, ${sector.lon.toFixed(3)}°E · ${sector.elevationMeters}M</span><br/>
          <div style="margin:5px 0;padding:5px 0;border-top:1px solid rgba(255,255,255,0.1);display:grid;grid-template-columns:1fr 1fr;gap:4px;">
            <span>STATUS: <b style="color:${color};">${sector.statusLabel}</b></span>
            <span>RADAR: <b style="color:${color};">${sector.reflectivityDbz} dBZ</b></span>
            <span>RAIN: <b style="color:#38BDF8;">${sector.rainRateMmHr} mm/hr</b></span>
            <span>CAPE: <b>${sector.capeJkg} J/kg</b></span>
            <span>GUST: <b>${sector.windGustKmh} km/h</b></span>
            <span>HAIL: <b>${sector.hailProbability}%</b></span>
          </div>
          <div style="font-size:10px;color:#38BDF8;margin-top:4px;">
            <b>+30M CONVGRU:</b> ${sector.predictions.t30.reflectivityDbz} dBZ (${sector.predictions.t30.trend.toUpperCase()})<br/>
            <em>${sector.predictions.summary}</em>
          </div>
          <div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(56,189,248,0.25);">
            <button
              onclick="window.vajraOpen3DDigitalTwin && window.vajraOpen3DDigitalTwin('${sector.id}')"
              style="width:100%;padding:6px 10px;background:linear-gradient(135deg, rgba(6,182,212,0.3), rgba(14,165,233,0.45));border:1.5px solid #38BDF8;border-radius:4px;color:#FFFFFF;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:4px;"
            >
              ⚡ LAUNCH 3D STREET VIEW & AREA PERSPECTIVE
            </button>
          </div>
        </div>
      `);

      dot.on("click", () => {
        onSelectSector?.(sector);
        onSelectDot?.(null, sector.lat, sector.lon);
        dot.openPopup();
      });
      layer.addLayer(dot);
    });
  }, [
    airports, nationwideSectors, gridResolutionKm, selectedSector,
    selectedDotCoords, stormCells, onSelectSector, onSelectDot, liveSectors,
  ]);

  // Global window bridge for 3D digital twin modal trigger
  useEffect(() => {
    (window as any).vajraOpen3DDigitalTwin = (sectorId: string) => {
      const pool = [
        ...(liveSectorsRef.current || []),
        ...nationwideSectors,
      ];
      const found = pool.find((s) => s.id === sectorId);
      if (found && onOpen3DViewRef.current) {
        onOpen3DViewRef.current(found);
      }
    };
    return () => {
      delete (window as any).vajraOpen3DDigitalTwin;
    };
  }, [nationwideSectors]);

  // Re-render on map move / data change
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    renderMarkers();
    renderTrails();
    const onMove = () => { renderMarkers(); renderTrails(); };
    map.on("moveend", onMove);
    return () => { map.off("moveend", onMove); };
  }, [renderMarkers, renderTrails]);

  const currentRadarFrame = radarFrames[currentRadarFrameIndex];
  const currentSatFrame = satelliteFrames[currentSatFrameIndex];

  return (
    <div
      className={`relative w-full h-full ${className}`}
      style={{ position: "absolute", inset: 0, background: "#07111D", overflow: "hidden" }}
    >
      <div
        ref={containerRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />

      {/* ── Live radar frame label HUD ── */}
      {currentRadarFrame && (
        <div className="radar-frame-hud">
          <span className={`rfh-badge ${currentRadarFrame.isNowcast ? "rfh-nowcast" : "rfh-past"}`}>
            {currentRadarFrame.isNowcast ? "NOWCAST" : "RADAR"}
          </span>
          <span className="rfh-time">
            {new Date(currentRadarFrame.timestamp * 1000).toLocaleTimeString("en-GB", {
              hour: "2-digit", minute: "2-digit", timeZone: "UTC"
            })} UTC
          </span>
          <span className="rfh-label">{currentRadarFrame.label}</span>
        </div>
      )}

      {/* ── Satellite IR label ── */}
      {showSatelliteIR && currentSatFrame && (
        <div className="sat-ir-hud">
          <span>🛰️ INSAT IR</span>
          <span>{new Date(currentSatFrame.timestamp * 1000).toLocaleTimeString("en-GB", {
            hour: "2-digit", minute: "2-digit", timeZone: "UTC"
          })} UTC</span>
        </div>
      )}

      {/* ── Storm legend ── */}
      {showStormTrails && stormCells.length > 0 && (
        <div className="storm-legend-hud">
          <div className="slh-row"><span style={{ background: "#38BDF8" }} />TRAIL (PAST)</div>
          <div className="slh-row slh-dashed"><span style={{ borderColor: "#F97316" }} />PROJECTED PATH</div>
          <div className="slh-row"><span style={{ background: "#EF4444" }} />30MIN ETA RING</div>
        </div>
      )}

      {/* ── Dynamic Animated Cloud Cover & Lightning Overlays ── */}
      <CloudLayerOverlay
        map={mapRef.current}
        engine="leaflet"
        cells={cloudCells}
        leadMinutes={leadMinutes}
        enabled={hazards.thunderstorm || hazards.lightning}
      />
      <LightningCanvasOverlay
        map={mapRef.current}
        engine="leaflet"
        cells={lightningCells}
        enabled={hazards.lightning}
      />

      {/* ── Map info readout ── */}
      <div className="map-readout-hud">
        TACTICAL LEAFLET · ZOOM: <span style={{ color: "#38BDF8" }}>{currentZoom}</span>
        {" "}· MESH: <span style={{ color: "#38BDF8" }}>{gridResolutionKm} KM</span>
        {" "}· SECTORS: <span style={{ color: "#38BDF8" }}>{nationwideSectors.length}</span>
      </div>
    </div>
  );
}
