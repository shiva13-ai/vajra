/*
 * VAJRA Atmospheric Command Grid
 * Map-first mission-control composition. Space Grotesk labels + IBM Plex Mono telemetry.
 * Cyan is reserved for live system state; hazard colors remain semantic.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BatteryCharging,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  CloudLightning,
  CloudRain,
  Crosshair,
  Download,
  Gauge,
  Layers3,
  MapPin,
  Maximize2,
  Menu,
  Minimize2,
  Minus,
  Pause,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Search,
  ScanLine,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Target,
  Wind,
  X,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { IndiaMap, type WeatherPoint } from "@/components/IndiaMap";
import { MapplsMap } from "@/components/MapplsMap";
import { WeatherGridDots } from "@/components/WeatherGridDots";
import { type ForecastData } from "@/hooks/useWeatherForecast";
import { TelemetryPanels } from "@/components/TelemetryPanels";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type HazardKey = "thunderstorm" | "cloudburst" | "hail" | "lightning" | "downburst";
type StormCell = { id: number; x: number; y: number; size: number; intensity: number; driftX: number; driftY: number; phase: number };

type Frame = {
  lead: number;
  reflectivity: number;
  rainRate: number;
  hail: number;
  lightning: number;
  wind: number;
  loss: number;
  cells: StormCell[];
};

type MapLocation = { name: string; region: string; lat: number; lon: number; x: number; y: number; };
type WeatherStatus = "idle" | "loading" | "live" | "error";
type WeatherState = { status: WeatherStatus; points: WeatherPoint[]; updatedAt: string; message: string };
type RadarFrame = { url: string; time: string };
type RadarState = { status: WeatherStatus; url: string; frames: RadarFrame[]; updatedAt: string; message: string };
type HistoryPoint = { time: string; temperature: number; precipitation: number; wind: number };
type Thresholds = { rainRate: number; hail: number; lightning: number; wind: number };

const initialThresholds: Thresholds = { rainRate: 80, hail: 70, lightning: 10, wind: 90 };
const ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive";

const hazardMeta: Record<HazardKey, { label: string; color: string; bg: string; icon: typeof CloudRain }> = {
  thunderstorm: { label: "Thunderstorm genesis", color: "#EAB308", bg: "rgba(234,179,8,.16)", icon: CloudLightning },
  cloudburst: { label: "Cloudburst zones", color: "#EF4444", bg: "rgba(239,68,68,.18)", icon: CloudRain },
  hail: { label: "Hail probability core", color: "#F97316", bg: "rgba(249,115,22,.16)", icon: CircleDot },
  lightning: { label: "Lightning density", color: "#A855F7", bg: "rgba(168,85,247,.18)", icon: Zap },
  downburst: { label: "Downburst wind shear", color: "#38BDF8", bg: "rgba(56,189,248,.15)", icon: Wind },
};

const initialHazards: Record<HazardKey, boolean> = { thunderstorm: true, cloudburst: true, hail: true, lightning: true, downburst: true };
const timeline = Array.from({ length: 37 }, (_, index) => index * 10);
const INDIA_BOUNDS = { north: 37.5, south: 6.5, west: 67, east: 98.5 };
const LIVE_WEATHER_URL = "https://api.open-meteo.com/v1/forecast";
const RADAR_METADATA_URL = "https://api.rainviewer.com/public/weather-maps.json";
const OFFICIAL_DISTRICT_GEOJSON_URL = "/manus-storage/india-districts-official_3a81e00d.geojson";
const radarImageCache = new Map<string, HTMLImageElement>();
type DistrictBoundary = { name: string; polygons: Array<Array<[number, number]>> };
function pointInPolygon(point: [number, number], polygon: Array<[number, number]>) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [x, y] = point; const [px, py] = polygon[previous]; const [cx, cy] = polygon[index];
    if ((cy > y) !== (py > y) && x < ((px - cx) * (y - cy)) / (py - cy) + cx) inside = !inside;
  }
  return inside;
}
function normalizeDistrictName(name: string) { return name.toLowerCase().replace(/[^a-z]/g, "").replace("hydrabad", "hyderabad"); }

function preloadRadarFrame(url: string) {
  if (!url || radarImageCache.has(url)) return;
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  radarImageCache.set(url, image);
}
const weatherStations = [
  { id: "hyderabad", label: "Hyderabad", lat: 17.385, lon: 78.486 },
  { id: "delhi", label: "New Delhi", lat: 28.614, lon: 77.209 },
  { id: "mumbai", label: "Mumbai", lat: 19.076, lon: 72.878 },
  { id: "kolkata", label: "Kolkata", lat: 22.573, lon: 88.364 },
  { id: "bengaluru", label: "Bengaluru", lat: 12.972, lon: 77.595 },
  { id: "chennai", label: "Chennai", lat: 13.083, lon: 80.271 },
];
const districtsByState: Record<string, string[]> = {
  "ALL STATES": ["ALL DISTRICTS", "Hyderabad", "Secunderabad", "New Delhi", "Mumbai", "Pune", "Bengaluru Urban", "Chennai", "Kolkata", "Ahmedabad", "Guwahati"],
  "Andhra Pradesh": ["ALL DISTRICTS"], Assam: ["ALL DISTRICTS", "Guwahati"], Delhi: ["ALL DISTRICTS", "New Delhi"], Gujarat: ["ALL DISTRICTS", "Ahmedabad"], Karnataka: ["ALL DISTRICTS", "Bengaluru Urban"], Kerala: ["ALL DISTRICTS"], Maharashtra: ["ALL DISTRICTS", "Mumbai", "Pune"], Odisha: ["ALL DISTRICTS"], "Tamil Nadu": ["ALL DISTRICTS", "Chennai"], Telangana: ["ALL DISTRICTS", "Hyderabad", "Secunderabad"], "Uttar Pradesh": ["ALL DISTRICTS"], "West Bengal": ["ALL DISTRICTS", "Kolkata"],
};
const districtCenters: Record<string, { lat: number; lon: number }> = { Hyderabad: { lat: 17.385, lon: 78.486 }, Secunderabad: { lat: 17.44, lon: 78.498 }, "New Delhi": { lat: 28.614, lon: 77.209 }, Mumbai: { lat: 19.076, lon: 72.878 }, Pune: { lat: 18.52, lon: 73.857 }, "Bengaluru Urban": { lat: 12.972, lon: 77.595 }, Chennai: { lat: 13.083, lon: 80.271 }, Kolkata: { lat: 22.573, lon: 88.364 }, Ahmedabad: { lat: 23.023, lon: 72.571 }, Guwahati: { lat: 26.145, lon: 91.736 } };

function indiaPercent(lat: number, lon: number) {
  return {
    x: ((lon - INDIA_BOUNDS.west) / (INDIA_BOUNDS.east - INDIA_BOUNDS.west)) * 100,
    y: ((INDIA_BOUNDS.north - lat) / (INDIA_BOUNDS.north - INDIA_BOUNDS.south)) * 100,
  };
}

function cellCoordinates(cell: StormCell) {
  return {
    lat: INDIA_BOUNDS.north - (cell.y / 100) * (INDIA_BOUNDS.north - INDIA_BOUNDS.south),
    lon: INDIA_BOUNDS.west + (cell.x / 100) * (INDIA_BOUNDS.east - INDIA_BOUNDS.west),
  };
}

const baseCells: StormCell[] = [
  { id: 14, x: 37, y: 65, size: 13, intensity: 0.9, driftX: 0.46, driftY: -0.12, phase: 0.2 },
  { id: 18, x: 66, y: 40, size: 11, intensity: 0.72, driftX: 0.18, driftY: -0.36, phase: 1.4 },
  { id: 21, x: 18, y: 60, size: 10, intensity: 0.54, driftX: 0.38, driftY: -0.18, phase: 2.1 },
  { id: 25, x: 47, y: 78, size: 8, intensity: 0.44, driftX: -0.28, driftY: -0.22, phase: 3.2 },
  { id: 31, x: 35, y: 28, size: 7, intensity: 0.31, driftX: 0.35, driftY: 0.2, phase: 4.7 },
];

const mapLocations: MapLocation[] = [
  { name: "Hyderabad", region: "Telangana / active sector", lat: 17.385, lon: 78.486, ...indiaPercent(17.385, 78.486) },
  { name: "Secunderabad", region: "Telangana / railway corridor", lat: 17.4399, lon: 78.4983, ...indiaPercent(17.4399, 78.4983) },
  { name: "New Delhi", region: "Delhi NCR / airport corridor", lat: 28.6139, lon: 77.209, ...indiaPercent(28.6139, 77.209) },
  { name: "Bengaluru", region: "Karnataka / southern sector", lat: 12.9716, lon: 77.5946, ...indiaPercent(12.9716, 77.5946) },
  { name: "Chennai", region: "Tamil Nadu / east coast sector", lat: 13.0827, lon: 80.2707, ...indiaPercent(13.0827, 80.2707) },
  { name: "Mumbai", region: "Maharashtra / west coast sector", lat: 19.076, lon: 72.8777, ...indiaPercent(19.076, 72.8777) },
  { name: "Pune", region: "Maharashtra / plateau sector", lat: 18.5204, lon: 73.8567, ...indiaPercent(18.5204, 73.8567) },
  { name: "Kolkata", region: "West Bengal / east corridor", lat: 22.5726, lon: 88.3639, ...indiaPercent(22.5726, 88.3639) },
  { name: "Ahmedabad", region: "Gujarat / western sector", lat: 23.0225, lon: 72.5714, ...indiaPercent(23.0225, 72.5714) },
  { name: "Guwahati", region: "Assam / northeast sector", lat: 26.1445, lon: 91.7362, ...indiaPercent(26.1445, 91.7362) },
];

function simulateConvGRU(lead: number): Frame {
  const t = lead / 60;
  const step = Math.round(lead / 10);
  const cells = baseCells.map((cell) => ({
    ...cell,
    x: Math.max(10, Math.min(90, cell.x + cell.driftX * step + Math.sin(step * 0.35 + cell.phase) * 1.4)),
    y: Math.max(13, Math.min(87, cell.y + cell.driftY * step + Math.cos(step * 0.28 + cell.phase) * 1.2)),
    size: cell.size * (0.92 + 0.12 * Math.sin(t * 1.2 + cell.phase)),
    intensity: Math.max(0.18, Math.min(1, cell.intensity + 0.1 * Math.sin(t * 1.5 + cell.phase) + (cell.id === 14 ? t * 0.03 : 0))),
  }));
  const attention = 0.94 + Math.sin(t * 0.8) * 0.018;
  return {
    lead,
    reflectivity: Number((54.2 * attention + Math.sin(t) * 1.6).toFixed(1)),
    rainRate: Number((112.5 * attention + Math.sin(t * 1.3) * 6.2).toFixed(1)),
    hail: Math.round(Math.min(97, 88 + Math.sin(t * 1.1) * 4)),
    lightning: Number((12 + Math.max(0, Math.sin(t * 1.5)) * 4.4).toFixed(1)),
    wind: Math.round(68 + Math.sin(t * 0.9) * 7),
    loss: Number((0.0124 + Math.abs(Math.sin(t * 0.74)) * 0.0022).toFixed(4)),
    cells,
  };
}

function formatLead(lead: number) {
  if (lead === 0) return "NOW";
  return `+${String(lead).padStart(2, "0")}M`;
}

function getCellTelemetry(cell: StormCell, frame: Frame) {
  const intensity = Math.max(0.55, cell.intensity);
  return {
    reflectivity: (frame.reflectivity * (0.72 + intensity * 0.3)).toFixed(1),
    rainRate: (frame.rainRate * (0.62 + intensity * 0.38)).toFixed(1),
    hail: Math.min(98, Math.round(frame.hail * (0.7 + intensity * 0.34))),
    wind: Math.round(frame.wind * (0.76 + intensity * 0.23)),
    lightning: (frame.lightning * (0.55 + intensity * 0.58)).toFixed(1),
  };
}

function formatDataTimestamp(iso: string) {
  if (!iso) return "WAITING";
  return new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC" }) + "Z";
}

function archiveDate(daysAgo = 3) {
  const date = new Date(Date.now() - daysAgo * 86400000);
  return date.toISOString().slice(0, 10);
}

let widgetSequence = 0;
function Panel({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const [minimized, setMinimized] = useState(false);
  const widgetKey = useRef(`${className.split(" ")[0] || "widget"}-${widgetSequence++}`);
  const [offset, setOffset] = useState(() => { try { return JSON.parse(localStorage.getItem(`vajra-position-${widgetKey.current}`) || "{\"x\":0,\"y\":0}"); } catch { return { x: 0, y: 0 }; } });
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
  useEffect(() => { if (!dragRef.current) return; const move = (event: PointerEvent) => { const drag = dragRef.current; if (!drag) return; const snap = (value: number) => Math.round(value / 16) * 16; const element = document.querySelector<HTMLElement>(`.widget[data-widget-key="${widgetKey.current}"]`); let next = { x: snap(drag.ox + event.clientX - drag.x), y: snap(drag.oy + event.clientY - drag.y) }; if (element) { const base = element.getBoundingClientRect(); const candidate = { left: base.left + next.x - offset.x, top: base.top + next.y - offset.y, right: base.right + next.x - offset.x, bottom: base.bottom + next.y - offset.y }; document.querySelectorAll<HTMLElement>(".widget").forEach((other) => { if (other === element) return; const rect = other.getBoundingClientRect(); const overlap = candidate.left < rect.right && candidate.right > rect.left && candidate.top < rect.bottom && candidate.bottom > rect.top; if (overlap) { next = { ...next, y: snap(next.y + rect.bottom - candidate.top + 16) }; candidate.top = base.top + next.y - offset.y; candidate.bottom = base.bottom + next.y - offset.y; } }); } setOffset(next); localStorage.setItem(`vajra-position-${widgetKey.current}`, JSON.stringify(next)); }; const up = () => { dragRef.current = null; document.body.classList.remove("is-dragging-widget"); }; window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); }; }, [offset]);
  const startDrag = (event: React.PointerEvent<HTMLElement>) => { if ((event.target as HTMLElement).closest("button, input, select, textarea, a")) return; dragRef.current = { x: event.clientX, y: event.clientY, ox: offset.x, oy: offset.y }; document.body.classList.add("is-dragging-widget"); };
  return <section data-widget-key={widgetKey.current} onPointerDown={startDrag} className={`glass-panel widget ${className} ${minimized ? "widget-minimized" : ""}`} style={{ translate: `${offset.x}px ${offset.y}px` }}><span className="widget-drag-hint" aria-hidden="true">⋮⋮</span><button className="widget-minimize" aria-label={minimized ? "Expand widget" : "Minimize widget"} onClick={() => setMinimized((value) => !value)}>{minimized ? <Plus size={11} /> : <Minus size={11} />}</button>{!minimized && children}</section>;
}

function StatusPill({ icon: Icon, label, state, color }: { icon: typeof CloudRain; label: string; state: string; color: string }) {
  return <div className="status-pill" style={{ borderColor: `${color}42` }}><Icon size={13} style={{ color }} /><span>{label}</span><b style={{ color }}>{state}</b></div>;
}

function AppLogo() {
  return <div className="brand-mark" aria-label="VAJRA mark"><div className="brand-diamond"><span /></div><div><div className="brand-name">VAJRA</div><div className="brand-sub">CONVECTIVE NOWCASTING ENGINE <i /> 1.5 KM MESH</div></div></div>;
}

export default function Home() {
  const [lead, setLead] = useState(30);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [hazards, setHazards] = useState(initialHazards);
  const [selectedCell, setSelectedCell] = useState<StormCell | null>(baseCells[0]);
  const [hoveredCell, setHoveredCell] = useState<StormCell | null>(null);
  const [search, setSearch] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [focusedLocation, setFocusedLocation] = useState<MapLocation | null>(null);
  const [showLayers, setShowLayers] = useState(true);
  const [showStates, setShowStates] = useState(true);
  const [showDistricts, setShowDistricts] = useState(false);
  const [showRadar, setShowRadar] = useState(true);
  const [mapLayer, setMapLayer] = useState<"radar" | "satellite" | "admin">("radar");
  const [stateFilter, setStateFilter] = useState("ALL STATES");
  const [districtFilter, setDistrictFilter] = useState("ALL DISTRICTS");
  const [districtBoundaries, setDistrictBoundaries] = useState<DistrictBoundary[]>([]);
  const [boundaryStatus, setBoundaryStatus] = useState<WeatherStatus>("loading");
  const officialDistrictNames = useMemo(() => districtBoundaries.map((boundary) => boundary.name).filter(Boolean).sort((a, b) => a.localeCompare(b)), [districtBoundaries]);
  const districtOptions = officialDistrictNames.length ? officialDistrictNames : (districtsByState[stateFilter] ?? ["ALL DISTRICTS"]).filter((name) => name !== "ALL DISTRICTS");
  const [showInspector, setShowInspector] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [utc, setUtc] = useState(new Date());
  const [weather, setWeather] = useState<WeatherState>({ status: "loading", points: [], updatedAt: "", message: "Connecting to Open-Meteo" });
  const [radar, setRadar] = useState<RadarState>({ status: "loading", url: "", frames: [], updatedAt: "", message: "Connecting to RainViewer" });
  const [radarPlaying, setRadarPlaying] = useState(false);
  const [radarFrameIndex, setRadarFrameIndex] = useState(0);
  const [gridResolutionKm, setGridResolutionKm] = useState<1 | 1.5 | 2 | 3>(1.5);
  const [historyDate, setHistoryDate] = useState(archiveDate());
  const [history, setHistory] = useState<{ status: WeatherStatus; points: HistoryPoint[]; message: string }>({ status: "loading", points: [], message: "Loading historical archive" });
  const [historyHour, setHistoryHour] = useState(12);
  const [thresholds, setThresholds] = useState<Thresholds>(initialThresholds);
  const [alarmsEnabled, setAlarmsEnabled] = useState(true);
  const [alertLog, setAlertLog] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showForecast, setShowForecast] = useState(false);
  const [etaSeconds, setEtaSeconds] = useState({ airport: 31 * 60 + 45, rail: 52 * 60 + 10 });
  const mapStageRef = useRef<HTMLElement>(null);
  const frame = useMemo(() => simulateConvGRU(lead), [lead]);
  const hoveredTelemetry = hoveredCell ? getCellTelemetry(hoveredCell, frame) : null;
  const activeCell = selectedCell ? frame.cells.find((cell) => cell.id === selectedCell.id) ?? frame.cells[0] : frame.cells[0];
  const selectedTelemetry = activeCell ? getCellTelemetry(activeCell, frame) : null;
  const selectedCoordinates = activeCell ? cellCoordinates(activeCell) : null;
  const primaryWeather = weather.points[0];
  const selectedBoundary = districtBoundaries.find((boundary) => normalizeDistrictName(boundary.name) === normalizeDistrictName(districtFilter));
  const districtCenter = districtCenters[districtFilter];
  const districtCenterPercent = districtCenter ? indiaPercent(districtCenter.lat, districtCenter.lon) : null;
  const visibleCells = selectedBoundary ? frame.cells.filter((cell) => { const { lat, lon } = cellCoordinates(cell); return selectedBoundary.polygons.some((polygon) => pointInPolygon([lon, lat], polygon)); }) : districtCenterPercent ? frame.cells.filter((cell) => Math.hypot(cell.x - districtCenterPercent.x, cell.y - districtCenterPercent.y) <= 9 + gridResolutionKm) : frame.cells;
  const thresholdAlerts = selectedTelemetry ? [
    Number(selectedTelemetry.rainRate) >= thresholds.rainRate ? `RAIN RATE ${selectedTelemetry.rainRate} MM/HR` : "",
    selectedTelemetry.hail >= thresholds.hail ? `HAIL PROBABILITY ${selectedTelemetry.hail}%` : "",
    Number(selectedTelemetry.lightning) >= thresholds.lightning ? `LIGHTNING ${selectedTelemetry.lightning}/KM²/10M` : "",
    selectedTelemetry.wind >= thresholds.wind ? `WIND GUST ${selectedTelemetry.wind} KM/H` : "",
  ].filter(Boolean) : [];
  const searchResults = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return mapLocations.slice(0, 5);
    return mapLocations.filter((location) => `${location.name} ${location.region}`.toLowerCase().includes(query)).slice(0, 5);
  }, [search]);
  const trend = useMemo(() => timeline.map((minute) => Math.round(42 + Math.sin(minute / 29) * 11 + (minute > 100 ? (minute - 100) / 18 : 0))), []);

  useEffect(() => {
    if (!radar.frames.length) return;
    const adjacent = [radarFrameIndex - 1, radarFrameIndex, radarFrameIndex + 1].map((index) => radar.frames[(index + radar.frames.length) % radar.frames.length]);
    adjacent.forEach((item) => item && preloadRadarFrame(item.url));
  }, [radar.frames, radarFrameIndex]);

  useEffect(() => {
    const controller = new AbortController();
    setBoundaryStatus("loading");
    fetch(OFFICIAL_DISTRICT_GEOJSON_URL, { signal: controller.signal }).then((response) => response.ok ? response.json() : Promise.reject(new Error("District geometry unavailable"))).then((payload) => {
      const boundaries = (payload.features ?? []).map((feature: { properties?: Record<string, unknown>; geometry?: { type?: string; coordinates?: unknown } }) => {
        const name = String(feature.properties?.shapeName ?? feature.properties?.NAME_2 ?? feature.properties?.district ?? "");
        const coordinates = feature.geometry?.coordinates as any;
        const polygons = feature.geometry?.type === "Polygon" ? (coordinates ?? []) : feature.geometry?.type === "MultiPolygon" ? (coordinates ?? []).flat(1) : [];
        return { name, polygons: polygons.filter((polygon: unknown) => Array.isArray(polygon) && polygon.length > 2) };
      }).filter((boundary: DistrictBoundary) => boundary.name && boundary.polygons.length);
      setDistrictBoundaries(boundaries);
      setBoundaryStatus(boundaries.length ? "live" : "error");
    }).catch((error) => { if (error instanceof DOMException && error.name === "AbortError") return; setDistrictBoundaries([]); setBoundaryStatus("error"); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const clock = window.setInterval(() => setUtc(new Date()), 1000);
    return () => window.clearInterval(clock);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setEtaSeconds((current) => ({ airport: Math.max(0, current.airport - 1), rail: Math.max(0, current.rail - 1) })), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setLead((current) => current >= 360 ? 0 : current + 10), Math.max(260, 1200 / speed));
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  useEffect(() => {
    if (!activeCell) return;
    const controller = new AbortController();
    const { lat, lon } = cellCoordinates(activeCell);
    const url = `${ARCHIVE_WEATHER_URL}?latitude=${lat.toFixed(3)}&longitude=${lon.toFixed(3)}&start_date=${historyDate}&end_date=${historyDate}&hourly=temperature_2m,precipitation,wind_gusts_10m&timezone=UTC`;
    setHistory((current) => ({ ...current, status: "loading", message: `Loading ${historyDate} archive` }));
    fetch(url, { signal: controller.signal }).then((response) => {
      if (!response.ok) throw new Error("Archive request failed");
      return response.json();
    }).then((payload) => {
      const points = (payload.hourly?.time ?? []).map((time: string, index: number) => ({
        time,
        temperature: Number(payload.hourly.temperature_2m?.[index] ?? 0),
        precipitation: Number(payload.hourly.precipitation?.[index] ?? 0),
        wind: Number(payload.hourly.wind_gusts_10m?.[index] ?? 0),
      }));
      setHistory({ status: points.length ? "live" : "error", points, message: points.length ? "Historical archive linked" : "No archive data returned" });
    }).catch((error) => {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setHistory({ status: "error", points: [], message: "Historical archive unavailable" });
    });
    return () => controller.abort();
  }, [activeCell?.id, historyDate]);

  const loadLiveData = useCallback(async (signal?: AbortSignal) => {
    const coordinates = weatherStations.map((station) => station.lat).join(",");
    const longitudes = weatherStations.map((station) => station.lon).join(",");
    const weatherUrl = `${LIVE_WEATHER_URL}?latitude=${coordinates}&longitude=${longitudes}&current=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,weather_code&timezone=auto`;
    try {
      const response = await fetch(weatherUrl, { signal });
      if (!response.ok) throw new Error("Weather request failed");
      const payload = await response.json();
      const records = Array.isArray(payload) ? payload : [payload];
      const points = records.map((record, index) => ({
        ...weatherStations[index],
        temperature: Number(record.current?.temperature_2m ?? 0),
        precipitation: Number(record.current?.precipitation ?? 0),
        wind: Number(record.current?.wind_gusts_10m ?? record.current?.wind_speed_10m ?? 0),
        code: Number(record.current?.weather_code ?? 0),
      })).filter((point) => Boolean(point.id) && Number.isFinite(point.temperature));
      setWeather({ status: points.length ? "live" : "error", points, updatedAt: records[0]?.current?.time ?? new Date().toISOString(), message: points.length ? "Live observations linked" : "No current observations returned" });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setWeather((current) => ({ ...current, status: "error", message: "Open-Meteo unavailable · simulation retained" }));
    }

    try {
      const response = await fetch(RADAR_METADATA_URL, { signal });
      if (!response.ok) throw new Error("Radar request failed");
      const payload = await response.json();
      const sourceFrames = [...(payload.radar?.past ?? []).slice(-8), ...(payload.radar?.nowcast ?? []).slice(0, 4)];
      const frame = sourceFrames.at(-1);
      if (!frame?.path || !payload.host) throw new Error("No radar frame returned");
      const frames = sourceFrames.filter((item: { path?: string }) => item.path).map((item: { path: string; time: number }) => ({ url: `${payload.host}${item.path}/512/3/22.500/79.500/2/1_1.png`, time: new Date(item.time * 1000).toISOString() }));
      frames.forEach((item) => preloadRadarFrame(item.url));
      setRadarFrameIndex(0);
      setRadar({ status: "live", url: frames.at(-1)?.url ?? "", frames, updatedAt: new Date(frame.time * 1000).toISOString(), message: `${frames.length} RainViewer frames linked` });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setRadar((current) => ({ ...current, status: "error", message: "RainViewer unavailable · radar hidden" }));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadLiveData(controller.signal);
    const refresh = window.setInterval(() => loadLiveData(controller.signal), 5 * 60 * 1000);
    return () => {
      controller.abort();
      window.clearInterval(refresh);
    };
  }, [loadLiveData]);

  useEffect(() => {
    if (!alarmsEnabled || thresholdAlerts.length === 0) return;
    const signature = `${activeCell?.id}-${formatLead(lead)}-${thresholdAlerts.join("|")}`;
    setAlertLog((current) => current[0] === signature ? current : [signature, ...current].slice(0, 3));
    toast.warning("Threshold alarm triggered", { description: thresholdAlerts.join(" · ") });
    if ("Notification" in window && Notification.permission === "granted") new Notification("VAJRA severe weather alarm", { body: thresholdAlerts.join(" · ") });
  }, [alarmsEnabled, activeCell?.id, lead, thresholdAlerts.join("|")]);

  useEffect(() => {
    const syncFullscreen = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", syncFullscreen);
    document.addEventListener("webkitfullscreenchange", syncFullscreen);
    return () => {
      document.removeEventListener("fullscreenchange", syncFullscreen);
      document.removeEventListener("webkitfullscreenchange", syncFullscreen);
    };
  }, []);

  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (mapStageRef.current?.requestFullscreen) await mapStageRef.current.requestFullscreen({ navigationUI: "hide" });
      else toast.error("Fullscreen mode is not supported in this browser");
    } catch {
      toast.error("Fullscreen mode could not be opened", { description: "Use your browser's fullscreen control instead." });
    }
  };

  const toggleHazard = (key: HazardKey) => setHazards((current) => ({ ...current, [key]: !current[key] }));
  const focusLocation = (location: MapLocation) => {
    setFocusedLocation(location);
    setSearch(location.name);
    setSearchOpen(false);
    toast.success(`${location.name} in focus`, { description: `${location.lat.toFixed(3)}°N, ${location.lon.toFixed(3)}°E · forecast frame ${formatLead(lead)}` });
  };
  const submitSearch = () => {
    const match = searchResults[0];
    if (match) focusLocation(match);
    else toast.error("No mapped region found", { description: "Try Hyderabad, Secunderabad, New Delhi, Bengaluru, Chennai, Mumbai, or Pune." });
  };
  const local = utc.toLocaleTimeString("en-IN", { hour12: false, timeZone: "Asia/Kolkata" });
  const utcClock = utc.toLocaleTimeString("en-GB", { hour12: false, timeZone: "UTC" });
  const formatCountdown = (seconds: number) => `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;

  return (
    <main ref={mapStageRef} className={`dashboard-shell ${isFullscreen ? "is-fullscreen" : ""}`}>
      <IndiaMap className={`map-underlay map-layer-${mapLayer}`} mapLayer={mapLayer} showStates={showStates || mapLayer === "admin"} showDistricts={showDistricts || mapLayer === "admin"} stateFilter={stateFilter} districtFilter={districtFilter} radarUrl={mapLayer === "radar" && showRadar && radar.status === "live" ? radar.url : undefined} radarFrames={mapLayer === "radar" && showRadar && radar.status === "live" ? radar.frames : []} radarPlaying={radarPlaying} radarFrameIndex={radarFrameIndex} gridResolutionKm={gridResolutionKm} weatherPoints={weather.points} onRadarError={() => setRadar((current) => ({ ...current, status: "error", message: "Radar image unavailable · radar hidden" }))} />
      <div className="map-atmosphere" />
      <div className="map-grid" />
      <div className="map-vignette" />
      <div className="map-label label-ne">INDIA / NATIONAL WEATHER MOSAIC</div>
      <div className="map-label label-sw">06°30′N — 37°30′N <span>•</span> 2° GRID RESOLUTION</div>
      <div className="reticle" style={{ left: `${activeCell.x}%`, top: `${activeCell.y}%` }}><span>{activeCell.id}_88 / LOCK</span></div>
      <div className="reticle reticle-secondary" style={{ left: `${frame.cells[1].x}%`, top: `${frame.cells[1].y}%` }}><span>18_62</span></div>
      {visibleCells.map((cell) => <div key={cell.id} className={`storm-cell cell-${cell.id} ${selectedCell?.id === cell.id ? "selected-cell" : ""}`} role="button" tabIndex={0} aria-pressed={selectedCell?.id === cell.id} aria-label={`Inspect grid cell ${cell.id} in ${districtFilter}`} style={{ left: `${cell.x}%`, top: `${cell.y}%`, width: `${cell.size}%`, height: `${cell.size * .72}%`, opacity: hazards.thunderstorm ? cell.intensity : 0 }} onMouseEnter={() => setHoveredCell(cell)} onMouseLeave={() => setHoveredCell(null)} onFocus={() => setHoveredCell(cell)} onBlur={() => setHoveredCell(null)} onClick={() => { setSelectedCell(cell); setShowInspector(true); }} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedCell(cell); setShowInspector(true); } }}><div className="cell-core" style={{ opacity: hazards.cloudburst ? cell.intensity : 0 }} /><div className="cell-halo" style={{ opacity: hazards.hail ? cell.intensity : 0 }} /></div>)}
      {hoveredCell && hoveredTelemetry && <div className="cell-tooltip" style={{ left: `${hoveredCell.x}%`, top: `${hoveredCell.y}%` }}><div className="tooltip-kicker"><span><span className="tooltip-live" /> CELL HOVER / LIVE FRAME</span><b>{formatLead(lead)}</b></div><div className="tooltip-title"><strong>GRID #{hoveredCell.id}_88</strong><span>17.{String(385 + hoveredCell.id).slice(-3)}°N · 78.486°E</span></div><div className="tooltip-metrics"><span>REFLECTIVITY <b>{hoveredTelemetry.reflectivity}<i>dBZ</i></b></span><span>RAIN RATE <b>{hoveredTelemetry.rainRate}<i>mm/hr</i></b></span><span>HAIL PROB. <b>{hoveredTelemetry.hail}<i>%</i></b></span><span>WIND GUST <b>{hoveredTelemetry.wind}<i>km/h</i></b></span><span>LIGHTNING <b>{hoveredTelemetry.lightning}<i>/km²/10m</i></b></span></div><div className="tooltip-footer"><span>1.5 KM × 1.5 KM</span><span>SPATIAL ATTENTION {Math.round(hoveredCell.intensity * 100)}%</span></div></div>}
      {hazards.lightning && <div className="lightning-trace trace-a"><Zap size={18} /></div>}
      {hazards.lightning && <div className="lightning-trace trace-b"><Zap size={13} /></div>}

      <header className="topbar">
        <AppLogo />
        <div className="status-rail">
          <StatusPill icon={Zap} label="Lightning" state="ACTIVE" color="#A855F7" />
          <StatusPill icon={CloudRain} label="Cloudburst" state="WARNING" color="#EF4444" />
          <StatusPill icon={CircleDot} label="Hail risk" state="ELEVATED" color="#F97316" />
          <StatusPill icon={Wind} label="Downburst" state="LOW" color="#38BDF8" />
          <StatusPill icon={CloudLightning} label="Thunderstorm" state="TRACKING" color="#EAB308" />
        </div>
        <div className="top-actions">
          <div className="live-status"><span className={`live-dot ${weather.status === "error" && radar.status === "error" ? "offline" : ""}`} /> LIVE FEED <b>{weather.status === "live" || radar.status === "live" ? "ONLINE" : weather.status === "loading" || radar.status === "loading" ? "LINKING" : "SIMULATION"}</b></div>
          <div className="clock-block"><span>UTC {utcClock}</span><span>IST {local}</span></div>
          <button className="export-button" onClick={() => toast.success("Threat field packaged for export", { description: "GeoJSON + CAP alert payload ready." })}><Download size={14} /> EXPORT <span>GEOJSON / CAP</span></button>
          <button className="icon-button mobile-menu" aria-label="Open menu"><Menu size={18} /></button>
        </div>
      </header>

      <div className="map-toolbar"><button className="icon-button" aria-label="Layers" onClick={() => setShowLayers((value) => !value)}><Layers3 size={16} /></button><button className="icon-button" aria-label="Recenter on India" onClick={() => { setLead(30); setFocusedLocation(null); toast.info("India view recentered", { description: "National grid locked to the India operating area." }); }}><Crosshair size={16} /></button><button className="icon-button" aria-label={isFullscreen ? "Exit fullscreen" : "Open fullscreen map"} onClick={toggleFullscreen}>{isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}</button></div>
      <div className="map-search" role="search"><div className="search-input-wrap"><Search size={15} /><input aria-label="Search city or region" value={search} placeholder="LOCATE CITY / REGION" onFocus={() => setSearchOpen(true)} onChange={(event) => { setSearch(event.target.value); setSearchOpen(true); }} onKeyDown={(event) => { if (event.key === "Enter") submitSearch(); if (event.key === "Escape") setSearchOpen(false); }} /><kbd>⌘ K</kbd></div>{searchOpen && <div className="search-results">{searchResults.length ? searchResults.map((location) => <button key={location.name} onMouseDown={(event) => event.preventDefault()} onClick={() => focusLocation(location)}><span className="result-pin"><MapPin size={13} /></span><span><strong>{location.name}</strong><small>{location.region}</small></span><em>{location.lat.toFixed(2)}°N</em></button>) : <div className="no-results">NO MAPPED SECTOR MATCHES</div>}<div className="search-hint">ENTER TO FOCUS <span>ESC TO CLOSE</span></div></div>}</div>
      {focusedLocation && <div className="location-focus" style={{ left: `${focusedLocation.x}%`, top: `${focusedLocation.y}%` }}><span>{focusedLocation.name.toUpperCase()}</span></div>}
      <div className="radar-dock" aria-label="Radar timeline controls"><div><span>RADAR TIMELINE</span><b>{radar.frames.length ? `${radarFrameIndex + 1}/${radar.frames.length}` : "--/--"}</b></div><button onClick={() => setRadarFrameIndex((current) => radar.frames.length ? (current - 1 + radar.frames.length) % radar.frames.length : 0)} aria-label="Previous radar frame"><ChevronLeft size={13} /></button><button className="radar-play" onClick={() => setRadarPlaying((value) => !value)} aria-label={radarPlaying ? "Pause radar animation" : "Play radar animation"}>{radarPlaying ? <Pause size={13} /> : <Play size={13} />}</button><button onClick={() => setRadarFrameIndex((current) => radar.frames.length ? (current + 1) % radar.frames.length : 0)} aria-label="Next radar frame"><ChevronRight size={13} /></button><span className="radar-dock-time">{radar.frames[radarFrameIndex]?.time ? formatDataTimestamp(radar.frames[radarFrameIndex].time) : "WAITING"}</span><label>GRID <select value={gridResolutionKm} onChange={(event) => setGridResolutionKm(Number(event.target.value) as 1 | 1.5 | 2 | 3)}><option value="1">1 KM</option><option value="1.5">1.5 KM</option><option value="2">2 KM</option><option value="3">3 KM</option></select></label></div>

      {showLayers && <Panel className="layers-panel">
        <div className="panel-kicker"><span>01 / OVERLAYS</span><button className="panel-collapse" onClick={() => setShowLayers(false)}><ChevronLeft size={14} /></button></div>
        <div className="panel-title-row"><h2>Hazard layers</h2><SlidersHorizontal size={16} /></div>
        <div className="layer-list">{(Object.keys(hazardMeta) as HazardKey[]).map((key) => { const item = hazardMeta[key]; const Icon = item.icon; return <button key={key} className={`layer-row ${hazards[key] ? "active" : ""}`} onClick={() => toggleHazard(key)}><span className="layer-icon" style={{ color: item.color, background: item.bg }}><Icon size={14} /></span><span className="layer-copy"><b>{item.label}</b><small>{key === "thunderstorm" ? "CAPE / REFLECTIVITY" : key === "cloudburst" ? ">100 MM / HR" : key === "hail" ? "PROBABILITY CORE" : key === "lightning" ? "STRIKE DENSITY" : "SHEAR VECTORS"}</small></span><span className={`switch ${hazards[key] ? "on" : ""}`}><i /></span></button> })}</div>
        <div className="layer-divider"><span>MAP DETAIL</span><i /></div>
        <div className="map-view-toggle" aria-label="Map layer view"><span>VIEW</span>{([['radar', 'RADAR'], ['satellite', 'SATELLITE'], ['admin', 'ADMIN']] as const).map(([value, label]) => <button key={value} className={mapLayer === value ? "active" : ""} onClick={() => setMapLayer(value)}>{label}</button>)}</div>
        <div className="boundary-filters"><label>STATE FILTER<select value={stateFilter} onChange={(event) => { const nextState = event.target.value; setStateFilter(nextState); setDistrictFilter("ALL DISTRICTS"); }}><option>ALL STATES</option>{Object.keys(districtsByState).filter((name) => name !== "ALL STATES").map((name) => <option key={name}>{name}</option>)}</select></label><label>DISTRICT FILTER<select value={districtFilter} onChange={(event) => setDistrictFilter(event.target.value)}><option>ALL DISTRICTS</option>{districtOptions.map((name) => <option key={name}>{name}</option>)}</select></label></div><div className={`grid-scope-readout ${boundaryStatus === "error" ? "boundary-error" : ""}`}>{boundaryStatus === "loading" && <RefreshCw size={10} className="boundary-spinner" />}{boundaryStatus === "error" && <AlertTriangle size={10} />}{districtFilter === "ALL DISTRICTS" ? `ALL ${stateFilter} · NATIONAL GRID` : `${districtFilter.toUpperCase()} · ${gridResolutionKm} KM GRID CELLS · ${boundaryStatus === "loading" ? "BOUNDARY LOADING" : boundaryStatus === "error" ? "BOUNDARY ERROR" : selectedBoundary ? "OFFICIAL POLYGON CLIP" : "DISTRICT NOT FOUND"}`}</div>
        <div className="layer-list map-layer-list">
          <button className={`layer-row ${showStates ? "active" : ""}`} onClick={() => setShowStates((value) => !value)}><span className="layer-icon layer-icon-cyan"><Layers3 size={14} /></span><span className="layer-copy"><b>State boundaries</b><small>ADM 1 / REGIONAL FOCUS</small></span><span className={`switch ${showStates ? "on" : ""}`}><i /></span></button>
          <button className={`layer-row ${showDistricts ? "active" : ""}`} onClick={() => setShowDistricts((value) => !value)}><span className="layer-icon layer-icon-cyan"><MapPin size={14} /></span><span className="layer-copy"><b>District boundaries</b><small>ADM 2 / FINE GRID CONTEXT</small></span><span className={`switch ${showDistricts ? "on" : ""}`}><i /></span></button>
          <button className={`layer-row ${showRadar ? "active" : ""}`} onClick={() => setShowRadar((value) => !value)}><span className="layer-icon layer-icon-radar"><ScanLine size={14} /></span><span className="layer-copy"><b>Live radar field</b><small>{radar.status === "live" ? "RAINVIEWER / 10 MIN FRAME" : radar.message.toUpperCase()}</small></span><span className={`switch ${showRadar ? "on" : ""}`}><i /></span></button>
        </div>
        <div className="layer-note">BOUNDARIES: DATAMEET CC BY 2.5 IN · RADAR: RAINVIEWER</div>
        <div className="legend"><div className="legend-title"><span>REFLECTIVITY / DBZ</span><span>10 — 65</span></div><div className="legend-bar" /><div className="legend-scale"><span>10</span><span>25</span><span>42</span><span>50</span><span>65</span></div></div>
      </Panel>}

      <aside className="alert-stack">
        <div className="panel-kicker"><span>02 / PRIORITY TARGETS</span><span className="target-count"><Target size={12} /> 02</span></div>
        <Panel className="alert-card critical"><div className="alert-card-top"><span className="alert-badge red"><span /> RED ALERT</span><span className="alert-id">TGT-014</span></div><h3>International Airport</h3><div className="alert-location">VIDP / VOHS <span>·</span> 17.24°N, 78.43°E</div><div className="alert-hazard"><CloudRain size={15} /><span>Cloudburst core</span><span className="slash">/</span><CircleDot size={14} /><span>Hail</span></div><div className="eta-line"><span>ETA</span><strong>{formatCountdown(etaSeconds.airport)}</strong></div><div className="trajectory"><ArrowUpRight size={14} /><span>HEADING NE</span><b>42 KM/H</b></div></Panel>
        <Panel className="alert-card warning"><div className="alert-card-top"><span className="alert-badge orange"><span /> ORANGE ALERT</span><span className="alert-id">TGT-021</span></div><h3>Central Railway Terminal</h3><div className="alert-location">SECUNDERABAD <span>·</span> 17.44°N, 78.50°E</div><div className="alert-hazard"><CloudLightning size={15} /><span>Severe storm</span><span className="slash">/</span><Zap size={14} /><span>Lightning</span></div><div className="eta-line"><span>ETA</span><strong>{formatCountdown(etaSeconds.rail)}</strong></div><div className="trajectory"><ArrowUpRight size={14} /><span>HEADING NNE</span><b>28 KM/H</b></div></Panel>
      </aside>

      <Panel className={`inspector-panel ${showInspector ? "visible" : "hidden-panel"}`}>
        <div className="panel-kicker"><span>03 / GRID CELL INSPECTOR</span><button className="panel-collapse" onClick={() => setShowInspector(false)}><X size={14} /></button></div>
        <div className="inspector-heading"><div><h2>Grid cell <em>#{selectedCell?.id ?? 14}_88</em></h2><span>{selectedCoordinates ? `${selectedCoordinates.lat.toFixed(3)}°N, ${selectedCoordinates.lon.toFixed(3)}°E` : "—"} <i /> 1.5 × 1.5 KM</span></div><MapPin size={18} /></div>
        <div className="metric-grid"><div><span>REFLECTIVITY</span><b className="metric-red">{selectedTelemetry?.reflectivity ?? frame.reflectivity}<small> dBZ</small></b><em>SEVERE CORE</em></div><div><span>EST. RAIN RATE</span><b className="metric-red">{selectedTelemetry?.rainRate ?? frame.rainRate}<small> mm/hr</small></b><em>CLOUDBURST ALERT</em></div><div><span>HAIL PROBABILITY</span><b className="metric-orange">{selectedTelemetry?.hail ?? frame.hail}<small>%</small></b><em>FREEZING LAYER 4,100M</em></div><div><span>SURFACE WIND GUST</span><b>{selectedTelemetry?.wind ?? frame.wind}<small> km/h</small></b><em className="metric-cyan">NE TRAJECTORY</em></div><div><span>LIGHTNING DENSITY</span><b className="metric-purple">{selectedTelemetry?.lightning ?? frame.lightning}<small> /km²/10m</small></b><em>HIGH ACTIVITY</em></div><div><span>MODEL CONFIDENCE</span><b className="metric-green">{primaryWeather ? Math.max(86, Math.round(100 - primaryWeather.precipitation * 2)) : 94}<small>%</small></b><em>{weather.status === "live" ? "LIVE OBSERVATION LINKED" : "SIMULATION MODE"}</em></div></div>
      </Panel>

      <Panel className="telemetry-panel"><div className="panel-kicker"><span>04 / MODEL TELEMETRY</span><span className={`telemetry-live ${weather.status === "error" && radar.status === "error" ? "telemetry-error" : ""}`}><span /> {weather.status === "live" || radar.status === "live" ? "STREAMING" : "FALLBACK"}</span></div><div className="model-row"><div className="model-icon"><Sparkles size={15} /></div><div><h2>ConvGRU <span>+ Spatial Attention</span></h2><p>SELF-LEARNING INFERENCE BACKBONE</p></div><RefreshCw size={15} className="spin-slow" /></div><div className="telemetry-stats"><div><span>INFERENCE LATENCY</span><b>32 <small>MS</small></b></div><div><span>LIVE TEMPERATURE</span><b>{primaryWeather ? primaryWeather.temperature.toFixed(1) : "—"}<small> °C</small></b><em>{primaryWeather?.label ?? "WAITING FOR OBS"}</em></div><div><span>WIND GUST</span><b>{primaryWeather ? primaryWeather.wind.toFixed(0) : "—"}<small> KM/H</small></b><em>OPEN-METEO</em></div></div><div className="confidence"><span>FRAME CONFIDENCE</span><div className="confidence-track"><i style={{ width: `${94 - (frame.loss * 100)}%` }} /></div><b>94.0%</b></div><div className="source-strip"><span className="data-provenance">ALERT BASIS: THRESHOLD RULES + SIMULATED CONVGRU FIELD · NOT OFFICIAL WARNING FEED</span><span className={weather.status}><i /> WEATHER {weather.status.toUpperCase()}</span><span className={radar.status}><i /> RADAR {radar.status.toUpperCase()}</span><small>{formatDataTimestamp(weather.updatedAt)} · {formatDataTimestamp(radar.updatedAt)}</small></div></Panel>

      <TelemetryPanels cellId={activeCell?.id ?? 14} date={historyDate} maxDate={archiveDate(3)} hour={historyHour} history={history} thresholds={thresholds} alarmsEnabled={alarmsEnabled} activeAlerts={thresholdAlerts} alertLog={alertLog} historyOpen={showHistory} onToggleHistory={() => setShowHistory((value) => !value)} onDateChange={(value) => { setHistoryDate(value); setHistoryHour(12); }} onHourChange={setHistoryHour} onThresholdChange={(key, value) => setThresholds((current) => ({ ...current, [key]: value }))} onToggleAlarms={() => setAlarmsEnabled((value) => !value)} onEnableNotifications={() => { if ("Notification" in window && Notification.permission === "default") Notification.requestPermission(); toast.info("Browser alerts ready", { description: "VAJRA will notify when the selected cell crosses an armed threshold." }); }} />
      <section className={`playback-dock ${showForecast ? "" : "collapsed-playback"}`}><button className="playback-collapse" aria-label={showForecast ? "Collapse forecast playback" : "Expand forecast playback"} onClick={() => setShowForecast((value) => !value)}>{showForecast ? "−" : "+ 05 / FORECAST"}</button>{showForecast && <div className="playback-content"><div className="playback-head"><div><span className="panel-kicker">05 / FORECAST PLAYBACK</span><h2>{formatLead(lead)} <i>·</i> {new Date(Date.now() + lead * 60000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })} UTC</h2></div><div className="frame-status"><Radio size={13} /> FRAME {String(lead / 10 + 1).padStart(2, "0")} / 37</div></div><div className="trend-chart">{trend.map((height, index) => <div key={index} className={`trend-bar ${timeline[index] <= lead ? "filled" : ""}`} style={{ height: `${height}%` }} />)}<div className="trend-line" style={{ left: `${(lead / 360) * 100}%` }} /></div><div className="timeline"><span className="timeline-label now">NOW</span><div className="timeline-track"><input aria-label="Forecast timeline" type="range" min="0" max="360" step="10" value={lead} onChange={(event) => setLead(Number(event.target.value))} /><div className="timeline-ticks">{timeline.filter((minute) => minute % 60 === 0).map((minute) => <span key={minute} style={{ left: `${(minute / 360) * 100}%` }}>{minute === 0 ? "T+0" : `T+${minute / 60}H`}</span>)}</div></div><span className="timeline-label end">+6H</span></div><div className="playback-controls"><div className="transport"><button className="transport-button" onClick={() => setLead((value) => Math.max(0, value - 10))}><ChevronLeft size={16} /><span>−10M</span></button><button className="play-button" aria-label={playing ? "Pause forecast" : "Play forecast"} onClick={() => setPlaying((value) => !value)}>{playing ? <Pause size={19} fill="currentColor" /> : <Play size={19} fill="currentColor" />}</button><button className="transport-button" onClick={() => setLead((value) => Math.min(360, value + 10))}><span>+10M</span><ChevronRight size={16} /></button></div><div className="speed-toggle"><span>SPEED</span>{[1, 2, 5].map((value) => <button key={value} className={speed === value ? "active" : ""} onClick={() => setSpeed(value)}>{value}×</button>)}</div><div className="playback-right"><span><Gauge size={14} /> THREAT PEAK</span><b>{frame.reflectivity} <small>dBZ</small></b><span className="forecast-window"><BatteryCharging size={14} /> 06:00 WINDOW</span></div></div></div>}</section>

      <footer className="bottom-footer"><span>VAJRA / ATMOSPHERIC INTELLIGENCE SYSTEM</span><span><ShieldAlert size={12} /> {weather.status === "live" || radar.status === "live" ? "LIVE WEATHER + RADAR" : "SIMULATION FALLBACK"} <i /> REFRESH 05:00</span><span>BUILD 2.4.18 <Settings2 size={12} /></span></footer>
    </main>
  );
}
