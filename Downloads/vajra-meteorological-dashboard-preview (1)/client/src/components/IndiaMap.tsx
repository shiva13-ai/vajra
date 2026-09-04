import { useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";

const INDIA_BOUNDS = { north: 37.5, south: 6.5, west: 67, east: 98.5 };
const INDIA_GEOJSON_URL = "/manus-storage/india-land-simplified_0823e409.geojson";
const INDIA_STATES_URL = "/manus-storage/india-states_5adaa029.geojson";
const INDIA_DISTRICTS_URL = "/manus-storage/india-districts-official_3a81e00d.geojson";

type Coordinate = [number, number];
type GeoJsonGeometry = { type?: string; coordinates?: unknown };
type GeoJsonFeature = { geometry?: GeoJsonGeometry; properties?: Record<string, unknown> };
type GeoJsonCollection = { features?: GeoJsonFeature[] };
type NamedPath = { path: string; name: string; anchor?: { x: number; y: number } };

export type WeatherPoint = {
  id: string;
  label: string;
  lat: number;
  lon: number;
  temperature: number;
  precipitation: number;
  wind: number;
  code: number;
};

type IndiaMapProps = {
  className?: string;
  showStates?: boolean;
  showDistricts?: boolean;
  radarUrl?: string;
  radarFrames?: Array<{ url: string; time: string }>;
  radarPlaying?: boolean;
  radarFrameIndex?: number;
  gridResolutionKm?: 1 | 1.5 | 2 | 3;
  mapLayer?: "radar" | "satellite" | "admin";
  stateFilter?: string;
  districtFilter?: string;
  weatherPoints?: WeatherPoint[];
  onRadarError?: () => void;
};

function project([lon, lat]: Coordinate) {
  return {
    x: ((lon - INDIA_BOUNDS.west) / (INDIA_BOUNDS.east - INDIA_BOUNDS.west)) * 1000,
    y: ((INDIA_BOUNDS.north - lat) / (INDIA_BOUNDS.north - INDIA_BOUNDS.south)) * 900,
  };
}

function pathFromCoordinates(coords: Coordinate[], close = false) {
  const path = coords.map((coordinate, index) => {
    const point = project(coordinate);
    return `${index === 0 ? "M" : "L"}${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
  }).join(" ");
  return close ? `${path} Z` : path;
}

function pathsFromGeometry(geometry?: GeoJsonGeometry) {
  if (!geometry?.type || !geometry.coordinates) return [];
  const coordinates = geometry.coordinates as any;
  if (geometry.type === "LineString") return [pathFromCoordinates(coordinates as Coordinate[])];
  if (geometry.type === "MultiLineString") return (coordinates as Coordinate[][]).map((line) => pathFromCoordinates(line));
  if (geometry.type === "Polygon") return (coordinates as Coordinate[][]).map((ring) => pathFromCoordinates(ring, true));
  if (geometry.type === "MultiPolygon") return (coordinates as Coordinate[][][]).flatMap((polygon) => polygon.map((ring) => pathFromCoordinates(ring, true)));
  return [];
}

function featureName(feature: GeoJsonFeature) {
  const values = Object.entries(feature.properties ?? {}).filter(([key]) => /name|state|district|st_nm|dt_nm/i.test(key)).map(([, value]) => String(value));
  return values.find((value) => value && value !== "null" && value !== "undefined") ?? "";
}
function normalizeBoundaryName(name: string) { return name.toLowerCase().replace(/[^a-z]/g, "").replace("hydrabad", "hyderabad"); }

function geometryCenter(geometry?: GeoJsonGeometry) {
  const points: Coordinate[] = [];
  const collect = (value: unknown) => { if (Array.isArray(value) && typeof value[0] === "number" && typeof value[1] === "number") points.push([Number(value[0]), Number(value[1])]); else if (Array.isArray(value)) value.forEach(collect); };
  collect(geometry?.coordinates);
  if (!points.length) return undefined;
  const center = points.reduce((sum, [lon, lat]) => [sum[0] + lon, sum[1] + lat], [0, 0]);
  return project([center[0] / points.length, center[1] / points.length]);
}

function extractPaths(data: GeoJsonCollection) {
  return (data.features ?? []).flatMap((feature) => pathsFromGeometry(feature.geometry));
}

function extractNamedPaths(data: GeoJsonCollection): NamedPath[] {
  return (data.features ?? []).flatMap((feature) => pathsFromGeometry(feature.geometry).map((path) => ({ path, name: featureName(feature), anchor: geometryCenter(feature.geometry) })));
}

export function IndiaMap({ className, showStates = false, showDistricts = false, radarUrl, radarFrames = [], radarPlaying = false, radarFrameIndex = 0, gridResolutionKm = 1.5, stateFilter = "ALL STATES", districtFilter = "ALL DISTRICTS", weatherPoints = [], onRadarError }: IndiaMapProps) {
  const [boundaryPath, setBoundaryPath] = useState("");
  const [statePaths, setStatePaths] = useState<NamedPath[]>([]);
  const [districtPaths, setDistrictPaths] = useState<NamedPath[]>([]);
  const [boundaryStatus, setBoundaryStatus] = useState<"loading" | "live" | "error">("loading");
  const [visibleRadarIndex, setVisibleRadarIndex] = useState(0);

  useEffect(() => {
    if (!radarPlaying || radarFrames.length < 2) { setVisibleRadarIndex(radarFrameIndex); return; }
    const timer = window.setInterval(() => setVisibleRadarIndex((current) => (current + 1) % radarFrames.length), 650);
    return () => window.clearInterval(timer);
  }, [radarPlaying, radarFrames.length, radarFrameIndex]);

  useEffect(() => {
    let active = true;
    setBoundaryStatus("loading");
    Promise.all([INDIA_GEOJSON_URL, INDIA_STATES_URL, INDIA_DISTRICTS_URL].map((url) => fetch(url).then((response) => response.ok ? response.json() : Promise.reject(new Error("Boundary unavailable")))))
      .then(([country, states, districts]) => {
        if (!active) return;
        const countryPath = extractPaths(country as GeoJsonCollection)[0] ?? "";
        setBoundaryPath(countryPath);
        setStatePaths(extractNamedPaths(states as GeoJsonCollection));
        setDistrictPaths(extractNamedPaths(districts as GeoJsonCollection));
        setBoundaryStatus("live");
      })
      .catch(() => {
        if (!active) return;
        setBoundaryPath("");
        setStatePaths([]);
        setDistrictPaths([]);
        setBoundaryStatus("error");
      });
    return () => { active = false; };
  }, []);

  const grid = useMemo(() => {
    const lines: Array<{ x1: number; y1: number; x2: number; y2: number; label: string }> = [];
    const latStep = gridResolutionKm / 111;
    const lonStep = gridResolutionKm / (111 * Math.cos((INDIA_BOUNDS.south + INDIA_BOUNDS.north) * Math.PI / 360));
    for (let lat = INDIA_BOUNDS.south; lat <= INDIA_BOUNDS.north; lat += latStep) {
      const y = project([INDIA_BOUNDS.west, lat]).y;
      lines.push({ x1: 0, y1: y, x2: 1000, y2: y, label: `${lat.toFixed(1)}N` });
    }
    for (let lon = INDIA_BOUNDS.west; lon <= INDIA_BOUNDS.east; lon += lonStep) {
      const x = project([lon, INDIA_BOUNDS.south]).x;
      lines.push({ x1: x, y1: 0, x2: x, y2: 900, label: `${lon.toFixed(1)}E` });
    }
    return lines;
  }, [gridResolutionKm]);

  return (
    <div className={cn("india-map-layer", className)} aria-label="India map with state, district, and weather radar layers">
      <svg className="india-map-svg" viewBox="0 0 1000 900" role="img" aria-labelledby="india-map-title">
        <title id="india-map-title">India national weather operating grid</title>
        <defs>
          <linearGradient id="indiaLandFill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#17334a" stopOpacity=".82" />
            <stop offset="1" stopColor="#0d1c2a" stopOpacity=".96" />
          </linearGradient>
          <filter id="indiaBoundaryGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <pattern id="microDots" width="18" height="18" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="2" r=".7" fill="#83dfff" opacity=".18" />
          </pattern>
          <clipPath id="indiaClip">{boundaryPath && <path d={`${boundaryPath} Z`} />}</clipPath>
        </defs>
        <rect width="1000" height="900" fill="#091522" />
        <g className="india-grid-lines">
          {grid.map((line, index) => <line key={index} x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} />)}
        </g>
        {(radarUrl || radarFrames.length > 0) && <g className="india-radar-stack">{(radarFrames.length ? radarFrames : [{ url: radarUrl ?? "", time: "" }]).map((frame, index) => <image key={`${frame.url}-${index}`} className="india-radar-image" href={frame.url} x="60" y="90" width="880" height="710" preserveAspectRatio="none" opacity={index === visibleRadarIndex ? ".48" : "0"} onError={onRadarError} aria-hidden="true" />)}</g>}
        {weatherPoints.length > 0 && <g className="india-weather-points" aria-label="Live weather observations">{weatherPoints.map((point) => { const projected = project([point.lon, point.lat]); return <g key={point.id} transform={`translate(${projected.x} ${projected.y})`}><circle r="11" className="weather-point-pulse" /><circle r="4" className="weather-point-core" /><text x="10" y="-9">{point.label.toUpperCase()}</text><text x="10" y="5">{point.temperature.toFixed(0)}° · {point.precipitation.toFixed(1)}MM</text></g>; })}</g>}
        {showDistricts && <g className="india-district-lines" aria-label="District boundaries">{districtPaths.filter((item) => districtFilter === "ALL DISTRICTS" || normalizeBoundaryName(item.name).includes(normalizeBoundaryName(districtFilter))).map((item, index) => <g key={index}><path d={item.path} /><text x={item.anchor?.x ?? 0} y={item.anchor?.y ?? 0}>{item.name.replace(/^Hydrabad$/i, "Hyderabad")}</text></g>)}</g>}
        {showStates && <g className="india-state-lines" aria-label="State boundaries">{statePaths.filter((item) => stateFilter === "ALL STATES" || item.name.toLowerCase().includes(stateFilter.toLowerCase())).map((item, index) => <path key={index} d={item.path} />)}</g>}
        {boundaryPath && <>
          <path d={`${boundaryPath} Z`} fill="url(#indiaLandFill)" stroke="#38BDF8" strokeOpacity=".24" strokeWidth="7" vectorEffect="non-scaling-stroke" filter="url(#indiaBoundaryGlow)" />
          <g clipPath="url(#indiaClip)"><rect width="1000" height="900" fill="url(#microDots)" opacity=".42" /></g>
          <path d={boundaryPath} fill="none" stroke="#b9efff" strokeOpacity=".95" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        </>}
        <g className="india-grid-labels">
          {grid.filter((_, index) => index % Math.max(1, Math.round(grid.length / 18)) === 0).map((line, index) => <text key={index} x={line.x1 + 8} y={line.y1 - 5}>{line.label}</text>)}
        </g>
      </svg>
      <div className="india-map-readout"><span>BOUNDARY / INDIA</span><i /> <span>ADM2 {boundaryStatus === "loading" ? "LOADING" : boundaryStatus === "error" ? "ERROR" : "OFFICIAL"}</span><i className="grid-readout-dot" /> <span>FULL GRID {gridResolutionKm} KM</span>{(radarUrl || radarFrames.length > 0) && <><i className="radar-readout-dot" /> <span>RADAR {radarFrames.length > 1 ? `${visibleRadarIndex + 1}/${radarFrames.length}` : "LINKED"}</span></>}{weatherPoints.length > 0 && <><i className="weather-readout-dot" /> <span>LIVE OBS {weatherPoints.length}</span></>}{boundaryStatus === "loading" && <b className="boundary-status-spinner" aria-label="Loading official district boundaries" />}{boundaryStatus === "error" && <b className="boundary-status-error" role="status">!</b>}</div>
    </div>
  );
}
