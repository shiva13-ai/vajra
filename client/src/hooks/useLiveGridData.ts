/**
 * useLiveGridData — Replaces hardcoded 64-sector grid values with LIVE Open-Meteo data.
 *
 * Strategy:
 *  - On mount, takes the existing NATIONWIDE_INDIAN_GRID (64 sectors with real lat/lon)
 *  - Batches them into Open-Meteo multi-coordinate API calls (max 50 per call)
 *  - Fetches live: temperature, precipitation, wind_speed, wind_gusts, cloud_cover, weather_code,
 *    CAPE, lifted_index, freezing_level_height
 *  - Overwrites the static values in each sector with live data
 *  - Refreshes every 5 minutes
 *  - Falls back to original hardcoded values if API fails
 */
import { useEffect, useState, useRef, useCallback } from "react";
import {
  NATIONWIDE_INDIAN_GRID,
  type IndianGridSector,
  type SeverityLevel,
} from "@/lib/indiaMeteorologicalGrid";
import { zToRainRate } from "@/lib/mlNowcastingEngine";

const OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast";
const REFRESH_MS = 5 * 60 * 1000; // 5 minutes
const BATCH_SIZE = 50; // Open-Meteo max coordinates per call

type LiveGridState = {
  sectors: IndianGridSector[];
  status: "loading" | "live" | "stale" | "error";
  lastUpdated: string;
  liveCount: number; // How many sectors got live data
  errorMsg: string;
};

function computeSeverity(
  dbz: number,
  rainRate: number,
  cape: number,
  li: number,
  windGust: number
): { severity: SeverityLevel; statusLabel: string } {
  if (rainRate >= 100 || dbz >= 55 || (cape >= 2500 && li <= -5)) {
    return { severity: "severe", statusLabel: "CLOUDBURST ALERT (≥100 MM/HR)" };
  }
  if (dbz >= 45 || cape >= 2000 || li <= -4 || windGust >= 70) {
    return { severity: "high", statusLabel: "ACTIVE CONVECTIVE CORE" };
  }
  if (dbz >= 35 || cape >= 1200 || li <= -2.5 || windGust >= 50) {
    return { severity: "moderate", statusLabel: "THUNDERSTORM CELL ACTIVE" };
  }
  if (dbz >= 22 || cape >= 700 || windGust >= 30) {
    return { severity: "mild", statusLabel: "SCATTERED CONVECTIVE SHOWERS" };
  }
  return { severity: "clear", statusLabel: "STABLE ATMOSPHERE" };
}

function computePrimaryHazard(
  dbz: number,
  rainRate: number,
  cape: number,
  li: number,
  windGust: number,
  freezingLevel: number,
  hailProb: number
): IndianGridSector["predictions"]["primaryHazard"] {
  if (rainRate >= 80 || dbz >= 55) return "Cloudburst";
  if (hailProb >= 70 && freezingLevel < 4000) return "Hail";
  if (windGust >= 65 && dbz >= 45) return "Downburst";
  if (cape >= 1500 && li <= -3) return "Lightning";
  if (dbz >= 35) return "Thunderstorm";
  return "None";
}

export function useLiveGridData(): LiveGridState {
  const [state, setState] = useState<LiveGridState>({
    sectors: NATIONWIDE_INDIAN_GRID, // start with static fallback
    status: "loading",
    lastUpdated: "",
    liveCount: 0,
    errorMsg: "",
  });
  const abortRef = useRef<AbortController | null>(null);

  const fetchLiveData = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const sectors = [...NATIONWIDE_INDIAN_GRID];
      const liveUpdates = new Map<string, Partial<IndianGridSector>>();

      // Batch sectors into groups of BATCH_SIZE
      for (let i = 0; i < sectors.length; i += BATCH_SIZE) {
        const batch = sectors.slice(i, i + BATCH_SIZE);
        const lats = batch.map((s) => s.lat.toFixed(4)).join(",");
        const lons = batch.map((s) => s.lon.toFixed(4)).join(",");

        const url =
          `${OPEN_METEO_URL}?latitude=${lats}&longitude=${lons}` +
          `&current=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,relative_humidity_2m,cloud_cover,weather_code` +
          `&hourly=cape,lifted_index,freezing_level_height` +
          `&forecast_hours=2&timezone=auto`;

        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`Open-Meteo batch: ${res.status}`);

        const payload = await res.json();

        // Open-Meteo returns an array when multiple coordinates, or a single object for one
        const results: any[] = Array.isArray(payload) ? payload : [payload];

        results.forEach((result: any, idx: number) => {
          const sector = batch[idx];
          if (!sector || !result?.current) return;

          const temp = Number(result.current.temperature_2m ?? sector.temperatureC);
          const precip = Number(result.current.precipitation ?? 0);
          const windSpeed = Number(result.current.wind_speed_10m ?? 0);
          const windGust = Number(result.current.wind_gusts_10m ?? sector.windGustKmh);
          const humidity = Number(result.current.relative_humidity_2m ?? sector.humidityPercent);
          const cloudCover = Number(result.current.cloud_cover ?? 0);
          const weatherCode = Number(result.current.weather_code ?? 0);
          const elevation = Number(result.elevation ?? sector.elevationMeters);

          // Hourly atmospheric data (first hour)
          const cape = Number(result.hourly?.cape?.[0] ?? sector.capeJkg);
          const li = Number(result.hourly?.lifted_index?.[0] ?? sector.liftedIndex);
          const freezingLevel = Number(
            result.hourly?.freezing_level_height?.[0] ?? sector.freezingLevelMeters
          );

          // Derive radar reflectivity from precipitation rate via Marshall-Palmer inverse
          // Z = 200 * R^1.6 → dBZ = 10 * log10(Z)
          let dbz = 12.0;
          if (precip > 0.05) {
            // Marshall-Palmer equation: Z = 200 * R^1.6 => dBZ = 10 * log10(Z)
            const Z = 200 * Math.pow(precip, 1.6);
            dbz = Number((10 * Math.log10(Math.max(1, Z))).toFixed(1));
          } else if (weatherCode >= 95) {
            // Severe convective thunderstorm code (WMO 95/96/99)
            const capeBonus = Math.min(12, Math.max(0, (cape - 1000) / 150));
            dbz = Number((46.0 + capeBonus).toFixed(1));
          } else if (weatherCode >= 80) {
            // Convective rain showers (WMO 80/81/82)
            dbz = Number((34.0 + (humidity > 80 ? 4 : 0)).toFixed(1));
          } else if (weatherCode >= 61) {
            // Rain (WMO 61/63/65)
            dbz = Number((26.0 + (cloudCover > 80 ? 3 : 0)).toFixed(1));
          } else if (weatherCode >= 51) {
            // Drizzle (WMO 51/53/55)
            dbz = Number((18.0 + (humidity > 85 ? 2 : 0)).toFixed(1));
          } else {
            // Stable atmosphere / light boundary layer scatter
            dbz = Number((10.0 + (humidity > 70 ? 2 : 0)).toFixed(1));
          }
          dbz = Number(dbz.toFixed(1));

          const rainRate = precip > 0 ? precip : zToRainRate(dbz);

          // Compute lightning density from CAPE: updraft = sqrt(2 * CAPE)
          const updraftV = Math.sqrt(2 * Math.max(100, cape));
          const lightningDensity =
            cape >= 800 && li <= -2
              ? Number(((updraftV / 35) * (dbz / 40) * (Math.abs(li) / 2.5)).toFixed(1))
              : 0;

          // Hail probability
          const hailTempFactor =
            freezingLevel < 3900 ? 1.3 : freezingLevel > 4600 ? 0.7 : 1.0;
          const hailProb = Math.round(
            Math.min(
              98,
              Math.max(
                2,
                (dbz > 50
                  ? (dbz - 45) * 5
                  : dbz > 35
                  ? (dbz - 30) * 2
                  : 2) *
                  hailTempFactor *
                  (Math.abs(Math.min(0, li)) / 3)
              )
            )
          );

          const { severity, statusLabel } = computeSeverity(dbz, rainRate, cape, li, windGust);
          const primaryHazard = computePrimaryHazard(
            dbz, rainRate, cape, li, windGust, freezingLevel, hailProb
          );

          // Build prediction steps based on live intensity
          const decayRate = severity === "severe" ? 0.06 : severity === "high" ? 0.08 : 0.12;
          const t15Dbz = Number((dbz * (1 + (severity === "severe" ? 0.05 : 0.02))).toFixed(1));
          const t30Dbz = Number((dbz * (1 - decayRate * 0.5)).toFixed(1));
          const t45Dbz = Number((dbz * (1 - decayRate * 1.5)).toFixed(1));
          const t60Dbz = Number((dbz * (1 - decayRate * 3)).toFixed(1));

          liveUpdates.set(sector.id, {
            temperatureC: temp,
            rainRateMmHr: Number(rainRate.toFixed(1)),
            windGustKmh: Math.round(windGust),
            humidityPercent: Math.round(humidity),
            elevationMeters: Math.round(elevation),
            capeJkg: Math.round(cape),
            liftedIndex: Number(li.toFixed(1)),
            freezingLevelMeters: Math.round(freezingLevel),
            reflectivityDbz: dbz,
            lightningDensity: Math.max(0, lightningDensity),
            hailProbability: hailProb,
            severity,
            statusLabel,
            predictions: {
              t15: {
                minute: 15,
                reflectivityDbz: t15Dbz,
                rainRateMmHr: Number((rainRate * 1.05).toFixed(1)),
                trend: severity === "severe" || severity === "high" ? "intensifying" : "steady",
                hazardNote:
                  severity === "severe"
                    ? "Extreme convective moisture pooling"
                    : severity === "high"
                    ? "Active updraft intensification"
                    : "Stable advection",
              },
              t30: {
                minute: 30,
                reflectivityDbz: t30Dbz,
                rainRateMmHr: Number((rainRate * 0.9).toFixed(1)),
                trend: "steady",
                hazardNote: "Core persistence projected",
              },
              t45: {
                minute: 45,
                reflectivityDbz: t45Dbz,
                rainRateMmHr: Number((rainRate * 0.6).toFixed(1)),
                trend: "decaying",
                hazardNote: "Decay phase onset",
              },
              t60: {
                minute: 60,
                reflectivityDbz: t60Dbz,
                rainRateMmHr: Number((rainRate * 0.3).toFixed(1)),
                trend: "decaying",
                hazardNote: "Stratiform transition",
              },
              summary: `LIVE: ${temp.toFixed(1)}°C, ${Math.round(windGust)} km/h gusts, CAPE ${Math.round(cape)} J/kg, LI ${li.toFixed(1)}. ${
                severity === "severe"
                  ? "CRITICAL: Extreme convective instability detected."
                  : severity === "high"
                  ? "Strong convective development in progress."
                  : severity === "moderate"
                  ? "Moderate thunderstorm conditions."
                  : "Conditions stable."
              }`,
              modelConfidence: severity === "severe" ? 96 : severity === "high" ? 93 : 89,
              primaryHazard,
              spatialAttentionScore: Number(
                Math.min(0.99, Math.max(0.1, dbz / 60)).toFixed(2)
              ),
            },
          });
        });
      }

      // Merge live data into sectors
      let updatedSectors = sectors.map((sector) => {
        const live = liveUpdates.get(sector.id);
        if (!live) return sector;
        return { ...sector, ...live } as IndianGridSector;
      });

      // Prepare comprehensive observation payload for ALL 64 nationwide Indian sectors
      const allSectorsObservations = updatedSectors.map((sector) => ({
        sector_id: sector.id,
        id: sector.id,
        reflectivity: sector.reflectivityDbz ?? 30,
        rain_rate: sector.rainRateMmHr ?? 5,
        cape: sector.capeJkg ?? 1500,
        li: sector.liftedIndex ?? -3,
        freezing_level: sector.freezingLevelMeters ?? 4200,
        wind_gust: sector.windGustKmh ?? 35,
        hail_prob: sector.hailProbability ?? 10,
        temp: sector.temperatureC ?? 28,
        humidity: sector.humidityPercent ?? 75,
        elevation: sector.elevationMeters ?? 400,
      }));

      // 1. Continuous ML Self-Learning: Ingest real soundings for all 64 sectors into training engine
      fetch("/api/ml/continuous-learn/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ observations: allSectorsObservations }),
      }).catch(() => {});

      // 2. Nationwide Batch Inference: Evaluate latest self-learned checkpoints across ALL 64 grids
      try {
        const predRes = await fetch("/api/ml/predict-all-grids", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sectors: allSectorsObservations }),
          signal: controller.signal,
        });

        if (predRes.ok) {
          const predData = await predRes.json();
          const predictionsMap = predData.predictions_by_sector || {};

          updatedSectors = updatedSectors.map((sec) => {
            const p = predictionsMap[sec.id];
            if (p) {
              return {
                ...sec,
                mlThunderstormProb: p.thunderstorm_prob,
                mlCloudburstProb: p.cloudburst_prob,
                mlHailProb: p.hail_prob,
                mlMicroburstProb: p.microburst_prob,
                mlProvenHazard: p.is_proven_hazard,
                mlDominantThreat: p.dominant_threat,
                mlEvaluatedAt: new Date().toISOString(),
              };
            }
            return sec;
          });
        }
      } catch {
        // Non-blocking fallback if ML inference service takes longer than network tick
      }

      setState({
        sectors: updatedSectors,
        status: "live",
        lastUpdated: new Date().toISOString(),
        liveCount: liveUpdates.size,
        errorMsg: "",
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setState((prev) => ({
        ...prev,
        status: prev.liveCount > 0 ? "stale" : "error",
        errorMsg: err instanceof Error ? err.message : "Live grid fetch failed",
      }));
    }
  }, []);

  useEffect(() => {
    fetchLiveData();
    const timer = setInterval(fetchLiveData, REFRESH_MS);
    return () => {
      abortRef.current?.abort();
      clearInterval(timer);
    };
  }, [fetchLiveData]);

  return state;
}
