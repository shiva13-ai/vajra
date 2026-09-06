/**
 * useRainViewer — Comprehensive RainViewer API integration
 * Provides: Live radar frames, satellite IR (cloud-top temperature), storm timing metadata
 * Refresh: Every 2 minutes — continuous live data stream
 */
import { useState, useEffect, useRef, useCallback } from "react";

const RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json";
const REFRESH_MS = 2 * 60 * 1000;

export type RVFrame = {
  timestamp: number;
  path: string;
  isoString: string;
  label: string; // "−40m", "−30m", "NOW", "+10m", "+20m" etc
  isPast: boolean;
  isNowcast: boolean;
};

export type StormTiming = {
  frameTimestamp: number;
  frameIndex: number;
  totalFrames: number;
  minutesFromNow: number; // negative = past
  isoString: string;
  label: string;
};

export type RainViewerState = {
  host: string;
  radarFrames: RVFrame[];       // past + nowcast merged
  satelliteFrames: RVFrame[];   // infrared satellite frames
  currentRadarIndex: number;
  currentSatIndex: number;
  isPlaying: boolean;
  status: "loading" | "live" | "error";
  lastUpdated: string;
  errorMsg: string;
  /** Move to next radar frame */
  nextFrame: () => void;
  prevFrame: () => void;
  setFrameIndex: (i: number) => void;
  togglePlay: () => void;
};

export function useRainViewer(): RainViewerState {
  const [host, setHost] = useState("");
  const [radarFrames, setRadarFrames] = useState<RVFrame[]>([]);
  const [satelliteFrames, setSatelliteFrames] = useState<RVFrame[]>([]);
  const [currentRadarIndex, setCurrentRadarIndex] = useState(0);
  const [currentSatIndex, setCurrentSatIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [status, setStatus] = useState<"loading" | "live" | "error">("loading");
  const [lastUpdated, setLastUpdated] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(RAINVIEWER_API, { signal: controller.signal });
      if (!res.ok) throw new Error(`RainViewer API: ${res.status}`);
      const payload = await res.json();

      const h: string = payload.host ?? "";
      const now = Date.now();

      // --- Radar frames (past + nowcast) ---
      const pastRaw: Array<{ time: number; path: string }> = payload.radar?.past ?? [];
      const nowcastRaw: Array<{ time: number; path: string }> = payload.radar?.nowcast ?? [];

      const buildFrame = (
        item: { time: number; path: string },
        isPast: boolean,
        isNowcast: boolean
      ): RVFrame => {
        const ts = item.time * 1000;
        const diffMin = Math.round((ts - now) / 60000);
        const iso = new Date(ts).toISOString();
        let label = "";
        if (Math.abs(diffMin) <= 5) label = "NOW";
        else if (diffMin < 0) label = `${diffMin}m`;
        else label = `+${diffMin}m`;
        return { timestamp: item.time, path: item.path, isoString: iso, label, isPast, isNowcast };
      };

      const radar: RVFrame[] = [
        ...pastRaw.slice(-10).map((f) => buildFrame(f, true, false)),
        ...nowcastRaw.slice(0, 4).map((f) => buildFrame(f, false, true)),
      ];

      // --- Satellite IR frames ---
      const satRaw: Array<{ time: number; path: string }> =
        payload.satellite?.infrared ?? [];

      const satellite: RVFrame[] = satRaw
        .slice(-8)
        .map((f) => buildFrame(f, true, false));

      setHost(h);
      setRadarFrames(radar);
      setSatelliteFrames(satellite);
      setCurrentRadarIndex(radar.findIndex((f) => !f.isPast) - 1 || radar.length - 5);
      setCurrentSatIndex(satellite.length > 0 ? satellite.length - 1 : 0);
      setStatus("live");
      setLastUpdated(new Date().toISOString());
      setErrorMsg("");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "RainViewer unavailable");
    }
  }, []);

  // Initial fetch + refresh every 2 minutes
  useEffect(() => {
    fetchData();
    const refresh = setInterval(fetchData, REFRESH_MS);
    return () => {
      abortRef.current?.abort();
      clearInterval(refresh);
    };
  }, [fetchData]);

  // Auto-play: advance radar frame every 600ms
  useEffect(() => {
    if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    if (!isPlaying || radarFrames.length === 0) return;

    playIntervalRef.current = setInterval(() => {
      setCurrentRadarIndex((prev) => (prev + 1) % radarFrames.length);
    }, 600);

    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, radarFrames.length]);

  const nextFrame = useCallback(() => {
    setCurrentRadarIndex((prev) => (prev + 1) % Math.max(1, radarFrames.length));
  }, [radarFrames.length]);

  const prevFrame = useCallback(() => {
    setCurrentRadarIndex((prev) =>
      prev === 0 ? Math.max(0, radarFrames.length - 1) : prev - 1
    );
  }, [radarFrames.length]);

  const setFrameIndex = useCallback((i: number) => {
    setCurrentRadarIndex(Math.max(0, Math.min(i, radarFrames.length - 1)));
  }, [radarFrames.length]);

  const togglePlay = useCallback(() => setIsPlaying((p) => !p), []);

  return {
    host,
    radarFrames,
    satelliteFrames,
    currentRadarIndex,
    currentSatIndex,
    isPlaying,
    status,
    lastUpdated,
    errorMsg,
    nextFrame,
    prevFrame,
    setFrameIndex,
    togglePlay,
  };
}
