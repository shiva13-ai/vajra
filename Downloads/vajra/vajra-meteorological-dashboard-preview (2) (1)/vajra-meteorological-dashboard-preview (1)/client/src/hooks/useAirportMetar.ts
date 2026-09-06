/**
 * useAirportMetar — Fetches live METAR ground-truth anemometer and weather observations
 * from AviationWeather.gov for Indian international airports (VOHS, VIDP, VABB, VOBL, VECC, VOMM).
 * Free, public, zero-auth endpoint.
 */
import { useEffect, useState, useCallback, useRef } from "react";

const METAR_URL = "https://aviationweather.gov/api/data/metar?ids=VIDP,VOHS,VABB,VOBL,VECC,VOMM&format=json";
const REFRESH_INTERVAL_MS = 2 * 60 * 1000; // 2 minutes

export type AirportMetar = {
  icaoId: string;
  name: string;
  lat: number;
  lon: number;
  elevationMeters: number;
  reportTime: string;
  temp: number;
  dewp: number;
  windDirection: number; // degrees
  windSpeedKt: number;  // knots
  windSpeedKm: number;  // km/h
  windGustKt: number | null;
  windGustKm: number;   // km/h
  altimeter: number;    // hPa / QNH
  visibilityMiles: string | number;
  wxString: string;
  flightCategory: string;
  rawOb: string;
  isShearAlert: boolean; // Wind gust delta >= 15 kt or gusts >= 50 km/h
};

export function useAirportMetar() {
  const [airports, setAirports] = useState<Record<string, AirportMetar>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);

  const fetchMetar = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(METAR_URL, { signal: controller.signal });
      if (!response.ok) throw new Error(`AviationWeather API returned ${response.status}`);

      const data = await response.json();
      if (!Array.isArray(data)) throw new Error("Invalid METAR response array");

      const map: Record<string, AirportMetar> = {};

      data.forEach((item: any) => {
        const wspd = Number(item.wspd ?? 0);
        const wgst = item.wgst != null ? Number(item.wgst) : null;
        const windSpeedKm = Math.round(wspd * 1.852);
        const windGustKm = wgst != null ? Math.round(wgst * 1.852) : windSpeedKm;
        const isShearAlert = (wgst != null && wgst - wspd >= 15) || windGustKm >= 50;

        map[item.icaoId] = {
          icaoId: item.icaoId,
          name: item.name ?? item.icaoId,
          lat: Number(item.lat ?? 0),
          lon: Number(item.lon ?? 0),
          elevationMeters: Number(item.elev ?? 0),
          reportTime: item.reportTime ?? new Date().toISOString(),
          temp: Number(item.temp ?? 0),
          dewp: Number(item.dewp ?? 0),
          windDirection: Number(item.wdir ?? 0),
          windSpeedKt: wspd,
          windSpeedKm,
          windGustKt: wgst,
          windGustKm,
          altimeter: Number(item.altim ?? 1013),
          visibilityMiles: item.visib ?? 10,
          wxString: item.wxString ?? "",
          flightCategory: item.fltCat ?? "VFR",
          rawOb: item.rawOb ?? "",
          isShearAlert,
        };
      });

      setAirports(map);
      setLastUpdated(new Date().toISOString());
      setLoading(false);
      setError(null);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "METAR feed unavailable");
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetar();
    const interval = setInterval(fetchMetar, REFRESH_INTERVAL_MS);
    return () => {
      abortRef.current?.abort();
      clearInterval(interval);
    };
  }, [fetchMetar]);

  return { airports, loading, error, lastUpdated, refetch: fetchMetar };
}
