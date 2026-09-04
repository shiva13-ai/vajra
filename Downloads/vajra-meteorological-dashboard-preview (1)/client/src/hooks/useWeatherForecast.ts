/**
 * useWeatherForecast — fetches current weather + 6-hour hourly forecast
 * from the Open-Meteo Forecast API for a given lat/lon.
 * Results are cached for 5 minutes to avoid redundant calls.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const FORECAST_URL = "https://api.open-meteo.com/v1/forecast";
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

export type CurrentWeather = {
  temperature: number;       // °C
  precipitation: number;     // mm
  windSpeed: number;         // km/h
  windGusts: number;         // km/h
  humidity: number;          // %
  cloudCover: number;        // %
  weatherCode: number;
  description: string;
  time: string;
};

export type HourlyForecast = {
  time: string;
  temperature: number;
  precipitation: number;
  windSpeed: number;
  windGusts: number;
  humidity: number;
  cloudCover: number;
  weatherCode: number;
  description: string;
  precipitationProbability: number;
};

export type ForecastData = {
  current: CurrentWeather | null;
  forecast: HourlyForecast[];
  severity: "clear" | "mild" | "moderate" | "severe";
};

type CacheEntry = {
  data: ForecastData;
  timestamp: number;
};

// WMO Weather interpretation codes → description
function weatherCodeToDescription(code: number): string {
  const map: Record<number, string> = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
  };
  return map[code] ?? "Unknown";
}

// Determine severity from weather code + metrics
function computeSeverity(
  code: number,
  precipitation: number,
  windGusts: number,
  forecasts: HourlyForecast[],
): ForecastData["severity"] {
  // Check forecasts for any severe upcoming conditions
  const maxForecastCode = Math.max(code, ...forecasts.map((f) => f.weatherCode));
  const maxPrecip = Math.max(precipitation, ...forecasts.map((f) => f.precipitation));
  const maxWind = Math.max(windGusts, ...forecasts.map((f) => f.windGusts));

  if (maxForecastCode >= 95 || maxPrecip >= 20 || maxWind >= 80) return "severe";
  if (maxForecastCode >= 61 || maxPrecip >= 5 || maxWind >= 50) return "moderate";
  if (maxForecastCode >= 51 || maxPrecip >= 1 || maxWind >= 30) return "mild";
  return "clear";
}

// Global cache shared across hook instances
const cache = new Map<string, CacheEntry>();

function cacheKey(lat: number, lon: number): string {
  // Round to ~1 km precision to allow nearby points to share cache
  return `${lat.toFixed(2)}_${lon.toFixed(2)}`;
}

export function useWeatherForecast(lat: number | null, lon: number | null) {
  const [data, setData] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchForecast = useCallback(async (latitude: number, longitude: number) => {
    // Check cache first
    const key = cacheKey(latitude, longitude);
    const cached = cache.get(key);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
      setData(cached.data);
      setLoading(false);
      setError(null);
      return;
    }

    // Abort any ongoing request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const url =
        `${FORECAST_URL}?latitude=${latitude.toFixed(4)}` +
        `&longitude=${longitude.toFixed(4)}` +
        `&current=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,relative_humidity_2m,cloud_cover,weather_code` +
        `&hourly=temperature_2m,precipitation,precipitation_probability,wind_speed_10m,wind_gusts_10m,relative_humidity_2m,cloud_cover,weather_code` +
        `&forecast_hours=7&timezone=auto`;

      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`Weather API returned ${response.status}`);

      const payload = await response.json();

      const current: CurrentWeather = {
        temperature: Number(payload.current?.temperature_2m ?? 0),
        precipitation: Number(payload.current?.precipitation ?? 0),
        windSpeed: Number(payload.current?.wind_speed_10m ?? 0),
        windGusts: Number(payload.current?.wind_gusts_10m ?? 0),
        humidity: Number(payload.current?.relative_humidity_2m ?? 0),
        cloudCover: Number(payload.current?.cloud_cover ?? 0),
        weatherCode: Number(payload.current?.weather_code ?? 0),
        description: weatherCodeToDescription(Number(payload.current?.weather_code ?? 0)),
        time: String(payload.current?.time ?? ""),
      };

      const hourlyTimes: string[] = payload.hourly?.time ?? [];
      const forecast: HourlyForecast[] = hourlyTimes.slice(1, 7).map((time: string, i: number) => {
        const idx = i + 1; // skip current hour
        return {
          time,
          temperature: Number(payload.hourly.temperature_2m?.[idx] ?? 0),
          precipitation: Number(payload.hourly.precipitation?.[idx] ?? 0),
          windSpeed: Number(payload.hourly.wind_speed_10m?.[idx] ?? 0),
          windGusts: Number(payload.hourly.wind_gusts_10m?.[idx] ?? 0),
          humidity: Number(payload.hourly.relative_humidity_2m?.[idx] ?? 0),
          cloudCover: Number(payload.hourly.cloud_cover?.[idx] ?? 0),
          weatherCode: Number(payload.hourly.weather_code?.[idx] ?? 0),
          description: weatherCodeToDescription(Number(payload.hourly.weather_code?.[idx] ?? 0)),
          precipitationProbability: Number(payload.hourly.precipitation_probability?.[idx] ?? 0),
        };
      });

      const severity = computeSeverity(current.weatherCode, current.precipitation, current.windGusts, forecast);

      const result: ForecastData = { current, forecast, severity };

      // Store in cache
      cache.set(key, { data: result, timestamp: Date.now() });

      setData(result);
      setLoading(false);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Forecast unavailable");
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (lat != null && lon != null) {
      fetchForecast(lat, lon);
    }
    return () => {
      abortRef.current?.abort();
    };
  }, [lat, lon, fetchForecast]);

  return { data, loading, error, refetch: fetchForecast };
}
