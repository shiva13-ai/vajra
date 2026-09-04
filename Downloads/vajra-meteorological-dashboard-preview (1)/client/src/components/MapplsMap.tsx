/**
 * MapplsMap — Loads the Mappls (MapMyIndia) SDK and renders an interactive dark-themed map.
 * Exposes the map instance via onMapReady callback.
 */
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

declare global {
  interface Window {
    mappls: any;
    mapplsClassObject?: any;
  }
}

const MAPPLS_API_KEY = import.meta.env.VITE_MAPPLS_API_KEY || "";

type MapplsMapProps = {
  className?: string;
  center?: { lat: number; lng: number };
  zoom?: number;
  onMapReady?: (map: any) => void;
  children?: React.ReactNode;
};

let scriptLoadPromise: Promise<void> | null = null;

function loadMapplsScript(): Promise<void> {
  if (scriptLoadPromise) return scriptLoadPromise;
  if (window.mappls) return Promise.resolve();

  scriptLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://apis.mappls.com/advancedmaps/api/${MAPPLS_API_KEY}/map_sdk?v=3.0&layer=vector`;
    script.async = true;
    script.onload = () => {
      // Mappls SDK needs a brief moment to initialize
      const check = () => {
        if (window.mappls) {
          resolve();
        } else {
          setTimeout(check, 100);
        }
      };
      check();
    };
    script.onerror = () => {
      scriptLoadPromise = null;
      reject(new Error("Failed to load Mappls SDK"));
    };
    document.head.appendChild(script);
  });

  return scriptLoadPromise;
}

export function MapplsMap({
  className,
  center = { lat: 20.5937, lng: 78.9629 }, // Center of India
  zoom = 5,
  onMapReady,
}: MapplsMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const onMapReadyRef = useRef(onMapReady);
  onMapReadyRef.current = onMapReady;

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await loadMapplsScript();
        if (cancelled || !containerRef.current) return;

        // Initialize Mappls map
        const map = new window.mappls.Map(containerRef.current, {
          center: [center.lat, center.lng],
          zoom: zoom,
          zoomControl: true,
          search: false,
          location: false,
          // Dark styling
          clickableIcons: false,
          disableDoubleClickZoom: false,
          scrollWheel: true,
          traffic: false,
          geolocation: false,
          tilt: false,
        });

        mapRef.current = map;

        map.addListener("load", () => {
          if (cancelled) return;
          setStatus("ready");
          onMapReadyRef.current?.(map);
        });
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setErrorMsg(err instanceof Error ? err.message : "Map load failed");
      }
    }

    init();

    return () => {
      cancelled = true;
      if (mapRef.current) {
        try {
          mapRef.current.remove?.();
        } catch {
          /* ignore cleanup errors */
        }
        mapRef.current = null;
      }
    };
  }, []); // Only initialize once

  // Update center/zoom when props change
  useEffect(() => {
    if (mapRef.current && status === "ready") {
      try {
        mapRef.current.setCenter([center.lat, center.lng]);
        mapRef.current.setZoom(zoom);
      } catch {
        /* ignore if map not ready */
      }
    }
  }, [center.lat, center.lng, zoom, status]);

  return (
    <div className={cn("mappls-map-container", className)} style={{ position: "relative", width: "100%", height: "100%" }}>
      <div
        ref={containerRef}
        style={{ width: "100%", height: "100%", position: "absolute", inset: 0 }}
      />
      {status === "loading" && (
        <div className="mappls-loading-overlay" style={{
          position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(9,21,34,0.95)", zIndex: 10, color: "#38BDF8", fontFamily: "'IBM Plex Mono', monospace",
          fontSize: "12px", letterSpacing: "0.08em", textTransform: "uppercase",
        }}>
          <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#38BDF8", animation: "pulse 1.2s infinite" }} />
            INITIALIZING MAPPLS MAP ENGINE
          </span>
        </div>
      )}
      {status === "error" && (
        <div className="mappls-error-overlay" style={{
          position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
          flexDirection: "column", gap: "8px", background: "rgba(9,21,34,0.95)", zIndex: 10,
          color: "#EF4444", fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px",
          letterSpacing: "0.08em", textTransform: "uppercase",
        }}>
          <span>MAP ENGINE ERROR</span>
          <span style={{ color: "#64748b", fontSize: "10px" }}>{errorMsg || "Check VITE_MAPPLS_API_KEY in .env"}</span>
        </div>
      )}
    </div>
  );
}
