import { useEffect, useRef, useState, useMemo } from "react";
import { useWeatherForecast, type ForecastData } from "@/hooks/useWeatherForecast";

type WeatherGridDotsProps = {
  map: any;
  centerLat: number;
  centerLon: number;
  gridResolutionKm: number;
  onSelectDot: (data: ForecastData | null, lat: number, lon: number) => void;
};

// Earth radius in km
const R = 6371;

function getLatLonOffset(lat: number, lon: number, dxKm: number, dyKm: number) {
  const dLat = (dyKm / R) * (180 / Math.PI);
  const dLon = (dxKm / R) * (180 / Math.PI) / Math.cos((lat * Math.PI) / 180);
  return { lat: lat + dLat, lon: lon + dLon };
}

function getDotColor(severity: string | undefined, loading: boolean) {
  if (loading) return "#38BDF8"; // Cyan (loading)
  switch (severity) {
    case "severe": return "#EF4444"; // Red
    case "moderate": return "#F97316"; // Orange
    case "mild": return "#EAB308"; // Yellow
    case "clear":
    default: return "#A855F7"; // Purple (matching screenshot)
  }
}

// Single Grid Dot Component
function GridDot({ 
  map, lat, lon, id, onSelect 
}: { 
  map: any; lat: number; lon: number; id: string;
  onSelect: (data: ForecastData | null, lat: number, lon: number) => void;
}) {
  const { data, loading } = useWeatherForecast(lat, lon);
  const markerRef = useRef<any>(null);
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    if (!map || !window.mappls) return;

    const color = getDotColor(data?.severity, loading);
    const size = hovered ? 18 : 12;
    
    const html = `
      <div style="
        width: ${size}px; 
        height: ${size}px; 
        background: ${color}; 
        border-radius: 50%; 
        border: 2px solid rgba(255,255,255,0.8);
        box-shadow: 0 0 ${hovered ? '15px' : '8px'} ${color};
        transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        opacity: 0.85;
      "></div>
    `;

    if (!markerRef.current) {
      markerRef.current = new window.mappls.Marker({
        map: map,
        position: { lat, lng: lon },
        html: html,
        width: size,
        height: size,
        offset: [0, 0]
      });

      const el = markerRef.current.getElement();
      if (el) {
        el.addEventListener("mouseenter", () => {
          setHovered(true);
        });
        el.addEventListener("mouseleave", () => {
          setHovered(false);
        });
        el.addEventListener("click", () => {
          onSelect(data, lat, lon);
        });
      }
    } else {
      // Update existing marker
      markerRef.current.setHtml(html);
      markerRef.current.setWidth(size);
      markerRef.current.setHeight(size);
    }
  }, [map, lat, lon, data, loading, hovered, onSelect]);

  useEffect(() => {
    return () => {
      if (markerRef.current) {
        try {
          markerRef.current.remove();
        } catch (e) {}
      }
    };
  }, []);

  return null; // Rendered via Mappls imperative API
}

export function WeatherGridDots({
  map,
  centerLat,
  centerLon,
  gridResolutionKm,
  onSelectDot
}: WeatherGridDotsProps) {
  
  // Generate a grid of points (e.g. 7x7 grid around center)
  const points = useMemo(() => {
    const pts = [];
    const steps = 3; // 3 steps each direction = 7x7 grid
    
    for (let x = -steps; x <= steps; x++) {
      for (let y = -steps; y <= steps; y++) {
        // Create a circular pattern by skipping corners
        const distance = Math.sqrt(x*x + y*y);
        if (distance > steps + 0.5) continue;
        
        const dx = x * gridResolutionKm;
        const dy = y * gridResolutionKm;
        const coords = getLatLonOffset(centerLat, centerLon, dx, dy);
        pts.push({
          id: `dot-${x}-${y}`,
          lat: coords.lat,
          lon: coords.lon
        });
      }
    }
    return pts;
  }, [centerLat, centerLon, gridResolutionKm]);

  if (!map) return null;

  return (
    <>
      {points.map((pt) => (
        <GridDot 
          key={pt.id} 
          id={pt.id}
          map={map} 
          lat={pt.lat} 
          lon={pt.lon} 
          onSelect={onSelectDot}
        />
      ))}
    </>
  );
}
