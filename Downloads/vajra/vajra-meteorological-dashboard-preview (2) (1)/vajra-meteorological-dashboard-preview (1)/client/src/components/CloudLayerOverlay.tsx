import React, { useEffect, useRef } from "react";

export type CloudCellData = {
  id: number;
  lat: number;
  lon: number;
  size: number;
  intensity: number;
  reflectivityDbz: number;
  driftVx: number;
  driftVy: number;
  temperatureC?: number;
  relativeHumidity?: number;
  lclCloudBaseMeters?: number;
  cloudCoveragePercent?: number;
};

interface CloudLayerOverlayProps {
  map: any; // google.maps.Map | L.Map | null;
  engine: "google" | "leaflet";
  cells: CloudCellData[];
  leadMinutes?: number;
  enabled?: boolean;
}

interface CloudPuff {
  offsetX: number;
  offsetY: number;
  radiusFactor: number;
  opacityFactor: number;
  driftSpeedRatio: number;
  swirlAngle: number;
}

// Generate deterministic cluster of cloud puffs per cell
function getCellPuffs(cellId: number): CloudPuff[] {
  const puffs: CloudPuff[] = [];
  const seed = cellId * 137.5;
  const count = 7 + (cellId % 5);

  for (let i = 0; i < count; i++) {
    const angle = (seed + i * 2.399) % (Math.PI * 2);
    const dist = (0.2 + 0.65 * ((seed * (i + 1) * 37) % 100) / 100);
    puffs.push({
      offsetX: Math.cos(angle) * dist,
      offsetY: Math.sin(angle) * dist,
      radiusFactor: 0.55 + 0.45 * Math.sin(seed + i),
      opacityFactor: 0.6 + 0.4 * Math.cos(seed * 2 + i),
      driftSpeedRatio: 0.85 + 0.3 * Math.sin(seed * 3 + i),
      swirlAngle: angle,
    });
  }
  return puffs;
}

export function CloudLayerOverlay({
  map,
  engine,
  cells,
  leadMinutes = 0,
  enabled = true,
}: CloudLayerOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animIdRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(performance.now());

  // Projection helper to convert LatLng to Canvas (x, y)
  const getPointFromLatLng = (lat: number, lon: number): { x: number; y: number } | null => {
    if (!map) return null;

    if (engine === "leaflet") {
      try {
        const pt = map.latLngToContainerPoint([lat, lon]);
        return { x: pt.x, y: pt.y };
      } catch {
        return null;
      }
    } else if (engine === "google" && window.google?.maps) {
      try {
        const projection = (map as any).__vajraOverlayProjection;
        if (projection) {
          const pt = projection.fromLatLngToDivPixel(new google.maps.LatLng(lat, lon));
          if (pt) return { x: pt.x, y: pt.y };
        }
        const bounds = map.getBounds();
        if (!bounds) return null;
        const ne = bounds.getNorthEast();
        const sw = bounds.getSouthWest();
        const canvas = canvasRef.current;
        if (!canvas) return null;

        const x = ((lon - sw.lng()) / (ne.lng() - sw.lng())) * canvas.width;
        const y = ((ne.lat() - lat) / (ne.lat() - sw.lat())) * canvas.height;
        return { x, y };
      } catch {
        return null;
      }
    }
    return null;
  };

  // Canvas Resizing
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    if (engine === "leaflet" && map) {
      map.on("resize", handleResize);
      map.on("zoom", handleResize);
      map.on("move", handleResize);
    } else if (engine === "google" && map) {
      map.addListener("bounds_changed", handleResize);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      if (engine === "leaflet" && map) {
        map.off("resize", handleResize);
        map.off("zoom", handleResize);
        map.off("move", handleResize);
      }
    };
  }, [engine, map]);

  // Main Render Animation Loop (Continuous Drift & Morphing)
  useEffect(() => {
    if (!enabled) {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const animate = () => {
      const now = performance.now();
      const elapsedSec = (now - startTimeRef.current) * 0.001;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      cells.forEach((cell) => {
        // Physical cloud scaling based on Relative Humidity and LCL
        const rh = cell.relativeHumidity ?? 72;
        const lcl = cell.lclCloudBaseMeters ?? 1200;
        const dbz = cell.reflectivityDbz ?? 35;

        // Higher humidity + lower LCL => denser, larger, darker cumulonimbus
        const rhMultiplier = Math.max(0.4, Math.min(1.8, rh / 65));
        const lclFactor = Math.max(0.7, Math.min(1.4, (3500 - lcl) / 2000));
        const storminess = Math.min(1.0, Math.max(0.2, (dbz - 20) / 45));

        // Projected center for future nowcast frames (+leadMinutes)
        const leadHours = leadMinutes / 60;
        const projLat = cell.lat + (cell.driftVy * leadHours * 0.52);
        const projLon = cell.lon + (cell.driftVx * leadHours * 0.52);

        const centerPt = getPointFromLatLng(projLat, projLon);
        if (!centerPt) return;

        // Approximate pixel radius based on cell size and humidity
        const baseRadiusPx = Math.max(35, Math.min(190, (cell.size || 10) * 11 * rhMultiplier * lclFactor));
        const puffs = getCellPuffs(cell.id);

        // Real-time continuous advection drift displacement (simulating internal circulation)
        const driftAngle = Math.atan2(cell.driftVy, cell.driftVx);
        const speed = Math.hypot(cell.driftVx, cell.driftVy);
        const internalDriftDist = (elapsedSec * speed * 8) % baseRadiusPx;

        ctx.save();
        ctx.globalCompositeOperation = "source-over";

        // ── 1. Base Cirrus / Anvil Outflow Shield (Translucent Outer Glow) ──
        const anvilRadius = baseRadiusPx * 1.55;
        const anvilGrad = ctx.createRadialGradient(
          centerPt.x,
          centerPt.y,
          baseRadiusPx * 0.2,
          centerPt.x,
          centerPt.y,
          anvilRadius
        );
        const anvilOpacity = Math.min(0.38, 0.15 * rhMultiplier + storminess * 0.15);
        anvilGrad.addColorStop(0, `rgba(148, 163, 184, ${anvilOpacity.toFixed(3)})`);
        anvilGrad.addColorStop(0.55, `rgba(71, 85, 105, ${(anvilOpacity * 0.65).toFixed(3)})`);
        anvilGrad.addColorStop(1, "rgba(15, 23, 42, 0)");

        ctx.fillStyle = anvilGrad;
        ctx.beginPath();
        ctx.arc(centerPt.x, centerPt.y, anvilRadius, 0, Math.PI * 2);
        ctx.fill();

        // ── 2. Multi-Layer Volumetric Billowing Cloud Puffs ──
        puffs.forEach((puff, idx) => {
          // Slow continuous morphing and circulation
          const morphPhase = elapsedSec * 0.4 + puff.swirlAngle;
          const puffWobbleX = Math.cos(morphPhase) * (baseRadiusPx * 0.12);
          const puffWobbleY = Math.sin(morphPhase * 0.8) * (baseRadiusPx * 0.12);

          const driftOffX = Math.cos(driftAngle) * (internalDriftDist * puff.driftSpeedRatio);
          const driftOffY = Math.sin(driftAngle) * (internalDriftDist * puff.driftSpeedRatio);

          const pX = centerPt.x + puff.offsetX * (baseRadiusPx * 0.75) + puffWobbleX + driftOffX;
          const pY = centerPt.y + puff.offsetY * (baseRadiusPx * 0.75) + puffWobbleY + driftOffY;
          const pRadius = baseRadiusPx * puff.radiusFactor * 0.82;

          const pGrad = ctx.createRadialGradient(pX, pY, pRadius * 0.15, pX, pY, pRadius);

          // Color shifts from stormy dark slate (severe core) to silvery-gray
          const coreDarkness = Math.min(0.85, 0.45 + storminess * 0.38);
          const puffAlpha = Math.min(0.72, (0.35 + puff.opacityFactor * 0.32) * rhMultiplier);

          if (storminess > 0.5) {
            // Cumulonimbus dark core
            pGrad.addColorStop(0, `rgba(15, 23, 42, ${(puffAlpha * coreDarkness).toFixed(3)})`);
            pGrad.addColorStop(0.45, `rgba(30, 41, 59, ${(puffAlpha * 0.85).toFixed(3)})`);
            pGrad.addColorStop(0.85, `rgba(71, 85, 105, ${(puffAlpha * 0.35).toFixed(3)})`);
            pGrad.addColorStop(1, "rgba(148, 163, 184, 0)");
          } else {
            // Fair weather / developing cumulus
            pGrad.addColorStop(0, `rgba(51, 65, 85, ${(puffAlpha * 0.8).toFixed(3)})`);
            pGrad.addColorStop(0.5, `rgba(100, 116, 139, ${(puffAlpha * 0.55).toFixed(3)})`);
            pGrad.addColorStop(1, "rgba(203, 213, 225, 0)");
          }

          ctx.fillStyle = pGrad;
          ctx.beginPath();
          ctx.arc(pX, pY, pRadius, 0, Math.PI * 2);
          ctx.fill();
        });

        // ── 3. High-Density Convective Core Underbelly ──
        if (dbz >= 38) {
          const coreRadius = baseRadiusPx * 0.48;
          const coreGrad = ctx.createRadialGradient(
            centerPt.x,
            centerPt.y,
            coreRadius * 0.1,
            centerPt.x,
            centerPt.y,
            coreRadius
          );
          const coreAlpha = Math.min(0.82, 0.4 + (dbz / 70) * 0.4);
          coreGrad.addColorStop(0, `rgba(2, 6, 23, ${coreAlpha.toFixed(3)})`);
          coreGrad.addColorStop(0.6, `rgba(15, 23, 42, ${(coreAlpha * 0.7).toFixed(3)})`);
          coreGrad.addColorStop(1, "rgba(30, 41, 59, 0)");

          ctx.fillStyle = coreGrad;
          ctx.beginPath();
          ctx.arc(centerPt.x, centerPt.y, coreRadius, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.restore();
      });

      animIdRef.current = requestAnimationFrame(animate);
    };

    animIdRef.current = requestAnimationFrame(animate);

    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
    };
  }, [map, engine, cells, leadMinutes, enabled]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 380,
      }}
    />
  );
}
