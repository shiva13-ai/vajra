import React, { useEffect, useRef } from "react";

export type LightningCellData = {
  id: number;
  lat: number;
  lon: number;
  size: number;
  intensity: number;
  reflectivityDbz: number;
  lightningRatePerMin: number;
  lightningProbability: number;
  temperatureC?: number;
  relativeHumidity?: number;
};

interface LightningCanvasOverlayProps {
  map: any; // google.maps.Map | L.Map | null
  engine: "google" | "leaflet";
  cells: LightningCellData[];
  enabled?: boolean;
}

interface ActiveBolt {
  id: string;
  cellId: number;
  startX: number;
  startY: number;
  segments: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[];
  branches: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[][];
  flashRadius: number;
  createdAt: number;
  durationMs: number;
  flashAlpha: number;
}

/**
 * Generates a procedural fractal lightning bolt with random bifurcations
 */
function createBolt(startX: number, startY: number, lengthPx: number, angleRad: number): {
  segments: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[];
  branches: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[][];
} {
  const segments: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[] = [];
  const branches: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[][] = [];

  const numSteps = 8 + Math.floor(Math.random() * 6);
  const stepDist = lengthPx / numSteps;

  let currX = startX;
  let currY = startY;
  let currAngle = angleRad;

  for (let i = 0; i < numSteps; i++) {
    // Angular deviation
    currAngle += (Math.random() - 0.5) * 0.75;
    const nextX = currX + Math.cos(currAngle) * stepDist * (0.8 + Math.random() * 0.4);
    const nextY = currY + Math.sin(currAngle) * stepDist * (0.8 + Math.random() * 0.4);

    const width = Math.max(1.2, 3.2 * (1 - i / numSteps));
    segments.push({
      x1: currX,
      y1: currY,
      x2: nextX,
      y2: nextY,
      width,
      alpha: 1.0,
    });

    // 35% chance to sprout a secondary branch
    if (Math.random() < 0.38 && i < numSteps - 2) {
      const branch: { x1: number; y1: number; x2: number; y2: number; width: number; alpha: number }[] = [];
      let bX = nextX;
      let bY = nextY;
      let bAngle = currAngle + (Math.random() < 0.5 ? 1 : -1) * (0.5 + Math.random() * 0.5);
      const bSteps = 3 + Math.floor(Math.random() * 4);
      const bDist = stepDist * 0.65;

      for (let j = 0; j < bSteps; j++) {
        bAngle += (Math.random() - 0.5) * 0.6;
        const bNextX = bX + Math.cos(bAngle) * bDist;
        const bNextY = bY + Math.sin(bAngle) * bDist;
        branch.push({
          x1: bX,
          y1: bY,
          x2: bNextX,
          y2: bNextY,
          width: Math.max(0.8, (width * 0.6) * (1 - j / bSteps)),
          alpha: 0.85,
        });
        bX = bNextX;
        bY = bNextY;
      }
      branches.push(branch);
    }

    currX = nextX;
    currY = nextY;
  }

  return { segments, branches };
}

export function LightningCanvasOverlay({
  map,
  engine,
  cells,
  enabled = true,
}: LightningCanvasOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const activeBoltsRef = useRef<ActiveBolt[]>([]);
  const animIdRef = useRef<number | null>(null);
  const nextStrikeTimeRef = useRef<Map<number, number>>(new Map());

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
        // Google Maps OverlayView projection
        const projection = (map as any).__vajraOverlayProjection;
        if (projection) {
          const pt = projection.fromLatLngToDivPixel(new google.maps.LatLng(lat, lon));
          if (pt) return { x: pt.x, y: pt.y };
        }
        // Fallback using map bounds
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

  // Google Maps OverlayView registration to hook projection
  useEffect(() => {
    if (engine !== "google" || !map || !window.google?.maps) return;

    const overlay = new google.maps.OverlayView();
    overlay.onAdd = () => {};
    overlay.draw = function () {
      (map as any).__vajraOverlayProjection = this.getProjection();
    };
    overlay.onRemove = () => {
      (map as any).__vajraOverlayProjection = null;
    };
    overlay.setMap(map);

    return () => {
      overlay.setMap(null);
    };
  }, [engine, map]);

  // Handle Canvas Resizing
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

  // Main Animation and Strike Scheduling Loop
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
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Strike trigger check for each active storm cell
      cells.forEach((cell) => {
        const rate = cell.lightningRatePerMin ?? 0;
        if (rate <= 0) return;

        // Base interval inversely proportional to rate (e.g., 20 strikes/min => average 3000ms)
        const avgIntervalMs = Math.max(700, (60000 / rate) * (0.6 + Math.random() * 0.8));
        const nextTime = nextStrikeTimeRef.current.get(cell.id) ?? 0;

        if (now >= nextTime) {
          nextStrikeTimeRef.current.set(cell.id, now + avgIntervalMs);

          const pt = getPointFromLatLng(cell.lat, cell.lon);
          if (!pt) return;

          // Strike geometry
          const lengthPx = Math.max(50, Math.min(180, (cell.size || 10) * 12));
          // Scatter start point within storm cell
          const offsetRadius = Math.random() * (cell.size * 3);
          const offsetAngle = Math.random() * Math.PI * 2;
          const sX = pt.x + Math.cos(offsetAngle) * offsetRadius;
          const sY = pt.y - lengthPx * 0.5 + Math.sin(offsetAngle) * (offsetRadius * 0.4);

          // Downward or diagonal angle
          const boltAngle = Math.PI * 0.5 + (Math.random() - 0.5) * 0.6;
          const { segments, branches } = createBolt(sX, sY, lengthPx, boltAngle);

          const flashRadius = Math.max(60, Math.min(220, (cell.size || 10) * 16));

          activeBoltsRef.current.push({
            id: `bolt-${cell.id}-${now}`,
            cellId: cell.id,
            startX: pt.x,
            startY: pt.y,
            segments,
            branches,
            flashRadius,
            createdAt: now,
            durationMs: 160 + Math.random() * 100,
            flashAlpha: Math.min(0.75, 0.35 + (cell.reflectivityDbz / 70) * 0.4),
          });
        }
      });

      // 2. Render and age active lightning bolts
      activeBoltsRef.current = activeBoltsRef.current.filter((bolt) => {
        const age = now - bolt.createdAt;
        if (age >= bolt.durationMs) return false;

        const progress = age / bolt.durationMs;

        // Double-strobe flash curve: rapid spike, dip, second peak, exponential fade
        let intensity = 1.0;
        if (progress < 0.2) {
          intensity = progress / 0.2;
        } else if (progress < 0.4) {
          intensity = 0.5 + Math.random() * 0.5; // flicker
        } else {
          intensity = Math.pow(1 - progress, 2);
        }

        ctx.save();

        // ── A. Ambient Illuminated Flash Glow (Cloud illumination) ──
        const flashGrad = ctx.createRadialGradient(
          bolt.startX,
          bolt.startY,
          5,
          bolt.startX,
          bolt.startY,
          bolt.flashRadius
        );
        const flashAlpha = bolt.flashAlpha * intensity;
        flashGrad.addColorStop(0, `rgba(255, 255, 255, ${flashAlpha.toFixed(3)})`);
        flashGrad.addColorStop(0.25, `rgba(56, 189, 248, ${(flashAlpha * 0.75).toFixed(3)})`);
        flashGrad.addColorStop(0.65, `rgba(14, 165, 233, ${(flashAlpha * 0.35).toFixed(3)})`);
        flashGrad.addColorStop(1, "rgba(2, 6, 23, 0)");

        ctx.fillStyle = flashGrad;
        ctx.beginPath();
        ctx.arc(bolt.startX, bolt.startY, bolt.flashRadius, 0, Math.PI * 2);
        ctx.fill();

        // ── B. Outer Cyan Glow Channel ──
        ctx.strokeStyle = `rgba(56, 189, 248, ${(0.85 * intensity).toFixed(3)})`;
        ctx.lineCap = "round";
        ctx.lineJoin = "miter";
        ctx.shadowColor = "#00F2FE";
        ctx.shadowBlur = 14;

        bolt.segments.forEach((seg) => {
          ctx.lineWidth = seg.width * 2.4;
          ctx.beginPath();
          ctx.moveTo(seg.x1, seg.y1);
          ctx.lineTo(seg.x2, seg.y2);
          ctx.stroke();
        });

        bolt.branches.forEach((branch) => {
          branch.forEach((bSeg) => {
            ctx.lineWidth = bSeg.width * 2.0;
            ctx.beginPath();
            ctx.moveTo(bSeg.x1, bSeg.y1);
            ctx.lineTo(bSeg.x2, bSeg.y2);
            ctx.stroke();
          });
        });

        // ── C. Hot Pure White Discharge Core ──
        ctx.strokeStyle = `rgba(255, 255, 255, ${(0.95 * intensity).toFixed(3)})`;
        ctx.shadowColor = "#FFFFFF";
        ctx.shadowBlur = 6;

        bolt.segments.forEach((seg) => {
          ctx.lineWidth = seg.width;
          ctx.beginPath();
          ctx.moveTo(seg.x1, seg.y1);
          ctx.lineTo(seg.x2, seg.y2);
          ctx.stroke();
        });

        bolt.branches.forEach((branch) => {
          branch.forEach((bSeg) => {
            ctx.lineWidth = bSeg.width * 0.75;
            ctx.beginPath();
            ctx.moveTo(bSeg.x1, bSeg.y1);
            ctx.lineTo(bSeg.x2, bSeg.y2);
            ctx.stroke();
          });
        });

        ctx.restore();
        return true;
      });

      animIdRef.current = requestAnimationFrame(animate);
    };

    animIdRef.current = requestAnimationFrame(animate);

    return () => {
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
    };
  }, [map, engine, cells, enabled]);

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
        zIndex: 420,
      }}
    />
  );
}
