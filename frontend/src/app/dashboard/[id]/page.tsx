"use client";

import { useEffect, useState, use, useCallback } from "react";

// Colores por slot — coinciden con los del tracker (test_tracker.py)
const SLOT = {
  "1": { label: "J1",  side: "near-izq", dot: "#60a5fa", ring: "ring-blue-500",   bg: "bg-blue-500/10",   text: "text-blue-400",   border: "border-blue-500/30" },
  "2": { label: "J2",  side: "near-der", dot: "#facc15", ring: "ring-yellow-500", bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/30" },
  "3": { label: "J3",  side: "far-izq",  dot: "#34d399", ring: "ring-emerald-500",bg: "bg-emerald-500/10",text: "text-emerald-400",border: "border-emerald-500/30" },
  "4": { label: "J4",  side: "far-der",  dot: "#e879f9", ring: "ring-fuchsia-500",bg: "bg-fuchsia-500/10",text: "text-fuchsia-400",border: "border-fuchsia-500/30" },
} as const;

type SlotKey = keyof typeof SLOT;

interface ShotEvent {
  id: string;
  video_id: string;
  frame: number;
  timestamp: number;
  player_id: number;
  shot_type: string;
  pos_x: number;
  pos_y: number;
}

interface PlayerStat {
  total_shots: number;
  smash_count: number;
  stroke_count: number;
}

interface BackendResult {
  video_id: string;
  filename?: string;
  status: string;
  total_frames: number;
  shots_count: number;
  player_stats: Record<string, PlayerStat>;
  events: ShotEvent[];
}

// ── Componente: tarjeta de jugador ───────────────────────────────────────────
function PlayerCard({ slotId, stats }: { slotId: SlotKey; stats: PlayerStat | undefined }) {
  const s = SLOT[slotId];
  const total  = stats?.total_shots  ?? 0;
  const smash  = stats?.smash_count  ?? 0;
  const stroke = stats?.stroke_count ?? 0;
  const smashPct = total > 0 ? Math.round((smash / total) * 100) : 0;

  return (
    <div className={`rounded-2xl border p-5 ${s.bg} ${s.border} flex flex-col gap-3`}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-full ring-2 ${s.ring} flex items-center justify-center font-black text-white text-sm`}
             style={{ background: s.dot + "33" }}>
          {s.label}
        </div>
        <div>
          <div className={`font-bold ${s.text}`}>{s.label}</div>
          <div className="text-xs text-slate-500">{s.side}</div>
        </div>
        <div className="ml-auto text-right">
          <div className={`text-2xl font-black ${s.text}`}>{total}</div>
          <div className="text-xs text-slate-500">golpes</div>
        </div>
      </div>

      {/* Barra smash vs stroke */}
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1">
          <span>Stroke <span className="text-slate-400">{stroke}</span></span>
          <span>Smash <span className="text-slate-400">{smash}</span></span>
        </div>
        <div className="h-2 rounded-full bg-slate-700/50 overflow-hidden flex">
          <div className="h-full bg-emerald-500/70 transition-all duration-700"
               style={{ width: `${100 - smashPct}%` }} />
          <div className="h-full bg-orange-400/70 transition-all duration-700"
               style={{ width: `${smashPct}%` }} />
        </div>
        <div className="flex justify-between text-xs text-slate-600 mt-0.5">
          <span>{100 - smashPct}%</span>
          <span>{smashPct}%</span>
        </div>
      </div>
    </div>
  );
}

// ── Componente: mapa de tiros SVG ─────────────────────────────────────────────
function ShotMap({ events, frameW, frameH }: {
  events: ShotEvent[];
  frameW: number;
  frameH: number;
}) {
  const W = 280, H = 420;  // Dimensiones del SVG del mapa
  // Normalizar coordenadas de frame al mapa
  const toX = (px: number) => (px / (frameW || 1920)) * W;
  const toY = (py: number) => (py / (frameH || 1080)) * H;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl bg-slate-800/60 border border-slate-700/40">
      {/* Fondo de pista */}
      <rect x="20" y="20" width={W - 40} height={H - 40} rx="4" fill="#1e3a5f" stroke="#334155" strokeWidth="1" />
      {/* Red */}
      <line x1="20" y1={H / 2} x2={W - 20} y2={H / 2} stroke="#94a3b8" strokeWidth="2" strokeDasharray="6 3" />
      <text x={W - 16} y={H / 2 + 4} fontSize="9" fill="#64748b" textAnchor="end">red</text>
      {/* Separación de zonas cámara/fondo */}
      <text x="4" y="30" fontSize="8" fill="#475569">fondo</text>
      <text x="4" y={H - 10} fontSize="8" fill="#475569">cámara</text>

      {/* Puntos de tiros */}
      {events.map((ev) => {
        const slot = String(ev.player_id) as SlotKey;
        const color = SLOT[slot]?.dot ?? "#94a3b8";
        return (
          <circle
            key={ev.id}
            cx={toX(ev.pos_x)}
            cy={toY(ev.pos_y)}
            r="5"
            fill={color}
            fillOpacity="0.75"
            stroke={color}
            strokeWidth="1"
          >
            <title>{`J${ev.player_id} · ${ev.shot_type} · t=${ev.timestamp.toFixed(1)}s`}</title>
          </circle>
        );
      })}
    </svg>
  );
}

// ── Componente: timeline de tiros ─────────────────────────────────────────────
function ShotTimeline({ events }: { events: ShotEvent[] }) {
  const recent = [...events].sort((a, b) => b.frame - a.frame).slice(0, 15);

  return (
    <div className="flex flex-col gap-1.5 overflow-y-auto max-h-[420px] pr-1">
      {recent.length === 0 && (
        <p className="text-slate-600 text-sm text-center py-8">Sin tiros detectados aún</p>
      )}
      {recent.map((ev) => {
        const slot = String(ev.player_id) as SlotKey;
        const s = SLOT[slot];
        return (
          <div key={ev.id}
               className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm border ${s.bg} ${s.border}`}>
            <div className="w-2 h-2 rounded-full shrink-0" style={{ background: s.dot }} />
            <span className={`font-bold w-6 shrink-0 ${s.text}`}>{s.label}</span>
            <span className="text-slate-400 grow">{ev.shot_type}</span>
            <span className="text-slate-600 font-mono text-xs">{ev.timestamp.toFixed(1)}s</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Inferir zona real de cada jugador a partir de sus tiros ───────────────────
// Usa la mediana de pos_y: > 55% del frame = cámara (near), < 55% = fondo (far).
// Así el layout se reordena automáticamente cuando las parejas cambian de lado.
function deriveZones(events: ShotEvent[], frameH: number): Record<string, "near" | "far"> {
  const yBySlot: Record<string, number[]> = {};
  for (const ev of events) {
    const k = String(ev.player_id);
    (yBySlot[k] ??= []).push(ev.pos_y);
  }
  const zones: Record<string, "near" | "far"> = {};
  for (const [k, ys] of Object.entries(yBySlot)) {
    if (ys.length === 0) continue;
    const sorted = [...ys].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    zones[k] = median > frameH * 0.55 ? "near" : "far";
  }
  // Fallback para slots sin tiros: convención inicial (1,2=near / 3,4=far)
  for (const k of ["1", "2", "3", "4"]) {
    if (!(k in zones)) zones[k] = k === "1" || k === "2" ? "near" : "far";
  }
  return zones;
}

// ── Página principal ───────────────────────────────────────────────────────────
export default function DashboardPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [data, setData]       = useState<BackendResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);

  const fetchResults = useCallback(async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/results/${id}`);
      if (!res.ok) throw new Error(`Error ${res.status}`);
      const json: BackendResult = await res.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error de conexión");
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    fetchResults();
    const interval = setInterval(() => {
      // Dejar de hacer polling cuando el análisis esté completo o haya fallado
      if (data?.status === "completed" || data?.status === "error") return;
      fetchResults();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchResults, data?.status]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-white">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-sky-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400">Cargando análisis…</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-white p-8 text-center">
        <div>
          <h2 className="text-2xl font-bold text-red-400 mb-2">Error</h2>
          <p className="text-slate-400">{error ?? "Video no encontrado"}</p>
          <a href="/" className="mt-6 inline-block text-sky-400 hover:underline text-sm">← Volver al inicio</a>
        </div>
      </div>
    );
  }

  const ps  = data.player_stats ?? {};
  const isProcessing = data.status !== "completed" && data.status !== "error";
  const frameW = 1920, frameH = 1080;  // Resolución por defecto (match.mov)

  // Zonas calculadas a partir de posiciones reales de los tiros
  const zones = deriveZones(data.events ?? [], frameH);
  const farSlots  = (["1","2","3","4"] as SlotKey[]).filter(k => zones[k] === "far");
  const nearSlots = (["1","2","3","4"] as SlotKey[]).filter(k => zones[k] === "near");

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6 md:p-10 flex flex-col gap-8 max-w-7xl mx-auto">

      {/* ── Header ── */}
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <a href="/" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">← Inicio</a>
          <h1 className="text-3xl font-black tracking-tight mt-1">MATCH ANALYSIS</h1>
          <p className="text-slate-500 text-sm mt-0.5 truncate max-w-xs">{data.filename ?? data.video_id}</p>
        </div>
        <div className="flex gap-3 flex-wrap">
          {/* Status */}
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-2 text-center">
            <div className="text-xs text-slate-500 uppercase font-semibold mb-0.5">Estado</div>
            <div className={`text-sm font-mono font-bold flex items-center gap-1.5
              ${data.status === "completed" ? "text-emerald-400" : isProcessing ? "text-sky-400" : "text-red-400"}`}>
              {isProcessing && <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse inline-block" />}
              {data.status.toUpperCase()}
            </div>
          </div>
          {/* Total shots */}
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-2 text-center">
            <div className="text-xs text-slate-500 uppercase font-semibold mb-0.5">Total golpes</div>
            <div className="text-2xl font-black text-white">{data.shots_count}</div>
          </div>
          {/* Frames */}
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-2 text-center">
            <div className="text-xs text-slate-500 uppercase font-semibold mb-0.5">Frames</div>
            <div className="text-2xl font-black text-white">{data.total_frames.toLocaleString()}</div>
          </div>
        </div>
      </header>

      {/* ── Grid de jugadores (layout de pista) + lateral ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Columna izquierda: jugadores dispuestos como en pista (zona inferida de tiros) */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <span>🎾 Fondo (lejos de cámara)</span>
          </div>

          {/* Jugadores del fondo — orden derivado de posición real */}
          <div className="grid grid-cols-2 gap-4">
            {farSlots.map(k => <PlayerCard key={k} slotId={k} stats={ps[k]} />)}
            {farSlots.length === 0 && <p className="col-span-2 text-xs text-slate-600 text-center py-4">Sin datos aún</p>}
          </div>

          {/* Separador = red */}
          <div className="flex items-center gap-3 py-1">
            <div className="flex-1 h-px bg-slate-700" />
            <span className="text-xs text-slate-600 uppercase tracking-widest px-2">Red</span>
            <div className="flex-1 h-px bg-slate-700" />
          </div>

          {/* Jugadores cercanos — orden derivado de posición real */}
          <div className="grid grid-cols-2 gap-4">
            {nearSlots.map(k => <PlayerCard key={k} slotId={k} stats={ps[k]} />)}
            {nearSlots.length === 0 && <p className="col-span-2 text-xs text-slate-600 text-center py-4">Sin datos aún</p>}
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-600">
            <span>📷 Cámara (lado cercano)</span>
          </div>
        </div>

        {/* Columna derecha: mapa + timeline */}
        <div className="flex flex-col gap-6">
          <div>
            <h2 className="text-xs font-semibold uppercase text-slate-500 mb-2 tracking-widest">Mapa de tiros</h2>
            <ShotMap events={data.events ?? []} frameW={frameW} frameH={frameH} />
          </div>
          <div>
            <h2 className="text-xs font-semibold uppercase text-slate-500 mb-2 tracking-widest">Últimos tiros</h2>
            <ShotTimeline events={data.events ?? []} />
          </div>
        </div>
      </div>

      <footer className="text-xs text-slate-700 text-center pt-4 border-t border-slate-800">
        Padel Stats © 2026 — Video ID: {data.video_id}
      </footer>
    </div>
  );
}
