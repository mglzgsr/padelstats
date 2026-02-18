"use client";

import { useEffect, useState, use } from "react";
import { StatsCard } from "@/components/StatsCard";
import { Player, PlayerMatchStats } from "@/types";

interface BackendResult {
    id: string;
    filename: string;
    status: string;
    total_frames: number;
    shots_count: number;
    player_stats: Record<string, {
        total_shots: number;
        smash_count: number;
        stroke_count: number;
    }>;
}

export default function DashboardPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = use(params);
    const [data, setData] = useState<BackendResult | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:8000/results/${id}`);
                if (!response.ok) throw new Error("Failed to fetch results");
                const json = await response.ok ? await response.json() : null;
                setData(json);
            } catch (err) {
                setError(err instanceof Error ? err.message : "Connection error");
            } finally {
                setLoading(false);
            }
        };

        fetchResults();
        const interval = setInterval(fetchResults, 5000); // Poll every 5s
        return () => clearInterval(interval);
    }, [id]);

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white">
                <div className="animate-pulse flex flex-col items-center">
                    <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
                    <p className="text-gray-400">Loading analysis data...</p>
                </div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white p-4 text-center">
                <div>
                    <h2 className="text-2xl font-bold text-danger mb-2">Error</h2>
                    <p className="text-gray-400">{error || "Video not found"}</p>
                </div>
            </div>
        );
    }

    // Map backend stats to frontend StatsCard format
    const players: PlayerMatchStats[] = Object.entries(data.player_stats).map(([pid, stats]) => ({
        player_id: pid,
        match_id: data.id,
        stats: {
            forehand_winners: stats.stroke_count, // Fallback mapping
            forehand_errors: 0,
            backhand_winners: 0,
            backhand_errors: 0,
            volley_winners: 0,
            volley_errors: 0,
            smash_winners: stats.smash_count,
            smash_errors: 0,
            total_points_won: stats.total_shots,
            total_points_lost: 0,
            unforced_errors: 0
        }
    }));

    return (
        <div className="min-h-screen p-8 bg-slate-950 font-[family-name:var(--font-geist-sans)] flex flex-col items-center">
            <header className="mb-12 text-center w-full max-w-6xl flex justify-between items-center">
                <div className="text-left">
                    <h1 className="text-3xl font-black text-white mb-2 tracking-tight">
                        MATCH ANALYSIS
                    </h1>
                    <p className="text-gray-400 text-sm">Video: {data.filename}</p>
                </div>
                <div className="flex gap-4">
                    <div className="bg-slate-800/50 px-4 py-2 rounded-xl border border-slate-700/50 text-center">
                        <div className="text-xs text-gray-500 uppercase font-bold">Status</div>
                        <div className={`text-sm font-mono ${data.status === 'completed' ? 'text-secondary' : 'text-primary'}`}>
                            {data.status.toUpperCase()}
                        </div>
                    </div>
                    <div className="bg-slate-800/50 px-4 py-2 rounded-xl border border-slate-700/50 text-center">
                        <div className="text-xs text-gray-500 uppercase font-bold">Total Shots</div>
                        <div className="text-xl font-black text-white">{data.shots_count}</div>
                    </div>
                </div>
            </header>

            <main className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-7xl">
                {players.map((p) => (
                    <StatsCard
                        key={p.player_id}
                        player={{ id: p.player_id, name: `Player ${p.player_id}`, ranking: 0 }}
                        matchStats={p}
                    />
                ))}
                {players.length === 0 && (
                    <div className="col-span-full py-20 text-center text-gray-500 border-2 border-dashed border-slate-800 rounded-3xl">
                        No shots detected yet. Processing frames... ({data.total_frames} total)
                    </div>
                )}
            </main>

            <footer className="mt-20 text-gray-500 text-xs">
                Padel Stats &copy; 2026 - Video ID: {data.id}
            </footer>
        </div>
    );
}
