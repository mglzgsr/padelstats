import React from 'react';
import { Player, PlayerMatchStats } from '@/types';

interface StatsCardProps {
    player: Player;
    matchStats: PlayerMatchStats;
}

const StatRow = ({ label, winners, errors }: { label: string; winners: number; errors: number }) => {
    const total = winners + errors;
    const winnerPct = total > 0 ? (winners / total) * 100 : 0;
    const errorPct = total > 0 ? (errors / total) * 100 : 0;

    return (
        <div className="mb-4">
            <div className="flex justify-between mb-1 text-sm font-medium">
                <span className="text-gray-300">{label}</span>
                <span className="text-gray-400 text-xs">{winners}W | {errors}E</span>
            </div>
            <div className="h-2.5 w-full bg-slate-700/50 rounded-full overflow-hidden flex">
                <div
                    className="h-full bg-secondary transition-all duration-500"
                    style={{ width: `${winnerPct}%` }}
                    title={`${winners} Winners`}
                />
                <div
                    className="h-full bg-danger transition-all duration-500"
                    style={{ width: `${errorPct}%` }}
                    title={`${errors} Errors`}
                />
            </div>
        </div>
    );
};

const CircularProgress = ({ value, label, subLabel }: { value: number; label: string; subLabel: string }) => {
    const radius = 30;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (value / 100) * circumference;

    return (
        <div className="flex flex-col items-center justify-center relative">
            <svg className="transform -rotate-90 w-24 h-24">
                <circle
                    className="text-slate-700"
                    strokeWidth="6"
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="48"
                    cy="48"
                />
                <circle
                    className="text-primary transition-all duration-1000 ease-out"
                    strokeWidth="6"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="48"
                    cy="48"
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-xl font-bold text-white">{value}%</span>
            </div>
            <span className="text-sm font-semibold text-gray-200 mt-1">{label}</span>
            <span className="text-xs text-gray-400">{subLabel}</span>
        </div>
    );
};

export const StatsCard: React.FC<StatsCardProps> = ({ player, matchStats }) => {
    const { stats } = matchStats;
    const totalPoints = stats.total_points_won + stats.total_points_lost;
    const winRate = totalPoints > 0 ? Math.round((stats.total_points_won / totalPoints) * 100) : 0;

    return (
        <div className="w-full max-w-sm rounded-[24px] bg-card backdrop-blur-md border border-slate-700/50 p-6 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1">
            {/* Header */}
            <div className="flex items-center space-x-4 mb-6 border-b border-slate-700/50 pb-4">
                <div className="w-14 h-14 rounded-full bg-gradient-to-br from-primary to-blue-600 flex items-center justify-center text-xl font-bold text-white shadow-lg">
                    {player.name.charAt(0)}
                </div>
                <div>
                    <h2 className="text-xl font-bold text-white tracking-tight">{player.name}</h2>
                    <p className="text-sm text-primary font-medium">Player Stats</p>
                </div>
                <div className="ml-auto text-right">
                    <div className="text-xs text-gray-400 uppercase tracking-widest">Ranking</div>
                    <div className="font-mono text-lg font-bold text-white">#{player.ranking || '-'}</div>
                </div>
            </div>

            {/* Main Stat */}
            <div className="flex justify-center mb-8">
                <CircularProgress value={winRate} label="Points Won" subLabel={`${stats.total_points_won} / ${totalPoints}`} />
            </div>

            {/* Detailed Stats */}
            <div className="space-y-2">
                <StatRow label="Forehand" winners={stats.forehand_winners} errors={stats.forehand_errors} />
                <StatRow label="Backhand" winners={stats.backhand_winners} errors={stats.backhand_errors} />
                <StatRow label="Volley" winners={stats.volley_winners} errors={stats.volley_errors} />
                <StatRow label="Smash" winners={stats.smash_winners} errors={stats.smash_errors} />
            </div>

            {/* Footer / Extra Stats */}
            <div className="mt-6 pt-4 border-t border-slate-700/50 grid grid-cols-2 gap-4">
                <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                    <div className="text-2xl font-bold text-danger">{stats.unforced_errors}</div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">Unforced Errors</div>
                </div>
                <div className="bg-slate-800/50 rounded-xl p-3 text-center">
                    <div className="text-2xl font-bold text-secondary">{stats.total_points_won}</div>
                    <div className="text-xs text-gray-400 uppercase tracking-wide">Total Points</div>
                </div>
            </div>
        </div>
    );
};
