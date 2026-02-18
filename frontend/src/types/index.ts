export type ShotType = "forehand" | "backhand" | "volley" | "smash" | "lob" | "serve";
export type Outcome = "winner" | "error" | "in_play";
export type Side = "left" | "right";

export interface Player {
    id: string;
    name: string;
    ranking?: number;
}

export interface Stat {
    forehand_winners: number;
    forehand_errors: number;
    backhand_winners: number;
    backhand_errors: number;
    volley_winners: number;
    volley_errors: number;
    smash_winners: number;
    smash_errors: number;
    total_points_won: number;
    total_points_lost: number;
    unforced_errors: number;
}

export interface PlayerMatchStats {
    player_id: string;
    match_id: string;
    stats: Stat;
}

export interface Event {
    id: string;
    match_id: string;
    timestamp: number;
    player_id?: string;
    shot_type?: ShotType;
    outcome?: Outcome;
    coordinates_x?: number;
    coordinates_y?: number;
}

export interface Video {
    id: string;
    filename: string;
    duration: number;
    upload_date: string; // ISO date string
    processed: boolean;
}

export interface Match {
    id: string;
    date: string; // ISO date string
    video_id?: string;
    players: Player[];
    events: Event[];
}
