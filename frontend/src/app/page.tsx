"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface Video {
  id: string;
  filename: string;
  status: string;
  created_at: string;
}

export default function Home() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/videos`)
      .then((res) => res.json())
      .then((data) => {
        setVideos(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen p-8 sm:p-20 bg-slate-950 font-[family-name:var(--font-geist-sans)] flex flex-col items-center">
      <header className="mb-16 text-center">
        <div className="inline-block px-3 py-1 bg-primary/10 border border-primary/20 rounded-full text-primary text-xs font-bold tracking-widest uppercase mb-6">
          AI Video Intelligence
        </div>
        <h1 className="text-5xl md:text-7xl font-black bg-clip-text text-transparent bg-gradient-to-br from-white to-slate-400 mb-4 tracking-tighter">
          PADEL STATS PRO
        </h1>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto">
          Turn your game footage into professional-grade performance insights with our advanced CV engine.
        </p>
      </header>

      <main className="w-full max-w-4xl">
        <div className="flex justify-between items-end mb-8">
          <h2 className="text-xl font-bold text-white tracking-tight">Available Matches</h2>
          <Link href="/upload" className="text-sm font-bold text-primary hover:text-blue-400 transition-colors">
            + Upload New Video
          </Link>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[1, 2].map((i) => (
              <div key={i} className="h-24 bg-slate-900/50 rounded-2xl animate-pulse border border-slate-800" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {videos.length > 0 ? (
              videos.map((vid) => (
                <Link
                  key={vid.id}
                  href={`/dashboard/${vid.id}`}
                  className="group bg-slate-900/40 backdrop-blur-sm border border-slate-800/50 p-6 rounded-3xl flex items-center justify-between hover:bg-slate-800/60 hover:border-primary/30 transition-all duration-300"
                >
                  <div className="flex items-center gap-6">
                    <div className="w-12 h-12 bg-slate-800 rounded-2xl flex items-center justify-center text-2xl group-hover:bg-primary/20 transition-colors">
                      🎾
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-white group-hover:text-primary transition-colors">
                        {vid.filename}
                      </h3>
                      <p className="text-xs text-gray-500 font-mono">
                        ID: {vid.id.slice(0, 8)}... • {new Date(vid.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={`text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wider ${vid.status === 'completed' ? 'bg-secondary/10 text-secondary border border-secondary/20' :
                        vid.status === 'processing' ? 'bg-primary/10 text-primary border border-primary/20' :
                          'bg-slate-800 text-gray-400'
                      }`}>
                      {vid.status}
                    </span>
                    <div className="w-8 h-8 rounded-full bg-slate-800/50 flex items-center justify-center text-gray-400 group-hover:bg-primary group-hover:text-black transition-all">
                      →
                    </div>
                  </div>
                </Link>
              ))
            ) : (
              <div className="py-20 text-center text-gray-500 border-2 border-dashed border-slate-800 rounded-3xl">
                No matches analyzed yet. Head to "Upload" to get started.
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="mt-20 text-gray-600 text-[10px] uppercase tracking-[0.2em]">
        Powered by Google DeepMind Technology
      </footer>
    </div>
  );
}
