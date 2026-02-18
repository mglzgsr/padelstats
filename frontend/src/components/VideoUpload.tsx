'use client';

import React, { useState, useRef } from 'react';

export const VideoUpload = () => {
    const [dragging, setDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragEnter = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragging(false);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
        }
    };

    const uploadFile = async () => {
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', 'http://localhost:8000/upload', true);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    setProgress(percentComplete);
                }
            };

            xhr.onload = async () => {
                if (xhr.status === 200) {
                    const result = JSON.parse(xhr.responseText);
                    console.log('Upload success:', result);

                    // Trigger analysis automatically
                    try {
                        const analyzeRes = await fetch(`http://localhost:8000/analyze/${result.id}`, { method: 'POST' });
                        if (analyzeRes.ok) {
                            // Redirect to dashboard
                            window.location.href = `/dashboard/${result.id}`;
                        } else {
                            alert('Video uploaded but analysis failed to start.');
                        }
                    } catch (e) {
                        console.error('Analysis trigger failed:', e);
                        window.location.href = `/dashboard/${result.id}`; // Attempt redirect anyway
                    }
                } else {
                    alert('Upload failed.');
                }
                setUploading(false);
            };

            xhr.send(formData);
        } catch (error) {
            console.error('Error uploading file:', error);
            setUploading(false);
        }
    };

    return (
        <div className="w-full max-w-xl mx-auto mt-10">
            {!file ? (
                <div
                    className={`border-4 border-dashed rounded-[2rem] h-64 flex flex-col items-center justify-center transition-all duration-300 cursor-pointer
                    ${dragging ? 'border-primary bg-primary/10 scale-105' : 'border-slate-700 bg-slate-800/50 hover:border-primary/50'}`}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        accept="video/*"
                        onChange={handleFileSelect}
                    />
                    <svg className="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-xl font-semibold text-gray-300">Drag & Drop your match video</p>
                    <p className="text-sm text-gray-500 mt-2">or click to browse</p>
                </div>
            ) : (
                <div className="bg-slate-800 rounded-xl p-6 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-4 mb-4">
                        <div className="p-3 bg-blue-500/20 rounded-lg text-blue-400">
                            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-lg font-medium text-white truncate">{file.name}</p>
                            <p className="text-sm text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </div>
                        {!uploading && (
                            <button
                                onClick={() => setFile(null)}
                                className="text-gray-400 hover:text-red-400 transition-colors"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        )}
                    </div>

                    {uploading ? (
                        <div className="w-full bg-slate-700 rounded-full h-4 overflow-hidden mb-2 relative">
                            <div
                                className="bg-secondary h-full rounded-full transition-all duration-300 relative overflow-hidden"
                                style={{ width: `${progress}%` }}
                            >
                                <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]"></div>
                            </div>
                        </div>
                    ) : (
                        <button
                            onClick={uploadFile}
                            className="w-full py-3 bg-primary text-slate-900 font-bold rounded-xl hover:bg-sky-300 transition-colors shadow-lg hover:shadow-primary/20"
                        >
                            Start Analysis
                        </button>
                    )}
                </div>
            )}
        </div>
    );
};
