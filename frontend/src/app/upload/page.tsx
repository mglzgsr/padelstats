import { VideoUpload } from "@/components/VideoUpload";

export default function UploadPage() {
    return (
        <div className="min-h-screen p-8 flex flex-col items-center justify-center font-[family-name:var(--font-geist-sans)]">
            <div className="text-center mb-10">
                <h1 className="text-3xl font-bold text-white mb-2">Upload Match Video</h1>
                <p className="text-gray-400">Supported formats: MP4, MOV</p>
            </div>
            <VideoUpload />
        </div>
    );
}
