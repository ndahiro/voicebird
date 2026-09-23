import * as React from "react"
import {
    Upload,
    Mic,
    FileAudio,
    FileVideo,
    X,
    Loader2,
    Globe,
    Link,
    Radio,
    Sparkles
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select } from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { InputMode, BackendMode } from "@/lib/types"
import { SUPPORTED_LANGUAGES, ALL_SUPPORTED_FORMATS, SUPPORTED_VIDEO_FORMATS, LOCAL_TRANSLATION_LANGS, ASR_MODELS } from "@/lib/config"
import { AudioRecorder } from "./AudioRecorder"
import { LiveCaption } from "./LiveCaption"
import type { TranscriptionSegment } from "@/lib/api"

interface UploadCardProps {
    inputMode: InputMode
    setInputMode: (mode: InputMode) => void
    file: File | null
    setFile: (file: File | null) => void
    videoUrl: string
    setVideoUrl: (url: string) => void
    language: string
    setLanguage: (lang: string) => void
    enableTranslation: boolean
    onTranscribe: () => void
    isTranscribing: boolean
    backendMode: BackendMode
    onTranslateToggle: (active: boolean) => void
    asrModel: string
    setAsrModel: (model: string) => void
    recordedAudio: File | null
    setRecordedAudio: (file: File | null) => void
    serverUrl: string
    progress?: { phase: string; current: number; total: number } | null
    onLiveStart: () => void
    onLiveText: (text: string, segments: TranscriptionSegment[]) => void
    hasAnyTranslationProvider: boolean
    isValidSunbirdPair: boolean
    separateSpeech: boolean
    setSeparateSpeech: (on: boolean) => void
    diarize: boolean
    setDiarize: (on: boolean) => void
}

export function UploadCard({
    inputMode,
    setInputMode,
    file,
    setFile,
    videoUrl,
    setVideoUrl,
    language,
    setLanguage,
    enableTranslation,
    onTranscribe,
    isTranscribing,
    backendMode,
    onTranslateToggle,
    asrModel,
    setAsrModel,
    recordedAudio,
    setRecordedAudio,
    serverUrl,
    progress,
    onLiveStart,
    onLiveText,
    hasAnyTranslationProvider,
    isValidSunbirdPair,
    separateSpeech,
    setSeparateSpeech,
    diarize,
    setDiarize
}: UploadCardProps) {
    const translationUnavailable =
        language === "auto" || !hasAnyTranslationProvider || !isValidSunbirdPair
    const translationUnavailableHint =
        language === "auto"
            ? "Select a source language (not Auto Detect) to enable translation."
            : !hasAnyTranslationProvider
            ? "Translation is unavailable — no translation service is configured. Add NEXT_PUBLIC_SUNBIRD_API_TOKEN or NEXT_PUBLIC_LOCAL_TRANSLATION_URL to your .env file."
            : "Translation only supports English ↔ African language pairs."
    const fileInputRef = React.useRef<HTMLInputElement>(null)
    const [isDragging, setIsDragging] = React.useState(false)
    const [micMode, setMicMode] = React.useState<"record" | "live">("record")

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile) setFile(droppedFile)
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0]
        if (selectedFile) setFile(selectedFile)
    }

    const isVideo = file?.type.startsWith("video/") ||
        SUPPORTED_VIDEO_FORMATS.includes(file?.name.split(".").pop()?.toLowerCase() || "")

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-accent" />
                    Upload Audio or Video
                </CardTitle>
                <CardDescription>
                    Upload a file or paste a URL (YouTube, TikTok, etc.)
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Input Mode Toggle */}
                <div className="grid grid-cols-3 gap-1 rounded-lg border-2 border-input p-1">
                    <button
                        type="button"
                        onClick={() => setInputMode("file")}
                        className={cn(
                            "flex items-center justify-center gap-2 rounded-md px-3 py-2.5 text-sm font-medium transition-all",
                            inputMode === "file" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                        )}
                    >
                        <Upload className="h-4 w-4" />
                        Upload File
                    </button>
                    <button
                        type="button"
                        onClick={() => setInputMode("url")}
                        className={cn(
                            "flex items-center justify-center gap-2 rounded-md px-3 py-2.5 text-sm font-medium transition-all",
                            inputMode === "url" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                        )}
                    >
                        <Link className="h-4 w-4" />
                        Paste URL
                    </button>
                    <button
                        type="button"
                        onClick={() => setInputMode("microphone")}
                        className={cn(
                            "flex items-center justify-center gap-2 rounded-md px-3 py-2.5 text-sm font-medium transition-all",
                            inputMode === "microphone" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                        )}
                    >
                        <Mic className="h-4 w-4" />
                        Record
                    </button>
                </div>


                {/* File Upload */}
                {inputMode === "file" && (
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                        className={cn("upload-zone text-center animate-fade-in", isDragging && "dragging")}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept={ALL_SUPPORTED_FORMATS.map((f) => `.${f}`).join(",")}
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                        {file ? (
                            <div className="space-y-4">
                                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                                    {isVideo ? <FileVideo className="h-8 w-8 text-primary" /> : <FileAudio className="h-8 w-8 text-primary" />}
                                </div>
                                <div>
                                    <p className="font-medium">{file.name}</p>
                                    <p className="text-sm text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                                </div>
                                <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); setFile(null); }}>
                                    <X className="mr-2 h-4 w-4" /> Remove
                                </Button>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                                    <Upload className="h-8 w-8 text-primary" />
                                </div>
                                <p className="font-medium">Drop your file here or click to browse</p>
                                <p className="text-sm text-muted-foreground">
                                    MP3, WAV, OGG, M4A, MP4, WebM, FLAC
                                </p>
                                <p className="text-xs text-muted-foreground/70 mt-1">
                                    Up to 1024 MB (1 GB)
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* URL Input */}
                {inputMode === "url" && (
                    <div className="space-y-4 animate-fade-in">
                        <div className="upload-zone text-center">
                            <div className="space-y-4">
                                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                                    <Globe className="h-8 w-8 text-primary" />
                                </div>
                                <p className="font-medium">Paste a video URL</p>
                                <p className="text-sm text-muted-foreground">
                                    YouTube, TikTok, Instagram, Facebook, X, and more
                                </p>
                                <p className="text-xs text-muted-foreground/70">
                                    Any length · Long videos are transcribed in chunks and joined
                                </p>
                            </div>
                        </div>
                        <input
                            type="url"
                            value={videoUrl}
                            onChange={(e) => setVideoUrl(e.target.value)}
                            placeholder="https://youtube.com/watch?v=... or https://tiktok.com/..."
                            className="w-full rounded-xl border-2 border-input py-4 px-4 text-sm focus:border-primary focus:outline-none"
                        />
                        {videoUrl && (
                            <Button variant="ghost" size="sm" onClick={() => setVideoUrl("")}>
                                <X className="mr-2 h-4 w-4" /> Clear URL
                            </Button>
                        )}
                    </div>
                )}

                {/* Microphone Recording / Live Captioning */}
                {inputMode === "microphone" && (
                    <div className="animate-fade-in space-y-4">
                        <div className="grid grid-cols-2 gap-1 rounded-lg border-2 border-input p-1">
                            <button
                                type="button"
                                onClick={() => setMicMode("record")}
                                className={cn(
                                    "flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-all",
                                    micMode === "record" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                <Mic className="h-4 w-4" />
                                Record &amp; Transcribe
                            </button>
                            <button
                                type="button"
                                onClick={() => setMicMode("live")}
                                className={cn(
                                    "flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-all",
                                    micMode === "live" ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                <Radio className="h-4 w-4" />
                                Live Captioning
                            </button>
                        </div>
                        {micMode === "record" ? (
                            <AudioRecorder
                                onRecordingComplete={setRecordedAudio}
                                recordedAudio={recordedAudio}
                                onClearRecording={() => setRecordedAudio(null)}
                            />
                        ) : (
                            <LiveCaption
                                language={language}
                                serverUrl={serverUrl}
                                asrModel={asrModel}
                                onLiveStart={onLiveStart}
                                onLiveText={onLiveText}
                            />
                        )}
                    </div>
                )}

                {/* Language Selection */}
                <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Source Language</label>
                        <Select value={language} onChange={setLanguage} options={SUPPORTED_LANGUAGES} />
                    </div>

                    {/* Model Selection (Only visible for Local Backend) */}
                    {backendMode === "local" && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium">Options</label>
                            <Select value={asrModel} onChange={setAsrModel} options={ASR_MODELS} />
                        </div>
                    )}
                </div>

                {/* Audio enhancement toggles (Local backend only) — borrowed from Buzz:
                    demucs speech separation (noise reduction) and pyannote speaker ID */}
                {backendMode === "local" && (
                    <div className="grid gap-3 sm:grid-cols-2">
                        <div className="flex items-center justify-between gap-2 rounded-lg border border-input p-3">
                            <div className="min-w-0">
                                <p className="text-sm font-medium">Noise Reduction</p>
                                <p className="text-xs text-muted-foreground">Separate vocals from background noise before transcribing (slower)</p>
                            </div>
                            <button
                                type="button"
                                role="switch"
                                aria-checked={separateSpeech}
                                aria-label="Enable noise reduction (speech separation)"
                                onClick={() => setSeparateSpeech(!separateSpeech)}
                                className={cn(
                                    "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
                                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                                    separateSpeech ? "bg-primary" : "bg-muted"
                                )}
                            >
                                <span
                                    className={cn(
                                        "inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform",
                                        separateSpeech ? "translate-x-6" : "translate-x-1"
                                    )}
                                />
                            </button>
                        </div>
                        <div className="flex items-center justify-between gap-2 rounded-lg border border-input p-3">
                            <div className="min-w-0">
                                <p className="text-sm font-medium">Speaker ID</p>
                                <p className="text-xs text-muted-foreground">Identify different speakers and label segments (SPEAKER_00, ...)</p>
                            </div>
                            <button
                                type="button"
                                role="switch"
                                aria-checked={diarize}
                                aria-label="Enable speaker identification"
                                onClick={() => setDiarize(!diarize)}
                                className={cn(
                                    "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
                                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                                    diarize ? "bg-primary" : "bg-muted"
                                )}
                            >
                                <span
                                    className={cn(
                                        "inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform",
                                        diarize ? "translate-x-6" : "translate-x-1"
                                    )}
                                />
                            </button>
                        </div>
                    </div>
                )}

                {/* Translation Toggle */}
                <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">Translate</span>
                            {enableTranslation && !translationUnavailable && (
                                <span className="text-xs font-medium text-primary">On</span>
                            )}
                        </div>
                        <button
                            type="button"
                            role="switch"
                            aria-checked={enableTranslation}
                            aria-label="Enable translation"
                            onClick={() => onTranslateToggle(!enableTranslation)}
                            disabled={translationUnavailable}
                            className={cn(
                                "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
                                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                                "disabled:cursor-not-allowed disabled:opacity-50",
                                enableTranslation ? "bg-primary" : "bg-muted"
                            )}
                        >
                            <span
                                className={cn(
                                    "inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform",
                                    enableTranslation ? "translate-x-6" : "translate-x-1"
                                )}
                            />
                        </button>
                    </div>
                    {translationUnavailable && (
                        <p className="text-xs text-muted-foreground">{translationUnavailableHint}</p>
                    )}
                </div>

                {/* Transcribe Button */}
                <Button
                    variant="hero"
                    size="xl"
                    className="w-full"
                    onClick={onTranscribe}
                    disabled={(inputMode === "file" && !file) || (inputMode === "url" && !videoUrl.trim()) || isTranscribing}
                >
                    {isTranscribing ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Transcribing...</> : <><Mic className="mr-2 h-5 w-5" /> Transcribe Now</>}
                </Button>

                {isTranscribing && progress && progress.total > 1 && (
                    <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>Transcribing chunk {Math.min(progress.current, progress.total)} of {progress.total}…</span>
                            <span>{Math.round((progress.current / progress.total) * 100)}%</span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                            <div
                                className="h-full rounded-full bg-primary transition-all duration-500"
                                style={{ width: `${Math.round((progress.current / progress.total) * 100)}%` }}
                            />
                        </div>
                    </div>
                )}

                {isTranscribing && (
                    <div className="flex items-center justify-center gap-1 py-4">
                        {Array.from({ length: 20 }).map((_, i) => (
                            <div key={i} className="waveform-bar h-8 w-1" style={{ animationDelay: `${i * 0.05}s` }} />
                        ))}
                    </div>
                )}
            </CardContent>
        </Card >
    )
}
