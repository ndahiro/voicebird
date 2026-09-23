"use client"

import * as React from "react"
import { Mic, Square, Radio } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { localAPI, TranscriptionSegment } from "@/lib/api"

const SLICE_MS = 6000

interface LiveCaptionProps {
    language: string
    serverUrl: string
    asrModel: string
    onLiveStart: () => void
    onLiveText: (text: string, segments: TranscriptionSegment[]) => void
}

export function LiveCaption({ language, serverUrl, asrModel, onLiveStart, onLiveText }: LiveCaptionProps) {
    const [isCapturing, setIsCapturing] = React.useState(false)
    const [elapsed, setElapsed] = React.useState(0)
    const [error, setError] = React.useState<string | null>(null)
    const [lastCapture, setLastCapture] = React.useState<string | null>(null)

    const streamRef = React.useRef<MediaStream | null>(null)
    const recorderRef = React.useRef<MediaRecorder | null>(null)
    const chunksRef = React.useRef<Blob[]>([])
    const sessionRef = React.useRef<string>(`live-${Date.now()}`)
    const sliceTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(null)
    const clockRef = React.useRef<ReturnType<typeof setInterval> | null>(null)
    const capturingRef = React.useRef(false)
    const inFlightRef = React.useRef(false)

    // Cleanup on unmount
    React.useEffect(() => {
        return () => stopAll()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const sendSlice = async (file: File) => {
        if (inFlightRef.current) return
        inFlightRef.current = true
        try {
            const result = await localAPI.transcribeLive(file, language, sessionRef.current, serverUrl, asrModel)
            if (result.text) {
                onLiveText(result.text, result.segments)
                setLastCapture(result.text.slice(0, 60) + (result.text.length > 60 ? "…" : ""))
            }
        } catch (e) {
            setError(e instanceof Error ? e.message : "Live transcription failed")
        } finally {
            inFlightRef.current = false
        }
    }

    const startRecorder = () => {
        if (!streamRef.current || !capturingRef.current) return
        const recorder = new MediaRecorder(streamRef.current)
        recorderRef.current = recorder
        chunksRef.current = []
        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunksRef.current.push(e.data)
        }
        recorder.onstop = () => {
            if (!capturingRef.current) return
            const blob = new Blob(chunksRef.current, { type: "audio/webm" })
            const file = new File([blob], `live-${Date.now()}.webm`, { type: "audio/webm" })
            void sendSlice(file)
            // Keep the capture loop going
            startRecorder()
        }
        recorder.start()
    }

    const sliceTick = () => {
        if (recorderRef.current && recorderRef.current.state !== "inactive") {
            try {
                recorderRef.current.stop()
            } catch {
                // recorder may already be stopping
            }
        }
    }

    const stopAll = () => {
        capturingRef.current = false
        if (sliceTimerRef.current) clearInterval(sliceTimerRef.current)
        if (clockRef.current) clearInterval(clockRef.current)
        sliceTimerRef.current = null
        clockRef.current = null
        if (recorderRef.current && recorderRef.current.state !== "inactive") {
            try {
                recorderRef.current.stop()
            } catch {
                // ignore
            }
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop())
            streamRef.current = null
        }
        setIsCapturing(false)
    }

    const start = async () => {
        try {
            setError(null)
            setLastCapture(null)
            onLiveStart()
            sessionRef.current = `live-${Date.now()}`
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            streamRef.current = stream
            capturingRef.current = true
            setElapsed(0)
            setIsCapturing(true)
            startRecorder()
            sliceTimerRef.current = setInterval(sliceTick, SLICE_MS)
            clockRef.current = setInterval(() => setElapsed((p) => p + 1), 1000)
        } catch (err) {
            console.error("Error accessing microphone:", err)
            setError("Could not access microphone. Please check permissions.")
        }
    }

    const stop = () => {
        stopAll()
        void localAPI.clearLiveSession(sessionRef.current, serverUrl)
    }

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, "0")}`
    }

    return (
        <div className="upload-zone text-center space-y-4">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                <Radio className={cn("h-8 w-8", isCapturing ? "text-red-500 animate-pulse" : "text-primary")} />
            </div>

            {error && (
                <div className="text-sm text-red-500 bg-red-50 p-3 rounded-lg">{error}</div>
            )}

            {!isCapturing ? (
                <>
                    <p className="font-medium">Live Captioning</p>
                    <p className="text-sm text-muted-foreground">
                        Microphone audio is transcribed in ~6 second slices and appended in real time
                    </p>
                    <Button variant="hero" size="lg" onClick={start} className="mt-2">
                        <Mic className="mr-2 h-5 w-5" />
                        Start Live Captioning
                    </Button>
                </>
            ) : (
                <>
                    <div className="flex items-center justify-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                        <p className="font-medium text-red-500">Listening…</p>
                    </div>
                    <p className="text-2xl font-mono font-bold">{formatTime(elapsed)}</p>
                    {lastCapture && (
                        <p className="text-xs text-muted-foreground truncate max-w-full px-4" title={lastCapture}>
                            Last: {lastCapture}
                        </p>
                    )}
                    <p className="text-xs text-muted-foreground">
                        Text appears in the Transcription card below as it is captured
                    </p>
                    <Button variant="outline" size="lg" onClick={stop} className="mt-2">
                        <Square className="mr-2 h-5 w-5" />
                        Stop
                    </Button>
                </>
            )}
        </div>
    )
}
