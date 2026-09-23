import * as React from "react"
import { Mic, Square, Play, Pause, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface AudioRecorderProps {
    onRecordingComplete: (file: File) => void
    recordedAudio: File | null
    onClearRecording: () => void
}

export function AudioRecorder({ onRecordingComplete, recordedAudio, onClearRecording }: AudioRecorderProps) {
    const [isRecording, setIsRecording] = React.useState(false)
    const [recordingTime, setRecordingTime] = React.useState(0)
    const [isPlaying, setIsPlaying] = React.useState(false)
    const [error, setError] = React.useState<string | null>(null)

    const mediaRecorderRef = React.useRef<MediaRecorder | null>(null)
    const audioChunksRef = React.useRef<Blob[]>([])
    const timerRef = React.useRef<NodeJS.Timeout | null>(null)
    const audioRef = React.useRef<HTMLAudioElement | null>(null)
    const streamRef = React.useRef<MediaStream | null>(null)

    // Cleanup on unmount
    React.useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current)
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop())
            }
        }
    }, [])

    const startRecording = async () => {
        try {
            setError(null)

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
            streamRef.current = stream

            // Create MediaRecorder
            const mediaRecorder = new MediaRecorder(stream)
            mediaRecorderRef.current = mediaRecorder
            audioChunksRef.current = []

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data)
                }
            }

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
                const audioFile = new File([audioBlob], `recording-${Date.now()}.webm`, {
                    type: 'audio/webm',
                    lastModified: Date.now()
                })
                onRecordingComplete(audioFile)

                // Stop all tracks
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop())
                    streamRef.current = null
                }
            }

            mediaRecorder.start()
            setIsRecording(true)
            setRecordingTime(0)

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1)
            }, 1000)

        } catch (err) {
            console.error('Error accessing microphone:', err)
            setError('Could not access microphone. Please check permissions.')
        }
    }

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop()
            setIsRecording(false)

            if (timerRef.current) {
                clearInterval(timerRef.current)
                timerRef.current = null
            }
        }
    }

    const togglePlayback = () => {
        if (!recordedAudio) return

        if (!audioRef.current) {
            audioRef.current = new Audio(URL.createObjectURL(recordedAudio))
            audioRef.current.onended = () => setIsPlaying(false)
        }

        if (isPlaying) {
            audioRef.current.pause()
            setIsPlaying(false)
        } else {
            audioRef.current.play()
            setIsPlaying(true)
        }
    }

    const handleClearRecording = () => {
        if (audioRef.current) {
            audioRef.current.pause()
            audioRef.current = null
        }
        setIsPlaying(false)
        setRecordingTime(0)
        onClearRecording()
    }

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    return (
        <div className="space-y-4">
            {/* Recording Interface */}
            {!recordedAudio ? (
                <div className="upload-zone text-center">
                    <div className="space-y-4">
                        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                            <Mic className={cn(
                                "h-8 w-8",
                                isRecording ? "text-red-500 animate-pulse" : "text-primary"
                            )} />
                        </div>

                        {error && (
                            <div className="text-sm text-red-500 bg-red-50 p-3 rounded-lg">
                                {error}
                            </div>
                        )}

                        {!isRecording ? (
                            <>
                                <p className="font-medium">Record Audio</p>
                                <p className="text-sm text-muted-foreground">
                                    Click the button below to start recording
                                </p>
                                <Button
                                    variant="hero"
                                    size="lg"
                                    onClick={startRecording}
                                    className="mt-4"
                                >
                                    <Mic className="mr-2 h-5 w-5" />
                                    Start Recording
                                </Button>
                            </>
                        ) : (
                            <>
                                <div className="flex items-center justify-center gap-2">
                                    <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                                    <p className="font-medium text-red-500">Recording...</p>
                                </div>
                                <p className="text-2xl font-mono font-bold">{formatTime(recordingTime)}</p>
                                <Button
                                    variant="outline"
                                    size="lg"
                                    onClick={stopRecording}
                                    className="mt-4"
                                >
                                    <Square className="mr-2 h-5 w-5" />
                                    Stop Recording
                                </Button>
                            </>
                        )}
                    </div>
                </div>
            ) : (
                /* Playback Interface */
                <div className="upload-zone text-center">
                    <div className="space-y-4">
                        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                            <Mic className="h-8 w-8 text-primary" />
                        </div>
                        <div>
                            <p className="font-medium">{recordedAudio.name}</p>
                            <p className="text-sm text-muted-foreground">
                                {(recordedAudio.size / 1024).toFixed(2)} KB • {formatTime(recordingTime)}
                            </p>
                        </div>
                        <div className="flex items-center justify-center gap-2">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={togglePlayback}
                            >
                                {isPlaying ? (
                                    <>
                                        <Pause className="mr-2 h-4 w-4" />
                                        Pause
                                    </>
                                ) : (
                                    <>
                                        <Play className="mr-2 h-4 w-4" />
                                        Play
                                    </>
                                )}
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={handleClearRecording}
                            >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
