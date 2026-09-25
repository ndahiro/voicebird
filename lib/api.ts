/**
 * Backend API clients for VoiceBird.
 *
 * Three backends are supported (see lib/config.ts DEFAULT_BACKEND):
 * - localAPI:       self-hosted ASR (:8100) and translation (:8200) services
 * - huggingFaceAPI: Hugging Face inference API (optional, for ASR only)
 *
 * All functions return normalized TranscriptionResult / translation strings so
 * the UI components don't care which backend produced them.
 */
import { ENV } from "./config"

export interface TranscriptionSegment {
    start: number
    end: number
    text: string
    speaker?: string
}

export interface TranscriptionResult {
    text: string
    segments: TranscriptionSegment[]
}

export interface TranscriptionProgress {
    phase: string
    current: number
    total: number
}

// Normalize the ASR server response (string or {text, segments}) into a result
export const parseTranscription = (data: any): TranscriptionResult => {
    const text = typeof data === "string" ? data : data?.text || "No transcription"
    if (typeof text !== "string") {
        console.error("Local server response:", data)
        throw new Error("Unexpected response format")
    }
    const segments: TranscriptionSegment[] = Array.isArray(data?.segments)
        ? data.segments.map((s: any) => ({
              start: Number(s.start) || 0,
              end: Number(s.end) || 0,
              text: String(s.text || ""),
              speaker: s.speaker ? String(s.speaker) : undefined,
          }))
        : []
    return { text, segments }
}

export class TranscriptionError extends Error {
    constructor(message: string, public status?: number) {
        super(message)
        this.name = "TranscriptionError"
    }
}

// Helper to validate response
const validateResponse = async (response: Response, errorMessage: string) => {
    if (!response.ok) {
        if (response.status === 401) throw new TranscriptionError("Unauthorized: Invalid token", 401)
        if (response.status === 503) throw new TranscriptionError("Service unavailable / Model loading", 503)
        let detail = ""
        try {
            const err = await response.json()
            detail = err.detail || err.error || ""
        } catch { } // ignore json parse error
        throw new TranscriptionError(`${errorMessage} (${response.status}) ${detail}`, response.status)
    }
}

export const huggingFaceAPI = {
    transcribe: async (file: File) => {
        const token = ENV.HF_API_TOKEN.trim()
        if (!token) throw new Error("Hugging Face token missing")

        const arrayBuffer = await file.arrayBuffer()

        const response = await fetch(
            `https://api-inference.huggingface.co/models/${ENV.HF_ASR_MODEL}`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "audio/wav",
                },
                body: arrayBuffer,
            }
        )

        await validateResponse(response, "Hugging Face API failed")

        const data = await response.json()
        const text = typeof data === 'string'
            ? data
            : (Array.isArray(data) ? data[0]?.text : data.text) || "No transcription"

        if (typeof text !== 'string') {
            console.error('HF ASR response:', data)
            throw new Error('Unexpected response format from Hugging Face')
        }
        return text
    }
}

export const localAPI = {
    checkHealth: async (serverUrl: string) => {
        try {
            const response = await fetch(`${serverUrl}/health`, { method: "GET" })
            return response.ok
        } catch {
            return false
        }
    },

    transcribe: async (file: File, language: string, serverUrl: string, model?: string, separateSpeech?: boolean, diarize?: boolean) => {
        const formData = new FormData()
        formData.append("file", file)
        if (language !== "auto") {
            formData.append("language", language)
        }
        if (model) {
            formData.append("model", model)
        }
        if (separateSpeech) {
            formData.append("separate_speech", "true")
        }
        if (diarize) {
            formData.append("diarize", "true")
        }

        const response = await fetch(`${serverUrl}/v1/audio/transcriptions`, {
            method: "POST",
            body: formData,
        })

        await validateResponse(response, "Local server failed")

        const data = await response.json()
        return parseTranscription(data)
    },

    transcribeUrl: async (url: string, language: string, serverUrl: string, model?: string, separateSpeech?: boolean, diarize?: boolean) => {
        const response = await fetch(`${serverUrl}/v1/audio/transcriptions/url`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                url,
                language: language === "auto" ? undefined : language,
                model,
                separate_speech: Boolean(separateSpeech),
                diarize: Boolean(diarize),
            }),
        })

        await validateResponse(response, "Local URL transcription failed")

        const data = await response.json()
        return parseTranscription(data)
    },

    transcribeLive: async (
        file: File,
        language: string,
        sessionId: string,
        serverUrl: string,
        model?: string
    ): Promise<TranscriptionResult> => {
        const formData = new FormData()
        formData.append("file", file)
        formData.append("session_id", sessionId)
        if (language !== "auto") {
            formData.append("language", language)
        }
        if (model) {
            formData.append("model", model)
        }

        const response = await fetch(`${serverUrl}/v1/audio/live`, {
            method: "POST",
            body: formData,
        })

        await validateResponse(response, "Local live transcription failed")

        const data = await response.json()
        return parseTranscription(data)
    },

    clearLiveSession: async (sessionId: string, serverUrl: string) => {
        try {
            await fetch(`${serverUrl}/v1/audio/live/${encodeURIComponent(sessionId)}`, {
                method: "DELETE",
            })
        } catch {
            // best-effort cleanup; ignore failures
        }
    },

    getProgress: async (serverUrl: string): Promise<TranscriptionProgress | null> => {
        try {
            const response = await fetch(`${serverUrl}/v1/audio/progress`)
            if (!response.ok) return null
            return await response.json()
        } catch {
            return null
        }
    },

    /**
     * OCR a video's on-screen text (captions, slides, ...) so it can be shown
     * as selectable text and translated. Backed by /v1/video/ocr on the local
     * ASR server (OpenCV frame sampling + Tesseract).
     */
    ocrVideo: async (
        file: File,
        serverUrl: string,
        frameInterval?: number
    ): Promise<{ text: string; frames: number }> => {
        const formData = new FormData()
        formData.append("file", file)
        if (frameInterval && frameInterval > 0) {
            formData.append("frame_interval", String(frameInterval))
        }

        const response = await fetch(`${serverUrl}/v1/video/ocr`, {
            method: "POST",
            body: formData,
        })

        await validateResponse(response, "Local OCR failed")

        const data = await response.json()
        const text = typeof data === "string" ? data : String(data?.text || "")
        return { text, frames: Number(data?.frames_scanned) || 0 }
    },

    translate: async (text: string, sourceLang: string, targetLang: string) => {
        const baseUrl = ENV.LOCAL_TRANSLATION_URL
        if (!baseUrl) {
            throw new Error("Local translation service not configured")
        }
        // Ensure URL ends without slash and append /translate endpoint
        const url = `${baseUrl.replace(/\/$/, '')}/translate`

        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                source_language: sourceLang,
                target_language: targetLang,
                text,
            }),
        })

        await validateResponse(response, "Local translation failed")

        const data = await response.json()
        let translatedText: string | undefined

        if (typeof data === "string") {
            translatedText = data
        } else if (data && typeof data === "object") {
            translatedText =
                data.translated_text ||
                data.translation ||
                data.text ||
                data.result ||
                data.output
        }

        if (!translatedText || typeof translatedText !== "string") {
            console.error("Local translation response:", data)
            throw new Error("Local translation returned an unexpected response")
        }
        return translatedText
    }
}
