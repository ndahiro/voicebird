"use client"

import * as React from "react"
import { useSession, signOut } from "next-auth/react"
import { useRouter } from "next/navigation"
import { ToastProvider, useToast } from "@/components/ui/toast"
import {
  BackendMode,
  InputMode
} from "@/lib/types"
import {
  ENV,
  DEFAULT_BACKEND,
  SUPPORTED_VIDEO_FORMATS,
  SUPPORTED_IMAGE_FORMATS,
  ALL_UPLOAD_FORMATS,
  LOCAL_TRANSLATION_LANGS,
  DEFAULT_ASR_MODEL
} from "@/lib/config"
import { huggingFaceAPI, localAPI, type TranscriptionSegment, type TranscriptionProgress } from "@/lib/api"

import { Header } from "@/components/voicebird/Header"
import { UploadCard } from "@/components/voicebird/UploadCard"
import { TranscriptionCard } from "@/components/voicebird/TranscriptionCard"
import { FeatureInfo } from "@/components/voicebird/FeatureInfo"

/** OCR only makes sense for video — audio has no frames to read text from. */
const isVideoMedia = (media: File | null): boolean =>
  Boolean(
    media &&
      (media.type.startsWith("video/") ||
        SUPPORTED_VIDEO_FORMATS.includes((media.name.split(".").pop() || "").toLowerCase()))
  )

/** Images have no audio — they are OCR-only input. */
const isImageMedia = (media: File | null): boolean =>
  Boolean(
    media &&
      (media.type.startsWith("image/") ||
        SUPPORTED_IMAGE_FORMATS.includes((media.name.split(".").pop() || "").toLowerCase()))
  )

/** Idle window after which the user is signed out (mirrors the NextAuth config). */
const IDLE_SIGNOUT_MS = Math.max(1, ENV.SESSION_IDLE_TIMEOUT_MINUTES) * 60 * 1000
/** User interactions that count as activity and keep the session alive. */
const ACTIVITY_EVENTS: (keyof WindowEventMap)[] = [
  "mousemove",
  "mousedown",
  "keydown",
  "scroll",
  "wheel",
  "touchstart",
]

interface SpeakerTurn {
  speaker?: string
  start: number
  end: number
  text: string
}

/**
 * Collapse consecutive segments that share a speaker into one "turn".
 * Each turn becomes a single translation call, so the SPEAKER_00/01 labels
 * survive into the translation exactly like they do in the transcript — and a
 * long monologue is one request instead of one per sentence.
 */
const groupBySpeaker = (segs: TranscriptionSegment[]): SpeakerTurn[] => {
  const turns: SpeakerTurn[] = []
  for (const s of segs) {
    const last = turns[turns.length - 1]
    if (last && last.speaker === s.speaker) {
      last.text = `${last.text} ${s.text}`.trim()
      last.end = s.end
    } else {
      turns.push({ speaker: s.speaker, start: s.start, end: s.end, text: s.text })
    }
  }
  return turns
}

/** "[SPEAKER_00] Hello" — the label format used by SRT/VTT exports. */
const withSpeakerLabels = (segs: TranscriptionSegment[]): string =>
  segs.map((s) => (s.speaker ? `[${s.speaker}] ${s.text}` : s.text)).join("\n\n")

function TranscriptionApp() {
  const { data: session, status } = useSession()
  const router = useRouter()

  // All hooks must be called before any conditional returns
  const backendMode: BackendMode = DEFAULT_BACKEND
  const serverUrl = ENV.LOCAL_SERVER_URL
  const [inputMode, setInputMode] = React.useState<InputMode>("file")
  const [file, setFile] = React.useState<File | null>(null)
  const [videoUrl, setVideoUrl] = React.useState("")
  const [language, setLanguage] = React.useState("auto")
  const [transcription, setTranscription] = React.useState("")
  const [segments, setSegments] = React.useState<TranscriptionSegment[]>([])
  const [transcriptionProgress, setTranscriptionProgress] = React.useState<TranscriptionProgress | null>(null)
  const [isTranscribing, setIsTranscribing] = React.useState(false)
  const [asrModel, setAsrModel] = React.useState(DEFAULT_ASR_MODEL)
  const [separateSpeech, setSeparateSpeech] = React.useState(false) // demucs noise reduction (Buzz-style)
  const [diarize, setDiarize] = React.useState(false)               // pyannote speaker identification
  const [recordedAudio, setRecordedAudio] = React.useState<File | null>(null)

  // OCR (on-screen text extraction from video) settings
  const [ocrEnabled, setOcrEnabled] = React.useState(false)
  const [ocrText, setOcrText] = React.useState("")
  const [ocrTranslation, setOcrTranslation] = React.useState("")
  const [isTranslatingOcr, setIsTranslatingOcr] = React.useState(false)

  // Translation/Interpretation settings
  const [enableTranslation, setEnableTranslation] = React.useState(false)
  const [targetLanguage, setTargetLanguage] = React.useState("eng")
  const [translation, setTranslation] = React.useState("")
  // Speaker-labelled segments of the translation (mirrors `segments` for the
  // transcript) so SPEAKER_00/... show up in the translation too.
  const [translatedSegments, setTranslatedSegments] = React.useState<TranscriptionSegment[]>([])
  const [isTranslating, setIsTranslating] = React.useState(false)

  const clearTranslation = () => {
    setTranslation("")
    setTranslatedSegments([])
  }

  // Local server settings
  const [serverStatus, setServerStatus] = React.useState<"unknown" | "connected" | "error">("unknown")

  const { addToast } = useToast()

  const isEnglishSource = language === "eng"
  const isEnglishTarget = targetLanguage === "eng"

  // The local translation service is the only provider, so it accepts its full
  // language set (any pair among the supported codes).
  const supportedCodes = [...LOCAL_TRANSLATION_LANGS, "eng"]

  const sourceSupported = supportedCodes.includes(language)
  const targetSupported = supportedCodes.includes(targetLanguage)

  const isValidTranslationPair = sourceSupported && targetSupported

  const hasAnyTranslationProvider = Boolean(ENV.LOCAL_TRANSLATION_URL)

  // Name shared by the audio/video and its saved text: "interview.mp3" ->
  // interview.txt / interview.srt / interview.translation.txt, so the text is
  // identifiable next to the media (used by lib/filesave.ts on download).
  const mediaBaseName = React.useMemo(() => {
    const stripExt = (name: string) => name.replace(/\.[^.]+$/, "")
    if (inputMode === "url") {
      try {
        const url = new URL(videoUrl.trim())
        const lastSegment = url.pathname.split("/").filter(Boolean).pop()
        const name = lastSegment || url.hostname.replace(/^www\./, "")
        return decodeURIComponent(name).slice(0, 120)
      } catch {
        return "url-transcription"
      }
    }
    const media = inputMode === "microphone" ? recordedAudio : file
    return media ? stripExt(media.name) : "transcription"
  }, [inputMode, videoUrl, recordedAudio, file])

  // Check local server connection (auto-run for local backend)
  const checkServerConnection = React.useCallback(async (): Promise<"connected" | "error" | "unknown"> => {
    if (backendMode !== "local") {
      return "unknown"
    }
    try {
      const isConnected = await localAPI.checkHealth(serverUrl)
      if (isConnected) {
        setServerStatus("connected")
        return "connected"
      }
      setServerStatus("error")
      return "error"
    } catch {
      setServerStatus("error")
      return "error"
    }
  }, [backendMode, serverUrl])

  React.useEffect(() => {
    if (backendMode === "local") {
      checkServerConnection()
    }
  }, [backendMode, checkServerConnection])

  // Poll ASR progress while a transcription is running (long audio chunks)
  React.useEffect(() => {
    if (!isTranscribing) {
      setTranscriptionProgress(null)
      return
    }
    const id = setInterval(async () => {
      const p = await localAPI.getProgress(serverUrl)
      if (p) setTranscriptionProgress(p)
    }, 1500)
    return () => clearInterval(id)
  }, [isTranscribing, serverUrl])

  const handleLiveStart = () => {
    setTranscription("")
    clearTranslation()
    setSegments([])
    setOcrText("")
    setOcrTranslation("")
  }

  const handleLiveText = (text: string, segs: TranscriptionSegment[]) => {
    setTranscription((prev) => (prev ? `${prev} ${text}` : text).trim())
    setSegments((prev) => [...prev, ...segs])
  }

  React.useEffect(() => {
    if (!enableTranslation) return
    if (isEnglishSource) {
      if (!LOCAL_TRANSLATION_LANGS.includes(targetLanguage)) {
        setTargetLanguage(LOCAL_TRANSLATION_LANGS[0])
      }
    } else if (targetLanguage !== "eng") {
      setTargetLanguage("eng")
    }
  }, [enableTranslation, isEnglishSource, targetLanguage])

  // Idle sign-out: echoes the server-side session timeout so a tab left open
  // also asks the user to sign in again after a long break.
  React.useEffect(() => {
    if (status !== "authenticated") return

    let timer: ReturnType<typeof setTimeout>
    let lastActivity = Date.now()

    const signOutForIdle = () => signOut({ callbackUrl: "/login?reason=idle" })

    const arm = () => {
      clearTimeout(timer)
      timer = setTimeout(signOutForIdle, IDLE_SIGNOUT_MS)
    }

    const onActivity = () => {
      lastActivity = Date.now()
      arm()
    }

    // Background tabs throttle timers, so re-check elapsed time on return
    // instead of trusting the timer alone.
    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") return
      if (Date.now() - lastActivity >= IDLE_SIGNOUT_MS) {
        signOutForIdle()
        return
      }
      onActivity()
    }

    arm()
    ACTIVITY_EVENTS.forEach((event) => window.addEventListener(event, onActivity, { passive: true }))
    document.addEventListener("visibilitychange", onVisibilityChange)

    return () => {
      clearTimeout(timer)
      ACTIVITY_EVENTS.forEach((event) => window.removeEventListener(event, onActivity))
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [status])

  // Redirect to login if not authenticated
  React.useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login")
    }
  }, [status, router])

  // Show loading while checking authentication
  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  // Don't render if not authenticated
  if (!session) {
    return null
  }

  const validateAndSetFile = (selectedFile: File | null) => {
    if (!selectedFile) {
      setFile(null)
      setTranscription("")
      clearTranslation()
      setOcrText("")
      setOcrTranslation("")
      return
    }

    const extension = selectedFile.name.split(".").pop()?.toLowerCase()
    if (!extension || !ALL_UPLOAD_FORMATS.includes(extension)) {
      addToast({
        title: "Unsupported format",
        description: `Please upload: ${ALL_UPLOAD_FORMATS.join(", ")}`,
        type: "error",
      })
      return
    }

    // Images are OCR-only (text picked from the picture, then translated) and
    // only the local backend runs OCR.
    const isImage = SUPPORTED_IMAGE_FORMATS.includes(extension)
    if (isImage && backendMode !== "local") {
      addToast({
        title: "Images need the local backend",
        description: "OCR runs on the local server — set NEXT_PUBLIC_DEFAULT_BACKEND=local.",
        type: "error",
      })
      return
    }

    // The self-hosted ASR service accepts up to 1024MB (1GB) per upload.
    const maxSize = 1024 * 1024 * 1024
    if (selectedFile.size > maxSize) {
      addToast({
        title: "File too large",
        description: "Maximum file size is 1024MB (1GB)",
        type: "error",
      })
      return
    }

    setFile(selectedFile)
    setTranscription("")
    clearTranslation()
    setOcrText("")
    setOcrTranslation("")
    addToast({ title: "File selected", description: selectedFile.name, type: "success" })
  }

  const handleTranscribe = async () => {
    // URL mode validation
    if (inputMode === "url") {
      if (!videoUrl.trim()) {
        addToast({ title: "No URL entered", description: "Please enter a video URL", type: "error" })
        return
      }
      // URL mode only works with local backend (uses yt-dlp)
      if (backendMode !== "local") {
        addToast({
          title: "URL mode unavailable",
          description: "Set NEXT_PUBLIC_DEFAULT_BACKEND=local in .env.local for URL transcription",
          type: "error",
        })
        return
      }
    } else if (inputMode === "microphone") {
      // Microphone mode validation
      if (!recordedAudio) {
        addToast({ title: "No recording", description: "Please record audio first", type: "error" })
        return
      }
    } else if (!file) {
      addToast({ title: "No file selected", description: "Please upload an audio, video or image file", type: "error" })
      return
    }

    // Validation based on backend mode
    if (backendMode === "huggingface" && !ENV.HF_API_TOKEN) {
      addToast({
        title: "Hugging Face token missing",
        description: "Set NEXT_PUBLIC_HF_API_TOKEN in .env.local",
        type: "error",
      })
      return
    }

    if (backendMode === "local") {
      let status = serverStatus
      if (status === "unknown") {
        status = await checkServerConnection()
      }
      if (status !== "connected") {
        addToast({
          title: "Local server unavailable",
          description: "Start python backend/server.py --port 8100 (app runs on 3100) or update NEXT_PUBLIC_LOCAL_SERVER_URL",
          type: "error",
        })
        return
      }
    }

    setIsTranscribing(true)

    try {
      // Which media is this run about? (mic recording, uploaded file, or URL)
      const mediaForOcr =
        inputMode === "microphone" ? recordedAudio : inputMode === "file" ? file : null
      // Images carry no audio: OCR is the whole job, so ASR is skipped.
      const imageOnly = backendMode === "local" && isImageMedia(mediaForOcr)

      let text = ""
      if (!imageOnly && backendMode === "huggingface") {
        const audioFile = inputMode === "microphone" ? recordedAudio! : file!
        text = await huggingFaceAPI.transcribe(audioFile)
      } else if (!imageOnly) {
        // Local backend - check if URL, microphone, or file mode
        if (inputMode === "url") {
          const result = await localAPI.transcribeUrl(videoUrl, language, serverUrl, asrModel, separateSpeech, diarize)
          text = result.text
          setSegments(result.segments)
        } else {
          const audioFile = inputMode === "microphone" ? recordedAudio! : file!
          const result = await localAPI.transcribe(audioFile, language, serverUrl, asrModel, separateSpeech, diarize)
          text = result.text
          setSegments(result.segments)
        }
      }
      setTranscription(text)
      if (!imageOnly) addToast({ title: "Transcription complete", type: "success" })

      // OCR pass: pull the text rendered in the video (captions, slides,
      // lower-thirds) so it can be picked and translated alongside speech.
      // Images always run it — it's the only thing they can produce.
      if (
        backendMode === "local" &&
        (imageOnly || (ocrEnabled && isVideoMedia(mediaForOcr)))
      ) {
        try {
          const ocr = await localAPI.ocrVideo(mediaForOcr!, serverUrl)
          if (ocr.text.trim()) {
            setOcrText(ocr.text)
            setOcrTranslation("")
            addToast({
              title: "OCR complete",
              description: "On-screen text found — pick or translate it in the OCR section below.",
              type: "success",
            })
          } else {
            setOcrText("")
            setOcrTranslation("")
            addToast({
              title: "OCR found no text",
              description: "No readable text was detected on screen.",
              type: "default",
            })
          }
        } catch (ocrError) {
          console.error("OCR error:", ocrError)
          addToast({
            title: "OCR failed",
            description: ocrError instanceof Error ? ocrError.message : "Could not extract on-screen text",
            type: "error",
          })
        }
      }
    } catch (error) {
      console.error("Transcription error:", error)
      addToast({
        title: "Transcription failed",
        description: error instanceof Error ? error.message : "Please try again",
        type: "error",
      })
    } finally {
      setIsTranscribing(false)
    }
  }

  // Shared validation for both speech and OCR translation: language pair +
  // provider availability, with a toast explaining what's wrong.
  const validateTranslationRequest = (): boolean => {
    if (!sourceSupported || language === "auto") {
      addToast({
        title: "Unsupported language",
        description: "Pick a specific source language (English, Luganda, Acholi, Ateso, Lugbara, or Runyankole).",
        type: "error",
      })
      return false
    }

    if (!isValidTranslationPair) {
      addToast({
        title: "Unsupported pair",
        description: "Pick a source and target language from the supported list.",
        type: "error",
      })
      return false
    }

    if (!ENV.LOCAL_TRANSLATION_URL) {
      addToast({
        title: "Translation unavailable",
        description: "Set NEXT_PUBLIC_LOCAL_TRANSLATION_URL in .env.local to use the local translation server.",
        type: "error",
      })
      return false
    }

    return true
  }

  // Translation always goes to the self-hosted translation server — there is no
  // cloud fallback that could silently receive the text.
  const runTranslation = async (text: string): Promise<string> => {
    return localAPI.translate(text, language, targetLanguage)
  }

  // Translate transcription to target language
  const handleTranslate = async () => {
    if (!transcription.trim()) {
      addToast({ title: "No text to translate", description: "Please transcribe first", type: "error" })
      return
    }

    if (!validateTranslationRequest()) return

    setIsTranslating(true)
    try {
      const hasSpeakers = segments.some((s) => s.speaker)
      if (hasSpeakers) {
        // Diarized transcript: translate one speaker turn at a time so the
        // SPEAKER_00/01 labels survive into the translation exactly like they
        // do in the transcript (each turn = one paragraph = one request).
        const turns = groupBySpeaker(segments)
        const translated: TranscriptionSegment[] = []
        for (const turn of turns) {
          const text = await runTranslation(turn.text)
          translated.push({ start: turn.start, end: turn.end, text, speaker: turn.speaker })
        }
        setTranslatedSegments(translated)
        // Labelled string drives Copy/TXT, matching the SRT/VTT label format.
        setTranslation(withSpeakerLabels(translated))
      } else {
        const text = await runTranslation(transcription)
        setTranslatedSegments([])
        setTranslation(text)
      }
      addToast({ title: "Translation complete", type: "success" })
    } catch (error) {
      console.error("Translation error:", error)
      addToast({
        title: "Translation failed",
        description: error instanceof Error ? error.message : "Please try again",
        type: "error",
      })
    } finally {
      setIsTranslating(false)
    }
  }

  // Translate the OCR-extracted (on-screen) text to the target language
  const handleTranslateOcr = async () => {
    if (!ocrText.trim()) {
      addToast({ title: "No text to translate", description: "Enable the OCR option and transcribe first", type: "error" })
      return
    }

    if (!validateTranslationRequest()) return

    setIsTranslatingOcr(true)
    try {
      const text = await runTranslation(ocrText)
      setOcrTranslation(text)
      addToast({ title: "Translation complete", type: "success" })
    } catch (error) {
      console.error("OCR translation error:", error)
      addToast({
        title: "Translation failed",
        description: error instanceof Error ? error.message : "Please try again",
        type: "error",
      })
    } finally {
      setIsTranslatingOcr(false)
    }
  }

  return (
    <div className="min-h-screen">
      <Header backendMode={backendMode} />

      <main className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-4xl space-y-6">
          {/* Language Pills */}
          <div className="flex flex-wrap justify-center gap-3">
            {["Luganda", "Acholi", "Ateso", "Lugbara", "Runyankole", "Lusoga", "Rutooro", "Kinyarwanda", "Lumasaba", "English"].map((lang) => (
              <span key={lang} className="rounded-full bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary">
                {lang}
              </span>
            ))}
          </div>

          <UploadCard
            inputMode={inputMode}
            setInputMode={setInputMode}
            file={file}
            setFile={validateAndSetFile} // Using wrapper to validate
            videoUrl={videoUrl}
            setVideoUrl={setVideoUrl}
            language={language}
            setLanguage={setLanguage}
            enableTranslation={enableTranslation}
            onTranslateToggle={setEnableTranslation}
            onTranscribe={handleTranscribe}
            isTranscribing={isTranscribing}
            backendMode={backendMode}
            asrModel={asrModel}
            setAsrModel={setAsrModel}
            recordedAudio={recordedAudio}
            setRecordedAudio={setRecordedAudio}
            serverUrl={serverUrl}
            progress={transcriptionProgress}
            onLiveStart={handleLiveStart}
            onLiveText={handleLiveText}
            hasAnyTranslationProvider={hasAnyTranslationProvider}
            isValidTranslationPair={isValidTranslationPair}
            separateSpeech={separateSpeech}
            setSeparateSpeech={setSeparateSpeech}
            diarize={diarize}
            setDiarize={setDiarize}
            ocrEnabled={ocrEnabled}
            setOcrEnabled={setOcrEnabled}
          />

          {/* Transcription Result */}
          {(transcription || ocrText) && (
            <TranscriptionCard
              transcription={transcription}
              segments={segments}
              translation={translation}
              clearTranslation={clearTranslation}
              translatedSegments={translatedSegments}
              language={language}
              targetLanguage={targetLanguage}
              setTargetLanguage={setTargetLanguage}
              enableTranslation={enableTranslation}
              handleTranslate={handleTranslate}
              isTranslating={isTranslating}
              hasAnyTranslationProvider={hasAnyTranslationProvider}
              isValidTranslationPair={isValidTranslationPair}
              mediaBaseName={mediaBaseName}
              ocrText={ocrText}
              ocrTranslation={ocrTranslation}
              onTranslateOcr={handleTranslateOcr}
              isTranslatingOcr={isTranslatingOcr}
            />
          )}

          <FeatureInfo />
        </div>
      </main>

      <footer className="border-t py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          VoiceBird - African Language Transcription
        </div>
      </footer>
    </div>
  )
}

export default function Home() {
  return (
    <ToastProvider>
      <TranscriptionApp />
    </ToastProvider>
  )
}
