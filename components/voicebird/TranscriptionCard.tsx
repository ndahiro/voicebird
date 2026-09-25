import {
    Copy,
    Download,
    X,
    Volume2,
    ArrowRight,
    Languages,
    Loader2,
    FolderOpen,
    ScanText
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
import { copyToClipboard } from "@/lib/utils"
import { saveText, pickSaveDirectory, fsaSupported, sanitizeFileName } from "@/lib/filesave"
import { SUPPORTED_LANGUAGES, LOCAL_TRANSLATION_LANGS } from "@/lib/config"
import type { TranscriptionSegment } from "@/lib/api"

interface TranscriptionCardProps {
    transcription: string
    translation: string
    clearTranslation: () => void
    /** Speaker-labelled segments of the translation (mirrors `segments`). */
    translatedSegments?: TranscriptionSegment[]
    language: string
    targetLanguage: string
    setTargetLanguage: (lang: string) => void
    enableTranslation: boolean
    handleTranslate: () => void
    isTranslating: boolean
    hasAnyTranslationProvider: boolean
    isValidSunbirdPair: boolean
    segments?: TranscriptionSegment[]
    /** Name of the audio/video (without extension) — saved text uses it. */
    mediaBaseName: string
    /** On-screen text extracted by OCR, if the OCR option was enabled. */
    ocrText?: string
    ocrTranslation?: string
    onTranslateOcr: () => void
    isTranslatingOcr: boolean
}

export function TranscriptionCard({
    transcription,
    translation,
    clearTranslation,
    translatedSegments = [],
    language,
    targetLanguage,
    setTargetLanguage,
    enableTranslation,
    handleTranslate,
    isTranslating,
    hasAnyTranslationProvider,
    isValidSunbirdPair,
    segments = [],
    mediaBaseName,
    ocrText = "",
    ocrTranslation = "",
    onTranslateOcr,
    isTranslatingOcr
}: TranscriptionCardProps) {
    const { addToast } = useToast()

    const translationTargetLabel =
        SUPPORTED_LANGUAGES.find((lang) => lang.value === targetLanguage)?.label || "English"

    // Every saved file carries the audio/video's name (interview.mp3 ->
    // interview.txt, interview.srt, ...) so the text is identifiable next to
    // the media it came from.
    const baseName = sanitizeFileName(mediaBaseName || "transcription")
    const fileName = (suffix: string, ext: string) =>
        sanitizeFileName(`${baseName}${suffix}.${ext}`)

    const handleCopy = async (text: string) => {
        const success = await copyToClipboard(text)
        if (success) {
            addToast({ title: "Copied!", type: "success" })
        }
    }

    const handleDownload = async (text: string, filename: string) => {
        const outcome = await saveText(text, filename)
        if (outcome.status === "error") {
            addToast({ title: "Could not save file", description: outcome.message, type: "error" })
            return
        }
        if (outcome.status === "saved") {
            addToast({
                title: "Saved!",
                description: `${filename} → "${outcome.location}" folder`,
                type: "success",
            })
        } else {
            addToast({
                title: "Downloaded!",
                description: `${filename} → Downloads folder`,
                type: "success",
            })
        }
    }

    // Optional: grant one folder (point it at the media's) so text is written
    // beside the audio/video instead of Downloads.
    const handlePickFolder = async () => {
        if (!fsaSupported()) {
            addToast({
                title: "Folder picking not supported",
                description: "This browser can't choose a save folder — text is saved to Downloads using the media's name.",
                type: "default",
            })
            return
        }
        const dir = await pickSaveDirectory()
        if (dir) {
            addToast({
                title: "Save folder set",
                description: `Text will be saved into "${dir.name}" instead of Downloads.`,
                type: "success",
            })
        }
    }

    const ocrTranslatable =
        Boolean(ocrText.trim()) &&
        language !== "auto" &&
        hasAnyTranslationProvider &&
        isValidSunbirdPair

    const toTimestamp = (seconds: number, msSeparator = ".") => {
        const ms = Math.max(0, seconds) * 1000
        const hr = Math.floor(ms / 3600000)
        let rem = ms - hr * 3600000
        const mn = Math.floor(rem / 60000)
        rem -= mn * 60000
        const sec = Math.floor(rem / 1000)
        const msec = Math.floor(rem - sec * 1000)
        const pad = (n: number, w = 2) => String(n).padStart(w, "0")
        return `${pad(hr)}:${pad(mn)}:${pad(sec)}${msSeparator}${pad(msec, 3)}`
    }

    const lineText = (s: TranscriptionSegment) => (s.speaker ? `[${s.speaker}] ${s.text}` : s.text)

    const toSRT = () =>
        segments
            .map((s, i) => `${i + 1}\n${toTimestamp(s.start, ",")} --> ${toTimestamp(s.end, ",")}\n${lineText(s)}\n`)
            .join("\n")

    const toVTT = () =>
        `WEBVTT\n\n${segments
            .map((s) => `${toTimestamp(s.start)} --> ${toTimestamp(s.end)}\n${lineText(s)}\n`)
            .join("\n")}`

    const handleExport = (format: "srt" | "vtt") => {
        const content = format === "srt" ? toSRT() : toVTT()
        handleDownload(content, fileName("", format))
    }

    return (
        <Card className="animate-fade-in">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 min-w-0">
                        <Volume2 className="h-5 w-5 text-primary shrink-0" />
                        Transcription
                        <span className="text-sm font-normal text-muted-foreground truncate">
                            ({SUPPORTED_LANGUAGES.find(l => l.value === language)?.label}
                            {mediaBaseName ? ` · ${mediaBaseName}` : ""})
                        </span>
                    </CardTitle>
                    <div className="flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" onClick={() => handleCopy(transcription)}><Copy className="mr-2 h-4 w-4" /> Copy</Button>
                        <Button variant="outline" size="sm" onClick={() => handleDownload(transcription, fileName("", "txt"))}><Download className="mr-2 h-4 w-4" /> TXT</Button>
                        <Button variant="outline" size="sm" disabled={!segments.length} onClick={() => handleExport("srt")} title="SubRip subtitles">SRT</Button>
                        <Button variant="outline" size="sm" disabled={!segments.length} onClick={() => handleExport("vtt")} title="WebVTT subtitles">VTT</Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handlePickFolder}
                            title="Choose the folder to save text into — point it at the media's folder so text lands next to the audio/video"
                        >
                            <FolderOpen className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="rounded-lg bg-muted/50 p-4">
                    {segments.some((s) => s.speaker) ? (
                        <div className="space-y-2">
                            {segments.map((s, i) => (
                                <p key={i} className="whitespace-pre-wrap leading-relaxed">
                                    {s.speaker && (
                                        <span className="mr-1.5 inline-block rounded bg-primary/15 px-1.5 py-0.5 text-xs font-medium text-primary">
                                            {s.speaker}
                                        </span>
                                    )}
                                    {s.text}
                                </p>
                            ))}
                        </div>
                    ) : (
                        <p className="whitespace-pre-wrap leading-relaxed">{transcription}</p>
                    )}
                </div>

                {/* OCR Section — on-screen text pulled from the video, selectable
                    and translatable through the same pipeline as the transcript */}
                {ocrText.trim() && (
                    <div className="space-y-3 animate-fade-in">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                            <span className="text-sm font-medium flex items-center gap-2">
                                <ScanText className="h-4 w-4 text-accent" />
                                OCR — Text found in the video
                            </span>
                            <div className="flex flex-wrap gap-2">
                                <Button variant="outline" size="sm" onClick={() => handleCopy(ocrText)}><Copy className="mr-2 h-4 w-4" /> Copy</Button>
                                <Button variant="outline" size="sm" onClick={() => handleDownload(ocrText, fileName(".ocr", "txt"))}><Download className="mr-2 h-4 w-4" /> TXT</Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={onTranslateOcr}
                                    disabled={!ocrTranslatable || isTranslatingOcr}
                                    title={
                                        ocrTranslatable
                                            ? "Translate the extracted on-screen text"
                                            : "Pick a source language (not Auto Detect) with a translation provider configured"
                                    }
                                >
                                    {isTranslatingOcr ? (
                                        <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Translating…</>
                                    ) : (
                                        <><Languages className="mr-2 h-4 w-4" /> Translate</>
                                    )}
                                </Button>
                            </div>
                        </div>
                        {/* select-text: the extracted text can be picked/quoted directly */}
                        <div className="rounded-lg bg-muted/50 p-4 select-text">
                            <p className="whitespace-pre-wrap leading-relaxed">{ocrText}</p>
                        </div>
                        {ocrTranslation && (
                            <div className="space-y-2 animate-fade-in">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium flex items-center gap-2">
                                        <Languages className="h-4 w-4 text-accent" />
                                        OCR Translation ({translationTargetLabel})
                                    </span>
                                    <div className="flex gap-2">
                                        <Button variant="ghost" size="sm" onClick={() => handleCopy(ocrTranslation)}><Copy className="h-4 w-4" /></Button>
                                        <Button variant="ghost" size="sm" onClick={() => handleDownload(ocrTranslation, fileName(".ocr.translation", "txt"))}><Download className="h-4 w-4" /></Button>
                                    </div>
                                </div>
                                <div className="rounded-lg bg-accent/10 border border-accent/20 p-4">
                                    <p className="whitespace-pre-wrap leading-relaxed">{ocrTranslation}</p>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Translation Section */}
                {enableTranslation && (
                    <div className="space-y-3 pt-2">
                        <div className="flex items-center gap-2">
                            <div className="flex-1 border-t border-border" />
                            <span className="text-xs text-muted-foreground flex items-center gap-1">
                                <ArrowRight className="h-3 w-3" />
                                Translate to {translationTargetLabel}
                            </span>
                            <div className="flex-1 border-t border-border" />
                        </div>

                        {language === "eng" ? (
                            <div className="flex items-center gap-3">
                                <label className="text-xs text-muted-foreground">Target language</label>
                                <select
                                    value={targetLanguage}
                                    onChange={(e) => {
                                        clearTranslation()
                                        setTargetLanguage(e.target.value)
                                    }}
                                    className="rounded-md border border-border bg-background px-2 py-1 text-sm"
                                >
                                    {LOCAL_TRANSLATION_LANGS.map((code) => {
                                        const label = SUPPORTED_LANGUAGES.find((l) => l.value === code)?.label || code
                                        return (
                                            <option key={code} value={code}>
                                                {label}
                                            </option>
                                        )
                                    })}
                                </select>
                            </div>
                        ) : (
                            <p className="text-xs text-muted-foreground">Target language: English</p>
                        )}

                        {language === "auto" ? (
                            <div className="rounded-lg bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-900 p-4">
                                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                                    <strong>Translation unavailable:</strong> Please select a specific source language (not "Auto Detect") to enable translation.
                                </p>
                            </div>
                        ) : !hasAnyTranslationProvider ? (
                            <div className="rounded-lg bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-900 p-4">
                                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                                    <strong>Translation unavailable:</strong> Configure NEXT_PUBLIC_SUNBIRD_API_TOKEN or NEXT_PUBLIC_LOCAL_TRANSLATION_URL in your .env file.
                                </p>
                            </div>
                        ) : !isValidSunbirdPair ? (
                            <div className="rounded-lg bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-900 p-4">
                                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                                    <strong>Translation unavailable:</strong> Translation only supports English ↔ African language pairs (Luganda, Acholi, Ateso, Lugbara, Runyankole, Lusoga, Rutooro, Kinyarwanda, Lumasaba).
                                </p>
                            </div>
                        ) : !translation ? (
                            <Button
                                variant="outline"
                                className="w-full"
                                onClick={handleTranslate}
                                disabled={isTranslating}
                            >
                                {isTranslating ? (
                                    <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Translating...</>
                                ) : (
                                    <><Languages className="mr-2 h-4 w-4" /> Translate to {translationTargetLabel}</>
                                )}
                            </Button>
                        ) : (
                            <div className="space-y-2 animate-fade-in">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium flex items-center gap-2">
                                        <Languages className="h-4 w-4 text-accent" />
                                        Translation ({translationTargetLabel})
                                    </span>
                                    <div className="flex gap-2">
                                        <Button variant="ghost" size="sm" onClick={() => handleCopy(translation)}><Copy className="h-4 w-4" /></Button>
                                        <Button variant="ghost" size="sm" onClick={() => handleDownload(translation, fileName(".translation", "txt"))}><Download className="h-4 w-4" /></Button>
                                        <Button variant="ghost" size="sm" onClick={clearTranslation}><X className="h-4 w-4" /></Button>
                                    </div>
                                </div>
                                <div className="rounded-lg bg-accent/10 border border-accent/20 p-4">
                                    {/* Speaker-labelled translation renders with the same
                                        SPEAKER_00 badges as the transcript above */}
                                    {translatedSegments.some((s) => s.speaker) ? (
                                        <div className="space-y-2">
                                            {translatedSegments.map((s, i) => (
                                                <p key={i} className="whitespace-pre-wrap leading-relaxed">
                                                    {s.speaker && (
                                                        <span className="mr-1.5 inline-block rounded bg-primary/15 px-1.5 py-0.5 text-xs font-medium text-primary">
                                                            {s.speaker}
                                                        </span>
                                                    )}
                                                    {s.text}
                                                </p>
                                            ))}
                                        </div>
                                    ) : (
                                        <p className="whitespace-pre-wrap leading-relaxed">{translation}</p>
                                    )}
                                </div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="w-full text-muted-foreground"
                                    onClick={handleTranslate}
                                    disabled={isTranslating}
                                >
                                    {isTranslating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Languages className="mr-2 h-4 w-4" />}
                                    Re-translate
                                </Button>
                            </div>
                        )}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
