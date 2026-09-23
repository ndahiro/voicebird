import {
    Copy,
    Download,
    X,
    Volume2,
    ArrowRight,
    Languages,
    Loader2
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/components/ui/toast"
import { copyToClipboard, downloadText, downloadFile } from "@/lib/utils"
import { SUPPORTED_LANGUAGES, LOCAL_TRANSLATION_LANGS } from "@/lib/config"
import type { TranscriptionSegment } from "@/lib/api"

interface TranscriptionCardProps {
    transcription: string
    translation: string
    setTranslation: (text: string) => void
    language: string
    targetLanguage: string
    setTargetLanguage: (lang: string) => void
    enableTranslation: boolean
    handleTranslate: () => void
    isTranslating: boolean
    hasAnyTranslationProvider: boolean
    isValidSunbirdPair: boolean
    segments?: TranscriptionSegment[]
}

export function TranscriptionCard({
    transcription,
    translation,
    setTranslation,
    language,
    targetLanguage,
    setTargetLanguage,
    enableTranslation,
    handleTranslate,
    isTranslating,
    hasAnyTranslationProvider,
    isValidSunbirdPair,
    segments = []
}: TranscriptionCardProps) {
    const { addToast } = useToast()

    const translationTargetLabel =
        SUPPORTED_LANGUAGES.find((lang) => lang.value === targetLanguage)?.label || "English"

    const handleCopy = async (text: string) => {
        const success = await copyToClipboard(text)
        if (success) {
            addToast({ title: "Copied!", type: "success" })
        }
    }

    const handleDownload = (text: string, prefix: string) => {
        downloadText(text, prefix)
        addToast({ title: "Downloaded!", type: "success" })
    }

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
        downloadFile(content, `transcription-${Date.now()}.${format}`)
        addToast({ title: `${format.toUpperCase()} downloaded`, type: "success" })
    }

    return (
        <Card className="animate-fade-in">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                        <Volume2 className="h-5 w-5 text-primary" />
                        Transcription
                        <span className="text-sm font-normal text-muted-foreground">
                            ({SUPPORTED_LANGUAGES.find(l => l.value === language)?.label})
                        </span>
                    </CardTitle>
                    <div className="flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" onClick={() => handleCopy(transcription)}><Copy className="mr-2 h-4 w-4" /> Copy</Button>
                        <Button variant="outline" size="sm" onClick={() => handleDownload(transcription, "transcription")}><Download className="mr-2 h-4 w-4" /> TXT</Button>
                        <Button variant="outline" size="sm" disabled={!segments.length} onClick={() => handleExport("srt")} title="SubRip subtitles">SRT</Button>
                        <Button variant="outline" size="sm" disabled={!segments.length} onClick={() => handleExport("vtt")} title="WebVTT subtitles">VTT</Button>
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
                                        setTranslation("")
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
                                        <Button variant="ghost" size="sm" onClick={() => handleDownload(translation, "translation")}><Download className="h-4 w-4" /></Button>
                                        <Button variant="ghost" size="sm" onClick={() => setTranslation("")}><X className="h-4 w-4" /></Button>
                                    </div>
                                </div>
                                <div className="rounded-lg bg-accent/10 border border-accent/20 p-4">
                                    <p className="whitespace-pre-wrap leading-relaxed">{translation}</p>
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
