import { BackendMode, Language } from "./types"

// Environment variables - accessed directly for Next.js compatibility
// These are replaced at build time by Next.js
export const ENV = {
    SUNBIRD_API_URL: process.env.NEXT_PUBLIC_SUNBIRD_API_URL || "https://api.sunbird.ai",
    SUNBIRD_API_TOKEN: process.env.NEXT_PUBLIC_SUNBIRD_API_TOKEN || "",
    HF_API_TOKEN: process.env.NEXT_PUBLIC_HF_API_TOKEN || "",
    HF_ASR_MODEL: process.env.NEXT_PUBLIC_HF_ASR_MODEL || "Sunbird/sunbird-mms",
    LOCAL_SERVER_URL: process.env.NEXT_PUBLIC_LOCAL_SERVER_URL || "http://localhost:8100",
    LOCAL_TRANSLATION_URL: process.env.NEXT_PUBLIC_LOCAL_TRANSLATION_URL || "",
    DEFAULT_BACKEND: process.env.NEXT_PUBLIC_DEFAULT_BACKEND || "local",
}

export const resolveBackendMode = (value: string): BackendMode => {
    if (value === "sunbird" || value === "huggingface" || value === "local") {
        return value
    }
    // Safe fallback: local avoids external API dependency failures by default.
    return "local"
}

export const DEFAULT_BACKEND = resolveBackendMode(ENV.DEFAULT_BACKEND)

export const SUPPORTED_LANGUAGES: Language[] = [
    { value: "auto", label: "Auto Detect", flag: "🌍" },
    { value: "lug", label: "Luganda", flag: "🇺🇬" },
    { value: "ach", label: "Acholi", flag: "🇺🇬" },
    { value: "teo", label: "Ateso", flag: "🇺🇬" },
    { value: "lgg", label: "Lugbara", flag: "🇺🇬" },
    { value: "nyn", label: "Runyankole", flag: "🇺🇬" },
    { value: "xog", label: "Lusoga", flag: "🇺🇬" },
    { value: "ttj", label: "Rutooro", flag: "🇺🇬" },
    { value: "kin", label: "Kinyarwanda", flag: "🇷🇼" },
    { value: "myx", label: "Lumasaba", flag: "🇺🇬" },
    { value: "eng", label: "English", flag: "🇬🇧" },
    { value: "fra", label: "French", flag: "🇫🇷" },
    { value: "swa", label: "Kiswahili", flag: "🇹🇿" },
    { value: "ara", label: "Arabic", flag: "🇸🇦" },
    { value: "spa", label: "Spanish", flag: "🇪🇸" },
    { value: "deu", label: "German", flag: "🇩🇪" },
    { value: "zho", label: "Chinese", flag: "🇨🇳" },
    { value: "hin", label: "Hindi", flag: "🇮🇳" },
    { value: "rus", label: "Russian", flag: "🇷🇺" },
    { value: "por", label: "Portuguese", flag: "🇵🇹" },
    { value: "ita", label: "Italian", flag: "🇮🇹" },
]

export const SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "ogg", "m4a", "aac"]
export const SUPPORTED_VIDEO_FORMATS = ["mp4", "webm", "mov"]
export const ALL_SUPPORTED_FORMATS = [...SUPPORTED_AUDIO_FORMATS, ...SUPPORTED_VIDEO_FORMATS]
export const SUNBIRD_TRANSLATION_CODES = ["lug", "ach", "teo", "lgg", "nyn", "eng"]
export const LOCAL_TRANSLATION_LANGS = ["lug", "ach", "teo", "lgg", "nyn", "fra", "swa", "ara", "spa", "deu", "zho", "hin", "rus", "por", "ita"]

export const ASR_MODELS = [
    { value: "Sunbird/asr-whisper-large-v3-salt", label: "African Languages" },
    { value: "openai/whisper-large-v3", label: "OpenAI Whisper Large V3 (Global - Best Quality)" },
    { value: "openai/whisper-medium", label: "OpenAI Whisper Medium (Global - Faster)" },
    { value: "openai/whisper-small", label: "OpenAI Whisper Small (Global - Fastest)" },
];

export const DEFAULT_ASR_MODEL = ASR_MODELS[0].value;
