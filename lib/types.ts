/**
 * Shared TypeScript types for the VoiceBird frontend.
 */

/** Which backend serves ASR/translation: self-hosted services or a cloud API. */
export type BackendMode = "sunbird" | "huggingface" | "local"
export type InputMode = "file" | "url" | "microphone"

export interface Language {
    value: string
    label: string
    flag: string
}
