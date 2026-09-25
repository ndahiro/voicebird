/**
 * Shared TypeScript types for the VoiceBird frontend.
 */

/** Which backend serves ASR: the self-hosted services or the Hugging Face API. */
export type BackendMode = "huggingface" | "local"
export type InputMode = "file" | "url" | "microphone"

export interface Language {
    value: string
    label: string
    flag: string
}
