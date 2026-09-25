/**
 * Saving transcript / translation text with the media's name.
 *
 * A browser cannot read the folder an uploaded file lives in, so the default
 * target is the browser's **Downloads** folder, always using the media-derived
 * filename (`interview.mp3` -> `interview.txt`) so the text is identifiable
 * next to the audio/video/image it came from.
 *
 * Optional: the "Save to folder" button grants one folder through the File
 * System Access API (Chrome/Edge) — point it at the media's folder and every
 * later transcript/translation is written straight into it instead of
 * Downloads. The handle is persisted in IndexedDB (the browser re-prompts for
 * permission on the first save after a reload).
 */

const DB_NAME = "voicebird"
const DB_STORE = "filesaves"
const SAVE_DIR_KEY = "save-directory"

export type SaveOutcome =
    | { status: "saved"; location: string } // written into the remembered folder
    | { status: "downloaded" } // written to the browser's Downloads folder
    | { status: "error"; message: string }

/** Strip characters that are illegal in file names and cap the length. */
export function sanitizeFileName(name: string): string {
    const cleaned = name
        .replace(/[\\/:*?"<>|]+/g, "-")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 180)
        .trim()
    return cleaned || "transcription"
}

/** True when a save folder can be granted (File System Access API). */
export function fsaSupported(): boolean {
    return typeof window !== "undefined" && "showDirectoryPicker" in window
}

function openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        if (typeof indexedDB === "undefined") {
            reject(new Error("IndexedDB unavailable"))
            return
        }
        const request = indexedDB.open(DB_NAME, 1)
        request.onupgradeneeded = () => {
            const db = request.result
            if (!db.objectStoreNames.contains(DB_STORE)) db.createObjectStore(DB_STORE)
        }
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error)
    })
}

async function idbGet<T>(key: string): Promise<T | null> {
    try {
        const db = await openDb()
        return await new Promise<T | null>((resolve, reject) => {
            const req = db.transaction(DB_STORE, "readonly").objectStore(DB_STORE).get(key)
            req.onsuccess = () => resolve((req.result as T) ?? null)
            req.onerror = () => reject(req.error)
        })
    } catch {
        return null // private mode / blocked storage — just don't remember
    }
}

async function idbSet(key: string, value: unknown): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const req = db.transaction(DB_STORE, "readwrite").objectStore(DB_STORE).put(value, key)
            req.onsuccess = () => resolve()
            req.onerror = () => reject(req.error)
        })
    } catch {
        // non-fatal: the folder simply won't be remembered
    }
}

async function idbDelete(key: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const req = db.transaction(DB_STORE, "readwrite").objectStore(DB_STORE).delete(key)
            req.onsuccess = () => resolve()
            req.onerror = () => reject(req.error)
        })
    } catch {
        // non-fatal
    }
}

interface PermissionCapableHandle {
    queryPermission?: (desc: { mode: "readwrite" }) => Promise<PermissionState>
    requestPermission?: (desc: { mode: "readwrite" }) => Promise<PermissionState>
}

/** The remembered save folder, but only while we hold permission for it. */
export async function getSaveDirectory(): Promise<FileSystemDirectoryHandle | null> {
    const handle = await idbGet<FileSystemDirectoryHandle & PermissionCapableHandle>(SAVE_DIR_KEY)
    if (!handle) return null
    try {
        const query = handle.queryPermission?.bind(handle)
        const request = handle.requestPermission?.bind(handle)
        if (!query) return handle
        let state = await query({ mode: "readwrite" })
        if (state === "prompt" && request) state = await request({ mode: "readwrite" })
        if (state === "granted") return handle
        await idbDelete(SAVE_DIR_KEY) // revoked or stale — stop writing into it
        return null
    } catch {
        return null
    }
}

/**
 * Let the user pick the folder transcripts should go to (ideally the folder
 * holding the media). Remembered for future saves.
 * Returns null when cancelled or unsupported.
 */
export async function pickSaveDirectory(): Promise<FileSystemDirectoryHandle | null> {
    if (!fsaSupported()) return null
    try {
        const dir = await (window as any).showDirectoryPicker({
            id: "voicebird-transcripts",
            mode: "readwrite",
        })
        await idbSet(SAVE_DIR_KEY, dir)
        return dir
    } catch {
        return null // cancelled
    }
}

async function writeIntoDirectory(
    dir: FileSystemDirectoryHandle,
    filename: string,
    text: string
): Promise<void> {
    const handle = await dir.getFileHandle(filename, { create: true })
    const writable = await handle.createWritable()
    await writable.write(text)
    await writable.close()
}

/**
 * Save ``text`` as ``filename``.
 *
 * Target: the remembered folder when one was granted, otherwise the browser's
 * Downloads folder. Either way the file keeps the media-derived ``filename``
 * so the text is identifiable next to the audio/video/image it belongs to.
 */
export async function saveText(text: string, filename: string): Promise<SaveOutcome> {
    const safeName = sanitizeFileName(filename)

    const dir = await getSaveDirectory()
    if (dir) {
        try {
            await writeIntoDirectory(dir, safeName, text)
            return { status: "saved", location: dir.name }
        } catch (err) {
            // NotAllowedError / NotFoundError / quota — still save it somewhere
            console.warn("Could not write to remembered folder; using Downloads:", err)
        }
    }

    try {
        // Plain download: the browser puts it in Downloads under our name.
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = safeName
        a.click()
        URL.revokeObjectURL(url)
        return { status: "downloaded" }
    } catch (err: any) {
        return { status: "error", message: err?.message || "Could not save the file" }
    }
}
