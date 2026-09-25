"""VoiceBird Local ASR Server
Runs Sunbird/asr-whisper-large-v3-salt model locally for African language transcription.
Supports file uploads and URL transcription (YouTube, TikTok, etc. via yt-dlp).

Usage:
    python server.py [--port 8100] [--host 0.0.0.0] [--device cuda|cpu]
    
Note: This model is gated. You must:
1. Request access at https://huggingface.co/Sunbird/asr-whisper-large-v3-salt
2. Set HF_TOKEN environment variable or run: huggingface-cli login

For URL transcription, install yt-dlp: pip install yt-dlp
"""

import argparse
import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import librosa
import numpy as np
import gc
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from functools import partial

# Load .env file from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import login, whoami
from pydantic import BaseModel
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import pipeline  # Standard pipeline

# Optional faster-whisper backend (CTranslate2) for ~4x faster inference
# Enable with ASR_BACKEND=faster-whisper (see README / requirements files)
try:
    import faster_whisper
    HAS_FASTER_WHISPER = True
    logger.info("faster-whisper (CTranslate2) backend is available.")
except ImportError:
    HAS_FASTER_WHISPER = False
    logger.info("faster-whisper not installed. Using standard Transformers backend (install faster-whisper to enable ASR_BACKEND=faster-whisper).")

# CTranslate2 is the runtime faster-whisper runs on. We also use it to convert
# custom HuggingFace Whisper fine-tunes (e.g. Sunbird SALT) that only ship
# PyTorch weights into the CTranslate2 format faster-whisper needs — this is
# what lets us borrow Buzz's fast CTranslate2 backend for the SALT model too.
try:
    from ctranslate2.converters import TransformersConverter
    HAS_CTRANSLATE2 = True
except ImportError:
    HAS_CTRANSLATE2 = False
    TransformersConverter = None

ASR_BACKEND = os.getenv("ASR_BACKEND", "transformers").strip().lower()

# Optional speech separation (demucs) and speaker diarization (pyannote.audio)
# — borrowed from Buzz (demucs vocals stem before transcription, pyannote
# speaker turns assigned to transcript segments). Both degrade gracefully when
# the package or a gated model is unavailable.
try:
    from demucs import api as demucs_api
    HAS_DEMUCS = True
except Exception:
    HAS_DEMUCS = False
    logger.info("demucs not installed — speech separation disabled (pip install demucs).")

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    HAS_PYANNOTE = True
except Exception:
    HAS_PYANNOTE = False
    logger.info("pyannote.audio not installed — speaker identification disabled (pip install pyannote.audio).")

# Optional OCR (on-screen text extraction from video): OpenCV samples frames
# and Tesseract reads any text rendered on screen (captions, slides,
# lower-thirds) so it can be shown as selectable text and translated alongside
# the spoken transcript. Needs the Python packages (opencv-python-headless,
# pytesseract) AND the tesseract system binary (`apt install tesseract-ocr`).
try:
    import cv2
    import pytesseract
    HAS_OCR = True
except Exception as e:
    HAS_OCR = False
    cv2 = None
    pytesseract = None
    logger.info("OCR disabled — install opencv-python-headless + pytesseract to enable it (%s).", e)

# OCR tuning: seconds between sampled frames, and a cap so a long video can't
# pin the CPU for hours.
OCR_FRAME_INTERVAL = float(os.getenv("OCR_FRAME_INTERVAL", "1.0"))
OCR_MAX_FRAMES = int(os.getenv("OCR_MAX_FRAMES", "300"))

SPEECH_SEPARATION = os.getenv("SPEECH_SEPARATION", "false").lower() == "true"
SPEECH_SEPARATION_DEVICE = os.getenv(
    "SPEECH_SEPARATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
DIARIZATION = os.getenv("DIARIZATION", "false").lower() == "true"
DIARIZATION_DEVICE = os.getenv("DIARIZATION_DEVICE", "cpu")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# Global model pipeline
asr_pipeline = None
CURRENT_MODEL_ID = None
MODEL_FALLBACK_MAP = {}

# Separate transformers pipeline kept ONLY for the SALT custom-language codes
# (lug/ach/teo/lgg/nyn/xog/ttj/kin/myx). faster-whisper can't be forced to those
# tokens (no named language code), so custom-language requests need the
# transformers backend where ``forced_decoder_ids`` works. The main ``asr_pipeline``
# stays on the fast faster-whisper backend for built-in languages.
_transformers_custom_pipeline = None
_TRANSFORMERS_CUSTOM_ID = None

# Concurrency control: serialize model access so multiple users queue up
# instead of racing / hanging. Only one transcription runs at a time.
_asr_lock = asyncio.Lock()
_asr_queue_size = 0          # how many requests are waiting or running
MAX_QUEUE_SIZE = 5           # reject requests beyond this to avoid OOM

# Live transcription sessions: keep the previous chunk's text so the next
# chunk can continue the transcript coherently (initial_prompt carry-over).
_live_sessions: dict = {}
_live_sessions_lock = threading.Lock()

# Transcription progress (polled by the frontend via /v1/audio/progress)
_progress = {"phase": "idle", "current": 0, "total": 1}

# Watch-folder auto-transcription state
_watch_stop: Optional[threading.Event] = None
_watch_thread: Optional[threading.Thread] = None
_seen_files: set = set()
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# File size limit for direct uploads (configurable via env var)
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "1024"))   # 1024 MB (1 GB) default

# Paragraph detection for the final transcript (pause-based, like Buzz's
# BUZZ_PARAGRAPH_SPLIT_TIME): a gap of at least PARAGRAPH_GAP_SECONDS between
# transcribed segments, or a run of audio silence of at least
# PARAGRAPH_SILENCE_SECONDS, starts a new paragraph.
PARAGRAPH_GAP_SECONDS = float(os.getenv("PARAGRAPH_GAP_SECONDS", "2.0"))
PARAGRAPH_SILENCE_SECONDS = float(os.getenv("PARAGRAPH_SILENCE_SECONDS", "1.5"))

# Default Model configuration
DEFAULT_MODEL_ID = os.getenv("ASR_MODEL_ID", "Sunbird/asr-whisper-large-v3-salt")
DEVICE = os.getenv("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Ignore placeholder values injected via sample env files. Some sample tokens use
# obvious text like `your_huggingface_token_here`, which causes a 401 loop when
# passed to `huggingface_hub.login()`. Users can still call `hf auth login`
# interactively; that session takes effect when HF_TOKEN is empty.
PLACEHOLDER_HF_TOKENS = {
    "your_huggingface_token_here",
    "paste_huggingface_token_here",
}


def _clean_hf_token(raw_value: Optional[str]) -> str:
    token = (raw_value or "").strip()
    if not token:
        return ""

    normalized = token.lower()
    if normalized in PLACEHOLDER_HF_TOKENS:
        return ""

    return token


HF_TOKEN = _clean_hf_token(os.getenv("HF_TOKEN"))

# Whether HF_TOKEN has been probed against the Hub yet (once per process).
_hf_token_checked = False


def _hf_login_if_needed() -> None:
    """Authenticate with Hugging Face once, dropping an unusable token.

    An expired token is not merely a failed login: ``from_pretrained`` and
    ``snapshot_download`` fall back to the ``HF_TOKEN`` environment variable, and
    Hugging Face answers 401 even for *public* repositories when a bad token is
    attached. That would fail every request on a machine whose models are
    already cached, so the token is validated first; if it is unusable it is
    dropped from the process environment and the server continues anonymously.
    """
    global HF_TOKEN, _hf_token_checked
    if _hf_token_checked or not HF_TOKEN:
        return
    _hf_token_checked = True

    try:
        account = whoami(token=HF_TOKEN)
        logger.info(
            "Hugging Face token accepted (account: %s).",
            account.get("name") if isinstance(account, dict) else account,
        )
    except Exception as e:
        logger.warning(
            "HF_TOKEN is not usable (%s). Dropping it and continuing anonymously — "
            "models already in the cache still load; gated models need a valid HF_TOKEN.",
            str(e)[:200],
        )
        HF_TOKEN = ""
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return

    try:
        login(token=HF_TOKEN)
    except Exception as e:
        logger.warning(
            "Storing the Hugging Face token failed (%s); it is still sent "
            "explicitly with each request.",
            str(e)[:200],
        )


def _is_complete_model_dir(model_dir: Path) -> bool:
    """Validate that local model files are complete enough to load."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            import json
            index_data = json.loads(index_path.read_text())
            shards = set(index_data.get("weight_map", {}).values())
            missing = [name for name in sorted(shards) if not (model_dir / name).exists()]
            if missing:
                logger.warning(
                    "Local model directory is incomplete at %s. Missing shard(s): %s",
                    model_dir,
                    ", ".join(missing),
                )
                return False
        except Exception as e:
            logger.warning(f"Failed to validate model index at {index_path}: {e}")
            return False
        return True

    # Non-sharded models typically ship this file.
    if (model_dir / "model.safetensors").exists():
        return True

    logger.warning(f"No model weights found in local directory: {model_dir}")
    return False


def _resolve_local_model_path(model_id: str) -> Optional[Path]:
    """
    Resolve a local on-disk model path if available.
    This avoids external downloads for preloaded Docker model volumes.
    """
    # Optional explicit override.
    explicit_model_path = os.getenv("ASR_MODEL_PATH", "").strip()
    if explicit_model_path:
        explicit = Path(explicit_model_path).expanduser()
        if explicit.exists() and _is_complete_model_dir(explicit):
            return explicit

    cache_root = Path.home() / ".cache" / "huggingface"

    # Project ships Sunbird as a local directory with this exact naming.
    slash_replaced = model_id.replace("/", "-")
    candidate_dirs = [
        cache_root / slash_replaced,
    ]

    for candidate in candidate_dirs:
        if candidate.exists() and _is_complete_model_dir(candidate):
            return candidate

    # Standard Hugging Face cache layout: hub/models--org--repo/snapshots/<sha>
    hub_model_dir = cache_root / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if hub_model_dir.exists():
        snapshots = sorted([p for p in hub_model_dir.iterdir() if p.is_dir()])
        if snapshots:
            latest_snapshot = snapshots[-1]
            if _is_complete_model_dir(latest_snapshot):
                return latest_snapshot

    return None


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def _unload_asr_pipeline():
    """Drop the loaded ASR pipeline and free its GPU memory.

    Handles both backends: the transformers pipeline wraps a torch nn.Module
    that has a ``to('cpu')`` method, but the faster-whisper (CTranslate2) model's
    ``.model`` is a ``ctranslate2._ext.Whisper`` object with no ``to()`` — calling
    it raised ``AttributeError`` (HTTP 500) on model switch / shutdown. CTranslate2
    releases its VRAM when the object is deleted, so just ``del`` it.
    """
    global asr_pipeline
    if asr_pipeline is None:
        return
    logger.info("Unloading current model...")
    if DEVICE == "cuda" and hasattr(asr_pipeline, "model"):
        model_obj = asr_pipeline.model
        # torch modules support device moves; CTranslate2 models do not.
        if hasattr(model_obj, "to"):
            try:
                model_obj.to("cpu")
            except Exception as e:
                logger.warning(f"Could not move model to CPU during unload: {e}")
        else:
            logger.info("CTranslate2 model: skipping .to('cpu'); releasing on delete.")
    # faster-whisper keeps an open CUDA context on the model; explicitly unload
    # it so the 8GB card is freed for the next model / other workloads.
    unload = getattr(asr_pipeline, "unload_model", None)
    if callable(unload):
        try:
            unload()
        except Exception as e:
            logger.warning(f"Model unload failed: {e}")
    del asr_pipeline
    asr_pipeline = None
    clear_gpu_memory()
    logger.info("Model unloaded.")
    

def log_memory_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def _update_progress(phase: str, current: int = 0, total: int = 1):
    """Update the global transcription progress dict (polled by the frontend)."""
    _progress["phase"] = phase
    _progress["current"] = current
    _progress["total"] = total


def _to_timestamp(seconds: float, ms_separator: str = ".") -> str:
    """Format seconds as HH:MM:SS.mmm (borrowed from Buzz's to_timestamp)."""
    ms = float(seconds) * 1000
    hr = int(ms / 3600000)
    ms -= hr * 3600000
    mn = int(ms / 60000)
    ms -= mn * 60000
    sec = int(ms / 1000)
    msec = int(ms - sec * 1000)
    return f"{hr:02d}:{mn:02d}:{sec:02d}{ms_separator}{msec:03d}"


def format_transcript(segments: list, text: str, output_format: str = "txt") -> str:
    """Render segments as TXT / SRT / VTT (port of Buzz's write_output)."""
    output_format = (output_format or "txt").lower()
    def _line_text(seg: dict) -> str:
        t = seg["text"]
        if seg.get("speaker"):
            t = f"[{seg['speaker']}] {t}"
        return t

    if output_format == "srt":
        lines = []
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(
                f"{_to_timestamp(seg['start'], ',')} --> {_to_timestamp(seg['end'], ',')}"
            )
            lines.append(_line_text(seg))
            lines.append("")
        return "\n".join(lines)
    if output_format == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            lines.append(f"{_to_timestamp(seg['start'])} --> {_to_timestamp(seg['end'])}")
            lines.append(_line_text(seg))
            lines.append("")
        return "\n".join(lines)
    # TXT (default)
    return text


# --- OCR: on-screen text extraction from video ------------------------------
# pytesseract imports fine even when the `tesseract` binary is missing, so the
# probe runs once at first use instead of failing every request with
# TesseractNotFoundError.
_ocr_probed = False
_ocr_ok = False


def _ocr_available() -> bool:
    global _ocr_probed, _ocr_ok
    if not HAS_OCR:
        return False
    if not _ocr_probed:
        try:
            pytesseract.get_tesseract_version()
            _ocr_ok = True
            logger.info("OCR enabled (tesseract %s).", pytesseract.get_tesseract_version())
        except Exception as e:
            _ocr_ok = False
            logger.warning(
                "OCR disabled: tesseract binary not usable (%s). "
                "Install it, e.g. `apt install tesseract-ocr` / `brew install tesseract`.",
                e,
            )
        _ocr_probed = True
    return _ocr_ok


def _ocr_frame(frame) -> str:
    """OCR one video frame. Small frames are upscaled first — Tesseract reads
    noticeably more text from larger glyphs."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    longest = max(height, width)
    if 0 < longest < 1600:
        scale = 1600 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray)
    if not text.strip():
        # psm 11 = sparse text — better for captions/lower-thirds over busy frames
        text = pytesseract.image_to_string(gray, config="--psm 11")
    return text


def ocr_video_frames(path: str, frame_interval: float) -> tuple:
    """Sample frames from ``path`` and OCR the text displayed on screen.

    Returns ``(text, frames_scanned)``. Identical readings are deduplicated —
    a caption held on screen for a minute produces one entry, not sixty.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for OCR: {os.path.basename(path)}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        step = max(1, int(round(fps * max(0.1, frame_interval))))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        expected = max(1, total_frames // step) if total_frames else 1
        pieces, seen = [], set()
        index = scanned = 0
        _update_progress("ocr", 0, expected)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                scanned += 1
                text = _ocr_frame(frame).strip()
                key = " ".join(text.split()).lower()
                if key and key not in seen:
                    seen.add(key)
                    pieces.append(text)
                _update_progress("ocr", min(scanned, expected), expected)
                if scanned >= OCR_MAX_FRAMES:
                    logger.info(f"OCR: stopping at OCR_MAX_FRAMES={OCR_MAX_FRAMES}")
                    break
            index += 1
        _update_progress("idle", 0, 1)
        logger.info(f"OCR complete: {scanned} frames scanned, {len(pieces)} produced text")
        return "\n\n".join(pieces), scanned
    finally:
        cap.release()


# --- Speech separation & speaker identification -----------------------------
# Borrowed from Buzz: demucs separates the vocals stem before transcription
# (better accuracy on noisy audio) and pyannote.audio assigns speaker turns to
# the transcript segments (speaker identification).
_diarization_pipeline = None
_diarization_lock = threading.Lock()


def _install_pyannote_plda_shim() -> None:
    """Make pyannote 4.x tolerate the missing (gated) community-1 PLDA.

    Background: the ``pyannote/speaker-diarization-3.1`` pipeline config uses
    AgglomerativeClustering, which never uses a PLDA model. But pyannote 4.x
    constructor defaults point ``plda`` at the *separately gated*
    ``pyannote/speaker-diarization-community-1`` repo, so ``Pipeline.from_pretrained
    (...3.1...)`` aborts with a 403 before it even starts, even when both 3.1
    models (speaker-diarization-3.1 + segmentation-3.0) were accepted on HF.

    Fix: wrap ``get_plda`` so a failed/gated PLDA load degrades to ``None``
    instead of raising. Harmless for the 3.1 config (PLDA is unused there) and
    still loads the real PLDA when access to community-1 exists.
    """
    import pyannote.audio.pipelines.speaker_diarization as _sd_module

    if getattr(_sd_module.get_plda, "_voicebird_shim", False):
        return  # already installed

    _original_get_plda = _sd_module.get_plda

    def _safe_get_plda(plda, token=None, cache_dir=None):
        try:
            return _original_get_plda(plda, token=token, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(
                "pyannote PLDA load skipped (%s). Only needed for VBx clustering; "
                "the speaker-diarization-3.1 pipeline uses AgglomerativeClustering.",
                str(e)[:200],
            )
            return None

    _safe_get_plda._voicebird_shim = True
    _sd_module.get_plda = _safe_get_plda


def separate_speech_file(audio_path: str, out_path: str) -> None:
    """Extract the vocals stem with demucs (Buzz-style pre-processing)."""
    _update_progress("separating", 0, 1)
    separator = demucs_api.Separator(device=SPEECH_SEPARATION_DEVICE, progress=False)
    _, separated = separator.separate_audio_file(Path(audio_path))
    demucs_api.save_audio(separated["vocals"], out_path, separator.samplerate)


def _get_diarization_pipeline():
    global _diarization_pipeline
    with _diarization_lock:
        if _diarization_pipeline is None:
            token = os.getenv("DIARIZATION_TOKEN", HF_TOKEN) or None
            _install_pyannote_plda_shim()
            _diarization_pipeline = PyannotePipeline.from_pretrained(
                DIARIZATION_MODEL, token=token
            )
            if DIARIZATION_DEVICE != "cpu" and torch.cuda.is_available():
                _diarization_pipeline.to(torch.device(DIARIZATION_DEVICE))
            logger.info("Speaker diarization pipeline loaded (%s)", DIARIZATION_MODEL)
        return _diarization_pipeline


def diarize_segments(audio_path: str, segments: list, words: list = None) -> list:
    """Assign a speaker label (SPEAKER_00, ...) to each transcript segment.

    Two strategies, mirroring Buzz:
    - Word-level (preferred): when the ASR produced word timestamps, each word
      is anchored to a pyannote speaker turn, mid-sentence flips are repaired by
      majority vote, and words are regrouped into speaker-labeled sentences
      (port of MahmoudAshraf97/whisper-diarization, which Buzz submodules).
    - Overlap fallback: assign each segment the speaker whose turn overlaps it
      the most. Segments with no overlap stay unlabeled.
    """
    _update_progress("diarizing", 0, 1)
    pipeline = _get_diarization_pipeline()
    output = pipeline(audio_path)
    # pyannote 3.x returns a plain Annotation; 4.x wraps it in a DiarizeOutput
    # object (with .speaker_diarization holding the Annotation). Support both.
    diarization = getattr(output, "speaker_diarization", output)
    turns = [
        (t.start, t.end, s) for t, _, s in diarization.itertracks(yield_label=True)
    ]
    if not turns:
        return segments

    if words:
        try:
            from diarization import diarize_words_to_segments

            labeled = diarize_words_to_segments(words, turns)
            if labeled:
                logger.info(
                    "Speaker identification: word-level assignment produced %d sentence segments",
                    len(labeled),
                )
                return labeled
            logger.warning(
                "Speaker identification: word-level assignment produced no segments; falling back to overlap"
            )
        except Exception as e:
            logger.warning(f"Speaker identification: word-level assignment failed ({e}); falling back to overlap")

    def _speaker_for(start: float, end: float):
        best, best_overlap = None, 0.0
        for t0, t1, s in turns:
            overlap = max(0.0, min(end, t1) - max(start, t0))
            if overlap > best_overlap:
                best, best_overlap = s, overlap
        return best if best_overlap > 0 else None

    for seg in segments:
        seg["speaker"] = _speaker_for(
            float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
        )
    return segments


# --- Punctuation & paragraph post-processing --------------------------------
# Whisper-family models already generate sentence punctuation (commas, periods)
# during decoding; this pass adds paragraph structure from pauses and polishes
# capitalization / terminal punctuation.

_SENTENCE_END = ".!?…"


def _find_silence_gaps(audio: np.ndarray, sr: int) -> list:
    """Return (start, end) second-ranges of quiet stretches long enough to
    separate paragraphs (frame RMS below 2% of the peak)."""
    min_gap = PARAGRAPH_SILENCE_SECONDS
    frame = int(0.02 * sr)  # 20 ms frames
    n = len(audio) // frame
    if n < 2:
        return []
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return []
    threshold = max(peak * 0.02, 1e-4)
    quiet = rms < threshold
    gaps = []
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            start_s = i * 0.02
            end_s = j * 0.02
            if end_s - start_s >= min_gap:
                gaps.append((start_s, end_s))
            i = j
        else:
            i += 1
    return gaps


def _polish_paragraph(p: str) -> str:
    """Capitalize the first letter and ensure terminal sentence punctuation."""
    p = p.strip()
    if not p:
        return p
    p = p[0].upper() + p[1:]
    if p[-1] not in _SENTENCE_END + '\u201d"':
        p += "."
    return p


def _sentence_boundaries(text: str) -> list:
    return [i for i, ch in enumerate(text) if ch in _SENTENCE_END]


def _nearest_boundary(bounds: list, target: int, length: int):
    """Nearest sentence-ending index within ~35% of the target position."""
    if not bounds:
        return None
    lo, hi = target - int(0.35 * length), target + int(0.35 * length)
    best = None
    for b in bounds:
        if lo <= b <= hi and (best is None or abs(b - target) < abs(best - target)):
            best = b
    return best


def _split_segment_at_silence(s: str, start: float, end: float, gaps: list) -> list:
    """Split one segment's text at long silences inside its time span.

    The Sunbird/transformers path often returns one coarse segment per 30s
    window without word timestamps, so sentence boundaries are matched to
    silence positions proportionally.
    """
    if end <= start or not gaps:
        return [s]
    span = end - start
    inside = [
        g
        for g in gaps
        if g[0] >= start and g[1] <= end and (g[1] - g[0]) >= PARAGRAPH_SILENCE_SECONDS
    ]
    if not inside:
        return [s]
    bounds = _sentence_boundaries(s)
    cuts = set()
    for gs, ge in inside:
        frac = ((gs + ge) / 2.0 - start) / span
        target = int(len(s) * frac)
        cut = _nearest_boundary(bounds, target, len(s))
        if cut is not None and 0 < cut < len(s) - 1:
            cuts.add(cut + 1)  # keep the punctuation with the preceding piece
    if not cuts:
        return [s]
    pieces = []
    prev = 0
    for cut in sorted(cuts):
        piece = s[prev:cut].strip()
        if piece:
            pieces.append(piece)
        prev = cut
    last = s[prev:].strip()
    if last:
        pieces.append(last)
    return pieces if pieces else [s]


def _silence_between(prev_end: float, start: float, gaps: list) -> bool:
    """True if a paragraph-length silence separates two adjacent segments."""
    mid = (prev_end + start) / 2.0
    for gs, ge in gaps:
        if gs <= mid <= ge or (gs >= prev_end and ge <= start):
            return True
    return False


def postprocess_text(
    segments: list,
    text: str,
    audio: Optional[np.ndarray] = None,
    sr: int = 16000,
) -> str:
    """Build the final transcript with pause-based paragraph breaks + polish.

    Falls back to the raw model text when no segments are available.
    """
    if not segments:
        return text
    gaps = _find_silence_gaps(audio, sr) if audio is not None else []
    parts = []  # (is_new_paragraph, piece)
    prev_end = None
    for seg in segments:
        s = (seg.get("text") or "").strip()
        if not s:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end < start:
            end = start

        break_before = False
        if prev_end is not None and (
            (start - prev_end) >= PARAGRAPH_GAP_SECONDS
            or _silence_between(prev_end, start, gaps)
        ):
            break_before = True

        for idx, piece in enumerate(_split_segment_at_silence(s, start, end, gaps)):
            parts.append((break_before if idx == 0 else True, piece))
            break_before = False
        prev_end = max(end, start)

    if not parts:
        return text
    out = ""
    for i, (is_break, piece) in enumerate(parts):
        out += piece if i == 0 else (("\n\n" if is_break else " ") + piece)
    paragraphs = [p for p in out.split("\n\n") if p]
    return "\n\n".join(_polish_paragraph(p) for p in paragraphs)


def transcribe_faster_whisper(
    model, audio: np.ndarray, generate_kwargs: dict, want_words: bool = False
) -> tuple:
    """Transcribe with the faster-whisper (CTranslate2) backend.

    faster-whisper runs a built-in Silero VAD filter (same idea as Buzz's
    silence-cut live transcription) and returns word/segment timestamps.

    Speed: we use Buzz's ``BatchedInferencePipeline`` so multiple 30s chunks
    decode concurrently on the GPU instead of one-at-a-time — this is the main
    source of Buzz's speed advantage over the sequential transformers pipeline.
    ``FASTER_WHISPER_BATCH_SIZE`` (default 16) controls the concurrency.

    Returns (text, segments, words). When ``want_words`` is set, ``words``
    carries [{start, end, text}] word timestamps used for Buzz-style
    word-level speaker diarization.
    """
    _update_progress("transcribing", 1, 1)
    language = generate_kwargs.get("language") or None
    initial_prompt = generate_kwargs.get("initial_prompt") or None
    beam_size = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "5"))
    batch_size = int(os.getenv("FASTER_WHISPER_BATCH_SIZE", "16"))
    vad_filter = os.getenv("FASTER_WHISPER_VAD_FILTER", "true").lower() == "true"

    # Buzz's batched pipeline: decode many chunks together on the GPU.
    # NOTE: BatchedInferencePipeline does not emit word-level timestamps, which
    # speaker diarization needs — so when words are requested we use the base
    # (sequential) model instead and keep the speedup for the normal case.
    if want_words:
        batched = model
    else:
        if getattr(model, "_voicebird_batched", None) is None:
            try:
                model._voicebird_batched = faster_whisper.BatchedInferencePipeline(model=model)
                logger.info("Using faster-whisper BatchedInferencePipeline (batch_size=%d).", batch_size)
            except Exception as e:
                logger.warning("BatchedInferencePipeline unavailable (%s); using sequential.", e)
                model._voicebird_batched = model
        batched = model._voicebird_batched

    def _run(lang):
        kwargs = dict(
            audio=audio,
            language=lang,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 400},
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            without_timestamps=False,
            word_timestamps=want_words,
            # Buzz's anti-hallucination knob: skip near-silent chunks early.
            no_speech_threshold=0.4,
        )
        # batch_size is only accepted by Buzz's BatchedInferencePipeline, not
        # the base WhisperModel (used when word timestamps are requested).
        if batched is not model:
            kwargs["batch_size"] = batch_size
        return batched.transcribe(**kwargs)

    try:
        segments_iter, info = _run(language)
    except ValueError as e:
        # African fine-tune languages (acholi, ateso, ...) aren't in the stock
        # whisper language list even though the model has their tokens — fall
        # back to auto-detection rather than failing the request.
        logger.warning(
            f"faster-whisper rejected language {language!r} ({e}); falling back to auto-detect"
        )
        segments_iter, info = _run(None)
    parts = []
    segments = []
    words = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        parts.append(text)
        segments.append(
            {"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "text": text}
        )
        if want_words:
            for w in (seg.words or []):
                wt = (w.word or "").strip()
                if wt:
                    words.append({"start": float(w.start), "end": float(w.end), "text": wt})
    return " ".join(parts), segments, words


def run_transcription(
    pipeline_instance,
    audio: np.ndarray,
    generate_kwargs: dict,
    sr: int = 16000,
    pipeline_kwargs: Optional[dict] = None,
    want_words: bool = False,
) -> tuple:
    """Dispatch to the right transcriber based on the loaded model type.
    Returns (text, segments, words)."""
    use_fw = HAS_FASTER_WHISPER and isinstance(pipeline_instance, faster_whisper.WhisperModel)
    if use_fw and CURRENT_MODEL_ID not in _CT2_BROKEN:
        try:
            return transcribe_faster_whisper(
                pipeline_instance, audio, generate_kwargs, want_words=want_words
            )
        except Exception as e:
            # Custom fine-tunes (e.g. the SALT model) can fail under faster-whisper
            # at transcription time (its CTranslate2 conversion drops the 128-mel
            # feature config). Fall back to the transformers pipeline so the request
            # still succeeds, and remember not to retry CT2 for this model.
            logger.warning(
                "faster-whisper transcription failed (%s); falling back to "
                "transformers backend for %s.",
                e, CURRENT_MODEL_ID,
            )
            _CT2_BROKEN.add(CURRENT_MODEL_ID)

    # transformers path: either natively, or after a faster-whisper failure
    if use_fw:
        pipeline_instance = _build_transformers_pipeline(CURRENT_MODEL_ID)
        # Cache it so subsequent requests reuse it instead of reloading the model.
        global asr_pipeline
        asr_pipeline = pipeline_instance
    return transcribe_audio_chunks(
        pipeline_instance, audio, generate_kwargs, sr, pipeline_kwargs, want_words=want_words
    )


async def _transcribe_bytes(
    content: bytes,
    file_ext: str,
    filename: str,
    language: Optional[str],
    model: Optional[str],
    initial_prompt: Optional[str] = None,
    separate_speech: bool = False,
    diarize: bool = False,
) -> dict:
    """Shared transcription core used by file upload, URL and live endpoints.

    Callers are responsible for holding ``_asr_lock``. Returns
    ``{"text": ..., "segments": [...], "speakers": bool}``.
    """
    loop = asyncio.get_event_loop()

    # Custom SALT languages (lug/ach/teo/lgg/nyn/xog/ttj/kin/myx) have no named
    # faster-whisper language code, so they can't be forced there. Route them to
    # the transformers pipeline (cached) where ``forced_decoder_ids`` applies the
    # correct token. Built-in languages stay on the fast faster-whisper backend.
    lang_full = LANGUAGE_MAP.get(language, language) if language else None
    is_custom_lang = bool(
        language
        and language in LANGUAGE_TOKEN_IDS
        and (lang_full not in WHISPER_SUPPORTED_LANGUAGES)
    )
    if is_custom_lang:
        logger.info(
            "Custom SALT language %s -> transformers backend (forced_decoder_ids).",
            language,
        )
        pipeline_instance = await loop.run_in_executor(
            None, partial(get_custom_lang_pipeline, model)
        )
    else:
        pipeline_instance = await loop.run_in_executor(None, partial(get_asr_pipeline, model))
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    separation_path = None
    try:
        # Speech separation (demucs) before decoding — better accuracy on noise
        if (separate_speech or SPEECH_SEPARATION) and HAS_DEMUCS:
            separation_path = f"{tmp_path}.vocals.mp3"
            try:
                await loop.run_in_executor(
                    None, partial(separate_speech_file, tmp_path, separation_path)
                )
                asr_audio_path = separation_path
                logger.info("Speech separation applied (demucs vocals stem).")
                # demucs is a second model sharing this GPU; its tensors are
                # freed when separate_speech_file returns, but the PyTorch
                # caching allocator keeps the reserved blocks. Return them to
                # the driver before Whisper decodes so an 8GB card doesn't OOM.
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"Speech separation failed, using original audio: {e}")
                separation_path = None
                asr_audio_path = tmp_path
        else:
            asr_audio_path = tmp_path

        lang_full = LANGUAGE_MAP.get(language, language) if language else None
        generate_kwargs = {}
        if language and language in LANGUAGE_TOKEN_IDS:
            logger.info(f"Using forced language token for {language}: {LANGUAGE_TOKEN_IDS[language]}")
            generate_kwargs["forced_decoder_ids"] = [[1, LANGUAGE_TOKEN_IDS[language]]]
        elif lang_full and lang_full in WHISPER_SUPPORTED_LANGUAGES:
            generate_kwargs["language"] = lang_full
        # Live-session context: pass the previous slice's text back as a prompt
        pipeline_kwargs = {}
        is_fw = HAS_FASTER_WHISPER and isinstance(pipeline_instance, faster_whisper.WhisperModel)
        if initial_prompt:
            if is_fw:
                # faster-whisper accepts the prompt string directly
                generate_kwargs["initial_prompt"] = initial_prompt
            else:
                # transformers: use the pipeline-level `prompt` kwarg (prompt_ids
                # in generate_kwargs hits a numpy/Tensor bug in transformers)
                pipeline_kwargs["prompt"] = initial_prompt
                logger.info(
                    "Live session: carrying %d chars of prior context as prompt",
                    len(initial_prompt),
                )

        # Anti-hallucination guard (borrowed from Buzz): stop the model from
        # looping on its own previous text over silence/music.
        if not is_fw and os.getenv("ANTI_HALLUCINATION", "true").lower() == "true":
            pipeline_kwargs["condition_on_previous_text"] = False
            rep = float(os.getenv("REPETITION_PENALTY", "1.0"))
            if rep > 1.0:
                generate_kwargs["repetition_penalty"] = rep
            logger.info("Anti-hallucination guard: condition_on_previous_text=False")

        logger.info(f"Transcribing: {filename}, language: {lang_full or 'auto'}")

        try:
            audio, _ = await loop.run_in_executor(
                None, partial(librosa.load, asr_audio_path, sr=16000)
            )
        except Exception as e:
            logger.error(f"Failed to load audio with librosa: {e}")
            raise RuntimeError(f"Could not decode audio file: {e}")

        duration_minutes = (len(audio) / 16000) / 60
        logger.info(f"Audio duration: {duration_minutes:.1f} min")

        # Speaker ID needs word timestamps (Buzz-style word-level assignment), so
        # request them from the transcriber when diarization is enabled.
        want_words = bool(diarize or DIARIZATION)

        def _run_inference():
            return run_transcription(
                pipeline_instance,
                audio,
                generate_kwargs,
                pipeline_kwargs=pipeline_kwargs,
                want_words=want_words,
            )

        text, segments, words = await loop.run_in_executor(None, _run_inference)

        # The fine-tuned model sometimes emits stray leading punctuation
        # (e.g. ". " or '".') — strip it from the text and every segment
        _strip_prefix = lambda v: re.sub(r"^[\s.\"'\u2018\u2019\u201c\u201d]+\s*", "", v or "")
        text = _strip_prefix(text)
        for s in segments:
            s["text"] = _strip_prefix(s["text"])

        # Speaker identification (pyannote) after transcription. Runs on the
        # ORIGINAL audio (not the separated vocals) so multiple speakers are
        # still detected even when demucs separation was applied. When word
        # timestamps are available, labels are assigned per word and regrouped
        # into sentences (Buzz-style); otherwise segments are labeled by overlap.
        speakers = False
        if (diarize or DIARIZATION) and HAS_PYANNOTE:
            try:
                segments = await loop.run_in_executor(
                    None, partial(diarize_segments, tmp_path, segments, words)
                )
                n_labeled = sum(1 for s in segments if s.get("speaker"))
                speakers = n_labeled > 0
                logger.info(
                    f"Speaker identification applied: {n_labeled}/{len(segments)} segments labeled"
                )
            except Exception as e:
                logger.warning(f"Speaker identification failed: {e}")

        # Punctuation & paragraphs: whisper already emits sentence punctuation;
        # this pass adds pause-based paragraph breaks and polishes casing/endings.
        text = postprocess_text(segments, text, audio, 16000)

        logger.info(f"Transcription complete: {len(text)} characters, {len(segments)} segments")
        clear_gpu_memory()
        return {"text": text, "segments": segments, "speakers": speakers}
    finally:
        for path in (separation_path, tmp_path):
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass
        # Always release GPU memory — including on failed requests. Without
        # this, a single OOM/triton error leaves the caching allocator holding
        # nearly the whole card and every later request keeps failing.
        clear_gpu_memory()


# Language mapping for the model
LANGUAGE_MAP = {
    "auto": None,
    "lug": "luganda",
    "ach": "acholi", 
    "teo": "ateso",
    "lgg": "lugbara",
    "nyn": "runyankole",
    "xog": "lusoga",
    "ttj": "rutooro",
    "kin": "kinyarwanda",
    "myx": "lumasaba",
    "eng": "english",
    "fra": "french",
    "swa": "swahili",
    "ara": "arabic",
    "spa": "spanish",
}

# Specific token IDs for Sunbird/Salt model languages
LANGUAGE_TOKEN_IDS = {
    'eng': 50259,  # English (Ugandan)
    'swa': 50318,  # Swahili
    'ach': 50357,  # Acholi
    'lgg': 50356,  # Lugbara
    'lug': 50355,  # Luganda
    'nyn': 50354,  # Runyankole
    'teo': 50353,  # Ateso
    'xog': 50352,  # Lusoga
    'ttj': 50351,  # Rutooro
    'kin': 50350,  # Kinyarwanda
    'myx': 50349,  # Lumasaba
}

# Languages supported by the base Whisper tokenizer
# African languages from Sunbird are handled by the fine-tuned model without explicit language param
WHISPER_SUPPORTED_LANGUAGES = {
    "english", "chinese", "german", "spanish", "russian", "korean", "french",
    "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic",
    "swedish", "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew",
    "ukrainian", "greek", "malay", "czech", "romanian", "danish", "hungarian",
    "tamil", "norwegian", "thai", "urdu", "croatian", "bulgarian", "lithuanian",
    "latin", "maori", "malayalam", "welsh", "slovak", "telugu", "persian", "latvian",
    "bengali", "serbian", "azerbaijani", "slovenian", "kannada", "estonian",
    "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian",
    "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi",
    "sinhala", "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan",
    "georgian", "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish",
    "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen", "nynorsk",
    "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan", "tagalog",
    "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir",
    "javanese", "sundanese", "cantonese", "burmese",
}


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    segments: Optional[List[Segment]] = None


class UrlTranscriptionRequest(BaseModel):
    url: str
    language: Optional[str] = None
    model: Optional[str] = None
    separate_speech: bool = False
    diarize: bool = False


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lazy loading - model loads on first request, cleanup on shutdown."""
    global _watch_stop, _watch_thread, _event_loop

    logger.info("ASR server starting (lazy loading enabled)")
    logger.info(f"Model will load on first transcription request: {DEFAULT_MODEL_ID}")
    logger.info(f"Using device: {DEVICE}")
    
    # Don't load model on startup - wait for first request (lazy loading)
    logger.info("Server ready. Model will load when first transcription is requested.")

    # Optional watch folder: auto-transcribe new audio files dropped into a dir
    _event_loop = asyncio.get_running_loop()
    watch_folder = os.getenv("WATCH_FOLDER", "").strip()
    if watch_folder:
        try:
            Path(watch_folder).mkdir(parents=True, exist_ok=True)
            _watch_stop = threading.Event()
            _watch_thread = threading.Thread(target=_watch_loop, args=(watch_folder,), daemon=True)
            _watch_thread.start()
            logger.info(f"Watch folder enabled: {watch_folder}")
        except Exception as e:
            logger.error(f"Failed to start watch folder: {e}")
    
    yield

    # Stop the watch thread before shutdown
    if _watch_thread is not None:
        _watch_stop.set()
        _watch_thread.join(timeout=5)
        logger.info("Watch folder stopped.")

    # Cleanup - properly release GPU memory
    logger.info("Shutting down and releasing memory...")
    _unload_asr_pipeline()

    # Clear GPU cache and run garbage collection
    clear_gpu_memory()
    logger.info("Cleanup complete")
    log_memory_usage()


app = FastAPI(
    title="VoiceBird Local ASR Server",
    description="Local speech-to-text server for African languages using Sunbird Whisper model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ct2_converted_dir(name: str, compute_type: str) -> Path:
    """Where a one-time CTranslate2 conversion of ``name`` is cached.

    ``name`` is the logical model id (e.g. ``Sunbird/asr-whisper-large-v3-salt``)
    so the cache is stable regardless of which local snapshot path happens to be
    resolved on disk.
    """
    return Path.home() / ".cache" / "huggingface" / "ctranslate2" / (
        name.replace("/", "--") + f"-{compute_type}"
    )


def _ensure_ct2_model(source: str, compute_type: str, logical_name: str = None) -> str:
    """Return a CTranslate2 model path for ``source``.

    faster-whisper can only load CTranslate2 weights (a ``model.bin``). We only
    *convert* when ``source`` is a local HuggingFace/PyTorch checkpoint (e.g. the
    Sunbird SALT fine-tune, which ships no pre-converted CTranslate2 weights) —
    Buzz ships these pre-converted, we do the equivalent one-time conversion on
    first use. For a plain HF repo id we return it unchanged and let
    faster-whisper download the pre-converted (Systran) weights itself; if that
    fails, ``get_asr_pipeline``'s load fallback drops to the transformers
    pipeline. Conversion is CPU-only so it never disturbs a loaded GPU model.
    """
    src = Path(source)
    if (src / "model.bin").exists() or (
        src.is_dir() and any(src.glob("*.bin"))
    ):
        return source  # already a CTranslate2 model

    # Local PyTorch checkpoint -> convert once (custom fine-tunes without CT2 weights).
    if src.is_dir() and (src / "config.json").exists():
        if not HAS_CTRANSLATE2:
            raise RuntimeError(
                "Model is a PyTorch checkpoint and ctranslate2 is not installed, "
                "so it cannot be converted for the faster-whisper backend."
            )
        cache_name = logical_name or source
        out_dir = _ct2_converted_dir(cache_name, compute_type)
    else:
        # A plain HF repo id (or other non-local source): let faster-whisper
        # download/resolve its pre-converted weights; get_asr_pipeline's load
        # fallback handles the case where none exist (e.g. custom fine-tunes).
        return source

    if out_dir.exists() and (out_dir / "model.bin").exists():
        logger.info("Using cached CTranslate2 conversion: %s", out_dir)
        # Make sure sidecar configs (128-mel preprocessor, SALT tokenizer) are
        # present even for a pre-existing cache, in case they were missed before.
        _copy_ct2_sidecar_configs(source, str(out_dir))
        return str(out_dir)

    logger.info(
        "Converting %s to CTranslate2 (%s). This is a one-time CPU conversion and "
        "may take a few minutes; the result is cached at %s.",
        source, compute_type, out_dir,
    )
    # Force CPU for the conversion trace so we don't touch the ASR GPU.
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        converter = TransformersConverter(str(source))
        converter.convert(
            str(out_dir),
            quantization=compute_type,
            force=True,
        )
    finally:
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
    logger.info("CTranslate2 conversion complete: %s", out_dir)

    # The CTranslate2 converter copies the weights but NOT the feature-extractor
    # config, so faster-whisper defaults to 80 mel bins and its encoder rejects
    # the 128-mel features this (large-v3) model expects — that mismatch is what
    # used to make SALT fall back to the slow transformers backend. Copy the
    # source preprocessor_config.json (feature_size=128) so faster-whisper builds
    # matching features. Also carry the tokenizer so the fine-tune's custom
    # (African-language) tokens decode correctly instead of falling back to
    # whisper-tiny's tokenizer.
    _copy_ct2_sidecar_configs(source, out_dir)

    return str(out_dir)


def _copy_ct2_sidecar_configs(source: str, out_dir: str) -> None:
    """Copy the feature-extractor + tokenizer configs next to a converted CT2
    model so faster-whisper loads it correctly.

    Only relevant for local PyTorch checkpoints (custom fine-tunes). Plain HF
    repo ids are returned unchanged by _ensure_ct2_model and handled by
    faster-whisper's built-in asset download.
    """
    src = Path(source)
    out = Path(out_dir)
    if not src.is_dir():
        return

    # Feature extractor: feature_size (mel bins) MUST match the model or the
    # encoder raises "Invalid input features shape: expected (1, 128, ...)".
    pre = src / "preprocessor_config.json"
    if pre.exists() and not (out / "preprocessor_config.json").exists():
        try:
            import shutil
            shutil.copy(pre, out / "preprocessor_config.json")
            logger.info("Copied preprocessor_config.json (128-mel) into CT2 dir.")
        except Exception as e:
            logger.warning(f"Failed to copy preprocessor_config.json: {e}")

    # Tokenizer: build a tokenizers.Tokenizer-compatible tokenizer.json from the
    # HF assets so SALT's added language tokens survive. faster-whisper only
    # reads tokenizer.json (or falls back to whisper-tiny), so build it when the
    # source ships the raw vocab/merges instead.
    if not (out / "tokenizer.json").exists() and (src / "vocab.json").exists():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(src))
            tok.save_pretrained(str(out))
            logger.info("Built tokenizer.json for CT2 dir from SALT assets.")
        except Exception as e:
            logger.warning(
                f"Could not build tokenizer.json ({e}); faster-whisper will fall "
                "back to whisper-tiny tokenizer (same shared vocab, but SALT's "
                "added tokens may not decode correctly)."
            )


# Models that fail at transcription time under faster-whisper (e.g. custom
# large-v3 fine-tunes whose CTranslate2 conversion loses the 128-mel feature
# config). Once flagged we stop converting/retrying them and serve them via the
# transformers pipeline instead, so requests still succeed.
_CT2_BROKEN = set()


def _build_transformers_pipeline(model_id: str):
    """Build a transformers ASR pipeline for ``model_id`` (bypasses the cache)."""
    local_path = _resolve_local_model_path(model_id)
    source = str(local_path) if local_path else model_id
    using_local = local_path is not None
    if HF_TOKEN and not using_local:
        _hf_login_if_needed()
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=source,
        torch_dtype=torch_dtype,
        device=DEVICE,
        token=HF_TOKEN if HF_TOKEN and not using_local else None,
    )


def get_asr_pipeline(model_id: str = None):
    """
    Get or load the ASR pipeline for the specified model.
    Dynamically switches models if a different one is requested.
    """
    global asr_pipeline, CURRENT_MODEL_ID
    
    # Use default if no model specified
    if not model_id or model_id == "default":
        model_id = DEFAULT_MODEL_ID

    # If this model previously failed and was mapped to a fallback, reuse it.
    if model_id in MODEL_FALLBACK_MAP:
        mapped_model = MODEL_FALLBACK_MAP[model_id]
        logger.warning(
            "Model %s is mapped to fallback model %s due to previous load failure.",
            model_id,
            mapped_model,
        )
        model_id = mapped_model
    
    # Check if already loaded
    if asr_pipeline is not None and CURRENT_MODEL_ID == model_id:
        return asr_pipeline
        
    logger.info(f"Switching model from {CURRENT_MODEL_ID} to {model_id}...")
    
    # Unload current model
    _unload_asr_pipeline()

    # Load new model
    try:
        logger.info(f"Loading new model: {model_id}")
        local_model_path = _resolve_local_model_path(model_id)
        model_source = str(local_model_path) if local_model_path else model_id
        using_local_model = local_model_path is not None
        if using_local_model:
            logger.info(f"Using local model files from: {local_model_path}")

        if DEVICE == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        def _load_faster_whisper(source: str):
            compute_type = os.getenv(
                "FASTER_WHISPER_COMPUTE_TYPE", "int8_float16" if DEVICE == "cuda" else "int8"
            )
            # Custom fine-tunes (SALT) need a one-time CTranslate2 conversion;
            # standard repos (openai/whisper-*) already have CT2 weights.
            ct2_source = _ensure_ct2_model(source, compute_type, model_id)
            return faster_whisper.WhisperModel(
                ct2_source,
                device=DEVICE,
                compute_type=compute_type,
                cpu_threads=max(1, (os.cpu_count() or 4) // 2),
                download_root=str(Path.home() / ".cache" / "huggingface" / "faster-whisper"),
                **({"token": HF_TOKEN} if HF_TOKEN and not using_local_model else {}),
            )

        def _load_pipeline(source: str, local_only: bool):
            # Authenticate only when remote fetch is expected.
            if HF_TOKEN and not local_only:
                _hf_login_if_needed()

            # Note: `local_files_only` is intentionally omitted here. Newer
            # transformers versions reject it as an unused model_kwarg in the
            # pipeline constructor, and when a local model path is used the
            # weights are loaded from disk anyway.
            return pipeline(
                "automatic-speech-recognition",
                model=source,
                torch_dtype=torch_dtype,
                device=DEVICE,
                token=HF_TOKEN if HF_TOKEN and not local_only else None,
            )

        if ASR_BACKEND == "faster-whisper":
            if not HAS_FASTER_WHISPER:
                raise RuntimeError(
                    "ASR_BACKEND=faster-whisper requested but faster-whisper is not installed. "
                    "Install it (see requirements files) or set ASR_BACKEND=transformers."
                )
            try:
                logger.info(f"Loading {model_id} with faster-whisper (CTranslate2)...")
                asr_pipeline = _load_faster_whisper(model_source)
            except Exception as fw_err:
                # The SALT fine-tune (or any custom model) may fail to load under
                # CTranslate2 — fall back to the transformers pipeline so requests
                # still succeed instead of hard-failing the whole service.
                logger.warning(
                    "faster-whisper load failed (%s); falling back to transformers backend.",
                    fw_err,
                )
                asr_pipeline = _load_pipeline(model_source, using_local_model)
        else:
            try:
                asr_pipeline = _load_pipeline(model_source, using_local_model)
            except Exception as local_model_error:
                if using_local_model:
                    logger.warning(
                        "Local model load failed (%s). Retrying with remote model id %s",
                        local_model_error,
                        model_id,
                    )
                    asr_pipeline = _load_pipeline(model_id, False)
                else:
                    raise

        # Anti-hallucination guard: applied per-request via condition_on_previous_text
        # (see _transcribe_bytes). Noting here that transformers 4.57 removed
        # no_speech_threshold from WhisperConfig, so the per-request knob is used.

        CURRENT_MODEL_ID = model_id
        logger.info(f"Model {model_id} loaded successfully!")
        log_memory_usage()
        return asr_pipeline
        
    except Exception as e:
        fallback_model_id = os.getenv("ASR_FALLBACK_MODEL_ID", "openai/whisper-small").strip()
        if fallback_model_id and model_id != fallback_model_id:
            logger.warning(
                "Failed to load model %s: %s. Trying fallback model %s",
                model_id,
                e,
                fallback_model_id,
            )
            try:
                asr_pipeline = _load_pipeline(fallback_model_id, False)
                MODEL_FALLBACK_MAP[model_id] = fallback_model_id
                CURRENT_MODEL_ID = fallback_model_id
                logger.warning(
                    "Using fallback model %s for requests targeting %s",
                    fallback_model_id,
                    model_id,
                )
                log_memory_usage()
                return asr_pipeline
            except Exception as fallback_error:
                logger.error(
                    "Fallback model %s failed after primary model %s error: %s",
                    fallback_model_id,
                    model_id,
                    fallback_error,
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Failed to load model {model_id}: {str(e)} | "
                        f"Fallback {fallback_model_id} also failed: {str(fallback_error)}"
                    ),
                )

        logger.error(f"Failed to load model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}: {str(e)}")



def get_custom_lang_pipeline(model_id: str = None):
    """Return a cached transformers ASR pipeline for SALT custom-language codes.

    Used when ``ASR_BACKEND=faster-whisper`` but the requested language is one of
    SALT's custom tokens (lug/ach/teo/lgg/nyn/xog/ttj/kin/myx) that faster-whisper
    cannot force. The transformers backend honors ``forced_decoder_ids`` so the
    right language token is applied. Kept separate from the main ``asr_pipeline``
    so built-in languages keep using the fast CTranslate2 backend.
    """
    global _transformers_custom_pipeline, _TRANSFORMERS_CUSTOM_ID
    if not model_id or model_id == "default":
        model_id = DEFAULT_MODEL_ID
    if _transformers_custom_pipeline is not None and _TRANSFORMERS_CUSTOM_ID == model_id:
        return _transformers_custom_pipeline
    logger.info(f"Loading transformers pipeline for custom SALT languages: {model_id}")
    _transformers_custom_pipeline = _build_transformers_pipeline(model_id)
    _TRANSFORMERS_CUSTOM_ID = model_id
    logger.info("Transformers (custom-language) pipeline loaded.")
    log_memory_usage()
    return _transformers_custom_pipeline


def _collect_segments(result: dict, base_offset: float = 0.0) -> list:
    """Pull timestamped chunks out of a transformers pipeline result.

    Whisper occasionally returns an open-ended (start, None) timestamp on the
    final chunk when the audio is cut off mid-word; fall back to start = end.
    """
    segments = []
    for ch in result.get("chunks", []) or []:
        ts = ch.get("timestamp")
        start, end = ts if isinstance(ts, (tuple, list)) else (None, None)
        text = (ch.get("text") or "").strip()
        if start is None or not text:
            continue
        if end is None:
            end = start
        segments.append(
            {
                "start": round(base_offset + float(start), 3),
                "end": round(base_offset + float(end), 3),
                "text": text,
            }
        )
    return segments


def _window_segments(result: dict, base_offset: float, window_seconds: float, text: str) -> list:
    """Segments for one transcribed window, with sane fallbacks when the model
    emits no timestamps (the Sunbird fine-tune often leaves the end open)."""
    segs = _collect_segments(result, base_offset=base_offset)
    if not segs and text:
        segs = [
            {
                "start": round(base_offset, 3),
                "end": round(base_offset + window_seconds, 3),
                "text": text,
            }
        ]
    else:
        # Clamp open-ended segments to the window end so SRT/VTT exports are valid
        for s in segs:
            if s["end"] <= s["start"]:
                s["end"] = round(base_offset + window_seconds, 3)
    return segs


def _collect_words(result: dict, base_offset: float = 0.0) -> list:
    """Pull word-level chunks out of a transformers pipeline result that was
    called with ``return_timestamps="word"``."""
    words = []
    for ch in result.get("chunks", []) or []:
        ts = ch.get("timestamp")
        start, end = ts if isinstance(ts, (tuple, list)) else (None, None)
        text = (ch.get("text") or "").strip()
        if start is None or not text:
            continue
        if end is None:
            end = start
        words.append(
            {
                "start": round(base_offset + float(start), 3),
                "end": round(base_offset + float(end), 3),
                "text": text,
            }
        )
    return words


def _words_to_segments(words: list, text: str) -> list:
    """Group word timestamps into sentence-level segments.

    Fallback used when word timestamps were requested (speaker ID enabled) but
    the diarization pipeline is unavailable: words are grouped on sentence
    punctuation so the transcript stays readable without raw per-word chunks.
    """
    if not words:
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []
    segments = []
    cur = {"start": words[0]["start"], "end": words[0]["end"], "text": words[0]["text"]}
    for w in words[1:]:
        prev_end = cur["end"]
        cur["end"] = w["end"]
        if cur["text"].endswith((" ", "-", "—")):
            cur["text"] += w["text"]
        else:
            cur["text"] += " " + w["text"]
        if cur["text"].rstrip()[-1:] in _SENTENCE_END or (w["start"] - prev_end) > 1.0:
            segments.append(cur)
            cur = {"start": w["start"], "end": w["end"], "text": w["text"]}
    if cur["text"]:
        segments.append(cur)
    return segments


def transcribe_audio_chunks(
    pipeline_instance,
    audio: np.ndarray,
    generate_kwargs: dict,
    sr: int = 16000,
    pipeline_kwargs: Optional[dict] = None,
    want_words: bool = False,
) -> tuple:
    """
    Transcribe audio in explicit 30s windows with a small overlap.

    Windowing ourselves (instead of the pipeline's internal chunking) guarantees
    timestamped segments on every model, keeps long audio from exhausting GPU
    memory, and makes per-window progress genuinely reportable.

    Returns (text, segments, words). When ``want_words`` is set the pipeline is
    called with ``return_timestamps="word"`` and ``words`` carries
    [{start, end, text}] word timestamps for Buzz-style speaker diarization;
    ``segments`` are then grouped from the words (sentence-level fallback).
    """
    pipeline_kwargs = pipeline_kwargs or {}
    total_samples = len(audio)
    window_samples = 30 * sr
    overlap_samples = 2 * sr  # 2-second overlap to avoid cutting words at boundaries
    all_segments = []
    all_words = []
    parts = []

    # Word-level timestamps make Whisper collect fp32 cross-attentions for all
    # decoder layers (output_attentions=True), needing ~4.5GB on top of the
    # model — it OOMs on an 8GB card (CUDA out of memory, 500). Only request
    # them when there's room; otherwise diarization falls back to segment-
    # overlap labeling and transcription still succeeds.
    use_word_ts = want_words
    if want_words and torch.cuda.is_available():
        free_mb, _ = torch.cuda.mem_get_info()
        if free_mb < 6 * 1024**3:
            logger.warning(
                "Word timestamps need ~4.5GB extra VRAM but only %.1fGB is free; "
                "degrading to segment timestamps (speaker ID will use segment overlap).",
                free_mb / 1024**3,
            )
            use_word_ts = False

    def _run_window(window_audio):
        """Call the pipeline, returning (result, word_mode) where word_mode is
        True only if word-level timestamps were actually produced."""
        try:
            result = pipeline_instance(
                {"array": window_audio, "sampling_rate": sr},
                generate_kwargs=generate_kwargs,
                return_timestamps="word" if use_word_ts else True,
                **pipeline_kwargs,
            )
            return result, use_word_ts
        except Exception as e:
            # Some pipelines reject extra kwargs (e.g. `prompt`) — retry without them
            logger.warning(f"Pipeline call failed with extra kwargs ({e}); retrying without.")
            try:
                result = pipeline_instance(
                    {"array": window_audio, "sampling_rate": sr},
                    generate_kwargs=generate_kwargs,
                    return_timestamps="word" if use_word_ts else True,
                )
                return result, use_word_ts
            except Exception as e2:
                # Word-level timestamps failed (OOM on small GPUs, or unsupported
                # by this model) — release the failed attempt's GPU memory, then
                # degrade to segment mode so transcription still works
                # (diarization then falls back to overlap labeling).
                if use_word_ts:
                    logger.warning(
                        f"Word timestamps failed ({e2}); degrading to segment timestamps."
                    )
                    clear_gpu_memory()
                    result = pipeline_instance(
                        {"array": window_audio, "sampling_rate": sr},
                        generate_kwargs=generate_kwargs,
                        return_timestamps=True,
                    )
                    return result, False
                raise

    def _window_out(result, base_offset, window_seconds, win_text, word_mode):
        """(segments, words) for one window."""
        if word_mode:
            words = _collect_words(result, base_offset=base_offset)
            return _words_to_segments(words, win_text), words
        return _window_segments(result, base_offset, window_seconds, win_text), []

    if total_samples <= window_samples:
        _update_progress("transcribing", 1, 1)
        result, word_mode = _run_window(audio)
        text = result.get("text", "").strip()
        segments, words = _window_out(result, 0.0, len(audio) / sr, text, word_mode)
        return text, segments, words

    total_minutes = total_samples / sr / 60
    num_windows = int(np.ceil(total_samples / (window_samples - overlap_samples)))
    logger.info(f"Long audio detected: {total_minutes:.1f} min. Splitting into {num_windows} windows of 30s.")

    start = 0
    win_idx = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        window_audio = audio[start:end]
        win_idx += 1
        logger.info(f"Transcribing window {win_idx}/{num_windows}...")
        _update_progress("transcribing", win_idx, num_windows)

        result, word_mode = _run_window(window_audio)
        win_text = result.get("text", "").strip()
        if win_text:
            parts.append(win_text)
        win_segments, win_words = _window_out(
            result, start / sr, len(window_audio) / sr, win_text, word_mode
        )
        all_segments.extend(win_segments)
        all_words.extend(win_words)

        start += window_samples - overlap_samples

    full_text = " ".join(parts)
    logger.info(f"Windowed transcription complete: {win_idx} windows, {len(full_text)} characters total.")
    _update_progress("transcribing", win_idx, num_windows)
    return full_text, all_segments, all_words


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server and model are ready."""
    # Return healthy even if model is not loaded, to allow frontend to connect
    # and trigger the lazy loading on first request.
    
    return HealthResponse(
        status="healthy" if asr_pipeline is not None else "ready_to_load",
        model=CURRENT_MODEL_ID or DEFAULT_MODEL_ID,
        device=DEVICE,
    )


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible endpoint to list available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": CURRENT_MODEL_ID or DEFAULT_MODEL_ID,
                "object": "model",
                "owned_by": "sunbird",
                "permission": [],
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None), # Dynamic model switching parameter
    separate_speech: bool = Form(False),  # demucs noise reduction before ASR
    diarize: bool = Form(False),          # pyannote speaker identification
):
    """
    Transcribe audio file to text.

    OpenAI Whisper API compatible endpoint.
    Requests are serialized via an asyncio.Lock so multiple users queue up
    gracefully instead of racing or hanging.

    Args:
        file: Audio file (mp3, wav, ogg, m4a, webm, mp4)
        language: Source language code (lug, ach, teo, lgg, nyn, eng)
        model: ASR model to use (optional)

    Returns:
        Transcription text
    """
    global _asr_queue_size

    # Reject early if too many requests are already waiting
    if _asr_queue_size >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail=f"Server is busy. {_asr_queue_size} transcription(s) already queued. Please try again shortly."
        )

    # Validate file type before reading the whole file
    allowed_extensions = {".mp3", ".wav", ".ogg", ".m4a", ".webm", ".mp4", ".flac"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Read the file into memory now (before acquiring the lock) so the upload
    # doesn't block other requests from even starting their upload.
    content = await file.read()

    # Enforce file size limit
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content) / 1024 / 1024:.1f} MB. Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    _asr_queue_size += 1
    logger.info(f"Transcription queued. Queue size: {_asr_queue_size}")
    try:
        # Acquire the lock — only one transcription runs at a time.
        async with _asr_lock:
            _update_progress("queued", 0, 1)
            result = await _transcribe_bytes(
                content,
                file_ext,
                file.filename,
                language,
                model,
                separate_speech=separate_speech,
                diarize=diarize,
            )
        _update_progress("idle", 0, 1)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Transcription error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _asr_queue_size -= 1
        logger.info(f"Transcription slot released. Queue size: {_asr_queue_size}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_simple(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Simple transcription endpoint (alternative to OpenAI format).
    
    Args:
        file: Audio file
        language: Source language code
    
    Returns:
        TranscriptionResponse with text and detected language
    """
    # `model` must be passed explicitly: when called directly (not via FastAPI
    # request routing) the `model: Form(None)` default is an unresolved Form
    # object, which crashes downstream in get_asr_pipeline().
    result = await transcribe_audio(file=file, language=language, model=None)
    data = result.body.decode()
    parsed = json.loads(data)
    return TranscriptionResponse(
        text=parsed["text"],
        language=language,
        segments=[Segment(**s) for s in parsed.get("segments", [])],
    )


@app.post("/v1/audio/transcriptions/url")
async def transcribe_audio_url(
    request: UrlTranscriptionRequest,
):
    """
    Transcribe audio from a URL (YouTube, TikTok, etc.).
    Requests are serialized via the same asyncio.Lock as file uploads.

    Args:
        request: UrlTranscriptionRequest with url and optional language

    Returns:
        Transcription text
    """
    global _asr_queue_size

    # Check if yt-dlp is available
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="yt-dlp not installed. Run: pip install yt-dlp"
        )

    url = request.url.strip()
    language = request.language

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    if _asr_queue_size >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail=f"Server is busy. {_asr_queue_size} transcription(s) already queued. Please try again shortly."
        )

    _asr_queue_size += 1
    logger.info(f"URL transcription queued. Queue size: {_asr_queue_size}")

    try:
        async with _asr_lock:
            loop = asyncio.get_event_loop()
            pipeline_instance = await loop.run_in_executor(
                None, partial(get_asr_pipeline, request.model)
            )

            if pipeline_instance is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # Create temp directory for download
            with tempfile.TemporaryDirectory() as tmp_dir:
                logger.info(f"Downloading audio from URL: {url}")

                output_template = os.path.join(tmp_dir, '%(title)s.%(ext)s')
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': output_template,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'noplaylist': True,
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-us,en;q=0.5',
                    },
                    'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
                }

                # Download audio in thread pool (blocking network I/O)
                def _download():
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        if 'entries' in info:
                            info = info['entries'][0]
                        filename = ydl.prepare_filename(info)
                        base, _ = os.path.splitext(filename)
                        return f"{base}.mp3", info

                try:
                    downloaded_path, _ = await loop.run_in_executor(None, _download)
                except yt_dlp.utils.DownloadError as e:
                    logger.error(f"yt-dlp download error: {e}")
                    raise HTTPException(status_code=400, detail=f"Failed to download: {str(e)}")

                if not os.path.exists(downloaded_path):
                    files = os.listdir(tmp_dir)
                    if not files:
                        raise RuntimeError("Failed to download audio from URL: No file created")
                    downloaded_path = os.path.join(tmp_dir, files[0])

                logger.info(f"Audio downloaded to {downloaded_path}, transcribing...")

                with open(downloaded_path, "rb") as f:
                    content = f.read()
                downloaded_ext = os.path.splitext(downloaded_path)[1].lower() or ".mp3"
                result = await _transcribe_bytes(
                    content,
                    downloaded_ext,
                    os.path.basename(downloaded_path),
                    language,
                    request.model,
                    separate_speech=request.separate_speech,
                    diarize=request.diarize,
                )
                logger.info(f"URL transcription complete: {len(result['text'])} characters")
                return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"URL transcription error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _asr_queue_size -= 1
        logger.info(f"URL transcription slot released. Queue size: {_asr_queue_size}")


OCR_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg"}
OCR_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OCR_EXTENSIONS = OCR_VIDEO_EXTENSIONS | OCR_IMAGE_EXTENSIONS


def ocr_media_text(path: str, frame_interval: float) -> tuple:
    """OCR a still image, or sample frames from a video.

    Images (jpg/png/...) carry no audio — they come in as OCR-only input and
    produce a single reading. Returns ``(text, frames_scanned)``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in OCR_IMAGE_EXTENSIONS:
        _update_progress("ocr", 1, 1)
        # np.fromfile + imdecode tolerates non-ASCII paths cv2.imread rejects.
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {os.path.basename(path)}")
        text = _ocr_frame(image).strip()
        _update_progress("idle", 0, 1)
        logger.info(f"Image OCR complete: {len(text)} characters")
        return text, 1
    return ocr_video_frames(path, frame_interval)


@app.post("/v1/video/ocr")
async def ocr_video(
    file: UploadFile = File(...),
    frame_interval: Optional[float] = Form(None),
):
    """Extract on-screen/printed text (OCR) from an uploaded video or image.

    Videos are sampled every ``frame_interval`` seconds (default
    OCR_FRAME_INTERVAL) and read with Tesseract — captions, slides,
    lower-thirds; images are read in one pass. Either way the frontend gets
    selectable text it can feed through the translation pipeline.

    Returns ``{"text": ..., "frames_scanned": ..., "filename": ...}``.
    """
    if not _ocr_available():
        raise HTTPException(
            status_code=501,
            detail=(
                "OCR unavailable: install opencv-python-headless + pytesseract and "
                "the tesseract binary (apt install tesseract-ocr / brew install tesseract)."
            ),
        )

    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in OCR_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type for OCR: {file_ext}. "
                f"Allowed: {', '.join(sorted(OCR_EXTENSIONS))}"
            ),
        )

    content = await file.read()
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content) / 1024 / 1024:.1f} MB. Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    interval = frame_interval if frame_interval and frame_interval > 0 else OCR_FRAME_INTERVAL
    loop = asyncio.get_event_loop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Runs in the thread pool: OCR is CPU-bound and must not block the event
        # loop (the ASR lock is untouched — OCR and transcription can overlap).
        text, scanned = await loop.run_in_executor(
            None, partial(ocr_media_text, tmp_path, interval)
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"OCR error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return JSONResponse(
        content={
            "text": text,
            "frames_scanned": scanned,
            "filename": file.filename,
        }
    )


@app.post("/v1/audio/live")
async def transcribe_live(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    session_id: Optional[str] = Form("default"),
    model: Optional[str] = Form(None),
):
    """Live (streamed) transcription endpoint.

    The frontend sends short microphone slices (~5-6s) with a shared
    ``session_id``. The server keeps the previous slice's text per session and
    feeds it back as ``initial_prompt`` so the running transcript stays
    coherent ("append-and-correct" behavior, borrowed from Buzz).
    """
    global _asr_queue_size

    allowed_extensions = {".mp3", ".wav", ".ogg", ".m4a", ".webm", ".mp4", ".flac", ".opus"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    content = await file.read()
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    if _asr_queue_size >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail="Server is busy. Please try again shortly.")

    with _live_sessions_lock:
        last_text = _live_sessions.get(session_id, {}).get("last_text", "")
    initial_prompt = last_text[-400:] if last_text else None

    _asr_queue_size += 1
    try:
        async with _asr_lock:
            result = await _transcribe_bytes(
                content, file_ext, file.filename, language, model, initial_prompt
            )
        with _live_sessions_lock:
            if result["text"]:
                _live_sessions[session_id] = {"last_text": result["text"]}
        return JSONResponse(content={**result, "session_id": session_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _asr_queue_size -= 1


@app.delete("/v1/audio/live/{session_id}")
async def clear_live_session(session_id: str):
    """Reset a live-transcription session (drop the initial_prompt context)."""
    with _live_sessions_lock:
        _live_sessions.pop(session_id, None)
    return JSONResponse(content={"status": "cleared"})


@app.get("/v1/audio/progress")
async def get_transcription_progress():
    """Poll endpoint: current transcription progress (phase / chunk / total)."""
    return _progress


def _watch_loop(folder: str):
    """Background watcher: queue new audio files in ``folder`` for transcription."""
    while not _watch_stop.is_set():
        try:
            for name in sorted(os.listdir(folder)):
                path = os.path.join(folder, name)
                if not os.path.isfile(path) or name.startswith("."):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in {".mp3", ".wav", ".ogg", ".m4a", ".webm", ".mp4", ".flac"}:
                    continue
                if path in _seen_files:
                    continue
                _seen_files.add(path)
                if _event_loop is not None:
                    asyncio.run_coroutine_threadsafe(_process_watch_file(path, folder), _event_loop)
                    logger.info(f"Watch folder: queued {name} for transcription")
        except Exception as e:
            logger.error(f"Watch folder scan error: {e}")
        _watch_stop.wait(int(os.getenv("WATCH_POLL_SECONDS", "10")))


async def _process_watch_file(path: str, folder: str):
    """Transcribe one watch-folder file and write TXT/SRT/VTT outputs next to it."""
    global _asr_queue_size
    if _asr_queue_size >= MAX_QUEUE_SIZE:
        logger.warning(f"Watch folder: queue full, skipping {path}")
        return
    _asr_queue_size += 1
    try:
        async with _asr_lock:
            logger.info(f"Watch folder: transcribing {path}")
            with open(path, "rb") as f:
                content = f.read()
            ext = os.path.splitext(path)[1].lower()
            result = await _transcribe_bytes(
                content,
                ext,
                os.path.basename(path),
                None,
                None,
                None,
                separate_speech=SPEECH_SEPARATION,
                diarize=DIARIZATION,
            )

        out_dir = os.getenv("WATCH_OUTPUT_FOLDER", folder)
        base = os.path.splitext(os.path.basename(path))[0]
        formats = [
            f.strip()
            for f in os.getenv("WATCH_OUTPUT_FORMATS", "txt,srt,vtt").split(",")
            if f.strip()
        ]
        for fmt in formats:
            out_path = os.path.join(out_dir, f"{base}.{fmt}")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(format_transcript(result["segments"], result["text"], fmt))
            logger.info(f"Watch folder: wrote {out_path}")

        if os.getenv("WATCH_DELETE_SOURCE", "false").lower() == "true":
            os.remove(path)
            logger.info(f"Watch folder: deleted source {path}")
    except Exception as e:
        logger.error(f"Watch folder: failed to transcribe {path}: {e}")
    finally:
        _asr_queue_size -= 1


def main():
    parser = argparse.ArgumentParser(description="VoiceBird Local ASR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind to")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--model", default=None, help="Model ID to load")
    
    args = parser.parse_args()
    
    # Override globals with CLI args
    global DEVICE, DEFAULT_MODEL_ID
    if args.device:
        DEVICE = args.device
        os.environ["ASR_DEVICE"] = args.device
    if args.model:
        DEFAULT_MODEL_ID = args.model
        os.environ["ASR_MODEL_ID"] = args.model
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  VoiceBird Local ASR Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  Model:  {DEFAULT_MODEL_ID:<50} ║
║  Device: {DEVICE:<50} ║
║  URL:    http://{args.host}:{args.port:<42} ║
║  HF Auth: {'Yes' if HF_TOKEN else 'No (set HF_TOKEN if needed)':<48} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
