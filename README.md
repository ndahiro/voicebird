# VoiceBird

VoiceBird is a local-first speech-to-text (ASR) and translation application focused on African languages, powered by Sunbird AI models running on your own GPU.

```
Browser ──► Next.js frontend (:3100)
              │
              ├──► ASR service  (:8100)  Sunbird/asr-whisper-large-v3-salt   (faster-whisper / CTranslate2)
              ├──► Translation   (:8200)  Sunbird/translate-nllb-3.3b-salt    (NLLB via transformers)
              └──► MySQL         (:3307) users + transcription history (Prisma)
```

## Project Structure

```
app/                    Next.js app router (page, login, register, providers)
components/
  voicebird/            Main UI: Header, UploadCard, AudioRecorder, LiveCaption,
                        TranscriptionCard, FeatureInfo
  ui/                   Shared primitives (button, card, toast, select)
lib/
  api.ts                Backend clients: local ASR/OCR/translation, Sunbird cloud, Hugging Face
  filesave.ts           Save transcript text with the media's name (Downloads by default; optional folder)
  config.ts             Env vars, supported languages, model list
  types.ts              Shared TypeScript types
  prisma.ts             Prisma client singleton
  utils.ts              cn() class-merge helper
prisma/                 Schema + seed script
backend/
  server.py             ASR service (FastAPI, port 8100)
  translation_server.py Translation service (FastAPI, port 8200)
  diarization.py        Word-level speaker diarization helpers (ported from Buzz)
  setup_models.py       One-time model download / cache warm-up
  start_servers.sh      Run both services locally without Docker
  requirements*.txt     Python deps (base / cpu / gpu variants)
docker-compose.yml      Full stack: db + asr-service + translation-service + frontend
Dockerfile.backend      Python image (GPU=1 build arg adds CUDA torch)
Dockerfile.frontend     Next.js production image
preload-docker-models.sh  Populate the model cache volume before first start
```

## 🚀 Quick Start (Docker — recommended)

Prerequisites: Docker with the NVIDIA container toolkit (GPU) or a machine with enough RAM for CPU inference.

1. **Configure environment**: copy `.env.example` to `.env` and fill in at least `HF_TOKEN` (needed for the gated Sunbird ASR model and pyannote speaker diarization).

2. **Preload models** (one-time, ~10 GB download into a persistent volume):
   ```bash
   ./preload-docker-models.sh
   ```

3. **Build and start**:
   ```bash
   docker compose up --build -d
   ```

4. **Access the app**: frontend at http://localhost:3100, ASR at http://localhost:8100, translation at http://localhost:8200.

5. **Stop**: `docker compose down` (models and database survive in the `voicebird_model_cache` and `mysql_data` volumes).

### GPU allocation

`docker-compose.yml` pins `asr-service` and `translation-service` to specific GPU ids via `deploy.resources.reservations.devices` (default: GPUs 6 and 7 on the original 8-GPU host). Adjust `device_ids` to match a free GPU on your machine.

## 🚀 Quick Start (manual, no Docker)

Prerequisites: Node.js ≥ 18, Python ≥ 3.10, MySQL.

1. `cp .env.example .env` and configure (see reference below). Generate a NextAuth secret with `openssl rand -base64 32`.

2. **Database**:
   ```bash
   mysql -u root -p -e "CREATE DATABASE voicebird;"
   npx prisma migrate dev     # create tables
   npx prisma db seed         # optional: seed test users
   ```

3. **Backend**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements-gpu.txt   # or requirements-cpu.txt
   python setup_models.py     # one-time model download
   ./start_servers.sh         # or run server.py / translation_server.py separately
   ```

4. **Frontend** (new terminal, project root):
   ```bash
   npm install
   npm run dev                # http://localhost:3100
   ```

### Production build

```bash
npm run build
npm start
```

## ⚙️ Environment Variables

### Frontend (Next.js)

| Variable | Purpose |
|----------|---------|
| `NEXT_PUBLIC_LOCAL_SERVER_URL` | Local ASR server URL (default `http://localhost:8100`) |
| `NEXT_PUBLIC_LOCAL_TRANSLATION_URL` | Local translation server URL (default `http://localhost:8200`) |
| `NEXT_PUBLIC_DEFAULT_BACKEND` | `local` (default), `sunbird`, or `huggingface` |
| `NEXT_PUBLIC_SUNBIRD_API_URL` / `NEXT_PUBLIC_SUNBIRD_API_TOKEN` | SunbirdAI cloud API (optional cloud mode) |
| `NEXT_PUBLIC_HF_API_TOKEN` / `NEXT_PUBLIC_HF_ASR_MODEL` | Hugging Face cloud inference (optional) |
| `NEXTAUTH_URL` / `NEXTAUTH_SECRET` | NextAuth.js session config |
| `NEXT_PUBLIC_SESSION_IDLE_TIMEOUT_MINUTES` | Idle minutes before the user must sign in again (default `30`). The session slides forward while the user is active; after this much inactivity the token expires and the open tab signs out to `/login?reason=idle` |

### ASR service (`backend/server.py`)

| Variable | Purpose |
|----------|---------|
| `ASR_MODEL_ID` | Model to load (default `Sunbird/asr-whisper-large-v3-salt`) |
| `ASR_MODEL_PATH` | Explicit local model directory override (skips HF download) |
| `ASR_DEVICE` | `cuda` (default when available) or `cpu` |
| `ASR_BACKEND` | `faster-whisper` (CTranslate2, fast, default in Docker) or `transformers` |
| `ASR_FALLBACK_MODEL_ID` | Model used if the primary model fails to load |
| `FASTER_WHISPER_BATCH_SIZE` | Chunk decode concurrency for `BatchedInferencePipeline` (default `16`) |
| `FASTER_WHISPER_COMPUTE_TYPE` | CTranslate2 compute type (default `int8_float16` on GPU) |
| `FASTER_WHISPER_BEAM_SIZE` | Beam size (default `5`) |
| `FASTER_WHISPER_VAD_FILTER` | `true` (default): Silero VAD skips silence |
| `MAX_FILE_SIZE_MB` | Upload size limit (default `1024`) |
| `ANTI_HALLUCINATION` | `true` (default): `condition_on_previous_text=False` on the transformers path |
| `REPETITION_PENALTY` | Optional penalty `> 1.0` when anti-hallucination is on (default `1.0` = off) |
| `PARAGRAPH_GAP_SECONDS` | Segment gap that starts a new paragraph (default `2.0`) |
| `PARAGRAPH_SILENCE_SECONDS` | Audio silence that starts a new paragraph (default `1.5`) |
| `SPEECH_SEPARATION` | `true` to always run demucs vocal separation before ASR (also toggleable per request) |
| `SPEECH_SEPARATION_DEVICE` | Device for demucs (default `cuda` when available; use `cpu` when the GPU is saturated) |
| `DIARIZATION` | `true` to always run pyannote speaker identification (also toggleable per request) |
| `DIARIZATION_DEVICE` / `DIARIZATION_MODEL` / `DIARIZATION_TOKEN` | pyannote device, pipeline model, and HF token (falls back to `HF_TOKEN`) |
| `WATCH_FOLDER` | Directory watched for new audio files; each is auto-transcribed on arrival |
| `WATCH_OUTPUT_FOLDER` | Where watch transcripts are written (default: alongside the audio) |
| `WATCH_OUTPUT_FORMATS` | Comma-separated: `txt,srt,vtt` (default) |
| `WATCH_DELETE_SOURCE` | `true` deletes the source file after transcription (default `false`) |
| `WATCH_POLL_SECONDS` | Watch-folder scan interval (default `10`) |
| `OCR_FRAME_INTERVAL` | Seconds between frames sampled by the OCR option (default `1.0`) |
| `OCR_MAX_FRAMES` | Cap on frames scanned per video by OCR (default `300`) |

### Translation service (`backend/translation_server.py`)

| Variable | Purpose |
|----------|---------|
| `TRANSLATION_MODEL_ID` | Default `Sunbird/translate-nllb-3.3b-salt` — full NLLB-200-3.3B fine-tuned for the five Ugandan languages + Swahili/Lusoga/Rutooro; retains all other NLLB-200 languages. Not gated; CC-BY-NC-4.0 (non-commercial). |
| `TRANSLATION_DEVICE` | `cuda` (default when available) or `cpu` |
| `TRANSLATION_NUM_BEAMS` | Beam width (default `4`; higher = better, slower) |
| `TRANSLATION_MAX_NEW_TOKENS` | Output cap per chunk (default `512`; ~1.6× the chunk's token count) |
| `TRANSLATION_MAX_INPUT_TOKENS` | Model input window, used to size chunks (default `512`) |
| `TRANSLATION_CHUNK_CHARS` | Character cap per chunk, on top of the token budget (default `1000`) |
| `TRANSLATION_MAX_SPLIT_DEPTH` | How many times a chunk may be halved and retried when it hits the output cap (default `6`) |
| `TRANSLATION_PRELOAD_MODEL` | `true` loads the model at startup instead of on first request |
| `TRANSLATION_CLEAR_CACHE_PER_REQUEST` | `true` frees GPU cache after each request (lower memory, higher latency) |
| `TRANSLATION_FALLBACK_MODEL_ID` | Second model cached by `setup_models.py` (default `facebook/nllb-200-distilled-1.3B`) |

## 🌍 Language Support

**ASR** — `Sunbird/asr-whisper-large-v3-salt` (a Whisper large-v3 fine-tune) is optimized for Luganda, Acholi, Ateso, Lugbara, Runyankole, Lusoga, Rutooro, Kinyarwanda, Lumasaba and English. Other languages work by switching `ASR_MODEL_ID` (e.g. `openai/whisper-large-v3`) — the UI's model selector offers these without config changes.

**Translation** — English ↔ the Ugandan languages via the SALT custom tokens, plus all other NLLB-200 languages (Swahili, Arabic, French, Spanish, German, Chinese, Hindi, Russian, Portuguese, Italian, ...) via standard NLLB codes — no separate fallback model needed at runtime. Long text is split on sentence **and newline** boundaries into chunks sized by the tokenizer's token window (not just characters), and any chunk whose translation hits the token cap is halved and retried — so transcripts and OCR output are translated end to end instead of being silently truncated.

## 🎙️ ASR Features

| Feature | How it works |
|---------|--------------|
| **Timestamps & exports** | Every transcription returns `{text, segments:[{start,end,text}]}`; the UI exports Copy / TXT / SRT / VTT. |
| **Media-named exports** | Saved text takes the audio/video/image's name — `interview.mp3` → `interview.txt`, `interview.srt`, `interview.translation.txt` — and lands in the browser's **Downloads** folder, so transcripts, translations and OCR output are easy to identify. The folder button (`lib/filesave.ts`) optionally grants one save folder via the File System Access API (Chrome/Edge) to write beside the media instead; elsewhere the filename is unchanged. |
| **OCR (on-screen text)** | Toggle on video uploads (images run it directly): frames are sampled every `OCR_FRAME_INTERVAL` seconds (OpenCV) and read with Tesseract — captions, slides and lower-thirds come back as selectable text with their own Translate button (`POST /v1/video/ocr`). Needs `opencv-python-headless` + `pytesseract` and the `tesseract-ocr` system package (`apt install tesseract-ocr`); the endpoint answers `501` when they're missing. |
| **Punctuation & paragraphs** | Whisper emits sentence punctuation; the server adds pause-based paragraph breaks (`PARAGRAPH_*` vars) and polishes casing. |
| **Live captioning** | The mic streams ~6 s slices to `/v1/audio/live`; the previous slice's text is carried back as a decoder prompt for coherent running transcripts. |
| **Progress reporting** | UI polls `/v1/audio/progress` (`{phase, current, total}`) for long files. |
| **Watch folder** | Drop audio into `WATCH_FOLDER` (mounted at `./watch` in Docker) → transcripts appear as `txt/srt/vtt`. |
| **faster-whisper backend** | CTranslate2 inference, ~4-10x faster warm than transformers. Custom fine-tunes (SALT) are converted once and cached (`_ensure_ct2_model`), including the 128-mel feature config and SALT tokenizer. |
| **Speech separation** | demucs extracts the vocals stem first — far fewer hallucinations on noisy audio. UI toggle, `separate_speech=true`, or `SPEECH_SEPARATION=true`. |
| **Speaker identification** | pyannote assigns `SPEAKER_00, ...` labels word-by-word (`backend/diarization.py`, ported from Buzz) and regroups words into speaker-labeled sentences. Requires accepting the gated `pyannote/speaker-diarization-3.1` + `pyannote/segmentation-3.0` models with your HF token. pyannote 4.x is required (3.x is incompatible with modern torchaudio); its optional community-1 PLDA stage is skipped gracefully when that repo wasn't accepted (`_install_pyannote_plda_shim` in `server.py`). |

### ASR endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription (`file`, optional `language`, `model`, `separate_speech`, `diarize`) |
| `POST /v1/audio/transcriptions/url` | Transcribe from a URL (YouTube, TikTok, ...) via yt-dlp |
| `POST /v1/audio/live` | Live slice transcription with per-session context |
| `DELETE /v1/audio/live/{session_id}` | Reset a live session's context |
| `POST /v1/video/ocr` | OCR a video's on-screen text or an image (`file`, optional `frame_interval`) → `{text, frames_scanned}` |
| `GET /v1/audio/progress` | Current transcription progress |
| `GET /health` | Health + current model + device |
| `GET /v1/models` | Loaded model (OpenAI-compatible shape) |
| `POST /transcribe` | Simple alias returning `{text, language, segments}` |

### Translation endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /translate` | `{text, source_language, target_language}` → `{translated_text, ...}` |
| `GET /health` | Health + model + device (503 until the model is loaded) |

## 🔧 Troubleshooting

- **Gated-model 403 / download failures**: accept `Sunbird/asr-whisper-large-v3-salt`, `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face with the account matching `HF_TOKEN`, then recreate the containers.
- **Health says `ready_to_load`**: normal — the ASR model lazy-loads on the first request. Send any transcription (or watch-folder file) to trigger it.
- **GPU memory**: the SALT 3.3B translation model needs ~7 GB VRAM in fp16; Whisper large-v3 fits an 8 GB card using faster-whisper int8_float16. On CPU everything works but is much slower and RAM-hungry (~13 GB for the 3.3B model in fp32).
- **`Invalid input features shape ... (1, 128, ...)`**: a CTranslate2 conversion missing its 128-mel config — delete the cached conversion under `~/.cache/huggingface/ctranslate2/` in the model volume and re-request.

## License

MIT
