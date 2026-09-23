"""VoiceBird Local Translation Server
Runs a local NLLB-based model for African language translation.

By default it loads Sunbird's SALT fine-tune on full NLLB-200-3.3B
(Sunbird/translate-nllb-3.3b-salt), which was trained for the five Ugandan
languages plus Swahili/Lusoga/Rutooro and retains all NLLB-200 languages.
It is not gated on Hugging Face (CC-BY-NC-4.0, non-commercial).

Usage:
    python translation_server.py [--port 8200] [--host 0.0.0.0] [--device cuda|cpu]
"""

import argparse
import logging
import os
import gc
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Load .env file from likely locations.
# In Docker, translation_server.py lives at /app and .env is usually /app/.env.
# In local dev, it often lives at repo root (../.env).
from dotenv import load_dotenv
possible_env_paths = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env",
]
for env_path in possible_env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = None
model = None
model_load_lock = threading.Lock()

# Model configuration
# Primary model: Sunbird's SALT fine-tune on full NLLB-200-3.3B, trained on the five
# Ugandan languages + Swahili/Lusoga/Rutooro, retaining all NLLB-200 languages.
# Not gated on Hugging Face (CC-BY-NC-4.0, non-commercial).
MODEL_ID = os.getenv("TRANSLATION_MODEL_ID", "Sunbird/translate-nllb-3.3b-salt")
# Whether the primary model is Sunbird's SALT fine-tune, which uses custom language tokens.
IS_SALT_MODEL = "salt" in MODEL_ID
DEVICE = os.getenv("TRANSLATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Falling back to %d", name, raw, default)
        return default


NUM_BEAMS = max(1, _int_env("TRANSLATION_NUM_BEAMS", 4))
MAX_INPUT_TOKENS = max(64, _int_env("TRANSLATION_MAX_INPUT_TOKENS", 512))
MAX_NEW_TOKENS = max(32, _int_env("TRANSLATION_MAX_NEW_TOKENS", 192))
MAX_CHUNK_CHARS = max(200, _int_env("TRANSLATION_CHUNK_CHARS", 1000))
PRELOAD_MODEL = os.getenv("TRANSLATION_PRELOAD_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}
CLEAR_CACHE_PER_REQUEST = os.getenv("TRANSLATION_CLEAR_CACHE_PER_REQUEST", "false").strip().lower() in {"1", "true", "yes", "on"}

# Helper to clean HF token
def _clean_hf_token(raw_value: Optional[str]) -> str:
    token = (raw_value or "").strip()
    if not token:
        return ""
    normalized = token.lower()
    if normalized in {"your_huggingface_token_here", "paste_huggingface_token_here"}:
        return ""
    return token

HF_TOKEN = _clean_hf_token(os.getenv("HF_TOKEN"))


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    

def log_memory_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


# NLLB language code mapping
# NLLB uses specific codes like "lug_Latn" for Luganda in Latin script
NLLB_LANGUAGE_MAP = {
    "lug": "lug_Latn",  # Luganda
    "ach": "ach_Latn",  # Acholi
    "teo": "teo_Latn",  # Ateso
    "lgg": "lgg_Latn",  # Lugbara
    "nyn": "nyn_Latn",  # Runyankole
    "eng": "eng_Latn",  # English
    "fra": "fra_Latn",  # French
    "swa": "swh_Latn",  # Swahili
    "ara": "arb_Arab",  # Arabic
    "spa": "spa_Latn",  # Spanish
    "deu": "deu_Latn",  # German
    "zho": "zho_Hans",  # Chinese (Simplified)
    "hin": "hin_Deva",  # Hindi
    "rus": "rus_Cyrl",  # Russian
    "por": "por_Latn",  # Portuguese
    "ita": "ita_Latn",  # Italian
}

# Sunbird SALT fine-tuned model: covers English + the five Ugandan languages.
# Its vocab was adapted from NLLB, so it uses its own language token IDs
# (see the usage example on https://huggingface.co/Sunbird/translate-nllb-1.3b-salt).
SALT_LANGUAGES = {"lug", "ach", "teo", "lgg", "nyn", "eng"}
SALT_LANGUAGE_TOKENS = {
    "eng": 256047,
    "ach": 256111,
    "lgg": 256008,
    "lug": 256110,
    "nyn": 256002,
    "teo": 256006,
}


class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lazy loading - models load on first request, cleanup on shutdown."""
    global tokenizer, model

    logger.info("Translation server starting (lazy loading enabled)")
    logger.info(f"Model will load on first translation request: {MODEL_ID}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Generation config: beams={NUM_BEAMS}, max_input_tokens={MAX_INPUT_TOKENS}, max_new_tokens={MAX_NEW_TOKENS}, max_chunk_chars={MAX_CHUNK_CHARS}")
    
    # Authenticate with Hugging Face if token provided
    if HF_TOKEN:
        logger.info("Authenticating with Hugging Face...")
        login(token=HF_TOKEN)
    else:
        logger.warning("No HF_TOKEN set. Public models will work, but gated models require authentication.")
    
    # Don't load model on startup - wait for first request (lazy loading)
    logger.info("Server ready. Model will load when first translation is requested.")
    if PRELOAD_MODEL:
        logger.info("Preloading translation model at startup...")
        get_translation_model()
        logger.info("Translation model preloaded.")
    
    yield
    
    # Cleanup - properly release GPU memory
    logger.info("Shutting down and releasing memory...")
    if model is not None and DEVICE == "cuda":
        model.to('cpu')
    if tokenizer is not None:
        del tokenizer
    if model is not None:
        del model
    
    # Clear GPU cache and run garbage collection
    clear_gpu_memory()
    logger.info("Cleanup complete")
    log_memory_usage()


app = FastAPI(
    title="VoiceBird Local Translation Server",
    description="Local translation server for African languages using NLLB model",
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


def _load_model(model_id: str, lock: threading.Lock, state: dict):
    """
    Lazy-load a translation model on first use and cache it in `state`.
    Returns (tokenizer, model). Safe to call from multiple requests.
    """
    # Fast path when model is already available.
    if state.get("tokenizer") is not None and state.get("model") is not None:
        return state["tokenizer"], state["model"]

    with lock:
        # Another request might have loaded it while we waited for lock.
        if state.get("tokenizer") is not None and state.get("model") is not None:
            return state["tokenizer"], state["model"]

        logger.info("Loading translation model (lazy loading)...")

        # Clear existing memory first just in case
        clear_gpu_memory()

        logger.info(f"Model: {model_id}")

        try:
            # Determine torch dtype based on device
            torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

            # Load tokenizer and model
            logger.info("Loading tokenizer...")
            tok = NllbTokenizer.from_pretrained(
                model_id,
                token=HF_TOKEN if HF_TOKEN else None,
            )

            logger.info("Loading model...")
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=HF_TOKEN if HF_TOKEN else None,
                low_cpu_mem_usage=True,
            )

            # Move model to device
            mdl = mdl.to(DEVICE)
            mdl.eval()

            state["tokenizer"] = tok
            state["model"] = mdl

            logger.info("Model loaded successfully!")
            log_memory_usage()

            return tok, mdl

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load translation model: {str(e)}")


def get_translation_model():
    """
    Lazy load the translation model (Sunbird SALT by default).
    Returns the tokenizer and model, loading them if not already loaded.
    """
    global tokenizer, model
    tokenizer, model = _load_model(MODEL_ID, model_load_lock, {"tokenizer": tokenizer, "model": model})
    return tokenizer, model


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the server and model are ready.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model=MODEL_ID,
        device=DEVICE,
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text between supported languages.
    
    Supports English ↔ Ugandan language pairs:
    - English (eng) ↔ Luganda (lug)
    - English (eng) ↔ Acholi (ach)
    - English (eng) ↔ Ateso (teo)
    - English (eng) ↔ Lugbara (lgg)
    - English (eng) ↔ Runyankole (nyn)
    
    Args:
        request: TranslationRequest with text, source_language, and target_language
    
    Returns:
        TranslationResponse with translated text
    """
    # Validate languages
    if request.source_language not in NLLB_LANGUAGE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source language: {request.source_language}. Supported: {list(NLLB_LANGUAGE_MAP.keys())}"
        )
    
    if request.target_language not in NLLB_LANGUAGE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language: {request.target_language}. Supported: {list(NLLB_LANGUAGE_MAP.keys())}"
        )
    
    # Map to NLLB codes
    src_lang = NLLB_LANGUAGE_MAP[request.source_language]
    tgt_lang = NLLB_LANGUAGE_MAP[request.target_language]

    # Use Sunbird SALT custom language tokens when the pair is within its trained
    # set; other pairs (e.g. French, Swahili, ...) use the same model via standard
    # NLLB language codes, which the full NLLB-200 fine-tune retains.
    use_salt = (
        IS_SALT_MODEL
        and request.source_language in SALT_LANGUAGES
        and request.target_language in SALT_LANGUAGES
    )
    if use_salt:
        logger.info("Using Sunbird SALT fine-tuned model (custom language tokens)")

    try:
        logger.info(f"Translating: {request.source_language} -> {request.target_language}")
        text_len = len(request.text)
        logger.info(f"Text length: {text_len} characters")
        
        # Helper to translate a single chunk
        def translate_chunk(chunk_text):
            if not chunk_text.strip():
                return ""

            if use_salt:
                # Sunbird SALT model: uses custom language tokens, so we do NOT set
                # tokenizer.src_lang. Instead the source language token is written
                # directly into the first input position (per the model card usage).
                salt_tok, salt_model = get_translation_model()
                chunk_inputs = salt_tok(
                    chunk_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                )
                chunk_inputs = {k: v.to(DEVICE) for k, v in chunk_inputs.items()}
                chunk_inputs["input_ids"][0][0] = SALT_LANGUAGE_TOKENS[request.source_language]
                forced_bos_token_id = SALT_LANGUAGE_TOKENS[request.target_language]
                gen_model = salt_model
                gen_tokenizer = salt_tok
            else:
                # Same model via standard NLLB src_lang + forced-BOS flow (the full
                # NLLB-200 fine-tune serves all NLLB-200 language pairs).
                nllb_tok, nllb_model = get_translation_model()
                nllb_tok.src_lang = src_lang
                chunk_inputs = nllb_tok(
                    chunk_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=MAX_INPUT_TOKENS,
                )
                chunk_inputs = {k: v.to(DEVICE) for k, v in chunk_inputs.items()}

                # Use proper language token ID for NLLB
                if not hasattr(nllb_tok, "lang_code_to_id"):
                    forced_bos_token_id = nllb_tok.convert_tokens_to_ids(tgt_lang)
                else:
                    forced_bos_token_id = nllb_tok.lang_code_to_id.get(tgt_lang)
                if forced_bos_token_id is None:
                    forced_bos_token_id = nllb_tok.encode(tgt_lang, add_special_tokens=False)[0]
                gen_model = nllb_model
                gen_tokenizer = nllb_tok

            token_count = int(chunk_inputs["input_ids"].shape[-1])

            # Dynamic output cap keeps latency predictable while preserving quality.
            max_new_tokens = min(MAX_NEW_TOKENS, max(32, int(token_count * 1.5)))

            # Generate
            with torch.inference_mode():
                chunk_translated_tokens = gen_model.generate(
                    **chunk_inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=max_new_tokens,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                    use_cache=True,
                )
            
            # Decode
            return gen_tokenizer.batch_decode(chunk_translated_tokens, skip_special_tokens=True)[0]

        if len(request.text) <= MAX_CHUNK_CHARS:
            translated_text = translate_chunk(request.text)
        else:
            logger.info("Text too long, splitting into chunks...")
            # Split by sentence endings including Arabic/Urdu question mark (؟)
            import re
            # Regex splits after ., !, ?, or ؟ followed by whitespace
            sentences = re.split(r'(?<=[.!?؟])\s+', request.text)
            chunks = []
            current_chunk = []
            current_len = 0
            
            for sentence in sentences:
                sent_len = len(sentence)
                # If a single sentence allows it, add to chunk
                if current_len + sent_len < MAX_CHUNK_CHARS:
                    current_chunk.append(sentence)
                    current_len += sent_len
                else:
                    # If current chunk has data, translate it
                    if current_chunk:
                        chunks.append(translate_chunk(" ".join(current_chunk)))
                        current_chunk = []
                        current_len = 0
                    
                    # If the new sentence itself is too long, we might need to split it further or just translate it alone (it will be truncated if > 512 tokens)
                    # For simplicity, we process it as its own chunk
                    if sent_len > MAX_CHUNK_CHARS:
                         # Fallback: translate huge sentence alone (will truncate)
                         chunks.append(translate_chunk(sentence))
                    else:
                         current_chunk.append(sentence)
                         current_len = sent_len
            
            # Process last chunk
            if current_chunk:
                chunks.append(translate_chunk(" ".join(current_chunk)))
                
            translated_text = " ".join(chunks)

        logger.info(f"Translation complete: {len(translated_text)} characters")
        
        # Optional: can help in constrained GPU environments at the expense of latency.
        if CLEAR_CACHE_PER_REQUEST:
            clear_gpu_memory()

        return TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Translation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="VoiceBird Local Translation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8200, help="Port to bind to")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--model", default=None, help="Model ID to load")
    
    args = parser.parse_args()
    
    # Override globals with CLI args
    global DEVICE, MODEL_ID, IS_SALT_MODEL
    if args.device:
        DEVICE = args.device
        os.environ["TRANSLATION_DEVICE"] = args.device
    if args.model:
        MODEL_ID = args.model
        IS_SALT_MODEL = "salt" in MODEL_ID
        os.environ["TRANSLATION_MODEL_ID"] = args.model
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            VoiceBird Local Translation Server                ║
╠══════════════════════════════════════════════════════════════╣
║  Model:  {MODEL_ID:<50} ║
║  Device: {DEVICE:<50} ║
║  URL:    http://{args.host}:{args.port:<42} ║
║  HF Auth: {'Yes' if HF_TOKEN else 'No (set HF_TOKEN if needed)':<48} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
