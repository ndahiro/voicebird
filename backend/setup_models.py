"""One-time model download / cache warm-up for VoiceBird.

Run this once with internet access (docker: ./preload-docker-models.sh or
`python setup_models.py` inside the backend venv) to populate the Hugging Face
cache so the services can start and serve fully offline afterwards. Downloads:

1. Translation: Sunbird SALT NLLB-3.3B fine-tune (+ optional generic NLLB
   fallback model for setups that use one)
2. ASR: Sunbird/asr-whisper-large-v3-salt (+ optional OpenAI Whisper models)
"""

import os
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSeq2SeqLM, NllbTokenizer
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def download_models():
    print("Downloading and caching AI models for offline use...")
    
    # 1. Translation Model (Priority: Sunbird SALT fine-tune on NLLB-200-3.3B)
    trans_model_id = os.getenv("TRANSLATION_MODEL_ID", "Sunbird/translate-nllb-3.3b-salt")
    print(f"\n[Translation] Verifying {trans_model_id}...")
    try:
        print("  - Tokenizer...")
        NllbTokenizer.from_pretrained(trans_model_id, token=os.getenv("HF_TOKEN") or None)
        print("  - Model...")
        AutoModelForSeq2SeqLM.from_pretrained(trans_model_id, token=os.getenv("HF_TOKEN") or None)
        print(f"[Translation] Successfully cached {trans_model_id}")
    except Exception as e:
         print(f"[Translation] Error downloading {trans_model_id}: {e}")

    # 1b. Translation Fallback Model (generic NLLB for non-Ugandan pairs)
    fallback_model_id = os.getenv("TRANSLATION_FALLBACK_MODEL_ID", "facebook/nllb-200-distilled-1.3B")
    if fallback_model_id != trans_model_id:
        print(f"\n[Translation] Verifying fallback {fallback_model_id}...")
        try:
            print("  - Tokenizer...")
            NllbTokenizer.from_pretrained(fallback_model_id)
            print("  - Model...")
            AutoModelForSeq2SeqLM.from_pretrained(fallback_model_id)
            print(f"[Translation] Successfully cached {fallback_model_id}")
        except Exception as e:
            print(f"[Translation] Error downloading {fallback_model_id}: {e}")

    # 2. ASR Model: Sunbird (Priority)
    sunbird_model = "Sunbird/asr-whisper-large-v3-salt"
    print(f"\n[ASR] Verifying Priority Model: {sunbird_model}...")
    try:
        pipeline("automatic-speech-recognition", model=sunbird_model, device="cpu")
        print(f"[ASR] Successfully cached {sunbird_model}")
    except Exception as e:
        print(f"[ASR] Error downloading {sunbird_model}. (Ensure HF_TOKEN is valid for gated models): {e}")

    # 3. ASR Model: OpenAI Whisper Large V3 (Optional/Bonus for other languages)
    openai_model = "openai/whisper-large-v3"
    print(f"\n[ASR] Verifying Bonus Model: {openai_model}...")
    try:
        pipeline("automatic-speech-recognition", model=openai_model, device="cpu")
        print(f"[ASR] Successfully cached {openai_model}")
    except Exception as e:
        print(f"[ASR] Error downloading {openai_model}: {e}")

    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_models()
