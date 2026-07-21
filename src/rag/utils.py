"""
utils.py
========
Utility functions bersama untuk semua komponen RAG:
  - init_vectordb()   : koneksi ke Qdrant + load embedding model
  - init_llm()        : inisialisasi LLM client berdasarkan config
  - call_llm()        : panggil LLM dengan retry logic
  - parse_json_safe() : parse JSON output LLM dengan fault tolerance
"""

import sys
import os
import time
import json
import re

# Pastikan config bisa diimport baik dari root maupun dari dalam src/rag/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from config import (  # type: ignore
    LLM_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    HF_TOKEN, HF_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    TOP_K,
)

try:
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL, JUDGE_PROVIDER, OLLAMA_JUDGE_MODEL  # type: ignore
except ImportError:
    OLLAMA_MODEL = ""
    OLLAMA_BASE_URL = ""
    JUDGE_PROVIDER = "openrouter"
    OLLAMA_JUDGE_MODEL = "llama3.3:70b"


# ── Vector DB ────────────────────────────────────────────────────────────────

def init_vectordb():
    """Load embedding model dan koneksi ke Qdrant."""
    from qdrant_client import QdrantClient  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    print("Memuat model embedding...")
    try:
        # Paksa CPU agar VRAM sepenuhnya tersedia untuk Ollama 70B
        # Embedding inference untuk 1 query tetap cepat di CPU
        model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
        print(f"   [OK] Embedding model loaded: {EMBEDDING_MODEL} (device: cuda)")
    except Exception as e:
        print(f"   [ERR] Gagal load embedding: {e}")
        sys.exit(1)

    client = QdrantClient(path=QDRANT_PATH)
    count = client.get_collection(COLLECTION_NAME).points_count
    print(f"   [OK] Terhubung ke Qdrant ({count} chunks)")
    return (client, model)


# ── LLM Client ───────────────────────────────────────────────────────────────

def init_llm(override_provider=None, override_model=None):
    """
    Inisialisasi LLM client berdasarkan provider di config.

    Returns: tuple (provider_str, client_obj, override_model)
    """
    provider_to_use = override_provider if override_provider else LLM_PROVIDER

    def _ret(p, c):
        return (p, c, override_model)

    if provider_to_use == "gemini":
        from google import genai  # type: ignore
        if not GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY tidak ditemukan!")
            sys.exit(1)
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"[OK] Gemini client ready ({GEMINI_MODEL})")
        return _ret("gemini", client)

    elif provider_to_use == "groq":
        if not GROQ_API_KEY:
            print("❌ GROQ_API_KEY tidak ditemukan!")
            sys.exit(1)
        print(f"[OK] Groq client ready ({GROQ_MODEL})")
        return _ret("groq", None)

    elif provider_to_use == "huggingface":
        from huggingface_hub import InferenceClient  # type: ignore
        if not HF_TOKEN:
            print("❌ HF_TOKEN tidak ditemukan!")
            sys.exit(1)
        client = InferenceClient(token=HF_TOKEN)
        print(f"[OK] HuggingFace client ready ({HF_MODEL})")
        return _ret("huggingface", client)

    elif provider_to_use == "openrouter":
        if not OPENROUTER_API_KEY:
            print("❌ OPENROUTER_API_KEY tidak ditemukan!")
            sys.exit(1)
        print(f"[OK] OpenRouter client ready ({OPENROUTER_MODEL})")
        return _ret("openrouter", None)

    elif provider_to_use == "ollama":
        print(f"[OK] Ollama client ready ({OLLAMA_MODEL})")
        return _ret("ollama", None)

    else:
        print(f"❌ LLM_PROVIDER '{provider_to_use}' tidak dikenal!")
        sys.exit(1)


# ── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(llm_client, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Panggil LLM dengan retry logic untuk rate limiting."""
    provider = llm_client[0]
    client = llm_client[1]
    override_model = llm_client[2] if len(llm_client) > 2 else None

    max_retries = 3
    retry_delays = [5, 15, 30]

    for attempt in range(max_retries + 1):
        try:
            if provider == "gemini":
                r = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return str(r.text or "")

            elif provider == "groq":
                import requests
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": GROQ_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "temperature": temperature, "max_tokens": max_tokens},
                    timeout=120,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                raise Exception(f"Groq Error: {r.status_code} {r.text}")

            elif provider == "huggingface":
                r = client.chat_completion(
                    model=HF_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                )
                return str(r.choices[0].message.content or "")

            elif provider == "openrouter":
                import requests
                used_model = override_model if override_model else OPENROUTER_MODEL
                payload = {
                    "model": used_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                )
                if r.status_code == 200 and "choices" in r.json():
                    return r.json()["choices"][0]["message"]["content"]
                error_msg = r.text
                try:
                    error_msg = r.json().get("error", {}).get("message", r.text)
                except Exception:
                    pass
                raise Exception(f"OpenRouter Error {r.status_code}: {error_msg}")

            elif provider == "ollama":
                import requests
                model_to_use = override_model if override_model else OLLAMA_MODEL
                r = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": model_to_use, "prompt": prompt, "stream": False,
                          "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}},
                    timeout=120,
                )
                if r.status_code == 200:
                    return r.json().get("response", "")
                raise Exception(f"Ollama Error: {r.status_code}")

        except Exception as e:
            if attempt < max_retries and any(
                k in str(e) for k in ["429", "RESOURCE_EXHAUSTED", "rate_limit"]
            ):
                time.sleep(retry_delays[attempt])
            else:
                raise e

    return ""


# ── JSON Parser ───────────────────────────────────────────────────────────────

def parse_json_safe(raw: str) -> dict:
    """Parse JSON output LLM dengan fault tolerance."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0].strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
