"""
config.py
=========
Konfigurasi terpusat untuk semua komponen RAG.
API key dibaca dari file .env di root project.
"""

import os
from dotenv import load_dotenv

# Load .env dari root project (TA_sirah/)
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
load_dotenv(os.path.normpath(_ENV_PATH))

# ═══════════════════════════════════════════════════════════════════
#  SETUP PER MESIN — hanya bagian ini yang perlu diubah
# ═══════════════════════════════════════════════════════════════════
#
#  💻 RTX 5080 (16GB VRAM):
#      LLM_PROVIDER   = "ollama"
#      OLLAMA_MODEL   = "qwen2.5:7b"
#      JUDGE_PROVIDER = "openrouter"     ← judge via cloud
#
#  💻 A6000 (48GB VRAM):
#      LLM_PROVIDER   = "ollama"
#      OLLAMA_MODEL   = "qwen2.5:7b"
#      JUDGE_PROVIDER = "ollama"         ← judge via lokal llama3.3:70b
#
# ═══════════════════════════════════════════════════════════════════

# Provider untuk RAG pipeline (generation + retrieval)
LLM_PROVIDER = "ollama"

# Provider untuk LLM Judge di eval_1 & eval_2
# "openrouter" → Llama 70B via cloud (RTX 5080, tidak cukup VRAM untuk 70B)
# "ollama"     → Llama 70B via lokal (A6000, 48GB VRAM cukup)
JUDGE_PROVIDER = "openrouter"    # ← RTX 5080
#JUDGE_PROVIDER = "ollama"       # ← A6000

# ── GEMINI CONFIG ───────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ── GROQ CONFIG ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── HUGGINGFACE CONFIG ──────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── OPENROUTER CONFIG ───────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"

# ── OLLAMA CONFIG ───────────────────────────────────────────────────
# Generator (RAG pipeline) — muat di RTX 5080 maupun A6000
#OLLAMA_MODEL = "llama3.1:8b"
#OLLAMA_MODEL = "qwen2.5:7b"
#OLLAMA_MODEL = "qwen3:8b"
OLLAMA_MODEL = "sealion-baseline:latest"
#OLLAMA_MODEL = "sahabat-ai-baseline:latest"

# Sufficiency evaluator — model fine-tuned Sirah (routing + gap analysis)
# Ganti ini untuk berpindah model:
#OLLAMA_EVALUATOR_MODEL = "qwen-sirah-evaluator:latest"      # ← Uji Qwen
OLLAMA_EVALUATOR_MODEL = "sealion-sirah-evaluator:latest"  # ← Uji Sea-LION  
#OLLAMA_EVALUATOR_MODEL = "sahabat-sirah-evaluator:latest"  # ← Uji Sahabat-AI
#OLLAMA_EVALUATOR_MODEL = "llama-sirah-evaluator:latest"
#OLLAMA_EVALUATOR_MODEL = "qwen3-sirah-evaluator:latest"

# Judge — hanya dipakai jika JUDGE_PROVIDER = "ollama" (A6000)
OLLAMA_JUDGE_MODEL = "llama3.3:70b"

# URL Ollama — localhost jika jalan di mesin yang sama
OLLAMA_BASE_URL = "http://localhost:11434"
#OLLAMA_BASE_URL = "https://047a-34-143-236-138.ngrok-free.app"  # ngrok (remote)

# ── RUNPOD / vLLM CONFIG ────────────────────────────────────────────
RUNPOD_BASE_URL    = os.getenv("RUNPOD_BASE_URL", "http://localhost:8000/v1")
RUNPOD_API_KEY     = os.getenv("RUNPOD_API_KEY", "runpod_token")
RUNPOD_MODEL       = "Qwen/Qwen2.5-7B-Instruct-AWQ"
RUNPOD_CONCURRENCY = 20

# ── VECTOR DB CONFIG ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
QDRANT_PATH = os.path.join(DATA_DIR, "vectordb", "qdrant_toc_baseline_BACKUP_CHUNK1000")
COLLECTION_NAME = "sirah_nabawiyah_toc"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
QDRANT_VECTOR_SIZE = 1024   # dimensi vector multilingual-e5-large

# ── RETRIEVAL CONFIG ────────────────────────────────────────────────
TOP_K = 5
