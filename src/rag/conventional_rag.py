"""
conventional_rag.py
====================
RAG Konvensional (Non-Agentic) — Pipeline sederhana:
    Query → Embedding → Top-k Retrieval → Satu LLM Call → Jawaban

Digunakan sebagai BASELINE untuk membandingkan dengan Agentic RAG.
Model LLM yang digunakan SAMA persis dengan agentic_rag.py
agar perbandingan fair (Ablation Study).

Usage:
    python src/rag/conventional_rag.py
"""

import re
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    QDRANT_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, QDRANT_VECTOR_SIZE,
    LLM_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    HF_TOKEN, HF_MODEL,
)

# Import OpenRouter jika tersedia di config
try:
    from config import OPENROUTER_API_KEY, OPENROUTER_MODEL  # pyre-ignore[21]
except ImportError:
    OPENROUTER_API_KEY = ""
    OPENROUTER_MODEL = ""

# Import Ollama jika tersedia di config
try:
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL  # pyre-ignore[21]
except ImportError:
    OLLAMA_MODEL = ""
    OLLAMA_BASE_URL = ""

import requests

# ── PROMPT (sama pinciple dengan agentic, tapi 1 shot) ──────────────
GEN_PROMPT = """\
Anda adalah Ahli Sejarah Sirah Nabawiyah yang sangat teliti.
Tugas: Jawab pertanyaan berdasarkan KONTEKS yang tersedia.

KONTEKS:
{context}

ATURAN GENERASI (STRICT):
1. JAWABAN LANGSUNG & PROPORSIONAL: Kalimat pertama harus langsung menjawab inti pertanyaan tanpa basa-basi pembuka. Panjang jawaban harus menyesuaikan dengan kedalaman pertanyaan (jangan potong penjelasan kronologis).
2. SINTESIS KOMPREHENSIF: Susunlah jawaban yang mengalir, runtut, dan selengkap mungkin berdasarkan konteks. Anda boleh memparafrasekan kalimat agar lebih mudah dipahami oleh pembaca.
3. SITASI WAJIB: Setiap fakta/klaim WAJIB diakhiri dengan nomor referensi chunk, misalnya [1] atau [1][2].
4. ANTI-HALUSINASI: DILARANG KERAS menambahkan fakta atau asumsi di luar konteks.
5. FALLBACK: Jika hanya sebagian informasi yang tersedia, jawablah sejauh fakta yang ada. Jika informasi benar-benar kosong total, barulah katakan: "Informasi tersebut tidak ditemukan dalam potongan kitab yang tersedia."

Pertanyaan: {question}

Jawaban (dengan sitasi [N]):"""

# ── KONFIGURASI RETRIEVAL ────────────────────────────────────────────
TOP_K = 5            # Jumlah chunks yang diambil
MAX_TOKENS = 512     # Maks token jawaban
TEMPERATURE = 0.1    # Rendah = lebih konsisten

# ── HEADER untuk bypass ngrok tunnel ────────────────────────────────
HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
    "Bypass-Tunnel-Reminder": "true",
    "User-Agent": "ConventionalRAG/1.0"
}


# ════════════════════════════════════════════════════════════════════
#  INISIALISASI
# ════════════════════════════════════════════════════════════════════

def init_vectordb():
    """Load embedding model dan koneksi ke Qdrant."""
    try:
        from nemo.collections.nlp.models.language_modeling.megatron_bert_model import (  # type: ignore
            MegatronBertModel,
        )
    except ImportError:
        pass

    from qdrant_client import QdrantClient  # pyre-ignore[21]

    print("Memuat model embedding...")

    # Gunakan cara yang sama dengan agentic_rag.py
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from sentence_transformers import SentenceTransformer  # pyre-ignore[21]
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"   [OK] Embedding model loaded: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"   [ERR] Gagal load embedding: {e}")
        sys.exit(1)

    client = QdrantClient(path=QDRANT_PATH)
    count = client.get_collection(COLLECTION_NAME).points_count
    print(f"   [OK] Terhubung ke Qdrant ({count} chunks)")
    return (client, embed_model)


def init_llm():
    """Inisialisasi LLM client berdasarkan provider di config."""
    if LLM_PROVIDER == "gemini":
        from google import genai  # pyre-ignore[21]
        if not GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY tidak ditemukan!"); sys.exit(1)
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"[OK] Gemini client ready ({GEMINI_MODEL})")
        return ("gemini", client)
    elif LLM_PROVIDER == "groq":
        from groq import Groq  # pyre-ignore[21]
        if not GROQ_API_KEY:
            print("❌ GROQ_API_KEY tidak ditemukan!"); sys.exit(1)
        client = Groq(api_key=GROQ_API_KEY)
        print(f"[OK] Groq client ready ({GROQ_MODEL})")
        return ("groq", client)
    elif LLM_PROVIDER == "huggingface":
        from huggingface_hub import InferenceClient  # pyre-ignore[21]
        if not HF_TOKEN:
            print("❌ HF_TOKEN tidak ditemukan!"); sys.exit(1)
        client = InferenceClient(token=HF_TOKEN)
        print(f"[OK] HuggingFace client ready ({HF_MODEL})")
        return ("huggingface", client)
    elif LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            print("❌ OPENROUTER_API_KEY tidak ditemukan!"); sys.exit(1)
        print(f"[OK] OpenRouter client ready ({OPENROUTER_MODEL})")
        return ("openrouter", None)
    elif LLM_PROVIDER == "ollama":
        print(f"[OK] Ollama client ready ({OLLAMA_MODEL}) via REST API")
        return ("ollama", None)
    else:
        print(f"❌ LLM_PROVIDER '{LLM_PROVIDER}' tidak dikenal!"); sys.exit(1)


# ════════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ════════════════════════════════════════════════════════════════════

def retrieve(query: str, embed_model, qdrant_client, top_k: int = TOP_K) -> list[dict]:
    """Ambil top-k chunks relevan dari Qdrant berdasarkan embedding similarity."""
    # Prefix E5 model
    query_vec = embed_model.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    ).tolist()[0]

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k
    ).points

    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append({
            "id": str(hit.id),
            "chunk_id": payload.get("chunk_id", str(hit.id)),
            "text": payload.get("text", ""),
            "score": hit.score,
            "metadata": {
                "bab_title": payload.get("bab_title", "?"),
                "subbab_title": payload.get("subbab_title", "?"),
                "page_start": payload.get("page_start", "?"),
                "page_end": payload.get("page_end", "?"),
            }
        })
    return chunks


# ════════════════════════════════════════════════════════════════════
#  TEXT CLEANING (sama dengan agentic_rag.py)
# ════════════════════════════════════════════════════════════════════




def clean_answer(answer: str) -> str:
    """Bersihkan format training data yang bocor dari output."""
    answer = re.sub(r'\[Trigger\]\s*', '', answer)
    answer = re.sub(r'\[Reaction/Action\]\s*', '', answer)
    answer = re.sub(r'\[Strategic/Political Outcome\]\s*', '', answer)
    answer = re.sub(r'\[Analisis\]\s*', '', answer)
    answer = re.sub(r'\[follow-up\].*?(\[/follow-up\]|$)', '', answer, flags=re.DOTALL)
    answer = re.sub(r'\[output\].*?(\[/output\]|$)', '', answer, flags=re.DOTALL)
    answer = re.sub(r'^\s*Analisis:.*$', '', answer, flags=re.MULTILINE)

    # Hapus baris duplikat
    lines = answer.split('\n')
    seen: set = set()
    unique_lines = []
    for line in lines:
        cl = line.strip()
        if not cl:
            unique_lines.append("")
            continue
        if cl not in seen:
            unique_lines.append(line)
            seen.add(cl)

    return re.sub(r'\n{3,}', '\n\n', "\n".join(unique_lines).strip())


# ════════════════════════════════════════════════════════════════════
#  GENERATION (1 LLM call saja!)
# ════════════════════════════════════════════════════════════════════

def call_llm(llm_client, prompt, temperature=0.1, max_tokens=512) -> str:
    """Panggil LLM dengan retry logic (Sesuai agentic_rag.py)."""
    provider = llm_client[0]
    client = llm_client[1]
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
                r = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                )
                return str(r.choices[0].message.content or "")
            elif provider == "huggingface":
                r = client.chat_completion(
                    model=HF_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                )
                return str(r.choices[0].message.content or "")
            elif provider == "openrouter":
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if "qwen" in OPENROUTER_MODEL.lower():
                    payload["provider"] = {
                        "order": ["DeepInfra"],
                        "allow_fallbacks": False
                    }
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=payload,
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"OpenRouter Error: {r.status_code}")
            elif provider == "ollama":
                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 8192,
                        "stop": ["\nPertanyaan:", "\nJawaban:", "<|im_end|>", "Human:"]
                    }
                }
                r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
                if r.status_code == 200:
                    return r.json().get("response", "")
                else:
                    print(f"DEBUG Ollama Error Body: {r.text}")
                    raise Exception(f"Ollama Error: {r.status_code}")
        except Exception as e:
            if attempt < max_retries and any(k in str(e) for k in ["429", "RESOURCE_EXHAUSTED", "rate_limit"]):
                time.sleep(retry_delays[attempt])
            else:
                raise e
    return ""


def generate(chunks: list[dict], question: str, llm_client) -> str:
    """Buat jawaban dari chunks menggunakan satu LLM call."""
    if not chunks:
        return "Maaf, tidak ditemukan informasi relevan di database."

    context = ""
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        cleaned = c["text"]  # Tidak perlu di-clean lagi karena sudah bersih di DB
        header = f"[{i}] Bab: {meta.get('bab_title','?')} | Hal: {meta.get('page_start','?')}"
        context += f"{header}\n{cleaned}\n\n"

    prompt = GEN_PROMPT.format(context=context, question=question)
    raw = call_llm(llm_client, prompt)
    return clean_answer(raw)


# ════════════════════════════════════════════════════════════════════
#  PIPELINE UTAMA
# ════════════════════════════════════════════════════════════════════

def conventional_rag_query(question: str, vectordb, llm_client) -> dict:
    """
    Pipeline RAG Konvensional:
    1. Retrieve top-k chunks
    2. Generate answer (1 LLM call)
    3. Return answer + sources
    """
    qdrant_client, embed_model = vectordb
    start = time.time()

    # Step 1: Retrieve
    chunks = retrieve(question, embed_model, qdrant_client, top_k=TOP_K)

    # Step 2: Generate
    answer = generate(chunks, question, llm_client)

    elapsed = time.time() - start
    return {
        "answer": answer,
        "chunks": chunks,
        "elapsed": elapsed,
        "latency": elapsed,  # alias agar kompatibel dengan eval_2_generator.py
    }


# ════════════════════════════════════════════════════════════════════
#  MAIN — Interactive Chatbot
# ════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  📚 Conventional RAG — Sirah Nabawiyah Chatbot")
    if LLM_PROVIDER == "groq":
        provider_name = GROQ_MODEL
    elif LLM_PROVIDER == "ollama":
        provider_name = OLLAMA_MODEL
    else:
        provider_name = LLM_PROVIDER
    print(f"  LLM: {LLM_PROVIDER.upper()} ({provider_name})")
    print(f"  Mode: CONVENTIONAL (1 Retrieval + 1 LLM Call)")
    print(f"  Top-K: {TOP_K}")
    print("="*60 + "\n")

    vectordb = init_vectordb()
    llm_client = init_llm()

    print("\n" + "="*60)
    print("  Mulai bertanya! (Ketik 'exit' untuk keluar)")
    print("="*60 + "\n")

    while True:
        try:
            question = input("❓ Pertanyaan: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Keluar.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "keluar"):
            print("👋 Keluar.")
            break

        print("\n🤖 Mencari dan menjawab...\n")
        result = conventional_rag_query(question, vectordb, llm_client)

        print("─" * 60)
        print("💬 JAWABAN:")
        print("─" * 60)
        print(result["answer"])
        print(f"\n⏱️  Waktu: {result['elapsed']:.1f} detik")

        print("\n─" * 60)
        print(f"📖 SUMBER ({len(result['chunks'])} chunks):")
        print("─" * 60)
        for i, c in enumerate(result["chunks"], 1):
            meta = c["metadata"]
            print(f"  [{i}] {meta.get('bab_title','?')} | {meta.get('subbab_title','?')} | "
                  f"Hal. {meta.get('page_start','?')}-{meta.get('page_end','?')} | "
                  f"Skor: {c.get('score', '?'):.3f}")
        print()


if __name__ == "__main__":
    main()
