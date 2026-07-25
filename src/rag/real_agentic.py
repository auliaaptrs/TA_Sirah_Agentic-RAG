"""
real_agentic.py (v7)
====================
"Real Agentic RAG" System.
Filosofi: True Iterative Retrieval with LLM-driven Sufficiency.

Arsitektur:
1.  Initial Retrieval (Top-5) + Neighbor Chunk Expansion
2.  Iterative Sufficiency Loop (max 3 iterasi, safety backstop 5):
    a.  Sufficiency Evaluator — checklist per aspek, aware of search history
    b.  Per-aspect LLM Query Reformulation (IRCoT-inspired)
    c.  Targeted Retrieval (top-3 per aspect) + Neighbor Expansion
    d.  Merge & Dedup
3.  Semantic Reranking (CrossEncoder mMARCO)
4.  Chronological Re-Ranking (Timeline Ordering)
5.  Grounded Generation (dengan sitasi [N])
6.  Hallucination Risk Estimator

Referensi:
- FLARE      : Jiang et al., EMNLP 2023 (arXiv:2305.06983)
               → Iterative active retrieval, bukan one-shot
- IRCoT      : Trivedi et al., 2022 (arXiv:2212.10509)
               → Per-sub-question retrieval > bulk concat
- Adaptive-RAG: Jeong et al., NAACL 2024 (arXiv:2403.14403)
               → Adaptive strategy berbasis evaluasi kebutuhan
- Lost in the Middle: Liu et al., 2023 (arXiv:2307.03172)
               → Ordering konteks mempengaruhi comprehension LLM,
                 memperkuat argumen chronological re-ranking
"""

import re
import sys
import os
import json
import time
from typing import List, Dict, Tuple

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.rag.config import (
    LLM_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    HF_TOKEN, HF_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    OLLAMA_MODEL, OLLAMA_EVALUATOR_MODEL,
    QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    OLLAMA_BASE_URL
)

# ── Konstanta Loop ────────────────────────────────────────────────────────────
MAX_ITERATIONS  = 3   # Maksimum iterasi sufficiency-check + targeted retrieval
SAFETY_BACKSTOP = 5   # Circuit breaker Python — tidak diekspos ke LLM
INITIAL_TOP_K   = 5   # Top-K untuk retrieval awal
ASPECT_TOP_K    = 3   # Top-K per aspek pada targeted retrieval (lebih presisi)

# ── Lazy Reranker ─────────────────────────────────────────────────────────────
_RERANKER_MODEL = None

def get_reranker():
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            
            _RERANKER_MODEL = CrossEncoder(
                'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
                device=device
            )
            print(f"[OK] Reranker loaded: mmarco-mMiniLMv2-L12-H384-v1 (device: {device})")
        except ImportError:
            print("❌ sentence-transformers tidak terinstall! Reranking dilewati.")
            return None
    return _RERANKER_MODEL


# ── PROMPTS ───────────────────────────────────────────────────────────────────

SUFFICIENCY_PROMPT = """\
Anda adalah Ahli Evaluasi Informasi Sirah Nabawiyah.
Tugas: Tentukan apakah konteks yang terkumpul sudah cukup untuk menjawab \
pertanyaan secara LENGKAP dan DETAIL.

PERTANYAAN: {question}

RIWAYAT PENCARIAN (query yang sudah pernah dilakukan):
{search_history}

KONTEKS TERKUMPUL SEJAUH INI:
{context}

INSTRUKSI — Metode Checklist (Deconstruct & Verify):
1. Pecah pertanyaan menjadi daftar fakta spesifik yang HARUS ada.
2. Verifikasi satu per satu apakah tiap item BENAR-BENAR TERDAPAT di konteks.
3. Kumpulkan item "found: false" ke dalam "missing_aspects".
4. "sufficient": true HANYA JIKA missing_aspects kosong.

Balas HANYA dengan JSON valid:
{{
  "checklist": [
    {{"item": "<aspek 1>", "found": true}},
    {{"item": "<aspek 2>", "found": false}}
  ],
  "missing_aspects": ["<aspek yang found: false>"],
  "sufficient": true/false
}}"""

# Prompt reformulasi: satu LLM call per aspek (IRCoT-inspired)
REFORMULATE_PROMPT = """\
Pertanyaan utama: {question}
Aspek yang belum ditemukan: {aspect}

Tulis SATU kalimat query pencarian yang singkat, spesifik, dan alami \
untuk mencari informasi tentang aspek tersebut dalam kitab sirah nabawiyah.
Balas HANYA dengan teks query saja, tanpa penjelasan apapun."""

CONTEXTUALIZE_PROMPT = """\
Diberikan riwayat percakapan berikut dan pertanyaan lanjutan dari pengguna, \
rumuskan ulang pertanyaan lanjutan menjadi pertanyaan mandiri (standalone question) \
yang dapat dipahami tanpa melihat riwayat percakapan. \
Jangan menjawab pertanyaannya, cukup rumuskan ulang. \
Jika pertanyaan sudah jelas tanpa riwayat, kembalikan pertanyaan aslinya.

Riwayat Percakapan:
{chat_history}

Pertanyaan Lanjutan: {question}

Pertanyaan Mandiri:"""

AGENTIC_GEN_PROMPT = """\
Anda adalah Ahli Sejarah Sirah Nabawiyah yang sangat teliti.
Tugas: Jawab pertanyaan berdasarkan KONTEKS yang tersedia.

KONTEKS:
{context}

ATURAN GENERASI (STRICT):
1. JAWABAN LANGSUNG & PROPORSIONAL: Kalimat pertama harus langsung menjawab \
inti pertanyaan tanpa basa-basi. Panjang jawaban proporsional dengan kedalaman \
pertanyaan — jangan potong penjelasan kronologis.
2. SINTESIS KOMPREHENSIF: Susunlah jawaban yang mengalir, runtut, dan selengkap mungkin berdasarkan konteks. Anda boleh memparafrasekan kalimat agar lebih mudah dipahami oleh pembaca.
3. SITASI WAJIB: Setiap fakta/klaim WAJIB diakhiri dengan nomor referensi \
chunk, contoh [1] atau [1][2].
4. ANTI-HALUSINASI: DILARANG KERAS menambahkan fakta di luar konteks.
5. FALLBACK: Jika sebagian informasi ada, jawab sejauh fakta yang tersedia. \
Jika kosong total, katakan: "Informasi tersebut tidak ditemukan dalam potongan \
kitab yang tersedia."

Pertanyaan: {question}

Jawaban (dengan sitasi [N]):"""

ROUTER_PROMPT = """\
Anda adalah Router Ahli RAG.
Tugas: Klasifikasikan pertanyaan pengguna ke dalam 2 kategori:
- FAKTUAL: Pertanyaan spesifik dan sederhana yang bisa dijawab dengan mencari 1 dokumen tunggal (contoh: "Siapa...", "Di mana...", "Apa nama...", "Berapa...").
- KOMPLEKS: Pertanyaan yang butuh banyak dokumen, kronologi, atau hubungan sebab-akibat (contoh: "Bagaimana proses...", "Ceritakan...", "Mengapa...").

Keluarkan HANYA satu kata: "FAKTUAL" atau "KOMPLEKS".

Pertanyaan: {question}
Kategori:"""


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────

def call_llm(llm_client, prompt: str, temperature: float = 0.1,
             max_tokens: int = 1024, model_override: str = None) -> str:
    provider = llm_client[0]
    client   = llm_client[1]
    try:
        if provider == "gemini":
            r = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            return str(r.text or "")
        elif provider == "groq":
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens
            )
            return str(r.choices[0].message.content or "")
        elif provider == "openrouter":
            import requests
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "max_tokens": max_tokens
            }
            if "qwen" in OPENROUTER_MODEL.lower():
                payload["provider"] = {
                    "order": ["DeepInfra"],
                    "allow_fallbacks": False
                }
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload
            )
            return res.json()['choices'][0]['message']['content']
        elif provider == "ollama":
            import requests
            model_to_use = model_override if model_override else OLLAMA_MODEL
            payload = {
                "model": model_to_use, "prompt": prompt,
                "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}
            }
            res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=600)
            return res.json().get("response", "")
    except Exception as e:
        print(f"LLM Error ({provider}): {e}")
    return ""


def retrieve(query: str, embed_model, qdrant_client,
             top_k: int = 5) -> List[Dict]:
    query_vec = embed_model.encode(
        [f"query: {query}"], normalize_embeddings=True
    ).tolist()[0]
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=query_vec, limit=top_k
    ).points
    chunks = []
    for hit in hits:
        p = hit.payload
        chunks.append({
            "id":       str(hit.id),
            "chunk_id": p.get("chunk_id", str(hit.id)),
            "text":     p["text"],
            "score":    hit.score,
            "metadata": {
                "bab_title":    p.get("bab_title",   "?"),
                "subbab_title": p.get("subbab_title","?"),
                "page_start":   p.get("page_start",  "?"),
                "page_end":     p.get("page_end",    p.get("page_start", "?"))
            }
        })
    return chunks


def expand_with_neighbors(chunks: List[Dict], qdrant_client,
                          window: int = 1) -> List[Dict]:
    """
    [NOVELTY] Neighbor Chunk Expansion.
    Untuk setiap chunk toc_ yang ditemukan, ambil juga chunk tetangga
    berdasarkan nomor urut di chunk_id (toc_XXXX_N → toc_XXXX_{N±1}).
    QASiNa tidak di-expand karena tidak berurutan.
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    seen_ids    = {c["chunk_id"] for c in chunks}
    extra_chunks = []

    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if not cid.startswith("toc_"):
            continue
        parts = cid.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prefix = parts[0]
        try:
            num = int(parts[1])
        except ValueError:
            continue

        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor_id = f"{prefix}_{num + delta}"
            if neighbor_id in seen_ids:
                continue
            try:
                results, _ = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(
                        must=[FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=neighbor_id)
                        )]
                    ),
                    limit=1, with_payload=True,
                )
                for hit in results:
                    p = hit.payload
                    extra_chunks.append({
                        "id":       str(hit.id),
                        "chunk_id": p.get("chunk_id", str(hit.id)),
                        "text":     p["text"],
                        "score":    chunk["score"] * 0.9,
                        "metadata": {
                            "bab_title":    p.get("bab_title",   "?"),
                            "subbab_title": p.get("subbab_title","?"),
                            "page_start":   p.get("page_start",  "?"),
                            "page_end":     p.get("page_end",    p.get("page_start","?"))
                        }
                    })
                    seen_ids.add(neighbor_id)
            except Exception:
                pass

    return chunks + extra_chunks


def _merge_dedup(base: List[Dict], new: List[Dict]) -> List[Dict]:
    """Gabungkan dua list chunks, deduplikasi berdasarkan 'id'."""
    seen_ids = {c["id"] for c in base}
    added = 0
    for nc in new:
        if nc["id"] not in seen_ids:
            base.append(nc)
            seen_ids.add(nc["id"])
            added += 1
    return base, added


def _sort_chronologically(chunks: List[Dict]) -> List[Dict]:
    """
    Mengurutkan chunks secara kronologis berdasarkan struktur ID.

    Urutan prioritas:
    1. qasina_* — Factual baseline (urut berdasarkan nomor)
    2. toc_*    — Buku Sirah (urut berdasarkan nomor bab/sub)
    3. Lainnya  — Diletakkan di akhir

    Argumen: Dalam sistem iteratif yang mengakumulasi chunks dari
    berbagai targeted query, chunk berasal dari bagian timeline yang
    berbeda-beda. Chronological re-ranking memastikan generator menerima
    narasi yang runtut, konsisten dengan temuan Liu et al. (2023) bahwa
    ordering konteks secara signifikan mempengaruhi kualitas output LLM.
    """
    def extract_sort_key(c):
        cid = str(c.get("chunk_id", c.get("id", "0")))
        if cid.startswith("qasina_"):
            nums = re.findall(r'\d+', cid)
            return (1, [int(n) for n in nums] if nums else [0])
        elif cid.startswith("toc_"):
            nums = re.findall(r'\d+', cid)
            return (2, [int(n) for n in nums] if nums else [0])
        else:
            return (3, [cid])

    return sorted(chunks, key=extract_sort_key)


def _estimate_hallucination_risk(answer: str, chunks: List[Dict]) -> float:
    """Heuristik sederhana: proporsi angka di jawaban yang tidak ada di konteks."""
    context_text = " ".join(c["text"] for c in chunks).lower()
    
    # Hapus sitasi [1], [2], dst agar tidak dihitung sebagai angka halusinasi
    answer_clean = re.sub(r'\[\d+\]', '', answer)
    
    numbers = re.findall(r'\b\d+\b', answer_clean)
    if not numbers:
        return 0.0
    found = sum(1 for n in numbers if n in context_text)
    return round(1.0 - (found / len(numbers)), 2)


def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """Mengurutkan ulang chunks menggunakan CrossEncoder mMARCO (multilingual)."""
    if not chunks:
        return []
    model = get_reranker()
    if not model:
        return chunks

    pairs  = [[query, c['text']] for c in chunks]
    scores = model.predict(pairs)
    for i, score in enumerate(scores):
        chunks[i]['rerank_score'] = float(score)

    return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)


def _reformulate_query(question: str, aspect: str, llm_client) -> str:
    """
    LLM-based query reformulation per aspek (IRCoT-inspired).
    Menghasilkan query yang natural dan spesifik, bukan sekadar
    konkatenasi string.

    Referensi: Trivedi et al. (2022) IRCoT — arXiv:2212.10509
    """
    prompt = REFORMULATE_PROMPT.format(question=question, aspect=aspect)
    result = call_llm(llm_client, prompt, temperature=0.1, max_tokens=80,
                      model_override=OLLAMA_EVALUATOR_MODEL)
    result = result.strip().strip('"\' ').strip()

    # Fallback jika LLM mengembalikan hasil kosong atau terlalu panjang
    if not result or len(result) > 300:
        return f"Dalam konteks {question}, jelaskan tentang: {aspect}"
    return result


def _run_sufficiency_check(
    question: str,
    chunks: List[Dict],
    search_history: List[str],
    llm_client,
    preview_chars: int = 300
) -> Tuple[Dict, str]:
    """
    Menjalankan sufficiency evaluator dengan konteks riwayat pencarian.
    Mengembalikan (eval_res_dict, raw_string).
    """
    context_preview = "\n".join([
        f"[{i+1}] {c['text'][:preview_chars]}..."
        for i, c in enumerate(chunks)
    ])
    history_str = "\n".join([f"- {q}" for q in search_history]) or "- (belum ada)"

    prompt = SUFFICIENCY_PROMPT.format(
        question=question,
        search_history=history_str,
        context=context_preview
    )
    raw = call_llm(llm_client, prompt, model_override=OLLAMA_EVALUATOR_MODEL)

    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group(0))
    except Exception:
        parsed = {"sufficient": True, "missing_aspects": [], "checklist": []}

    return parsed, raw

def classify_query_complexity(question: str, llm_client) -> str:
    """Mengklasifikasikan pertanyaan menjadi FAKTUAL atau KOMPLEKS.
    Menggunakan OLLAMA_EVALUATOR_MODEL (fine-tuned) untuk routing yang lebih presisi.
    """
    prompt = ROUTER_PROMPT.format(question=question)
    raw = call_llm(llm_client, prompt, temperature=0.0, max_tokens=10,
                   model_override=OLLAMA_EVALUATOR_MODEL)
    if "faktual" in raw.lower():
        return "FAKTUAL"
    return "KOMPLEKS"

def _contextualize_query(question: str, chat_history: List[Dict], llm_client) -> str:
    """Merumuskan ulang pertanyaan lanjutan menjadi standalone query berdasarkan chat history."""
    if not chat_history:
        return question
    history_str = ""
    for msg in chat_history:
        history_str += f"User: {msg['question']}\nSystem: {msg['answer'][:300]}...\n\n"
    
    prompt = CONTEXTUALIZE_PROMPT.format(
        chat_history=history_str.strip(),
        question=question
    )
    result = call_llm(llm_client, prompt, temperature=0.0, max_tokens=100,
                      model_override=OLLAMA_EVALUATOR_MODEL)
    result = result.strip()
    if result:
        return result
    return question


# ── FUNGSI UTAMA ──────────────────────────────────────────────────────────────

def real_agentic_rag_query(question: str, vectordb, llm_client,
                           use_chronological: bool = True,
                           chat_history: List[Dict] = None) -> Dict:
    """
    Pipeline Real Agentic RAG (v7).

    Parameters:
    - use_chronological: Aktifkan Chronological Re-Ranking (default True).
                         Set False untuk ablasi tanpa chronological sort.
    - chat_history     : List dict berisi 'question' dan 'answer' dari turn sebelumnya.

    Return dict mencakup:
    - answer          : str
    - chunks          : List[Dict]
    - hallucination_risk: float
    - sufficiency_analysis: Dict  (hasil evaluasi iterasi terakhir)
    - logs            : List[str]
    - elapsed         : float (detik)
    - llm_calls       : int   (total LLM call, dinamis)
    - iterations      : int   (iterasi loop yang dijalankan)
    - search_history  : List[str]
    """
    qdrant_client, embed_model = vectordb
    start_time      = time.time()
    logs            = []
    llm_call_counter = 0

    # ── 0. Contextualize Query (Multi-turn Support) ─────────────────────────
    if chat_history:
        logs.append(f"[Context] Memproses chat history ({len(chat_history)} turns).")
        question = _contextualize_query(question, chat_history, llm_client)
        llm_call_counter += 1
        logs.append(f"[Context] Standalone Query: '{question}'")

    # ── 1. Initial Retrieval + Neighbor Expansion ─────────────────────────────
    chunks = retrieve(question, embed_model, qdrant_client, top_k=INITIAL_TOP_K)
    logs.append(f"[Init] Retrieval awal: {len(chunks)} chunks.")

    chunks = expand_with_neighbors(chunks, qdrant_client, window=1)
    logs.append(f"[Init] Setelah neighbor expansion: {len(chunks)} chunks.")

    search_history = [question]
    eval_res       = {"sufficient": False, "missing_aspects": []}
    iteration      = 0

    # ── 1.5 Factual Routing (Adaptive RAG) ────────────────────────────────────
    complexity = classify_query_complexity(question, llm_client)
    logs.append(f"[Router] Kategori pertanyaan: {complexity}")
    current_max_iterations = 0 if complexity == "FAKTUAL" else MAX_ITERATIONS

    # ── 2. Iterative Sufficiency Loop ─────────────────────────────────────────
    #
    # Struktur yang dikoreksi (berbeda dari v6):
    #   Sufficiency check SELALU di AWAL loop → iterasi terakhir pun tetap
    #   mendapat verifikasi setelah retrieval sebelumnya selesai.
    #   Safety backstop (SAFETY_BACKSTOP) adalah penjaga Python murni —
    #   tidak diekspos ke LLM, setara circuit breaker LangChain/LlamaIndex.
    #
    # Referensi: FLARE (Jiang et al., 2023) — retrieval aktif berbasis
    #            kebutuhan, bukan jadwal tetap.

    while iteration < SAFETY_BACKSTOP:
        # 2a. Cek apakah ini soal faktual (iterasi 0). Jika iya, langsung skip loop.
        if current_max_iterations == 0:
            logs.append(f"[Iter {iteration}] Soal Faktual. Langsung menggunakan konteks awal.")
            break

        # 2b. Sufficiency check — selalu di atas, termasuk setelah retrieval
        eval_res, _ = _run_sufficiency_check(
            question, chunks, search_history, llm_client
        )
        llm_call_counter += 1

        missing_aspects = eval_res.get("missing_aspects", [])
        is_sufficient   = eval_res.get("sufficient", False)

        logs.append(
            f"[Iter {iteration}] Sufficient={is_sufficient} | "
            f"Missing={missing_aspects}"
        )

        # 2b. Berhenti jika sudah cukup
        if is_sufficient:
            logs.append(f"[Iter {iteration}] ✓ Sufficient — keluar dari loop.")
            break

        # 2c. Batas iterasi tercapai → lanjut generation dengan bukti yang ada
        #     Agen sudah melakukan current_max_iterations kali sufficiency check
        if iteration >= current_max_iterations - 1:
            logs.append(
                f"[Iter {iteration}] Max iterasi ({current_max_iterations}) tercapai. "
                f"Aspect yang masih hilang: {missing_aspects}. "
                f"Generate dengan bukti yang ada."
            )
            break

        # 2d. Per-aspect LLM Query Reformulation + Targeted Retrieval
        #     Referensi: IRCoT (Trivedi et al., 2022) — per-sub-question
        #     retrieval lebih efektif daripada bulk string concat.
        for aspect in missing_aspects:
            # LLM reformulasi query (1 call per aspek)
            new_query = _reformulate_query(question, aspect, llm_client)
            llm_call_counter += 1

            # Skip jika query duplikat (cegah loop sia-sia)
            if new_query in search_history:
                logs.append(f"  [Skip] Duplikat query: '{new_query[:70]}'")
                continue

            logs.append(f"  [Aspect] '{aspect[:60]}' → Query: '{new_query[:80]}'")

            # Targeted retrieval top-3 per aspek (lebih presisi dari top-5)
            new_chunks = retrieve(
                new_query, embed_model, qdrant_client, top_k=ASPECT_TOP_K
            )
            # Ekspansi tetangga dimatikan di dalam loop untuk mencegah Context Overload
            # new_chunks = expand_with_neighbors(new_chunks, qdrant_client, window=1)

            # Merge & dedup
            chunks, added = _merge_dedup(chunks, new_chunks)
            search_history.append(new_query)
            logs.append(
                f"  [Merge] +{added} chunk baru (total: {len(chunks)})"
            )

        iteration += 1

    # ── 3. Semantic Reranking & CRAG Filtering (Top-5) ────────────────────────
    chunks = rerank_chunks(question, chunks)
    chunks = chunks[:5]  # Filter ketat untuk mencegah Lost in the Middle
    top_score = chunks[0].get('rerank_score', 0.0) if chunks else 0.0
    logs.append(
        f"[Rerank] CrossEncoder applied — {len(chunks)} chunks, "
        f"top score: {top_score:.4f}"
    )

    # ── 4. Chronological Re-Ranking (opsional, default aktif) ────────────────
    #
    # Dampak lebih besar dalam sistem iteratif:
    # Chunk dari berbagai targeted query berasal dari bagian timeline berbeda.
    # Sorting kronologis memastikan generator menerima narasi yang runtut.
    # Referensi: Liu et al. (2023) "Lost in the Middle" — ordering konteks
    # mempengaruhi comprehension LLM secara signifikan.
    if use_chronological:
        original_ids = [c["id"] for c in chunks]
        chunks = _sort_chronologically(chunks)
        if [c["id"] for c in chunks] != original_ids:
            logs.append("[Sort] Chronological re-ranking applied — urutan berubah.")
    else:
        logs.append("[Sort] Chronological re-ranking DINONAKTIFKAN (ablasi).")

    # ── 5. Grounded Generation (sitasi [N] dipertahankan) ────────────────────
    context_full = ""
    for i, c in enumerate(chunks, 1):
        context_full += f"[{i}] {c['text']}\n\n"

    gen_prompt = AGENTIC_GEN_PROMPT.format(
        context=context_full, question=question
    )
    answer = call_llm(llm_client, gen_prompt)
    llm_call_counter += 1
    logs.append(f"[Gen] Jawaban dihasilkan. Total LLM calls: {llm_call_counter}")

    # ── 6. Hallucination Risk Estimator ───────────────────────────────────────
    risk = _estimate_hallucination_risk(answer, chunks)

    return {
        "answer":               answer,
        "chunks":               chunks,
        "hallucination_risk":   risk,
        "sufficiency_analysis": eval_res,
        "logs":                 logs,
        "elapsed":              time.time() - start_time,
        "llm_calls":            llm_call_counter,     # ← dinamis, bukan hardcode
        "iterations":           iteration,             # ← untuk analisis TA
        "search_history":       search_history,        # ← audit trail
    }


def real_agentic_retriever_only(question: str, vectordb, llm_client) -> List[Dict]:
    """
    Hanya menjalankan tahap retrieval agentic iteratif (tanpa generation).
    Digunakan untuk Eval 1 (retrieval quality) — konsisten dengan pipeline penuh.
    """
    qdrant_client, embed_model = vectordb

    # 1. Initial Retrieval + Neighbor Expansion
    chunks = retrieve(question, embed_model, qdrant_client, top_k=INITIAL_TOP_K)
    chunks = expand_with_neighbors(chunks, qdrant_client, window=1)

    search_history = [question]
    iteration      = 0

    complexity = classify_query_complexity(question, llm_client)
    current_max_iterations = 0 if complexity == "FAKTUAL" else MAX_ITERATIONS

    # 2. Iterative Sufficiency Loop (tanpa generation)
    while iteration < SAFETY_BACKSTOP:
        eval_res, _ = _run_sufficiency_check(
            question, chunks, search_history, llm_client
        )

        missing_aspects = eval_res.get("missing_aspects", [])
        is_sufficient   = eval_res.get("sufficient", False)

        if is_sufficient:
            break

        if iteration >= current_max_iterations - 1:
            break

        for aspect in missing_aspects:
            new_query = _reformulate_query(question, aspect, llm_client)
            if new_query in search_history:
                continue
            new_chunks = retrieve(
                new_query, embed_model, qdrant_client, top_k=ASPECT_TOP_K
            )
            # no expansion
            chunks, _ = _merge_dedup(chunks, new_chunks)
            search_history.append(new_query)

        iteration += 1

    # 3. Rerank, Filter (Top 5), dan kembalikan (tanpa sorting kronologis karena untuk eval retrieval semantik)
    chunks = rerank_chunks(question, chunks)
    return chunks[:5]


# ── BACKWARD COMPATIBILITY ALIAS ─────────────────────────────────────────────
# Agar eval_2_generator.py yang memanggil new_agentic.new_agentic_rag_query
# tetap bisa dipakai tanpa refactor. Cukup ganti import di eval script menjadi:
#   import src.rag.real_agentic as new_agentic
def new_agentic_rag_query(question: str, vectordb, llm_client) -> Dict:
    return real_agentic_rag_query(question, vectordb, llm_client)

def new_agentic_retriever_only(question: str, vectordb, llm_client) -> List[Dict]:
    return real_agentic_retriever_only(question, vectordb, llm_client)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.rag.utils import init_vectordb, init_llm
    vdb = init_vectordb()
    llm = init_llm()

    test_q = "Sebutkan penyebab dan hasil Perang Badar?"
    res    = real_agentic_rag_query(test_q, vdb, llm)

    print("\n" + "="*60)
    print("LOGS:")
    for log in res["logs"]:
        print(f"  {log}")
    print(f"\nIterasi: {res['iterations']} | LLM Calls: {res['llm_calls']}")
    print(f"Elapsed : {res['elapsed']:.1f}s | Halusinasi Risk: {res['hallucination_risk']}")
    print(f"\nSearch History:")
    for q in res["search_history"]:
        print(f"  → {q}")
    print(f"\nJawaban:\n{res['answer']}")