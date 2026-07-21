"""
generate_multihop_dataset.py
============================
Multi-Hop Golden Dataset Generator untuk evaluasi Agentic RAG Sirah Nabawiyah.

PEROMBAKAN TOTAL dari generate_goldendataset.py lama:
- LAMA : single-chunk context  → pertanyaan selalu bisa dijawab 1 chunk
- BARU : multi-chunk context   → pertanyaan butuh sintesis lintas bab

Dua mode generasi:
1. INTRA-BAB  : Gabungkan semua chunk dalam 1 bab → pertanyaan sintesis satu bab
2. CROSS-BAB  : Gabungkan 2 bab berdekatan → pertanyaan komparasi/lintas episode

Output:
  benchmark_multihop.jsonl : dataset evaluasi (BUKAN fine-tuning)
    - Setiap record menyertakan `gold_chunk_ids` (list) untuk eval retrieval
    - QASiNa tidak diikutkan sama sekali

Usage:
    python generate_multihop_dataset.py --mode both --target 150 --dry-run
    python generate_multihop_dataset.py --mode intra --target 100
    python generate_multihop_dataset.py --mode cross --target 50
"""

import asyncio
import json
import os
import sys
import re
import time
import argparse
import random
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict

# ── PATH CONFIG ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAG_DIR     = os.path.join(BASE_DIR, "src", "rag")

CHUNKS_PATH  = os.path.join(DATA_DIR, "vectordb", "chunks_toc_baseline_BACKUP_CHUNK1000.json")
OUTPUT_DIR   = os.path.join(DATA_DIR, "finetune_dataset")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "multihop_goldentruth.jsonl")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_multihop.json")

sys.path.insert(0, BASE_DIR)
from src.rag.config import OPENROUTER_API_KEY

# ── MODEL ──────────────────────────────────────────────────────────────────
GENERATOR_MODEL = "qwen/qwen-2.5-72b-instruct"  # Sangat pintar, murah, dan open-weight terbaik saat ini
JUDGE_MODEL     = "google/gemini-2.0-flash-001"

# ── KONSTANTA ──────────────────────────────────────────────────────────────
MAX_CHUNKS_PER_CONTEXT = 8  # 8 chunk = ~6000 kata (Sangat cukup untuk multi-hop)
MIN_CHUNKS_PER_BAB = 3      # Minimal chunk per bab agar layak digabung
MAX_CROSS_PAIRS    = 30     # Max pasangan cross-bab yang dibuat
JUDGE_PASS_SCORE   = 7      # Minimal skor judge agar QA diterima
GLOBAL_QUESTION_BANK = []

# ── MULTI-HOP GENERATE PROMPT ──────────────────────────────────────────────
# Kunci perbedaan: context panjang, dilarang single-paragraph answer
MULTIHOP_GENERATE_PROMPT = """\
Anda adalah pembuat dataset evaluasi RAG tingkat penelitian untuk buku Sirah Nabawiyah \
Ar-Raheeq Al-Makhtum.

Anda diberi TEKS PANJANG yang terdiri dari {num_chunks} potongan dari {context_desc}.
Buat TEPAT {num_questions} pasangan QA, dengan proporsi MERATA untuk keempat tipe di bawah ini (misal jika diminta 20 soal, buat masing-masing tipe 5 soal).

DEFINISI TIPE:
- factual      : fakta spesifik (tokoh/tempat/angka/waktu). Boleh dijawab dari 1-2 kalimat saja. Jawaban padat & komprehensif, tidak bertele-tele.
- synthesis    : mensintesis informasi dari beberapa chunk menjadi satu jawaban utuh
- causal_chain : rantai sebab-akibat yang melibatkan fakta dari beberapa chunk teks
- timeline     : urutan kejadian yang mencakup lebih dari satu episode/peristiwa

SYARAT KETAT (WAJIB SEMUA DIPENUHI):
1. KHUSUS factual: Jawaban maksimal 3 kalimat padat. Dilarang keras berhalusinasi. Boleh diambil dari SATU chunk saja.
2. KHUSUS synthesis, causal_chain, timeline: DILARANG KERAS membuat pertanyaan yang jawabannya hanya ada di SATU chunk. Wajib menggabungkan info dari minimal 2 chunk/potongan teks yang berbeda (terpisah oleh tanda [---]).
3. Pertanyaan HARUS menyebut nama tokoh/peristiwa/lokasi yang spesifik.
4. Pertanyaan HARUS self-contained (bisa dipahami tanpa membaca konteks).
5. Jawaban synthesis minimal 3 kalimat substantif.
6. Jawaban causal_chain wajib format: "Pertama... → Kemudian... → Akhirnya..."
7. Jawaban timeline wajib format bernomor: "1. ... 2. ... 3. ..."
8. Difficulty HARUS "easy" untuk factual, dan "hard" untuk tipe lainnya.

CONTOH pertanyaan YANG BENAR:
- [Factual] "Siapa tokoh kafir Quraisy yang bertugas menjaga sumur saat Perang Badar?"
- [Synthesis] "Apa peran Khalid bin Al-Walid sebelum dan sesudah memeluk Islam, dan bagaimana perubahannya mencerminkan dinamika kekuatan saat Fathu Makkah?"

METADATA:
Sumber: {context_desc}
Halaman: {page_range}

TEKS GABUNGAN:
\"\"\"{combined_text}\"\"\"

FORMAT OUTPUT JSON ARRAY ONLY (tanpa teks apapun di luar JSON):
[
  {{
    "type": "factual|synthesis|causal_chain|timeline",
    "difficulty": "easy|hard",
    "question": "...",
    "answer": "...",
    "requires_chunks": ["potongan_1", "potongan_2"]
  }}
]
"""

# ── JUDGE PROMPT ───────────────────────────────────────────────────────────
MULTIHOP_JUDGE_PROMPT = """\
Anda adalah evaluator ketat untuk dataset evaluasi RAG multi-hop.

TEKS GABUNGAN (konteks sumber):
\"\"\"{combined_text_preview}\"\"\"

TYPE: {q_type}
Q: {question}
A: {answer}

Nilai 1-10 berdasarkan kriteria KETAT:
1. Sesuai tipe: Jika tipe 'factual', tidak perlu multi-hop. Jika tipe lain, WAJIB multi-hop (minimal mengambil fakta dari 2 chunk/potongan teks yang berbeda).
2. Self-contained (pertanyaan bisa dipahami tanpa konteks)
3. Grounded (semua fakta ada di teks)
4. Kualitas jawaban (lengkap, substantif, sesuai tipe)
5. Natural bahasa Indonesia

Berikan penalti besar (-3 poin) jika:
- Pertanyaan bertipe SELAIN factual bisa dijawab dari satu kalimat saja
- Jawaban mengandung fakta yang tidak ada di teks (halusinasi)
- Jawaban factual terlalu bertele-tele (> 3 kalimat)
- Jawaban synthesis terlalu pendek

Jawab HANYA JSON tanpa markdown:
{{"score": 8, "reason": "...", "is_truly_multihop": true_or_false_sesuai_fakta}}
"""

# ── OPENROUTER CLIENT ──────────────────────────────────────────────────────
def build_async_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

# ── LLM CALL ──────────────────────────────────────────────────────────────
async def call_model(client, model, prompt, temperature=0.5, max_tokens=2500, retries=5):
    delays = [5, 10, 20, 40, 60]
    for attempt in range(retries + 1):
        try:
            extra = {}
            if "qwen" in model.lower():
                extra = {
                    "extra_body": {
                        "provider": {
                            "order": ["DeepInfra"],
                            "allow_fallbacks": False
                        }
                    }
                }
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **extra
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            if attempt < retries:
                wait = delays[min(attempt, len(delays)-1)]
                print(f"    ⚠ retry {attempt+1}/{retries} in {wait}s: {err[:80]}")
                await asyncio.sleep(wait)
            else:
                print(f"    ✗ permanent fail: {err[:100]}")
                return ""

# ── JSON PARSERS ──────────────────────────────────────────────────────────
def parse_json_list(text):
    clean = re.sub(r"^```(?:json)?", "", text.strip()).replace("```", "").strip()
    try:
        obj = json.loads(clean)
        if isinstance(obj, list):
            return obj
    except:
        pass
    m = re.search(r'\[.*\]', clean, re.DOTALL)
    if m:
        try:
            return json.loads(re.sub(r'[\x00-\x1F\x7F]', '', m.group()))
        except:
            pass
    return []

def parse_json_dict(text):
    clean = re.sub(r"^```(?:json)?", "", text.strip()).replace("```", "").strip()
    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return obj
    except:
        pass
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        try:
            return json.loads(re.sub(r'[\x00-\x1F\x7F]', '', m.group()))
        except:
            pass
    return {}

# ── DUPLICATE GUARD ───────────────────────────────────────────────────────
def is_duplicate(q, threshold=0.82):
    nq = q.lower().strip()
    for old in GLOBAL_QUESTION_BANK:
        if SequenceMatcher(None, nq, old).ratio() >= threshold:
            return True
    return False

def register_q(q):
    GLOBAL_QUESTION_BANK.append(q.lower().strip())

# ── CHUNK GROUPING ────────────────────────────────────────────────────────
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_chunks_by_bab(chunks):
    """Group chunks by bab_title, maintaining original order."""
    bab_map = defaultdict(list)
    for chunk in chunks:
        bab = chunk.get("metadata", {}).get("bab_title", "UNKNOWN")
        bab_map[bab].append(chunk)
    return bab_map

def build_intra_bab_contexts(bab_map, max_chunks=MAX_CHUNKS_PER_CONTEXT):
    """
    Untuk setiap bab, gabungkan chunk-chunk menjadi satu konteks panjang.
    Jika bab sangat panjang, bagi menjadi beberapa window yang overlap.
    Returns: list of context_dicts
    """
    contexts = []
    for bab_title, chunks in bab_map.items():
        if len(chunks) < MIN_CHUNKS_PER_BAB:
            continue

        # Gabungkan text semua chunk dalam bab
        full_text_chunks = []
        chunk_ids = []
        word_count = 0

        for chunk in chunks:
            words = chunk["text"].split()
            if len(full_text_chunks) >= max_chunks:
                # Simpan window ini
                contexts.append({
                    "context_type": "intra_bab",
                    "bab_title": bab_title,
                    "bab_titles": [bab_title],
                    "chunks": full_text_chunks.copy(),
                    "chunk_ids": chunk_ids.copy(),
                    "combined_text": "\n\n[---]\n\n".join(
                        f"[Chunk {i+1}] {c['text']}"
                        for i, c in enumerate(full_text_chunks)
                    ),
                    "context_desc": f"Bab: {bab_title}",
                    "page_range": f"{chunks[0]['metadata'].get('page_start','?')}-{chunks[-1]['metadata'].get('page_end','?')}",
                    "num_chunks": len(full_text_chunks),
                })
                # Rolling window: pertahankan chunk terakhir untuk overlap
                overlap_chunk = full_text_chunks[-1]
                overlap_id = chunk_ids[-1]
                full_text_chunks = [overlap_chunk, chunk]
                chunk_ids = [overlap_id, chunk.get("id", str(id(chunk)))]
                word_count = len(overlap_chunk["text"].split()) + len(words)
            else:
                full_text_chunks.append(chunk)
                chunk_ids.append(chunk.get("id", str(id(chunk))))
                word_count += len(words)

        # Simpan sisa window jika minimal chunks
        if len(full_text_chunks) >= MIN_CHUNKS_PER_BAB:
            contexts.append({
                "context_type": "intra_bab",
                "bab_title": bab_title,
                "bab_titles": [bab_title],
                "chunks": full_text_chunks,
                "chunk_ids": chunk_ids,
                "combined_text": "\n\n[---]\n\n".join(
                    f"[Chunk {i+1}] {c['text']}"
                    for i, c in enumerate(full_text_chunks)
                ),
                "context_desc": f"Bab: {bab_title}",
                "page_range": f"{full_text_chunks[0]['metadata'].get('page_start','?')}-{full_text_chunks[-1]['metadata'].get('page_end','?')}",
                "num_chunks": len(full_text_chunks),
            })

    return contexts

def build_cross_bab_contexts(bab_map, max_chunks=MAX_CHUNKS_PER_CONTEXT, max_pairs=MAX_CROSS_PAIRS):
    """
    Pasangkan bab-bab berdekatan (consecutive dalam kitab) untuk cross-chapter QA.
    Ambil potongan dari masing-masing bab agar total chunk tidak melebihi max_chunks.
    Returns: list of context_dicts
    """
    bab_list = list(bab_map.items())
    contexts = []

    for i in range(len(bab_list) - 1):
        if len(contexts) >= max_pairs:
            break

        bab_a_title, chunks_a = bab_list[i]
        bab_b_title, chunks_b = bab_list[i + 1]

        if len(chunks_a) < 2 or len(chunks_b) < 2:
            continue

        # Ambil subset chunks dari masing-masing bab (balanced)
        half = max_chunks // 2
        selected_a = chunks_a[:half]
        ids_a = [c.get("id", str(id(c))) for c in selected_a]
        
        selected_b = chunks_b[:half]
        ids_b = [c.get("id", str(id(c))) for c in selected_b]

        if not selected_a or not selected_b:
            continue

        all_chunks = selected_a + selected_b
        all_ids = ids_a + ids_b
        combined = (
            f"=== BAB: {bab_a_title} ===\n\n"
            + "\n\n[---]\n\n".join(f"[Chunk {j+1}] {c['text']}" for j, c in enumerate(selected_a))
            + f"\n\n=== BAB: {bab_b_title} ===\n\n"
            + "\n\n[---]\n\n".join(f"[Chunk {j+1+len(selected_a)}] {c['text']}" for j, c in enumerate(selected_b))
        )

        page_start = chunks_a[0]["metadata"].get("page_start", "?")
        page_end = chunks_b[-1]["metadata"].get("page_end", "?")

        contexts.append({
            "context_type": "cross_bab",
            "bab_title": f"{bab_a_title} + {bab_b_title}",
            "bab_titles": [bab_a_title, bab_b_title],
            "chunks": all_chunks,
            "chunk_ids": all_ids,
            "combined_text": combined,
            "context_desc": f"Bab: {bab_a_title} → {bab_b_title}",
            "page_range": f"{page_start}-{page_end}",
            "num_chunks": len(all_chunks),
        })

    return contexts

# ── GENERATE + JUDGE PIPELINE ─────────────────────────────────────────────
VALID_TYPES = {"factual", "synthesis", "causal_chain", "timeline"}

async def generate_and_judge(idx, total, ctx, client, semaphore, dry_run=False):
    """
    Satu siklus: generate QA dari konteks multi-chunk, judge tiap QA.
    Returns list of accepted QA dicts siap tulis ke JSONL.
    """
    async with semaphore:
        desc = ctx["context_desc"]
        print(f"\n[{idx+1}/{total}] {ctx['context_type'].upper()} | {desc[:60]}")
        print(f"           Chunks: {ctx['num_chunks']} | Words: {len(ctx['combined_text'].split())}")

        if dry_run:
            print("    [DRY-RUN] skip LLM call")
            return []

        # Tentukan jumlah QA: Lebih sedikit per window = Kualitas lebih tinggi
        # Intra-bab = 12 soal (3 per tipe)
        # Cross-bab = 8 soal (2 per tipe)
        num_q = 12 if ctx["context_type"] == "intra_bab" else 8

        prompt = MULTIHOP_GENERATE_PROMPT.format(
            num_chunks=ctx["num_chunks"],
            context_desc=ctx["context_desc"],
            num_questions=num_q,
            page_range=ctx["page_range"],
            combined_text=ctx["combined_text"],  # FULL TEXT, no truncation!
        )

        raw = await call_model(client, GENERATOR_MODEL, prompt, temperature=0.55, max_tokens=8000)
        parsed = parse_json_list(raw)

        if not parsed:
            print("    ✗ parse fail — no QA generated")
            return []

        accepted = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            t = item.get("type", "").lower().strip()

            if not q or not a or t not in VALID_TYPES:
                print(f"    ✗ skip invalid type={t}")
                continue

            if len(q) < 25:
                print(f"    ✗ skip too short Q: {q[:50]}")
                continue

            if is_duplicate(q):
                print(f"    ✗ skip duplicate Q: {q[:60]}")
                continue

            # Hard filter: jawaban harus cukup panjang untuk multi-hop
            min_words = {"factual": 1, "synthesis": 20, "causal_chain": 15, "timeline": 10}
            if len(a.split()) < min_words.get(t, 15):
                print(f"    ✗ skip too-short answer ({len(a.split())} words) for {t}")
                continue

            # Judge QA
            judge_prompt = MULTIHOP_JUDGE_PROMPT.format(
                combined_text_preview=ctx["combined_text"], # FULL TEXT untuk judge
                q_type=t,
                question=q,
                answer=a,
            )
            judge_raw = await call_model(
                client, JUDGE_MODEL, judge_prompt,
                temperature=0.0, max_tokens=200
            )
            judge = parse_json_dict(judge_raw)
            score = int(judge.get("score", 5))
            is_multihop = judge.get("is_truly_multihop", False)

            print(f"    QA [{t}] score={score} multihop={is_multihop} | {q[:60]}")

            if score < JUDGE_PASS_SCORE:
                print(f"    ✗ rejected (score {score} < {JUDGE_PASS_SCORE})")
                continue

            # Hanya tolak jika BUKAN factual tapi gagal multi-hop
            if t != "factual" and not is_multihop:
                print(f"    ✗ rejected (not truly multi-hop per judge untuk tipe {t})")
                continue

            register_q(q)
            accepted.append({
                "q": q,
                "a": a,
                "type": t,
                "score": score,
                "context_type": ctx["context_type"],
                "bab_titles": ctx["bab_titles"],
                "gold_chunk_ids": ctx["chunk_ids"],
                "context_desc": ctx["context_desc"],
                "page_range": ctx["page_range"],
                "num_source_chunks": ctx["num_chunks"],
            })

        print(f"    ✓ accepted {len(accepted)}/{len(parsed)} QA")
        return accepted

# ── WRITE TO JSONL ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Anda adalah asisten ahli Sirah Nabawiyah. "
    "Jawablah pertanyaan berdasarkan konteks yang diberikan secara akurat, "
    "ringkas, dan informatif."
)

def format_record(qa, ctx):
    """Format satu QA menjadi record benchmark JSONL."""
    context_text = ctx["combined_text"]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Konteks:\n{context_text}\n\nPertanyaan: {qa['q']}"},
            {"role": "assistant", "content": qa["a"]},
        ],
        "metadata": {
            "source": "multihop_generated",
            "type": qa["type"],
            "difficulty": "hard",
            "context_type": qa["context_type"],
            "bab_titles": qa["bab_titles"],
            "gold_chunk_ids": qa["gold_chunk_ids"],
            "expected_answer": qa["a"],
            "judge_score": qa["score"],
            "num_source_chunks": qa["num_source_chunks"],
            "context_desc": qa["context_desc"],
            "page_range": qa["page_range"],
        }
    }

def write_record(record, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── PROGRESS TRACKING ─────────────────────────────────────────────────────
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"done_indices": [], "total_written": 0}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

# ── MAIN ──────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["intra", "cross", "both"], default="both",
                        help="intra=satu bab, cross=dua bab, both=keduanya")
    parser.add_argument("--target", type=int, default=150,
                        help="Target jumlah QA yang dihasilkan")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Jumlah context yang diproses paralel (hati-hati rate limit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Jalankan tanpa memanggil LLM (test grouping logic)")
    parser.add_argument("--chunks-path", default=CHUNKS_PATH,
                        help="Path ke chunks_toc_baseline.json")
    parser.add_argument("--output", default=OUTPUT_JSONL,
                        help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("MULTI-HOP GOLDEN DATASET GENERATOR")
    print(f"Mode     : {args.mode.upper()}")
    print(f"Target   : {args.target} QA")
    print(f"Dry-run  : {args.dry_run}")
    print("=" * 65)

    # Load chunks
    if not os.path.exists(args.chunks_path):
        print(f"[ERROR] Chunks file not found: {args.chunks_path}")
        sys.exit(1)

    chunks = load_chunks(args.chunks_path)
    print(f"\nLoaded {len(chunks)} chunks")

    bab_map = group_chunks_by_bab(chunks)
    print(f"Unique babs: {len(bab_map)}")
    for bab, cs in list(bab_map.items())[:5]:
        print(f"  - {bab[:50]}: {len(cs)} chunks")
    if len(bab_map) > 5:
        print(f"  ... dan {len(bab_map)-5} bab lainnya")
    print()

    # Build contexts
    all_contexts = []
    if args.mode in ("intra", "both"):
        intra = build_intra_bab_contexts(bab_map)
        print(f"Intra-bab contexts: {len(intra)}")
        all_contexts.extend(intra)

    if args.mode in ("cross", "both"):
        cross = build_cross_bab_contexts(bab_map)
        print(f"Cross-bab contexts: {len(cross)}")
        all_contexts.extend(cross)

    print(f"Total contexts: {len(all_contexts)}")

    if args.dry_run:
        print("\n[DRY-RUN] Context grouping preview:")
        for ctx in all_contexts[:3]:
            print(f"  [{ctx['context_type']}] {ctx['context_desc'][:60]}")
            print(f"    chunks={ctx['num_chunks']}, words={len(ctx['combined_text'].split())}")
            print(f"    chunk_ids={ctx['chunk_ids'][:3]}...")
        print("\n[DRY-RUN] Done. No LLM calls made.")
        return

    # Load progress
    progress = load_progress()
    done_idx = set(progress["done_indices"])
    total_written = progress["total_written"]
    print(f"\nResuming: {total_written} QA sudah ditulis, {len(done_idx)} context sudah diproses")

    # Shuffle for variety
    random.seed(42)
    random.shuffle(all_contexts)

    # Build async pipeline
    client = build_async_client()
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        generate_and_judge(i, len(all_contexts), ctx, client, semaphore)
        for i, ctx in enumerate(all_contexts)
        if i not in done_idx
    ]

    print(f"\nMemulai generasi ({len(tasks)} contexts to process)...")
    start = time.time()

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        if total_written >= args.target:
            print(f"\n✓ Target {args.target} QA tercapai!")
            break

        results = await coro
        ctx_idx = i  # approximate; good enough for progress tracking

        for qa in results:
            # Cari context yang sesuai (by desc match)
            matching_ctx = next(
                (c for c in all_contexts if c["context_desc"] == qa["context_desc"]),
                all_contexts[0]
            )
            record = format_record(qa, matching_ctx)
            write_record(record, args.output)
            total_written += 1
            print(f"  ✅ Written #{total_written}: [{qa['type']}] {qa['q'][:60]}")

        done_idx.add(ctx_idx)
        progress["done_indices"] = list(done_idx)
        progress["total_written"] = total_written
        save_progress(progress)

        if total_written >= args.target:
            break

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"SELESAI")
    print(f"Total QA ditulis  : {total_written}")
    print(f"Output            : {args.output}")
    print(f"Waktu             : {elapsed/60:.1f} menit")
    print(f"{'='*65}")


if __name__ == "__main__":
    asyncio.run(main())