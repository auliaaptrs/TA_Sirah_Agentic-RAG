"""
generate_evaluator_dataset.py
==============================
Script untuk menghasilkan dataset fine-tuning Sufficiency Evaluator.

Strategi Augmentasi (3 Skenario per soal):
  1. SUFFICIENT=TRUE  : Semua gold_chunk_ids diberikan → model belajar bilang "Cukup"
  2. SUFFICIENT=FALSE : Sebagian gold_chunk_ids dihapus → model belajar bilang "Kurang"
  3. IRRELEVANT       : Chunk acak dari bab lain diberikan → model belajar menolak noise

Format output IDENTIK dengan SUFFICIENCY_PROMPT di real_agentic.py:
  {
    "checklist": [{"item": "...", "found": true/false}, ...],
    "missing_aspects": [...],
    "sufficient": true/false
  }

Cara pakai:
  python src/build_knowledge/generate_evaluator_dataset.py
      --input_dir data/finetune_dataset
      --chunk_db data/chunks/chunks_qasina.json  (path ke file chunk)
      --output   data/finetune_dataset/train_evaluator.jsonl
      --split    train
"""

import json
import os
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

# Load API key dari environment atau .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")

# Rasio pembagian skenario (akan di-sample dari setiap soal)
# True = buatkan skenario ini untuk soal ini
PROB_PARTIAL   = 0.8   # 80% soal multi-chunk dibuat skenario parsial
PROB_IRRELEVANT = 0.4  # 40% soal dibuat skenario konteks salah

# Minimum chunk untuk skenario parsial (butuh ≥ 2 chunk agar bisa dipotong)
MIN_CHUNKS_FOR_PARTIAL = 2

# ── Prompt Templates (sesuai real_agentic.py) ────────────────────────────────

# Prompt untuk menghasilkan LABEL (checklist + missing_aspects)
# diberikan: pertanyaan + konteks (penuh atau parsial) + jawaban emas
LABEL_GEN_PROMPT = """\
Anda adalah Ahli Evaluasi Informasi Sirah Nabawiyah.

Tugas: Buat label evaluasi kelengkapan konteks sesuai format JSON di bawah.

PERTANYAAN: {question}

KONTEKS YANG DIBERIKAN (sebagian atau seluruhnya):
{context}

APAKAH KONTEKS LENGKAP?: {is_complete}

JAWABAN EMAS (referensi kelengkapan):
{expected_answer}

INSTRUKSI:
1. Pecah PERTANYAAN menjadi daftar aspek/fakta spesifik yang HARUS ada untuk menjawabnya.
2. Verifikasi setiap aspek: apakah informasinya ADA di KONTEKS YANG DIBERIKAN?
3. Jika "APAKAH KONTEKS LENGKAP?" = YA, maka semua aspek harus "found": true dan "sufficient": true.
4. Jika "APAKAH KONTEKS LENGKAP?" = TIDAK atau KONTEKS TIDAK RELEVAN, 
   catat aspek yang tidak ditemukan sebagai "missing_aspects" dan set "sufficient": false.
5. Buat 3-6 item checklist yang relevan dan spesifik.

Balas HANYA dengan JSON valid (tanpa markdown, tanpa komentar):
{{
  "checklist": [
    {{"item": "<aspek 1>", "found": true}},
    {{"item": "<aspek 2>", "found": false}}
  ],
  "missing_aspects": ["<aspek yang found: false>"],
  "sufficient": true
}}"""

# Prompt yang akan digunakan saat INFERENCE (harus sama dengan real_agentic.py)
SUFFICIENCY_PROMPT = """\
Anda adalah Ahli Evaluasi Informasi Sirah Nabawiyah.
Tugas: Tentukan apakah konteks yang terkumpul sudah cukup untuk menjawab pertanyaan secara LENGKAP dan DETAIL.

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


# ── Helper: Load Chunk Database ──────────────────────────────────────────────

def load_chunk_db(chunk_paths: List[str]) -> Dict[str, str]:
    """
    Membaca semua file chunks JSON dan mengembalikan dict: {chunk_id: teks_chunk}
    Mendukung berbagai format struktur JSON.
    """
    db = {}
    for path in chunk_paths:
        if not os.path.exists(path):
            print(f"  [WARN] File chunk tidak ditemukan: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Format 1: list of {"chunk_id": ..., "text": ...}
        if isinstance(data, list):
            for item in data:
                cid = item.get("id") or item.get("chunk_id")
                txt = item.get("text") or item.get("content") or ""
                if cid:
                    db[cid] = txt
        # Format 2: dict {chunk_id: {text: ...}} atau {chunk_id: teks}
        elif isinstance(data, dict):
            for cid, val in data.items():
                if isinstance(val, dict):
                    txt = val.get("text") or val.get("content") or ""
                else:
                    txt = str(val)
                db[cid] = txt
    
    print(f"  [OK] Chunk DB dimuat: {len(db):,} chunks")
    return db


def get_chunk_text(chunk_id: str, db: Dict[str, str]) -> Optional[str]:
    return db.get(chunk_id)


def build_context_string(chunk_ids: List[str], db: Dict[str, str]) -> str:
    """Buat string konteks dari daftar chunk_id."""
    parts = []
    for i, cid in enumerate(chunk_ids, 1):
        txt = get_chunk_text(cid, db)
        if txt:
            parts.append(f"[{i}] (Sumber: {cid})\n{txt.strip()}")
    return "\n\n".join(parts) if parts else "[Tidak ada konteks tersedia]"


# ── LLM Client ───────────────────────────────────────────────────────────────

def get_llm_client():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    return client


def call_llm(client, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    try:
        extra = {}
        if "qwen" in OPENROUTER_MODEL.lower():
            extra = {
                "extra_body": {
                    "provider": {
                        "order": ["DeepInfra"],
                        "allow_fallbacks": False
                    }
                }
            }
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra
        )
        if not resp or not getattr(resp, "choices", None):
            print(f"  [ERR] LLM call gagal: response atau choices kosong. Response: {resp}")
            return ""
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [ERR] LLM call gagal: {e}")
        return ""


def parse_json_from_llm(raw: str) -> Optional[dict]:
    """Ekstrak JSON dari output LLM (tahan terhadap markdown fences)."""
    raw = raw.strip()
    # Coba langsung parse
    try:
        return json.loads(raw)
    except:
        pass
    # Coba cari blok JSON dalam teks
    match = re.search(r'\{[\s\S]+\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None


# ── Skenario Generator ────────────────────────────────────────────────────────

def make_scenario_sufficient(question: str, gold_ids: List[str],
                              expected_answer: str, db: Dict[str, str],
                              client, metadata: dict) -> Optional[dict]:
    """
    Skenario 1: Semua gold_chunk_ids diberikan → sufficient=True
    """
    # Filter hanya chunk yang ada di DB
    available = [cid for cid in gold_ids if get_chunk_text(cid, db)]
    if not available:
        return None

    context_str = build_context_string(available, db)
    
    # Generate label via LLM
    prompt = LABEL_GEN_PROMPT.format(
        question=question,
        context=context_str,
        is_complete="YA - semua konteks yang diperlukan sudah ada",
        expected_answer=expected_answer
    )
    raw = call_llm(client, prompt)
    label = parse_json_from_llm(raw)
    if not label:
        return None
    
    # Paksa konsisten: jika is_complete=YA, sufficient harus True
    label["sufficient"] = True
    label["missing_aspects"] = []
    for item in label.get("checklist", []):
        item["found"] = True

    # Format inference prompt (sama dengan real_agentic.py)
    input_prompt = SUFFICIENCY_PROMPT.format(
        question=question,
        search_history=gold_ids[0] if gold_ids else "-",
        context=context_str
    )

    return {
        "messages": [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)}
        ],
        "metadata": {
            **metadata,
            "scenario": "sufficient",
            "context_chunk_ids": available,
            "missing_chunk_ids": []
        }
    }


def make_scenario_partial(question: str, gold_ids: List[str],
                           expected_answer: str, db: Dict[str, str],
                           client, metadata: dict) -> Optional[dict]:
    """
    Skenario 2: Sebagian gold_chunk_ids dihapus → sufficient=False
    Strategi: berikan 50% pertama, sisanya dijadikan "yang hilang"
    """
    available = [cid for cid in gold_ids if get_chunk_text(cid, db)]
    if len(available) < MIN_CHUNKS_FOR_PARTIAL:
        return None

    # Acak urutan, lalu potong
    shuffled = available.copy()
    random.shuffle(shuffled)
    n_given  = max(1, len(shuffled) // 2)
    given    = shuffled[:n_given]
    missing  = shuffled[n_given:]

    context_str = build_context_string(given, db)
    missing_hint = ", ".join(missing)

    prompt = LABEL_GEN_PROMPT.format(
        question=question,
        context=context_str,
        is_complete=f"TIDAK - beberapa bagian konteks sengaja dihilangkan "
                    f"(chunk yang hilang: {missing_hint})",
        expected_answer=expected_answer
    )
    raw = call_llm(client, prompt)
    label = parse_json_from_llm(raw)
    if not label:
        return None

    # Validasi: harus sufficient=False dan ada missing_aspects
    label["sufficient"] = False
    if not label.get("missing_aspects"):
        label["missing_aspects"] = [f"Informasi dari chunk: {', '.join(missing)}"]

    # Format inference prompt
    search_hist = ", ".join(given)
    input_prompt = SUFFICIENCY_PROMPT.format(
        question=question,
        search_history=search_hist,
        context=context_str
    )

    return {
        "messages": [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)}
        ],
        "metadata": {
            **metadata,
            "scenario": "partial",
            "context_chunk_ids": given,
            "missing_chunk_ids": missing
        }
    }


def make_scenario_irrelevant(question: str, gold_ids: List[str],
                              expected_answer: str, db: Dict[str, str],
                              all_chunk_ids: List[str],
                              client, metadata: dict) -> Optional[dict]:
    """
    Skenario 3: Chunk acak (bukan gold) diberikan → sufficient=False
    Strategi: ambil chunk dari bab berbeda (beda prefix toc_XXXX)
    """
    # Dapatkan prefix bab dari gold chunks
    gold_prefixes = set()
    for cid in gold_ids:
        parts = cid.rsplit("_", 1)
        if len(parts) == 2:
            gold_prefixes.add(parts[0])  # e.g. "toc_0009"

    # Cari chunk dari bab lain
    irrelevant_pool = [
        cid for cid in all_chunk_ids
        if cid not in gold_ids and
        all(not cid.startswith(p) for p in gold_prefixes)
    ]

    if len(irrelevant_pool) < 3:
        return None

    # Pilih 3-5 chunk irrelevan secara acak
    n_irr = random.randint(3, min(5, len(irrelevant_pool)))
    chosen = random.sample(irrelevant_pool, n_irr)
    context_str = build_context_string(chosen, db)

    prompt = LABEL_GEN_PROMPT.format(
        question=question,
        context=context_str,
        is_complete="TIDAK - konteks yang diberikan sama sekali tidak relevan "
                    "dengan pertanyaan (berasal dari bab yang berbeda)",
        expected_answer=expected_answer
    )
    raw = call_llm(client, prompt)
    label = parse_json_from_llm(raw)
    if not label:
        return None

    # Paksa: semua found=False, sufficient=False
    label["sufficient"] = False
    for item in label.get("checklist", []):
        item["found"] = False
    if not label.get("missing_aspects"):
        label["missing_aspects"] = ["Seluruh informasi yang relevan dengan pertanyaan"]

    input_prompt = SUFFICIENCY_PROMPT.format(
        question=question,
        search_history="-",
        context=context_str
    )

    return {
        "messages": [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)}
        ],
        "metadata": {
            **metadata,
            "scenario": "irrelevant",
            "context_chunk_ids": chosen,
            "missing_chunk_ids": gold_ids
        }
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global OPENROUTER_MODEL
    parser = argparse.ArgumentParser(description="Generate Sufficiency Evaluator Dataset")
    parser.add_argument("--input_dir",  default="data/finetune_dataset",
                        help="Direktori berisi train_ok/valid_ok/test_ok.jsonl")
    parser.add_argument("--split",      default="train",
                        choices=["train", "valid", "test"],
                        help="Split yang akan diproses")
    parser.add_argument("--chunk_db",   nargs="+",
                        default=[
                            "data/vectordb/chunks_toc_baseline_BACKUP_CHUNK1000.json"
                        ],
                        help="Path ke file JSON chunk database")
    parser.add_argument("--output",     default=None,
                        help="Path file output .jsonl (default: auto dari split)")
    parser.add_argument("--max_samples",type=int, default=None,
                        help="Batasi jumlah soal yang diproses (untuk testing)")
    parser.add_argument("--model",      default=OPENROUTER_MODEL,
                        help="Model OpenRouter yang digunakan untuk generate label")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--workers",    type=int, default=5,
                        help="Jumlah worker threads untuk pemanggilan LLM paralel")
    args = parser.parse_args()

    random.seed(args.seed)

    # Tentukan path output
    if args.output is None:
        args.output = os.path.join(args.input_dir, f"{args.split}_evaluator.jsonl")

    OPENROUTER_MODEL = args.model

    input_file = os.path.join(args.input_dir, f"{args.split}_ok.jsonl")
    if not os.path.exists(input_file):
        print(f"[ERR] File tidak ditemukan: {input_file}")
        return

    print("=" * 55)
    print(f"GENERATE EVALUATOR DATASET — SPLIT: {args.split.upper()}")
    print("=" * 55)

    # Load chunk DB
    print("\n[1/4] Memuat Chunk Database...")
    db = load_chunk_db(args.chunk_db)
    all_chunk_ids = list(db.keys())
    
    if not db:
        print("[ERR] Chunk DB kosong! Pastikan path chunk_db benar.")
        print("      Jalankan dengan: --chunk_db <path_ke_chunks.json>")
        return

    # Load LLM client
    print("[2/4] Menghubungkan ke LLM (OpenRouter)...")
    if not OPENROUTER_API_KEY:
        print("[ERR] OPENROUTER_API_KEY tidak ditemukan di .env")
        return
    client = get_llm_client()
    print(f"  [OK] Model: {OPENROUTER_MODEL}")

    # Load dataset
    print(f"[3/4] Memuat dataset dari {input_file}...")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                samples.append(json.loads(line))
            except:
                pass
    print(f"  [OK] Dimuat: {len(samples):,} soal")

    if args.max_samples:
        random.shuffle(samples)
        samples = samples[:args.max_samples]
        print(f"  [INFO] Dibatasi ke {args.max_samples} soal (--max_samples)")

    # Generate dataset
    print(f"[4/4] Membuat dataset evaluator -> {args.output}")
    print("-" * 55)

    # Load checkpoint jika ada
    done_ids = set()
    results = []
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    src_id = item.get("metadata", {}).get("source_question_id", "")
                    sc = item.get("metadata", {}).get("scenario", "")
                    if src_id and sc:
                        done_ids.add(f"{src_id}__{sc}")
                    results.append(item)
                except:
                    pass
        print(f"  [RESUME] Sudah ada {len(results)} sample di file output.")

    out_file = open(args.output, "a", encoding="utf-8")

    n_sufficient  = sum(1 for r in results if r.get("metadata", {}).get("scenario") == "sufficient")
    n_partial     = sum(1 for r in results if r.get("metadata", {}).get("scenario") == "partial")
    n_irrelevant  = sum(1 for r in results if r.get("metadata", {}).get("scenario") == "irrelevant")

    import threading
    from concurrent.futures import ThreadPoolExecutor

    write_lock = threading.Lock()
    stats_lock = threading.Lock()
    processed_count = 0
    total_samples = len(samples)

    def worker(item):
        idx, sample = item
        nonlocal n_sufficient, n_partial, n_irrelevant, processed_count

        meta     = sample.get("metadata", {})
        q_type   = meta.get("type", "unknown")
        gold_ids = meta.get("gold_chunk_ids", [])
        exp_ans  = meta.get("expected_answer", "")

        # Ambil question dari messages
        question = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break

        # Buat ID unik per soal
        q_id = f"{args.split}_{idx:05d}"
        base_meta = {
            "source_question_id": q_id,
            "question_type": q_type,
            "gold_chunk_ids": gold_ids,
            "source": meta.get("source", ""),
            "context_type": meta.get("context_type", "")
        }

        new_count = 0
        local_suf = None
        local_par = None
        local_irr = None

        # Skenario 1: SUFFICIENT
        key_suf = f"{q_id}__sufficient"
        if key_suf not in done_ids:
            suf = make_scenario_sufficient(question, gold_ids, exp_ans, db, client, base_meta)
            if suf:
                local_suf = suf
                new_count += 1
                with stats_lock:
                    done_ids.add(key_suf)

        # Skenario 2: PARTIAL (hanya jika multi-chunk)
        key_par = f"{q_id}__partial"
        if key_par not in done_ids and len(gold_ids) >= MIN_CHUNKS_FOR_PARTIAL:
            if random.random() < PROB_PARTIAL:
                par = make_scenario_partial(question, gold_ids, exp_ans, db, client, base_meta)
                if par:
                    local_par = par
                    new_count += 1
                    with stats_lock:
                        done_ids.add(key_par)

        # Skenario 3: IRRELEVANT
        key_irr = f"{q_id}__irrelevant"
        if key_irr not in done_ids:
            if random.random() < PROB_IRRELEVANT:
                irr = make_scenario_irrelevant(
                    question, gold_ids, exp_ans, db, all_chunk_ids, client, base_meta
                )
                if irr:
                    local_irr = irr
                    new_count += 1
                    with stats_lock:
                        done_ids.add(key_irr)

        # Tulis ke file secara thread-safe
        if local_suf or local_par or local_irr:
            with write_lock:
                if local_suf:
                    out_file.write(json.dumps(local_suf, ensure_ascii=False) + "\n")
                if local_par:
                    out_file.write(json.dumps(local_par, ensure_ascii=False) + "\n")
                if local_irr:
                    out_file.write(json.dumps(local_irr, ensure_ascii=False) + "\n")
                out_file.flush()

        with stats_lock:
            processed_count += 1
            if local_suf: n_sufficient += 1
            if local_par: n_partial += 1
            if local_irr: n_irrelevant += 1
            total_now = n_sufficient + n_partial + n_irrelevant
            print(f"  [{processed_count}/{total_samples}] {q_id} | type={q_type} | chunks={len(gold_ids)} -> +{new_count} sample (Total: {total_now})")

    # Jalankan dengan ThreadPoolExecutor
    print(f"  [INFO] Menjalankan dengan {args.workers} worker threads...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        executor.map(worker, enumerate(samples))

    out_file.close()

    # Summary
    total = n_sufficient + n_partial + n_irrelevant
    print("\n" + "=" * 55)
    print(f"SELESAI! Dataset Evaluator berhasil dibuat.")
    print(f"  Output file   : {args.output}")
    print(f"  Total sample  : {total:,}")
    print(f"    - sufficient  : {n_sufficient:,} ({n_sufficient/max(total,1)*100:.1f}%)")
    print(f"    - partial     : {n_partial:,} ({n_partial/max(total,1)*100:.1f}%)")
    print(f"    - irrelevant  : {n_irrelevant:,} ({n_irrelevant/max(total,1)*100:.1f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
