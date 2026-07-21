"""
eval_retrieval.py  — Uji Coba 1 & 2 (Multihop Golden Dataset)
==============================================================
Skrip evaluasi retrieval yang mencakup dua skenario pengujian utama:
  - Uji Coba 1: Ablasi Model Embedding (Base vs Fine-tuned)
  - Uji Coba 2: Ablasi Strategi Pipeline & Seleksi Sufficiency Evaluator

Alur kerja:
  Pertanyaan bersih → RAG retrieve → cek: gold_chunk_id ∈ retrieved[:K]?

Metrik yang dihitung:
  - HIT@1, HIT@3, HIT@5  (apakah minimal 1 gold chunk ada di top-K)
  - MRR                   (Mean Reciprocal Rank)
  - Recall               (persentase gold chunks yang ditemukan dari seluruh chunk yang diambil)
  - Precision            (persentase chunk relevan dari seluruh chunk yang diambil)
  - P@5 LLM Judge        (presisi top-5 dinilai LLM sebagai juri relevansi)
  - Latency              (waktu pemrosesan per soal)

Format dataset:
  - Gold chunk IDs langsung dari metadata.gold_chunk_ids (string chunk_id)
  - Gunakan test_ok.jsonl (bab-based split, zero leakage)

==============================================================
PANDUAN PENGGUNAAN (untuk rekonstruksi eksperimen)
==============================================================

[UJI COBA 1 — Ablasi Model Embedding]
  Tujuan: Membandingkan kualitas retrieval antara Base Embedding dan
          Fine-tuned Embedding menggunakan pipeline konvensional sebagai kontrol.

  # Jalankan dengan Base Embedding (multilingual-e5-large):
  python -m src.evaluation.eval_retrieval --mode conv --embedding baseline

  # Jalankan dengan Fine-tuned Embedding (finetuned-e5-sirah):
  python -m src.evaluation.eval_retrieval --mode conv --embedding finetuned

  Output: data/evaluation/skenario_1/real_agentic_baseline_<evaluator>_v1.csv

--------------------------------------------------------------

[UJI COBA 2 — Ablasi Pipeline & Seleksi Sufficiency Evaluator]
  Tujuan: Membandingkan pipeline Konvensional (single-pass) dengan pipeline
          Agentik menggunakan berbagai model Sufficiency Evaluator.
  Catatan: Seluruh run menggunakan --embedding baseline (hasil terbaik Uji Coba 1)
           dan model generator Qwen 2.5 7B yang sama agar perbandingan fair.

  # Konvensional (baseline tanpa Sufficiency Evaluator):
  python -m src.evaluation.eval_retrieval --mode conv --embedding baseline

  # Agentik + SEA-LION (Gemma SEA-LION-v3 9B) — model terpilih:
  python -m src.evaluation.eval_retrieval --mode real_agentic --embedding baseline --evaluator sealion:latest

  # Agentik + Qwen 2.5 7B:
  python -m src.evaluation.eval_retrieval --mode real_agentic --embedding baseline --evaluator qwen2.5:7b

  # Agentik + Qwen 3 8B:
  python -m src.evaluation.eval_retrieval --mode real_agentic --embedding baseline --evaluator qwen3:8b

  # Agentik + Sahabat-AI 9B:
  python -m src.evaluation.eval_retrieval --mode real_agentic --embedding baseline --evaluator sahabat-ai:latest

  # Agentik + Llama 3.1 8B:
  python -m src.evaluation.eval_retrieval --mode real_agentic --embedding baseline --evaluator llama3.1:8b

  Output: data/evaluation/skenario_2/<evaluator_name>/real_agentic_baseline_<evaluator>_v1.csv

--------------------------------------------------------------

Argumen:
  --mode        : 'conv' untuk pipeline Konvensional,
                  'real_agentic' untuk pipeline Agentik
  --embedding   : 'baseline' untuk multilingual-e5-large (Qdrant: qdrant_toc_baseline_BACKUP_CHUNK1000),
                  'finetuned' untuk model fine-tuned (Qdrant: qdrant_toc_finetuned)
  --evaluator   : (Opsional, hanya untuk --mode real_agentic)
                  Nama model Ollama untuk Sufficiency Evaluator.
                  Jika tidak diisi, menggunakan OLLAMA_EVALUATOR_MODEL dari config.py.
"""

import os
import sys
import json
import re
import time
import pandas as pd
from tqdm import tqdm
import argparse
from typing import List, Optional

# ── PATH CONFIG ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.rag.utils import init_vectordb, init_llm, call_llm, parse_json_safe
import src.rag.conventional_rag as conv_rag
import src.rag.real_agentic as real_agentic
from src.rag.config import OPENROUTER_API_KEY, JUDGE_PROVIDER, OLLAMA_JUDGE_MODEL  # noqa: F401

# ── FILE PATHS ────────────────────────────────────────────────────────
# Base evaluation directory
EVAL_BASE_DIR = os.path.join(BASE_DIR, "data", "evaluation")

# Skenario output dirs (dibuat otomatis saat runtime berdasarkan mode)
SCENARIO_DIRS = {
    "conv":         os.path.join(EVAL_BASE_DIR, "skenario_1"),  # S1: embedding ablation
    "real_agentic": os.path.join(EVAL_BASE_DIR, "skenario_2"),  # S2: retrieval strategy
}

# Gunakan Llama-3.3-70B via Groq sebagai Judge
JUDGE_MODEL = "llama-3.3-70b-versatile" 

RELEVANCE_PROMPT = """
Anda adalah juri ahli Sirah Nabawiyah.
Tentukan apakah potongan teks (CHUNK) di bawah ini mengandung informasi yang RELEVAN untuk menjawab PERTANYAAN.

PERTANYAAN: {question}
CHUNK TEKS: {chunk_text}

Balas HANYA dengan format JSON yang valid:
{{
  "is_relevant": true/false,
  "reason": "singkat"
}}
"""


# ── METRIC FUNCTIONS ──────────────────────────────────────────────────
def calculate_mrr(retrieved_relevance: List[bool]) -> float:
    for i, rel in enumerate(retrieved_relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0

def calculate_hit_at_k(retrieved_relevance: List[bool], k: int) -> float:
    return 1.0 if any(retrieved_relevance[:k]) else 0.0

def calculate_precision_at_k(retrieved_relevance: List[bool], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(retrieved_relevance[:k]) / k

def calculate_recall_at_k(retrieved_relevance: List[bool], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(retrieved_relevance[:k]) / total_relevant


# ── QUESTION EXTRACTION ───────────────────────────────────────────────

def extract_clean_question(user_msg: str) -> str:
    """
    Ekstrak pertanyaan BERSIH dari pesan user yang berformat:
      'Konteks:\n[teks chunk]\n\nPertanyaan: [pertanyaan]'

    Mengembalikan hanya bagian pertanyaan, tanpa konteks.
    """
    # Coba pattern eksplisit \nPertanyaan:
    if "\nPertanyaan: " in user_msg:
        return user_msg.split("\nPertanyaan: ")[-1].strip()

    # Fallback: cari pattern "Pertanyaan:" di manapun
    match = re.search(r"Pertanyaan:\s*(.+?)$", user_msg, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: return seluruh user_msg (biarkan RAG bekerja)
    return user_msg.strip()


# ── CORE EVALUATION ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi Retrieval — Uji Coba 1 (ablasi embedding) & Uji Coba 2 (ablasi pipeline & Sufficiency Evaluator)"
    )
    parser.add_argument("--mode", type=str, choices=["conv", "real_agentic"], default="conv",
                        help="Pipeline yang diuji: 'conv' (Konvensional → Uji Coba 1) atau 'real_agentic' (Agentik → Uji Coba 2)")
    parser.add_argument("--embedding", type=str, choices=["baseline", "finetuned"], default="baseline",
                        help="Model embedding yang dipakai: 'baseline' (multilingual-e5-large) atau 'finetuned'. "
                             "Uji Coba 1 divariasikan di argumen ini.")
    parser.add_argument("--evaluator", type=str, default=None,
                        help="(Opsional) Nama model Ollama untuk Sufficiency Evaluator, contoh: 'sealion:latest', "
                             "'qwen3:8b', 'llama3.1:8b'. Hanya berlaku saat --mode real_agentic. "
                             "Jika tidak diisi, menggunakan nilai OLLAMA_EVALUATOR_MODEL dari config.py.")
    args = parser.parse_args()

    # Resolve output dir berdasarkan mode
    OUTPUT_DIR = SCENARIO_DIRS[args.mode]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    BENCHMARK_FILE = os.path.join(BASE_DIR, "data", "finetune_dataset", "test_ok.jsonl")

    # Override embedding config
    import src.rag.config as rag_config
    if args.embedding == "baseline":
        rag_config.QDRANT_PATH = os.path.join(BASE_DIR, "data", "vectordb", "qdrant_toc_baseline_BACKUP_CHUNK1000")
        rag_config.EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
        import src.rag.utils as rag_utils
        rag_utils.QDRANT_PATH = rag_config.QDRANT_PATH
        rag_utils.EMBEDDING_MODEL = rag_config.EMBEDDING_MODEL
        print(f"[INFO] Embedding: BASELINE (multilingual-e5-large)")
    else:
        print(f"[INFO] Embedding: FINETUNED ({rag_config.EMBEDDING_MODEL})")

    # Override Sufficiency Evaluator model untuk Uji Coba 2
    if args.mode == "real_agentic" and args.evaluator:
        rag_config.OLLAMA_EVALUATOR_MODEL = args.evaluator
        import src.rag.real_agentic as _ra
        _ra.OLLAMA_EVALUATOR_MODEL = args.evaluator
        print(f"[INFO] Sufficiency Evaluator: {args.evaluator}")
    elif args.mode == "real_agentic":
        print(f"[INFO] Sufficiency Evaluator: {rag_config.OLLAMA_EVALUATOR_MODEL} (dari config.py)")

    print("\n" + "="*60)
    print(f"   EVALUASI RETRIEVAL - MODE: {args.mode.upper()}")
    print(f"   Dataset: {os.path.basename(BENCHMARK_FILE)} | Metric: ALL-CHUNKS Recall/Prec")
    print("="*60)

    # 1. Load test dataset (format baru — tidak perlu gold_mapping.json)
    if not os.path.exists(BENCHMARK_FILE):
        print(f"Error: Test file tidak ditemukan: {BENCHMARK_FILE}")
        return

    with open(BENCHMARK_FILE, encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(f"Test dataset loaded: {len(data)} soal")
    print(f"File: {BENCHMARK_FILE}")

    # Nama file menyertakan mode + embedding (+ evaluator jika agentik) agar tidak overwrite
    evaluator_tag = f"_{args.evaluator.replace(':','_').replace('/','_')}" if (args.mode == "real_agentic" and args.evaluator) else ""
    run_tag = f"{args.mode}_{args.embedding}{evaluator_tag}"

    def get_latest_version(tag):
        v = 1
        while os.path.exists(os.path.join(OUTPUT_DIR, f"eval_1_{tag}_v{v}.csv")):
            v += 1
        return v - 1 if v > 1 else 1

    scenario_label = "Skenario 1" if args.mode == "conv" else "Skenario 2"
    print(f"   Output → {scenario_label}: {OUTPUT_DIR}")

    latest_v = get_latest_version(run_tag)
    output_csv = os.path.join(OUTPUT_DIR, f"eval_1_{run_tag}_v{latest_v}.csv")
    
    existing_results = []
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            # Jika sudah lengkap, buat versi baru
            if len(df_existing) >= len(data):
                latest_v += 1
                output_csv = os.path.join(OUTPUT_DIR, f"eval_1_{run_tag}_v{latest_v}.csv")
                print(f"Version v{latest_v-1} is complete. Starting NEW version: v{latest_v}")
            else:
                existing_results = df_existing.to_dict('records')
                print(f"Resuming v{latest_v}: {len(existing_results)}/{len(data)} questions done.")
        except Exception:
            print("Error reading CSV, starting fresh.")
    else:
        print(f"Starting fresh evaluation: {run_tag.upper()} v1")

    # 4. Init
    client, embed_model = init_vectordb()
    llm_client = init_llm()
    # Gunakan OpenRouter sebagai Judge agar stabil dan tidak kena limit harian Groq
    # Judge: ikut JUDGE_PROVIDER di config.py
    # "openrouter" → Llama 70B via cloud (RTX 5080)
    # "ollama"     → Llama 70B via lokal (A6000)
    if JUDGE_PROVIDER == "ollama":
        judge_client = init_llm(override_model=OLLAMA_JUDGE_MODEL)
        print(f"[INFO] Judge: Ollama lokal ({OLLAMA_JUDGE_MODEL})")
    else:
        judge_client = init_llm(override_provider="openrouter")
        print("[INFO] Judge: OpenRouter (Llama 3.3 70B)")
    vectordb = (client, embed_model)

    print(f"VDB & LLM Initialized. Running evaluation...")

    # 5. Evaluate
    results = existing_results
    processed_questions = {res['question'] for res in results}
    skipped = 0

    for i, row in enumerate(tqdm(data, desc=f"Evaluating {args.mode.upper()}")):
        messages = row.get("messages", [])
        meta     = row.get("metadata", {})

        # Ekstrak pertanyaan BERSIH dari user message
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        question = extract_clean_question(user_msg)
        
        if not question or question in processed_questions:
            if not question: skipped += 1
            continue

        q_type = meta.get("type", "unknown")
        context_type = meta.get("context_type", "intra_bab")

        # ✅ Gold chunk IDs langsung dari metadata (string chunk_id)
        gold_chunk_ids = set(meta.get("gold_chunk_ids", []))
        if not gold_chunk_ids:
            skipped += 1
            continue

        start_t = time.time()
        # ── Retrieval ────────────────────────────────────────────────
        try:
            if args.mode == "real_agentic":
                chunks = real_agentic.real_agentic_retriever_only(question, vectordb, llm_client)
                reasoning_type = "real_agentic"
            else:
                # Conventional: direct embedding retrieval
                from src.rag.conventional_rag import retrieve
                chunks = retrieve(question, embed_model, client, top_k=5)
                reasoning_type = "conventional"
        except Exception as e:
            print(f"\n[WARN] Error q={i}: {e}")
            continue

        latency = time.time() - start_t

        # ── Relevance check (Semua chunk yang diambil) ────────
        all_retrieved_ids = [str(c.get("chunk_id", c.get("id", ""))) for c in chunks]
        top5_retrieved_ids = all_retrieved_ids[:5]

        # List boolean untuk Hit@K (tetap top-5)
        top5_relevance = [rid in gold_chunk_ids for rid in top5_retrieved_ids]
        
        # List boolean untuk Recall & Precision (Semua chunk yang diretrieve)
        all_relevance = [rid in gold_chunk_ids for rid in all_retrieved_ids]

        # Calculate Metrics
        hit_1 = calculate_hit_at_k(top5_relevance, 1)
        hit_3 = calculate_hit_at_k(top5_relevance, 3)
        hit_5 = calculate_hit_at_k(top5_relevance, 5)
        mrr   = calculate_mrr(top5_relevance)
        
        # Recall: Pakai SEMUA chunk yang berhasil diambil
        recall_final = calculate_recall_at_k(all_relevance, len(gold_chunk_ids), len(all_relevance))
        
        # Precision: Pakai SEMUA chunk yang berhasil diambil
        precision_final = calculate_precision_at_k(all_relevance, len(all_relevance))


        # ── LLM Judge (10% sampling) ──────────────────────────────────
        p_at_5_llm: Optional[float] = None
        if True:  # judge semua soal (Ollama lokal, tidak ada rate limit)
            llm_rel_list = []
            for c in chunks[:5]:
                prompt = RELEVANCE_PROMPT.format(
                    question=question,
                    chunk_text=c.get("text", "")[:500]
                )
                try:
                    raw      = call_llm(judge_client, prompt, temperature=0.0, max_tokens=150)
                    judge_r  = parse_json_safe(raw)
                    llm_rel_list.append(bool(judge_r.get("is_relevant", False)))
                except Exception as e:
                    print(f"⚠️ [JUDGE ERROR] Soal {i}: {e}")
                    llm_rel_list.append(False)
            p_at_5_llm = calculate_precision_at_k(llm_rel_list, 5)

        current_res = {
            "question"       : question,
            "mode"           : args.mode,
            "reasoning_type" : reasoning_type,
            "q_type"         : q_type,
            "context_type"   : context_type,
            "num_gold_chunks": len(gold_chunk_ids),
            "hit_at_1"       : hit_1,
            "hit_at_3"       : hit_3,
            "hit_at_5"       : hit_5,
            "mrr"            : mrr,
            "recall_all"     : recall_final,
            "precision_all"  : precision_final,
            "p_at_5_llm"     : p_at_5_llm,
            "latency"        : latency,
            "num_chunks"     : len(chunks),
        }
        results.append(current_res)
        processed_questions.add(question)

        # Progressive Save (Auto-save)
        if (i + 1) % 5 == 0 or (i + 1) == len(data):
            pd.DataFrame(results).to_csv(output_csv, index=False)

    # 5. Statistics
    if not results:
        print("❌ Tidak ada hasil evaluasi.")
        return

    df = pd.DataFrame(results)

    print(f"\n{'='*45}")
    print(f"📈 SUMMARY STATISTICS ({args.mode.upper()}) [MULTI-GOLD]")
    print(f"{'='*45}")

    summary = {
        "Mode"              : args.mode.upper(),
        "Total Evaluated"   : len(df),
        "Hit@1"             : df["hit_at_1"].mean(),
        "Hit@3"             : df["hit_at_3"].mean(),
        "Hit@5"             : df["hit_at_5"].mean(),
        "MRR"               : df["mrr"].mean(),
        "Recall (All)"      : df["recall_all"].mean(),
        "Precision (All)"   : df["precision_all"].mean(),
        "Avg Latency (s)"   : df["latency"].mean(),
        "P@5 (LLM Judge)"   : df["p_at_5_llm"].dropna().mean()
                               if not df["p_at_5_llm"].isna().all() else 0.0,
    }

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")

    # Per-type breakdown
    print(f"\n  📊 HIT@5 Per Question Type:")
    for qt, grp in df.groupby("q_type"):
        h5 = grp["hit_at_5"].mean()
        n  = len(grp)
        print(f"    {qt:<15}: {h5:.3f}  (n={n})")

    # 6. Export
    output_file = os.path.join(OUTPUT_DIR, f"eval_1_retrieval_{args.mode}_v{latest_v}.xlsx")
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name="Full Results", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n✅ Results exported: {output_file}")


if __name__ == "__main__":
    main()
