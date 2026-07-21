"""
eval_generator.py  — Uji Coba 3 (Kualitas Generasi End-to-End)
===============================================================
Skrip evaluasi kualitas jawaban yang dihasilkan oleh pipeline RAG secara
end-to-end, mencakup tahap retrieval dan generasi sekaligus.

Alur kerja:
  Pertanyaan → RAG (retrieve + generate) → Jawaban sistem
  → Evaluasi otomatis vs gold answer + penilaian LLM Judge

Metrik yang dihitung:
  - Token F1           : Overlap token antara jawaban sistem vs gold answer
  - BERTScore          : Kemiripan semantik jawaban vs gold answer
                         (model: bert-base-multilingual-cased)
  - Faithfulness       : Skor 0–1 dari LLM Judge (apakah jawaban
                         didukung konteks / tidak berhalusinasi?)
  - Answer Relevancy   : Skor 0–1 dari LLM Judge (apakah jawaban
                         menjawab pertanyaan dengan tepat?)
  - Latency            : Waktu total pemrosesan per pertanyaan
  - LLM Calls          : Jumlah pemanggilan LLM per pertanyaan

Format dataset:
  - Gunakan test_ok.jsonl (bab-based split, zero leakage)
  - Gold answer diambil dari pesan assistant di dataset

==============================================================
PANDUAN PENGGUNAAN (untuk rekonstruksi eksperimen)
==============================================================

[UJI COBA 3 — Kualitas Generasi End-to-End]
  Tujuan: Membandingkan kualitas jawaban yang dihasilkan oleh pipeline
          Konvensional vs pipeline Agentik (SEA-LION sebagai Sufficiency
          Evaluator terpilih dari Uji Coba 2), menggunakan embedding
          baseline sebagai hasil terbaik Uji Coba 1.

  # Pipeline Konvensional (single-pass retrieval):
  python -m src.evaluation.eval_generator --mode conv --embedding baseline

  # Pipeline Agentik + SEA-LION:
  python -m src.evaluation.eval_generator --mode real_agentic --embedding baseline

  Output: data/evaluation/skenario_3/eval_2_generator_<mode>.xlsx

--------------------------------------------------------------

Argumen:
  --mode        : 'conv' untuk pipeline Konvensional,
                  'real_agentic' untuk pipeline Agentik
  --embedding   : 'baseline' (default) untuk multilingual-e5-large,
                  'finetuned' untuk model fine-tuned
  --no_chron    : (flag) Nonaktifkan Chronological Re-Ranking — hanya
                  untuk keperluan ablasi internal, tidak dilaporkan
                  sebagai uji coba terpisah di skripsi
"""

"""
CATATAN INTERNAL UNTUK PENGEMBANG:
  File ini juga mendukung argumen --max_iter untuk mengatur jumlah
  iterasi maksimum pada pipeline agentik (default: 3). Fitur ini
  digunakan untuk eksperimen ablasi awal dan hasilnya disimpan ke
  data/evaluation/skenario_4/. Fitur ini tidak dilaporkan sebagai
  uji coba tersendiri di skripsi akhir.
"""


import os
import sys
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re
from collections import Counter

# Setup Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src", "rag"))

from src.rag.utils import init_vectordb, init_llm, call_llm, parse_json_safe
import src.rag.conventional_rag as conv_rag
import src.rag.real_agentic as real_agentic
from src.rag.config import JUDGE_PROVIDER, OLLAMA_JUDGE_MODEL

# ── CONFIG ────────────────────────────────────────────────────────
BENCHMARK_FILE = os.path.join(BASE_DIR, "data", "finetune_dataset", "test_ok.jsonl")

# Base evaluation directory — output dipisah per skenario
EVAL_BASE_DIR = os.path.join(BASE_DIR, "data", "evaluation")
SCENARIO_3_DIR = os.path.join(EVAL_BASE_DIR, "skenario_3")  # Evaluasi generasi end-to-end
SCENARIO_4_DIR = os.path.join(EVAL_BASE_DIR, "skenario_4")  # Ablasi jumlah iterasi

JUDGE_PROMPT = """
Anda adalah juri ahli Sirah Nabawiyah.
Nilai kualitas jawaban sistem RAG berdasarkan dua dimensi:

1. FAITHFULNESS (0.0 - 1.0):
   Apakah semua klaim dalam JAWABAN SISTEM didukung oleh KONTEKS?
   1.0 = Tidak ada halusinasi sama sekali.
   0.0 = Jawaban sepenuhnya mengarang.

2. ANSWER RELEVANCY (0.0 - 1.0):
   Apakah JAWABAN SISTEM menjawab PERTANYAAN dengan tepat?
   1.0 = Sangat relevan dan lengkap.
   0.0 = Tidak nyambung.

PERTANYAAN: {question}
KONTEKS: {context}
JAWABAN SISTEM: {answer}

Balas HANYA dengan format JSON yang valid:
{{
  "faithfulness": <float 0.0-1.0>,
  "relevancy": <float 0.0-1.0>,
  "reason": "<singkat>"
}}
"""

# ── UTILITIES ─────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_citation_markers(text: str) -> str:
    """Hapus [1], [2], dll agar tidak mengganggu metrik kemiripan."""
    return re.sub(r"\s*\[\d+\]", "", text).strip()

def extract_clean_question(user_msg: str) -> str:
    """Ekstrak pertanyaan bersih dari prompt."""
    if "\nPertanyaan: " in user_msg:
        return user_msg.split("\nPertanyaan: ")[-1].strip()
    return user_msg.strip()

def clean_citations(text: str) -> str:
    """Menghapus sitasi seperti [1], [1][2] dari teks untuk evaluasi string."""
    return re.sub(r'\[\d+(?:,\d+)*\]|\[\d+\](?:\[\d+\])*', '', str(text)).strip()

def calculate_token_f1(pred: str, truth: str) -> float:
    pred = clean_citations(pred)
    pred_toks = normalize(pred).split()
    truth_toks = normalize(truth).split()
    if not pred_toks or not truth_toks: return 0.0
    common = Counter(pred_toks) & Counter(truth_toks)
    n_same = sum(common.values())
    if n_same == 0: return 0.0
    prec = n_same / len(pred_toks)
    rec = n_same / len(truth_toks)
    return 2 * prec * rec / (prec + rec)

def compute_bertscore(preds: list[str], refs: list[str]) -> list[float]:
    try:
        from bert_score import score as bs_score
        safe_preds = [p if p.strip() else "." for p in preds]
        safe_refs = [r if r.strip() else "." for r in refs]
        _, _, F = bs_score(safe_preds, safe_refs, lang="id", model_type="bert-base-multilingual-cased", verbose=False, batch_size=16)
        return [round(float(f), 4) for f in F.tolist()]
    except Exception as e:
        print(f"⚠️ BERTScore failed: {e}")
        return [0.0] * len(preds)

# ── MAIN ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eval 2: Generator Quality")
    parser.add_argument("--mode", choices=["conv", "real_agentic"], default="conv",
                        help="Mode RAG: 'conv' atau 'real_agentic'")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Override MAX_ITERATIONS di real_agentic (untuk Skenario 4). "
                             "Contoh: --max_iter 1 atau --max_iter 2. "
                             "Jika tidak diisi, pakai default (3) → output ke skenario_3.")
    parser.add_argument("--embedding", type=str, choices=["baseline", "finetuned"], default="finetuned",
                        help="Pilih embedding model: 'baseline' atau 'finetuned'")
    parser.add_argument("--no_chron", action="store_true", default=False,
                        help="Nonaktifkan Chronological Re-Ranking (ablasi). "
                             "Output diberi suffix '_no_chron'.")
    args = parser.parse_args()

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

    # Tentukan folder output:
    # Skenario 4 = real_agentic + max_iter eksplisit diisi
    # Skenario 3 = semua run tanpa --max_iter
    is_s4 = (args.mode == "real_agentic" and args.max_iter is not None)
    OUTPUT_DIR = SCENARIO_4_DIR if is_s4 else SCENARIO_3_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenario_label = f"Skenario 4 (iterasi={args.max_iter})" if is_s4 else "Skenario 3"
    print("="*60)
    print(f"EVAL 2: GENERATOR QUALITY - MODE: {args.mode.upper()}")
    print(f"Output → {scenario_label}: {OUTPUT_DIR}")
    print("="*60)

    # Patch MAX_ITERATIONS jika --max_iter diisi (untuk Skenario 4)
    if args.max_iter is not None and args.mode == "real_agentic":
        import src.rag.real_agentic as _ra
        _ra.MAX_ITERATIONS = args.max_iter
        print(f"[INFO] MAX_ITERATIONS di-override menjadi {args.max_iter}")

    client, embed_model = init_vectordb()
    llm_client = init_llm()
    # Judge using OpenRouter agar tidak kena rate limit harian Groq
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

    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ Benchmark file not found: {BENCHMARK_FILE}")
        return
        
    data = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    
    print(f"📄 Loaded {len(data)} benchmark questions.")
    results = []
    checkpoint_file = os.path.join(OUTPUT_DIR, f"eval_2_generator_{args.mode}_checkpoint.jsonl")
    done_indices = set()
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    res_row = json.loads(line)
                    results.append(res_row)
                    done_indices.add(res_row.get("_idx", -1))
        print(f"🔄 Resuming from checkpoint: {len(done_indices)} questions already processed.")

    for i, row in enumerate(tqdm(data, desc=f"Evaluating {args.mode.upper()}")):
        if i in done_indices:
            continue
            
        messages = row.get("messages", [])
        user_msg = messages[1].get("content", "") if len(messages) >= 2 else ""
        question = extract_clean_question(user_msg)
        expected_answer = messages[2].get("content", "") if len(messages) >= 3 else ""
        meta = row.get("metadata", {})

        try:
            if args.mode == "real_agentic":
                res_raw = real_agentic.real_agentic_rag_query(
                    question, vectordb, llm_client,
                    use_chronological=(not args.no_chron)
                )
                res = {
                    "answer": res_raw.get("answer", ""),
                    "chunks": res_raw.get("chunks", []),
                    "total_time": res_raw.get("elapsed", 0),
                    "llm_calls": res_raw.get("llm_calls", 3),
                }
            else:  # conv
                res_raw = conv_rag.conventional_rag_query(question, vectordb, llm_client)
                res = {
                    "answer": res_raw.get("answer", ""),
                    "chunks": res_raw.get("chunks", []),
                    "total_time": res_raw.get("latency", 0),
                    "llm_calls": 1,
                }
        except Exception as e:
            print(f"Error at question {i}: {e}")
            continue

        answer_raw = res["answer"]
        answer_clean = strip_citation_markers(answer_raw)
        
        # Susun konteks dengan nomor urut [1], [2], dst agar juri bisa memverifikasi sitasi
        context_chunks = []
        for idx, c in enumerate(res["chunks"], 1):
            text = c.get("text", "")[:800] # Ambil 800 karakter awal agar juri punya konteks cukup
            context_chunks.append(f"[{idx}] {text}")
        context_str = "\n\n".join(context_chunks)
        
        # LLM Judge (semua soal — adil antar semua run)
        faith, rel = None, None
        if True:  # judge semua soal (Ollama lokal, tidak ada rate limit)
            prompt = JUDGE_PROMPT.format(question=question, context=context_str, answer=answer_raw)
            try:
                raw = call_llm(judge_client, prompt, temperature=0.0, max_tokens=150)
                judge_res = parse_json_safe(raw)
                faith = float(judge_res.get("faithfulness", 0.0))
                rel = float(judge_res.get("relevancy", 0.0))
            except:
                faith, rel = 0.0, 0.0

        results.append({
            "_idx": i,
            "question": question,
            "mode": args.mode,
            "type": meta.get("type", "unknown"),
            "answer": answer_raw,
            "answer_clean": answer_clean,
            "_expected": expected_answer,
            "f1_score": calculate_token_f1(answer_clean, expected_answer),
            "faithfulness": faith,
            "relevancy": rel,
            "latency": res["total_time"],
            "llm_calls": res["llm_calls"],
        })

        # Save checkpoint per soal
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + "\n")

    # BERTScore
    print("\n🧠 Computing BERTScore...")
    all_preds = [r["answer_clean"] for r in results]
    all_expected = [r["_expected"] for r in results]
    bert_scores = compute_bertscore(all_preds, all_expected)
    for i, s in enumerate(bert_scores):
        results[i]["bert_score"] = s
        del results[i]["_expected"]

    # Export
    df = pd.DataFrame(results)
    summary = {
        "Mode": args.mode.upper(),
        "Avg F1": df["f1_score"].mean(),
        "Avg BERTScore": df["bert_score"].mean(),
        "Avg Faithfulness (LLM)": df["faithfulness"].dropna().mean(),
        "Avg Relevancy (LLM)": df["relevancy"].dropna().mean(),
        "Avg Latency (s)": df["latency"].mean(),
        "Avg LLM Calls": df["llm_calls"].mean(),
    }

    print("\n" + "="*30)
    print(f"📈 SUMMARY STATISTICS ({args.mode.upper()})")
    print("="*30)
    for k, v in summary.items():
        print(f"{k:<25}: {v:.4f}" if isinstance(v, float) else f"{k:<25}: {v}")

    # Per-type breakdown
    print(f"\n  📊 Metrik Per Question Type:")
    type_stats = []
    for qt, grp in df.groupby("type"):
        f1 = grp["f1_score"].mean()
        bs = grp["bert_score"].mean()
        n  = len(grp)
        type_stats.append({"Type": qt, "F1": f1, "BERTScore": bs, "N": n})
        print(f"    {qt:<15}: F1={f1:.3f} | BERTScore={bs:.3f} (n={n})")

    chron_suffix = "_no_chron" if (args.mode == "real_agentic" and args.no_chron) else ""
    iter_suffix = f"_iter{args.max_iter}" if args.max_iter is not None else ""
    output_file = os.path.join(OUTPUT_DIR, f"eval_2_generator_{args.mode}{iter_suffix}{chron_suffix}.xlsx")
    checkpoint_out = os.path.join(OUTPUT_DIR, f"eval_2_generator_{args.mode}{iter_suffix}{chron_suffix}_checkpoint.jsonl")
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name="Full Results", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(type_stats).to_excel(writer, sheet_name="Per Type", index=False)

    print(f"\n✅ Results exported to: {output_file}")

if __name__ == "__main__":
    main()
