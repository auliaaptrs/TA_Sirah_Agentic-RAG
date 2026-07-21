"""
eval_konsistensi.py  — Uji Coba 5 (Uji Konsistensi Sistem RAG)
================================================================
Skrip evaluasi untuk mengukur stabilitas dan keandalan (robustness) 
Sistem RAG Agentik terhadap variasi pertanyaan (Semantik) dan 
stabilitas generasi LLM saat ditanya berulang kali (Deterministik).

Dua jenis pengujian:
  1. SEMANTIK      : 4 tipe soal x 5 variasi parafrase 
                     -> mengukur kekebalan sistem terhadap ragam gaya bahasa
  2. DETERMINISTIK : 4 soal asli diulang N kali        
                     -> mengukur stabilitas LLM saat ditanya pertanyaan identik

Metrik yang dihitung:
  - PAIRWISE F1        : Konsistensi leksikal (overlap kata persis antar-jawaban)
  - PAIRWISE BERTScore : Konsistensi makna semantik antar-jawaban sistem

Catatan Desain:
  - Akurasi (jawaban sistem vs referensi/gold truth) sudah diukur di Uji Coba 3.
  - Pengujian ini MURNI mengukur konsistensi INTERNAL sistem (jawaban vs jawaban).
  - Setiap pasangan (a,b) dihitung dua arah [F(a->b) + F(b->a)] / 2 agar simetris.

==============================================================
PANDUAN PENGGUNAAN (untuk rekonstruksi eksperimen)
==============================================================

[UJI COBA 5 — Uji Konsistensi Semantik & Deterministik]
  Tujuan: Membuktikan bahwa pipeline RAG Agentik (Qwen 2.5 7B + SEA-LION) 
          tidak hanya akurat, tetapi juga stabil dan tahan terhadap 
          perubahan struktur kalimat dari pengguna.

  # Jalankan Pengujian Konsistensi (Pastikan OLLAMA_MODEL dan 
  # OLLAMA_EVALUATOR_MODEL di config.py diset ke model terbaik dari Uji Coba 2 & 3):
  python -m src.evaluation.eval_konsistensi

  Output: 
  - data/evaluation/skenario_6/eval_6_konsistensi_semantik_<model_generator>_eval_<model_evaluator>.xlsx
  - data/evaluation/skenario_6/eval_6_konsistensi_deterministik_<model_generator>_eval_<model_evaluator>.xlsx

--------------------------------------------------------------

Argumen:
  Tidak ada argumen khusus. Skrip otomatis membaca model dari config.py.
"""

import os, sys, json, time, re
from collections import Counter
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ── Setup Path ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src", "rag"))

from src.rag.config import OLLAMA_MODEL
from src.rag.utils import init_vectordb, init_llm
import src.rag.real_agentic as real_agentic

# ── Utilitas Token F1 ─────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_citations(text: str) -> str:
    return re.sub(r'\[\d+(?:,\d+)*\]|\[\d+\](?:\[\d+\])*', '', str(text)).strip()

def calculate_token_f1(pred: str, truth: str) -> float:
    pred       = clean_citations(pred)
    truth      = clean_citations(truth)
    pred_toks  = normalize(pred).split()
    truth_toks = normalize(truth).split()
    if not pred_toks or not truth_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(truth_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(pred_toks)
    rec  = n_same / len(truth_toks)
    return 2 * prec * rec / (prec + rec)

# ── BERTScore Batch Simetris ───────────────────────────────────────────────────

def compute_bertscore_symmetric(preds: list, refs: list) -> list:
    """
    Hitung BERTScore untuk semua pasangan dalam satu batch.
    Simetris: skor = rata-rata F1(a->b) dan F1(b->a)
    """
    try:
        from bert_score import score as bs_score
        safe_p = [p if p.strip() else "." for p in preds]
        safe_r = [r if r.strip() else "." for r in refs]
        _, _, F_ab = bs_score(safe_p, safe_r, lang="id", model_type="bert-base-multilingual-cased",
                              verbose=False, batch_size=16)
        _, _, F_ba = bs_score(safe_r, safe_p, lang="id", model_type="bert-base-multilingual-cased",
                              verbose=False, batch_size=16)
        return [round(float((a + b) / 2), 4) for a, b in zip(F_ab.tolist(), F_ba.tolist())]
    except Exception as e:
        print(f"  Warning BERTScore: {e}. Diisi 0.0.")
        return [0.0] * len(preds)

def stats(scores: list) -> dict:
    if not scores:
        return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)),  4),
        "min": round(float(np.min(scores)),  4),
        "max": round(float(np.max(scores)),  4),
    }

# ── Query RAG ─────────────────────────────────────────────────────────────────

def query_rag(question: str, vdb, llm) -> dict:
    start_t = time.time()
    try:
        res_raw = real_agentic.real_agentic_rag_query(question, vdb, llm)
        return {
            "answer":      res_raw.get("answer", ""),
            "llm_calls":   res_raw.get("llm_calls", 1),
            "latency_s":   round(time.time() - start_t, 2),
            "chunks_used": [c.get("chunk_id", "") for c in res_raw.get("chunks", [])],
            "error":       None
        }
    except Exception as e:
        return {
            "answer": "", "llm_calls": 0,
            "latency_s": round(time.time() - start_t, 2),
            "chunks_used": [], "error": str(e)
        }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("EVAL 6: UJI KONSISTENSI (BAB 4.5)")
    print(f"Model           : {OLLAMA_MODEL}")
    print(f"Metrik          : Pairwise F1 (Leksikal) | Pairwise BERTScore (Makna)")
    print(f"Fokus           : Konsistensi internal sistem (bukan vs referensi)")
    print("=" * 75)

    dataset_path = os.path.join(BASE_DIR, "data", "evaluation", "uji_konsistensi_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"File dataset tidak ditemukan: {dataset_path}"); return

    with open(dataset_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    sem_questions = ds.get("test_semantic", {}).get("questions", [])
    det_questions = ds.get("test_deterministic", {}).get("questions", [])
    N_REPEAT      = ds.get("test_deterministic", {}).get("n_repeat", 3)

    vdb = init_vectordb()
    llm = init_llm()

    timestamp  = time.strftime("%Y%m%d_%H%M")
    safe_model = OLLAMA_MODEL.replace(":", "_").replace("-", "_")
    out_dir    = os.path.join(BASE_DIR, "data", "evaluation")
    out_json   = os.path.join(out_dir, f"eval_konsistensi_{safe_model}_{timestamp}.json")
    out_csv    = os.path.join(out_dir, f"eval_konsistensi_{safe_model}_{timestamp}.csv")

    # ── FASE 1: Kumpulkan Semua Jawaban ───────────────────────────────────────
    print("\n--- FASE 1: KUMPULKAN JAWABAN ---")
    results_semantic = []
    for group in tqdm(sem_questions, desc="Semantik"):
        group_res = {
            "group_id": group["group_id"], "type": group["type"],
            "variants": []
        }
        for v_idx, q in enumerate(group["variants"]):
            res = query_rag(q, vdb, llm)
            res["variant_idx"] = v_idx
            res["question"]    = q
            if res["error"]:
                print(f"  Warning [{group['group_id']}] v{v_idx}: {res['error']}")
            group_res["variants"].append(res)
        results_semantic.append(group_res)

    results_deterministic = []
    for group in tqdm(det_questions, desc="Deterministik"):
        group_res = {
            "group_id": group["group_id"], "type": group["type"],
            "question": group["question"], "runs": []
        }
        for run_idx in range(N_REPEAT):
            res = query_rag(group["question"], vdb, llm)
            res["run_idx"] = run_idx
            if res["error"]:
                print(f"  Warning [{group['group_id']}] run{run_idx}: {res['error']}")
            group_res["runs"].append(res)
        results_deterministic.append(group_res)

    # ── FASE 2: Hitung Semua Skor dalam SATU Batch ────────────────────────────
    print("\n--- FASE 2: HITUNG METRIK PAIRWISE ---")
    print("🧠 Loading bert-base-multilingual-cased (sekali saja)...")

    # Kumpulkan semua pasangan pairwise dari semantik & deterministik
    all_preds, all_refs, pair_meta = [], [], []

    for grp in results_semantic:
        answers = [v["answer"] for v in grp["variants"]]
        for i, j in combinations(range(len(answers)), 2):
            all_preds.append(answers[i]); all_refs.append(answers[j])
            pair_meta.append({"scope": "sem", "gid": grp["group_id"], "i": i, "j": j})

    for grp in results_deterministic:
        answers = [r["answer"] for r in grp["runs"]]
        for i, j in combinations(range(len(answers)), 2):
            all_preds.append(answers[i]); all_refs.append(answers[j])
            pair_meta.append({"scope": "det", "gid": grp["group_id"], "i": i, "j": j})

    all_bert = compute_bertscore_symmetric(all_preds, all_refs)
    print(f"  Selesai: {len(all_bert)} pasangan dihitung.")

    # Pecah skor BERTScore kembali ke masing-masing grup
    bert_by_group = {}
    for idx, (meta, bscore) in enumerate(zip(pair_meta, all_bert)):
        key = (meta["scope"], meta["gid"])
        bert_by_group.setdefault(key, []).append(bscore)

    # ── Simpan JSON ───────────────────────────────────────────────────────────
    final_json = {
        "model": OLLAMA_MODEL, "timestamp": timestamp,
        "semantic": results_semantic, "deterministic": results_deterministic
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    # ── Ringkasan Semantik ────────────────────────────────────────────────────
    HDR = (f"{'Grup':<5} | {'Tipe':<15} | {'PW-F1 Avg':<10} | "
           f"{'PW-F1 Std':<10} | {'PW-BERT Avg':<12} | {'PW-BERT Std':<11}")
    SEP = "-" * 75

    print("\n" + "=" * 75)
    print("RINGKASAN UJI SEMANTIK — Pairwise Konsistensi Antar-Jawaban Sistem")
    print("C(5,2) = 10 pasang per grup")
    print("=" * 75)
    print(HDR); print(SEP)

    csv_rows = []
    for grp in results_semantic:
        gid     = grp["group_id"]
        answers = [v["answer"] for v in grp["variants"]]
        f1_pw   = [calculate_token_f1(a, b) for a, b in combinations(answers, 2)]
        bert_pw = bert_by_group.get(("sem", gid), [])
        sf1     = stats(f1_pw); sb = stats(bert_pw)
        print(f"{gid:<5} | {grp['type']:<15} | {sf1['avg']:<10.4f} | "
              f"{sf1['std']:<10.4f} | {sb['avg']:<12.4f} | {sb['std']:<11.4f}")
        for v in grp["variants"]:
            csv_rows.append({
                "Tipe_Uji": "Semantik", "Group": gid, "Soal_Tipe": grp["type"],
                "Query": v["question"],
                "PW_F1_Avg": sf1["avg"], "PW_F1_Std": sf1["std"],
                "PW_BERT_Avg": sb["avg"], "PW_BERT_Std": sb["std"],
                "Latency_s": v["latency_s"], "LLM_Calls": v["llm_calls"],
            })

    # ── Ringkasan Deterministik ───────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("RINGKASAN UJI DETERMINISTIK — Pairwise Konsistensi Antar-Jawaban Sistem")
    print("C(3,2) = 3 pasang per grup | Std Dev ~ 0 = deterministik sempurna")
    print("=" * 75)
    print(HDR); print(SEP)

    for grp in results_deterministic:
        gid     = grp["group_id"]
        answers = [r["answer"] for r in grp["runs"]]
        f1_pw   = [calculate_token_f1(a, b) for a, b in combinations(answers, 2)]
        bert_pw = bert_by_group.get(("det", gid), [])
        sf1     = stats(f1_pw); sb = stats(bert_pw)
        print(f"{gid:<5} | {grp['type']:<15} | {sf1['avg']:<10.4f} | "
              f"{sf1['std']:<10.4f} | {sb['avg']:<12.4f} | {sb['std']:<11.4f}")
        for idx, r in enumerate(grp["runs"]):
            csv_rows.append({
                "Tipe_Uji": "Deterministik", "Group": gid, "Soal_Tipe": grp["type"],
                "Query": f"Run-{idx+1}",
                "PW_F1_Avg": sf1["avg"], "PW_F1_Std": sf1["std"],
                "PW_BERT_Avg": sb["avg"], "PW_BERT_Std": sb["std"],
                "Latency_s": r["latency_s"], "LLM_Calls": r["llm_calls"],
            })

    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\nSelesai! Hasil diekspor ke:")
    print(f"  JSON : {out_json}")
    print(f"  CSV  : {out_csv}")

if __name__ == "__main__":
    main()
