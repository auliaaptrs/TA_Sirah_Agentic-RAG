"""
eval_multiturn.py  — Uji Coba 4 (Kualitas Percakapan Multi-Turn)
=================================================================
Skrip evaluasi kemampuan sistem RAG dalam menangani percakapan
bertingkat (multi-turn), di mana pertanyaan berikutnya bergantung
pada konteks jawaban sebelumnya dalam satu sesi.

Strategi dataset:
  Menggunakan dataset yang sudah ada (test_ok.jsonl) tanpa perlu
  membuat dataset baru. Soal-soal dari BAB yang sama dikelompokkan
  secara otomatis menjadi satu sesi percakapan (2–5 turns per sesi).

Alur kerja:
  [Sesi BAB] → Turn-by-turn RAG query (dengan/tanpa chat history)
  → Evaluasi otomatis per turn + penilaian LLM Judge per sesi

Metrik yang dihitung:
  - F1 / BERTScore     : Kualitas jawaban per turn vs gold answer
  - Context Retention  : Skor 0–1 (LLM Judge) — apakah jawaban Turn 2+
                         masih konsisten dengan topik sesi yang sedang berjalan
  - Session Coherence  : Skor 0–1 (LLM Judge) — kekoherensian
                         keseluruhan sesi percakapan
  - Avg Latency/Turn   : Kecepatan rata-rata per giliran percakapan

Format dataset:
  - Gunakan test_ok.jsonl (bab-based split, zero leakage)
  - Soal dari bab yang sama dikelompokkan otomatis menjadi satu sesi

==============================================================
PANDUAN PENGGUNAAN (untuk rekonstruksi eksperimen)
==============================================================

[UJI COBA 4 — Kualitas Percakapan Multi-Turn]
  Tujuan: Mengevaluasi kemampuan pipeline Agentik dalam mempertahankan
          konteks percakapan lintas giliran, dibandingkan pipeline
          Konvensional yang tidak memiliki memori sesi.
  Catatan: Model generator dan Sufficiency Evaluator menggunakan
           konfigurasi terpilih dari Uji Coba 2 & 3 (Qwen 2.5 7B +
           SEA-LION). Pastikan OLLAMA_MODEL dan OLLAMA_EVALUATOR_MODEL
           di config.py sudah diset dengan benar sebelum menjalankan.

  # Pipeline Agentik (dengan memori percakapan / chat history):
  python -m src.evaluation.eval_multiturn --mode agentic

  # Pipeline Konvensional (tanpa memori percakapan):
  python -m src.evaluation.eval_multiturn --mode conventional

  Output: data/evaluation/skenario_5/eval_5_multiturn_<mode>_<model>.xlsx

--------------------------------------------------------------

Argumen:
  --mode  : 'agentic' untuk pipeline Agentik dengan chat history,
            'conventional' untuk pipeline Konvensional tanpa memori sesi
"""


import os
import sys
import json
import time
import re
import random
import argparse
from collections import defaultdict
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

# ── Path Setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.rag.config import (
    OLLAMA_MODEL, OLLAMA_EVALUATOR_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    JUDGE_PROVIDER,
)
from src.rag.utils import init_vectordb, init_llm
from src.rag.real_agentic import real_agentic_rag_query
from src.rag.conventional_rag import conventional_rag_query

# ── Konfigurasi ───────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "finetune_dataset", "test_ok.jsonl")
OUTPUT_DIR        = os.path.join(BASE_DIR, "data", "evaluation", "skenario_5")
MIN_TURNS         = 2    # Minimal soal per BAB untuk dijadikan sesi
MAX_TURNS         = 5    # Maksimal soal per sesi (agar tidak terlalu panjang)
MAX_SESSIONS      = 50   # Batasi jumlah sesi agar evaluasi tidak terlalu lama
RANDOM_SEED       = 42

# ── PROMPTS ───────────────────────────────────────────────────────────────────

CONTEXT_RETENTION_PROMPT = """\
Anda adalah Evaluator Percakapan AI.

Sesi percakapan berlangsung dalam topik: "{topic}"

Giliran (Turn) sebelumnya membahas tentang:
{prev_summary}

Pertanyaan baru yang diajukan:
"{question}"

Jawaban yang diberikan sistem:
"{answer}"

Tugas: Nilai apakah jawaban sistem RELEVAN dan KONSISTEN dengan topik sesi.
Berikan skor dari 0.0 hingga 1.0:
- 1.0 = Jawaban konsisten dengan topik sesi dan menjawab pertanyaan dengan tepat
- 0.5 = Jawaban menjawab pertanyaan tapi kurang mempertimbangkan konteks sesi
- 0.0 = Jawaban melenceng dari topik sesi atau tidak koheren

Balas HANYA dengan angka desimal (contoh: 0.8), tanpa penjelasan."""

SESSION_COHERENCE_PROMPT = """\
Anda adalah Evaluator Percakapan AI.

Berikut adalah satu sesi percakapan lengkap tentang topik: "{topic}"

{conversation}

Tugas: Nilai KEKOHERENSIAN sesi percakapan ini secara keseluruhan (0.0-1.0):
- Apakah jawaban-jawaban saling berkaitan dan membangun pemahaman?
- Apakah sistem terlihat "mengingat" konteks dari turn sebelumnya?
- Apakah alur percakapan terasa alami dan runtut?

Berikan skor 0.0-1.0, lalu berikan penjelasan SINGKAT (1 kalimat).
Format: <skor>|<penjelasan>
Contoh: 0.85|Sistem menjawab runtut dan mempertahankan konteks Perang Badar."""


# ── Fungsi Helper ─────────────────────────────────────────────────────────────

def call_judge(prompt: str) -> str:
    """Memanggil LLM Judge (OpenRouter) untuk evaluasi."""
    import requests
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100
    }
    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=60
        )
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"  [Judge Error] {e}")
        return "0.5"


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 antara prediksi dan gold answer."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bertscore(preds: list[str], refs: list[str]) -> list[float]:
    """Menghitung BERTScore untuk prediksi vs referensi secara batch."""
    try:
        from bert_score import score as bs_score
        safe_preds = [p if p.strip() else "." for p in preds]
        safe_refs = [r if r.strip() else "." for r in refs]
        _, _, F = bs_score(safe_preds, safe_refs, lang="id", model_type="bert-base-multilingual-cased", verbose=False, batch_size=16)
        return [round(float(f), 4) for f in F.tolist()]
    except Exception as e:
        print(f"⚠️ BERTScore failed: {e}")
        return [0.0] * len(preds)


def parse_retention_score(raw: str) -> float:
    """Parsing skor context retention dari output LLM."""
    try:
        return float(re.search(r'\d+\.?\d*', raw).group())
    except:
        return 0.5


def parse_coherence(raw: str) -> tuple:
    """Parsing skor dan penjelasan coherence dari output LLM."""
    try:
        parts = raw.split("|", 1)
        score = float(re.search(r'\d+\.?\d*', parts[0]).group())
        explanation = parts[1].strip() if len(parts) > 1 else "-"
        return min(score, 1.0), explanation
    except:
        return 0.5, raw[:100]


# ── Fungsi Utama ──────────────────────────────────────────────────────────────

def load_and_group_questions(path: str) -> Dict[str, List[Dict]]:
    """
    Membaca test_ok.jsonl dan mengelompokkan soal berdasarkan bab_titles[0].
    Hanya mengambil soal-soal yang punya minimal MIN_TURNS soal per bab.
    """
    bab_groups = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = obj.get("metadata", {})
            bab_titles = meta.get("bab_titles", [])
            if not bab_titles:
                continue

            bab_key = bab_titles[0]

            # Ekstrak pertanyaan dari pesan user
            messages = obj.get("messages", [])
            question = ""
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    # Pertanyaan ada setelah kata "Pertanyaan:"
                    match = re.search(r'Pertanyaan:\s*(.+?)$', content, re.DOTALL)
                    if match:
                        question = match.group(1).strip()
                    break

            gold_answer = meta.get("expected_answer", "")
            q_type      = meta.get("type", "unknown")

            if question and gold_answer:
                bab_groups[bab_key].append({
                    "question":    question,
                    "gold_answer": gold_answer,
                    "type":        q_type,
                    "bab_title":   bab_key
                })

    # Filter: hanya bab yang punya >= MIN_TURNS soal
    filtered = {
        bab: items
        for bab, items in bab_groups.items()
        if len(items) >= MIN_TURNS
    }

    print(f"\n[INFO] Total BAB dengan >= {MIN_TURNS} soal: {len(filtered)}")
    for bab, items in sorted(filtered.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {bab}: {len(items)} soal")

    return filtered


def build_sessions(bab_groups: Dict[str, List[Dict]]) -> List[List[Dict]]:
    """
    Membangun sesi-sesi multi-turn dari kelompok per bab.
    Setiap sesi maksimal MAX_TURNS pertanyaan.
    """
    random.seed(RANDOM_SEED)
    sessions = []

    for bab, items in bab_groups.items():
        random.shuffle(items)
        # Potong ke MAX_TURNS soal per sesi
        session = items[:MAX_TURNS]
        sessions.append(session)

    # Acak dan batasi jumlah sesi
    random.shuffle(sessions)
    sessions = sessions[:MAX_SESSIONS]

    print(f"[INFO] Total sesi yang akan diuji: {len(sessions)}")
    print(f"[INFO] Total turns: {sum(len(s) for s in sessions)}\n")
    return sessions


def run_multiturn_evaluation(sessions: List[List[Dict]], vectordb, llm_client, mode: str) -> List[Dict]:
    """
    Menjalankan evaluasi multi-turn untuk semua sesi.
    """
    all_results = []
    session_summary = []

    for s_idx, session in enumerate(tqdm(sessions, desc="Evaluating Sessions")):
        topic     = session[0]["bab_title"]
        chat_history = []
        session_rows = []

        for t_idx, turn_data in enumerate(session):
            question    = turn_data["question"]
            gold_answer = turn_data["gold_answer"]
            q_type      = turn_data["type"]

            # Jalankan RAG sesuai mode
            if mode == "agentic":
                result = real_agentic_rag_query(
                    question=question,
                    vectordb=vectordb,
                    llm_client=llm_client,
                    chat_history=chat_history if t_idx > 0 else None
                )
            else:
                # Conventional RAG tidak punya memory (chat_history)
                result = conventional_rag_query(
                    question=question,
                    vectordb=vectordb,
                    llm_client=llm_client
                )

            answer  = result.get("answer", "")
            elapsed = result.get("elapsed", 0.0)

            # ── Metrik ────────────────────────────────────────────────────────
            f1      = compute_f1(answer, gold_answer)

            # Context Retention (hanya untuk turn >= 2)
            retention_score = 1.0  # Turn pertama selalu relevan
            if t_idx > 0 and chat_history:
                prev_summary = chat_history[-1]["question"]
                ret_prompt   = CONTEXT_RETENTION_PROMPT.format(
                    topic=topic,
                    prev_summary=prev_summary,
                    question=question,
                    answer=answer[:500]
                )
                raw_ret      = call_judge(ret_prompt)
                retention_score = parse_retention_score(raw_ret)

            row = {
                "session_id":        s_idx,
                "topic_bab":         topic,
                "turn":              t_idx + 1,
                "question_type":     q_type,
                "question":          question,
                "gold_answer":       gold_answer,
                "predicted_answer":  answer,
                "f1":                round(f1, 4),
                "context_retention": round(retention_score, 4),
                "latency_s":         round(elapsed, 2),
                "is_first_turn":     (t_idx == 0),
            }
            session_rows.append(row)
            all_results.append(row)

            # Simpan ke chat_history untuk turn berikutnya
            chat_history.append({
                "question": question,
                "answer":   answer
            })

        # ── Session Coherence (dinilai sekali di akhir sesi) ──────────────────
        conversation_str = ""
        for i, (row, turn_data) in enumerate(zip(session_rows, session)):
            conversation_str += (
                f"Turn {i+1} [{turn_data['type']}]:\n"
                f"  Q: {row['question']}\n"
                f"  A: {row['predicted_answer'][:300]}...\n\n"
            )

        coh_prompt  = SESSION_COHERENCE_PROMPT.format(
            topic=topic,
            conversation=conversation_str
        )
        raw_coh     = call_judge(coh_prompt)
        coh_score, coh_explanation = parse_coherence(raw_coh)

        # Tambahkan skor coherence ke semua row dalam sesi ini
        for row in session_rows:
            row["session_coherence"]     = round(coh_score, 4)
            row["coherence_explanation"] = coh_explanation

        session_summary.append({
            "session_id":       s_idx,
            "topic_bab":        topic,
            "n_turns":          len(session),
            "avg_f1":           round(sum(r["f1"] for r in session_rows) / len(session_rows), 4),
            "avg_retention":    round(sum(r["context_retention"] for r in session_rows) / len(session_rows), 4),
            "session_coherence": round(coh_score, 4),
            "avg_latency":      round(sum(r["latency_s"] for r in session_rows) / len(session_rows), 2),
        })

    return all_results, session_summary


def print_summary(all_results: List[Dict], session_summary: List[Dict], mode: str):
    """Mencetak ringkasan statistik evaluasi multi-turn."""
    if not all_results:
        print("Tidak ada hasil.")
        return

    # Pisahkan turn pertama dan turn lanjutan
    first_turns = [r for r in all_results if r["is_first_turn"]]
    later_turns = [r for r in all_results if not r["is_first_turn"]]

    avg = lambda lst, key: sum(x[key] for x in lst) / len(lst) if lst else 0

    print("="*65)
    print(f"  SKENARIO 5: MULTI-TURN CONVERSATION EVALUATION")
    print(f"  Mode              : {mode.upper()}")
    print("="*65)
    print(f"  Generator Model   : {OLLAMA_MODEL}")
    print(f"  Evaluator Model   : {OLLAMA_EVALUATOR_MODEL}")
    print(f"  Total Sesi        : {len(session_summary)}")
    print(f"  Total Turns       : {len(all_results)}")
    print(f"  Turn Pertama      : {len(first_turns)}")
    print(f"  Turn Lanjutan     : {len(later_turns)}")
    print("-"*65)
    f1_t1 = avg(first_turns, 'f1')
    f1_t2 = avg(later_turns, 'f1')
    f1_delta = f1_t2 - f1_t1

    bert_t1 = avg(first_turns, 'bert_score')
    bert_t2 = avg(later_turns, 'bert_score')
    bert_delta = bert_t2 - bert_t1

    print(f"  Avg F1 (All Turns)          : {avg(all_results, 'f1'):.4f}")
    print(f"  Avg F1 (Turn 1)             : {f1_t1:.4f}")
    print(f"  Avg F1 (Turn 2+)            : {f1_t2:.4f} [Delta: {f1_delta:+.4f}]")
    print(f"  Avg BERTScore (All)         : {avg(all_results, 'bert_score'):.4f}")
    print(f"  Avg BERTScore (Turn 1)      : {bert_t1:.4f}")
    print(f"  Avg BERTScore (Turn 2+)     : {bert_t2:.4f} [Delta: {bert_delta:+.4f}]")
    print(f"  Avg Context Retention (Turn 2+) : {avg(later_turns, 'context_retention'):.4f}")
    print(f"  Avg Session Coherence       : {avg(all_results, 'session_coherence'):.4f}")
    print(f"  Avg Latency/Turn (s)        : {avg(all_results, 'latency_s'):.2f}")
    print("="*65)


def main():
    parser = argparse.ArgumentParser(description="Jalankan Skenario 5 (Multi-Turn).")
    parser.add_argument("--mode", type=str, default="agentic", choices=["agentic", "conventional"],
                        help="Mode sistem: 'agentic' (dengan memori) atau 'conventional' (tanpa memori).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generator_name = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
    evaluator_name = OLLAMA_EVALUATOR_MODEL.replace(":", "_").replace("/", "_")
    output_file = os.path.join(
        OUTPUT_DIR,
        f"eval_5_multiturn_{args.mode}_{generator_name}_eval_{evaluator_name}.xlsx"
    )

    print("="*65)
    print(f"  EVAL 5: MULTI-TURN CONVERSATION — SKENARIO 5 ({args.mode.upper()})")
    print(f"  Output → {OUTPUT_DIR}")
    print("="*65)

    # 1. Load & Kelompokkan soal
    bab_groups = load_and_group_questions(GROUND_TRUTH_PATH)
    sessions   = build_sessions(bab_groups)

    # 2. Init model
    print("Memuat model embedding...")
    vectordb = init_vectordb()
    llm      = init_llm()
    print(f"[OK] Siap. Mode: {args.mode.upper()} | Generator: {OLLAMA_MODEL} | Evaluator: {OLLAMA_EVALUATOR_MODEL}\n")

    # 3. Jalankan evaluasi
    all_results, session_summary = run_multiturn_evaluation(sessions, vectordb, llm, args.mode)

    # 3.5 Hitung BERTScore
    print("\n🧠 Computing BERTScore...")
    all_preds = [r["predicted_answer"] for r in all_results]
    all_refs  = [r["gold_answer"] for r in all_results]
    bert_scores = compute_bertscore(all_preds, all_refs)
    
    for i, score in enumerate(bert_scores):
        all_results[i]["bert_score"] = score
        
    # Update session_summary dengan avg_bert_score
    for summary in session_summary:
        s_id = summary["session_id"]
        s_rows = [r for r in all_results if r["session_id"] == s_id]
        if s_rows:
            summary["avg_bert_score"] = round(sum(r["bert_score"] for r in s_rows) / len(s_rows), 4)

    # 4. Print ringkasan
    print_summary(all_results, session_summary, args.mode)

    # 5. Simpan ke Excel
    df_detail  = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(session_summary)

    first_turns = [r for r in all_results if r["is_first_turn"]]
    later_turns = [r for r in all_results if not r["is_first_turn"]]
    avg = lambda lst, key: sum(x[key] for x in lst) / len(lst) if lst else 0
    
    f1_t1 = avg(first_turns, 'f1')
    f1_t2 = avg(later_turns, 'f1')
    bert_t1 = avg(first_turns, 'bert_score')
    bert_t2 = avg(later_turns, 'bert_score')
    
    df_tabel1 = pd.DataFrame([
        {"Metrik Evaluasi": "Avg. BERTScore", "Turn 1 (Konteks Baru)": round(bert_t1, 4), "Turn 2+ (Lanjutan)": round(bert_t2, 4), "Selisih (Delta)": round(bert_t2 - bert_t1, 4)},
        {"Metrik Evaluasi": "Avg. F1 Score", "Turn 1 (Konteks Baru)": round(f1_t1, 4), "Turn 2+ (Lanjutan)": round(f1_t2, 4), "Selisih (Delta)": round(f1_t2 - f1_t1, 4)}
    ])
    
    df_tabel2 = pd.DataFrame([
        {"Metrik Kognitif Sesi": "Avg. Context Retention", "Skor Rata-rata": round(avg(later_turns, 'context_retention'), 4)},
        {"Metrik Kognitif Sesi": "Avg. Session Coherence", "Skor Rata-rata": round(avg(all_results, 'session_coherence'), 4)},
        {"Metrik Kognitif Sesi": "Avg. Latency/Turn (s)", "Skor Rata-rata": round(avg(all_results, 'latency_s'), 2)}
    ])

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_detail.to_excel(writer,  sheet_name="Detail Per Turn",   index=False)
        df_summary.to_excel(writer, sheet_name="Summary Per Sesi",  index=False)
        df_tabel1.to_excel(writer,  sheet_name="Tabel 1 - Akurasi", index=False)
        df_tabel2.to_excel(writer,  sheet_name="Tabel 2 - Sesi",    index=False)

    print(f"\n[OK] Hasil disimpan ke: {output_file}")


if __name__ == "__main__":
    main()
