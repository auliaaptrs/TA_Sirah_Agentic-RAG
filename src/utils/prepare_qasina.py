import json
import random
import os
import re

# --- CONFIG ---
QASINA_PATH = "TA_sirah/data/QASiNa.json"
OUTPUT_DIR = "TA_sirah/data/finetune_dataset"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
VALID_FILE = os.path.join(OUTPUT_DIR, "valid.jsonl")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.jsonl")

# Prompt Template (Context-Aware)
SYSTEM_PROMPT = "Anda adalah Ahli Sejarah Sirah Nabawiyah. Jawablah pertanyaan berdasarkan konteks yang diberikan dengan jujur dan akurat."

def format_to_messages(context, question, answer):
    user_content = f"KONTEKS:\n{context}\n\nPertanyaan: {question}"
    # Kita tambahkan [1] sebagai simulasi sitasi agar model terbiasa dengan format RAG kita
    assistant_content = f"{answer} [1]" 
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def main():
    if not os.path.exists(QASINA_PATH):
        print(f"Error: {QASINA_PATH} tidak ditemukan.")
        return

    with open(QASINA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Memproses {len(data)} konteks dari QASiNa...")

    # 1. Shuffle konteks (untuk mencegah bias urutan bab)
    random.seed(42)
    random.shuffle(data)

    # 2. Split data (80% Train, 10% Valid, 10% Test)
    n = len(data)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test": data[valid_end:]
    }

    counts = {"train": 0, "valid": 0, "test": 0}

    # 3. Proses dan simpan ke JSONL
    for mode, contexts in splits.items():
        file_path = os.path.join(OUTPUT_DIR, f"qasina_{mode}.jsonl")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for ctx_obj in contexts:
                context_text = ctx_obj['context']
                for qa in ctx_obj['question_answers']:
                    formatted = format_to_messages(context_text, qa['question'], qa['answer'])
                    f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                    counts[mode] += 1
        
        print(f"OK: Berhasil mengekstrak {counts[mode]} Q&A ke {file_path}")

    print("\n--- RINGKASAN ---")
    print(f"Total Q&A dari QASiNa: {sum(counts.values())}")
    print("Saran: Silakan gabungkan (append) file qasina_*.jsonl tersebut ke dataset utama Anda.")

if __name__ == "__main__":
    main()
