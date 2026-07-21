"""
finetune_evaluator.py
======================
Skrip untuk melakukan Fine-Tuning model LLM khusus untuk tugas
Sufficiency Evaluator (Agen Penilai Kecukupan Konteks).
Mendukung berbagai arsitektur base model: Qwen2.5, Llama-3, Gemma-2, dll.

Cara Pakai (di Jupyter / RunPod / Colab — RTX A6000):
    1. Ganti BASE_MODEL di bagian CONFIG sesuai model yang ingin di-finetune.
    2. Jalankan semua cell dari atas ke bawah.
    3. Setelah selesai, model LoRA tersimpan di OUTPUT_DIR.
    4. Export GGUF (cell paling bawah) lalu upload ke Ollama.

Model yang sudah diuji kompatibel dengan script ini:
    - "Qwen/Qwen2.5-7B-Instruct"      (Qwen 2.5  7B  — default TA)
    - "Qwen/Qwen2.5-14B-Instruct"     (Qwen 2.5  14B — jika VRAM cukup)
    - "Qwen/Qwen3-8B"                  (Qwen 3    8B  — eksperimen generasi baru)
    - "meta-llama/Llama-3.1-8B-Instruct"
    - "aisingapore/sea-lion-7b-instruct"
    - "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct"

Catatan VRAM RTX A6000 (48 GB):
    - 7B/8B/9B  4-bit LoRA  → ~12-18 GB  → aman, batch bisa dinaikkan
    - 14B        4-bit LoRA  → ~22-28 GB  → aman
    - 70B        4-bit LoRA  → ~48 GB     → mepet, gunakan GRAD_ACCUM tinggi
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ═══════════════════════════════════════════════════════════
#  CONFIG — HANYA BAGIAN INI YANG PERLU DIUBAH
# ═══════════════════════════════════════════════════════════

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"   # ← Ganti model di sini

# Nama pendek untuk penamaan folder output (bebas, tidak harus sama dengan HF name)
MODEL_SHORT_NAME = "llama3.1-8b"            # contoh: "llama3.1-8b", "gemma2-9b"

# Path data — sesuaikan jika menjalankan dari direktori lain
TRAIN_FILE = "data/finetune_dataset/train_evaluator.jsonl"
VALID_FILE = "data/finetune_dataset/valid_evaluator.jsonl"

# Output
OUTPUT_DIR = f"outputs/{MODEL_SHORT_NAME}-evaluator-lora"
GGUF_DIR   = f"outputs/{MODEL_SHORT_NAME}-evaluator-gguf"

# Hyperparameters
MAX_SEQ_LEN = 4096
LORA_RANK   = 16
LORA_ALPHA  = 32
BATCH_SIZE  = 1
GRAD_ACCUM  = 16   # Naikkan ke 32 jika model 14B+ dan VRAM mulai seret
EPOCHS      = 1
LR          = 2e-4

# ═══════════════════════════════════════════════════════════
#  HELPER: Deteksi chat template yang benar per model
# ═══════════════════════════════════════════════════════════

def get_chat_template_name(model_name: str) -> str:
    """
    Mengembalikan nama chat template yang sesuai untuk setiap keluarga model.

    Urutan pengecekan penting: Gemma dicek SEBELUM Sea-LION karena
    Gemma-SEA-LION-v3-9B menggunakan Gemma 2 sebagai base architecture.
    """
    name = model_name.lower()
    if "qwen" in name:
        return "qwen-2.5"          # Cocok untuk Qwen2.5 dan Qwen3
    elif "llama-3" in name or "llama3" in name:
        return "llama-3.1"         # Cocok untuk Llama 3.1 dan 3.2
    elif "gemma" in name or "sahabat" in name:
        return "gemma"             # Cocok untuk Gemma 2, Sahabat-AI
    elif "mistral" in name or "mixtral" in name:
        return "mistral"
    elif "sea-lion" in name or "sealion" in name:
        # Sea-LION instruct AI Singapore menggunakan ChatML
        return "chatml"
    else:
        print(f"[WARN] Model '{model_name}' tidak dikenali — menggunakan chat template default tokenizer.")
        return None

# ═══════════════════════════════════════════════════════════
#  STEP 1: Load Model & Tokenizer
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print(f"  Fine-tuning Sufficiency Evaluator")
print(f"  Model : {BASE_MODEL}")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,    # Auto-detect: bf16 jika didukung, fp16 jika tidak
    load_in_4bit   = True,    # QLoRA — hemat VRAM
)

# Terapkan chat template yang sesuai
chat_template = get_chat_template_name(BASE_MODEL)
if chat_template:
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    print(f"[OK] Chat template diterapkan: {chat_template}")
else:
    print(f"[OK] Menggunakan chat template bawaan tokenizer.")

# ═══════════════════════════════════════════════════════════
#  STEP 2: Tambahkan LoRA Adapter
# ═══════════════════════════════════════════════════════════

model = FastLanguageModel.get_peft_model(
    model,
    r               = LORA_RANK,
    target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    lora_alpha      = LORA_ALPHA,
    lora_dropout    = 0,           # 0 = direkomendasikan Unsloth untuk efisiensi
    bias            = "none",
    use_gradient_checkpointing = True,   # Ganti ke True agar tidak Pickling Error
    random_state    = 42,
)

print(f"[OK] LoRA adapter ditambahkan (rank={LORA_RANK}, alpha={LORA_ALPHA})")

# ═══════════════════════════════════════════════════════════
#  STEP 3: Load & Format Dataset
# ═══════════════════════════════════════════════════════════

def load_and_format(file_path: str) -> Dataset:
    """
    Membaca file JSONL (format messages ChatML) dan mengonversinya
    ke format teks menggunakan chat template tokenizer yang sudah diset.

    Format input setiap baris:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    data = []
    skipped = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                messages = record.get("messages", [])
                if not messages:
                    skipped += 1
                    continue
                # apply_chat_template: tokenize=False → kembalikan string teks
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                data.append({"text": text})
            except Exception as e:
                skipped += 1
                print(f"  [WARN] Baris dilewati: {e}")

    if skipped:
        print(f"  [WARN] {skipped} baris dilewati karena format tidak valid.")
    return Dataset.from_list(data)


print(f"\n[INFO] Memuat dataset...")
train_ds = load_and_format(TRAIN_FILE).shuffle(seed=42)
valid_ds = load_and_format(VALID_FILE)
print(f"[OK]  Train: {len(train_ds)} sampel | Valid: {len(valid_ds)} sampel")

# ═══════════════════════════════════════════════════════════
#  STEP 4: SFT Trainer
# ═══════════════════════════════════════════════════════════

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = valid_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    packing            = False,    # False lebih stabil untuk sequence panjang
    args = SFTConfig(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = 50,
        num_train_epochs            = EPOCHS,
        learning_rate               = LR,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 10,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 42,
        output_dir                  = OUTPUT_DIR,
        report_to                   = "none",      # Ganti "wandb" untuk tracking
        save_strategy               = "steps",
        save_steps                  = 50,
        save_total_limit            = 3,             # Simpan maksimal 3 checkpoint untuk hemat storage
        eval_strategy               = "no",          # Dinonaktifkan sementara karena memicu OOM
        eval_steps                  = 50,
        load_best_model_at_end      = False,         # Wajib False jika eval_strategy = "no"
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        per_device_eval_batch_size  = 1,
        eval_accumulation_steps     = 4,
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LEN,
    ),
)

# ═══════════════════════════════════════════════════════════
#  STEP 5: Training
# ═══════════════════════════════════════════════════════════

print(f"\n[INFO] Training dimulai...")
print(f"       Model     : {BASE_MODEL}")
print(f"       Train     : {len(train_ds)} sampel")
print(f"       Epochs    : {EPOCHS}")
print(f"       Batch size: {BATCH_SIZE} × {GRAD_ACCUM} (efektif = {BATCH_SIZE * GRAD_ACCUM})")
print(f"       Output    : {OUTPUT_DIR}")
print("-" * 60)

trainer_stats = trainer.train() 

# ═══════════════════════════════════════════════════════════
#  STEP 6: Simpan LoRA Adapter
# ═══════════════════════════════════════════════════════════

print(f"\n[INFO] Menyimpan LoRA adapter ke: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print(f"  SELESAI! LoRA tersimpan di: {OUTPUT_DIR}")
print(f"  Runtime: {trainer_stats.metrics.get('train_runtime', 0):.0f} detik")
print(f"  Train Loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
#  STEP 7: Selesai! (Export GGUF dilakukan terpisah di Colab)
# ═══════════════════════════════════════════════════════════
# Sesuai strategi hemat memori, script ini hanya menyimpan LoRA.
# File LoRA dapat di-zip dan di-export ke GGUF melalui Google Colab.
