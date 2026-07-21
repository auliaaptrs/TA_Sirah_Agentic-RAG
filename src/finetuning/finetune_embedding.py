import os
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, losses, evaluation, util,
    SentenceTransformerTrainer, SentenceTransformerTrainingArguments
)
from sentence_transformers.training_args import BatchSamplers
from transformers import TrainerCallback

# --- CONFIG ---
MODEL_NAME = "intfloat/multilingual-e5-large"
TRAIN_FILE = "data/finetune_dataset/train_ok.jsonl"
VAL_FILE = "data/finetune_dataset/valid_ok.jsonl"
CHUNKS_FILE = "data/vectordb/chunks_toc_baseline_BACKUP_CHUNK1000.json"
OUTPUT_MODEL_PATH = "models/finetuned-e5-sirah"
BATCH_SIZE = 64   # Naik 64→128: 2x lebih banyak in-batch negatives untuk MNRL!
EPOCHS = 3        # Lebih banyak kesempatan konvergen dengan LR yang lebih kecil
LEARNING_RATE = 2e-5  # Turun 2e-5→1e-5: mencegah overfitting yang terjadi di epoch 2
METRIC_NAME = "val-retrieval_cos_sim_mrr@10"

# ── Callback: simpan hanya saat metric membaik, timpa file sama ──────
class SaveBestOnlyCallback(TrainerCallback):
    def __init__(self, model, save_path, metric_name):
        self.model      = model
        self.save_path  = save_path
        self.metric_key = f"eval_{metric_name}"
        self.best       = -float("inf")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current = metrics.get(self.metric_key, -1.0)
        if current > self.best:
            self.best = current
            print(f"\n✅ Best model! {self.metric_key}: {current:.4f} → Saving...")
            self.model.save_pretrained(self.save_path)
            print(f"   Saved to {self.save_path}")
        else:
            print(f"\n⏭️  No improvement ({current:.4f} ≤ {self.best:.4f}), skip save.")


def load_chunks_map(filepath):
    print(f"Loading chunks mapping from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    chunks_map = {}
    for c in chunks:
        cid  = str(c.get('chunk_id', c.get('id')))
        meta = c.get("metadata", {})
        prefix = f"Bab {meta.get('bab_title', '')}. Subbab {meta.get('subbab_title', '')}. "
        chunks_map[cid] = prefix + c['text']
    return chunks_map


def prepare_examples(train_file, chunks_map):
    print(f"Preparing training examples from {train_file}...")
    anchors, positives = [], []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = ""
            for msg in data.get("messages", []):
                if msg["role"] == "user":
                    query = msg["content"]
                    break
            for sep in ["### Pertanyaan:", "Pertanyaan:", "pertanyaan:"]:
                if sep in query:
                    query = query.split(sep)[-1].strip()
                    break
            query    = f"query: {query}"
            gold_ids = data.get("metadata", {}).get("gold_chunk_ids", [])
            if gold_ids:
                pos_text = chunks_map.get(str(gold_ids[0]))
                if pos_text:
                    anchors.append(query)
                    positives.append(f"passage: {pos_text}")
    print(f"Total training examples: {len(anchors)}")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def build_ir_evaluator(val_file, chunks_map):
    print(f"Preparing evaluator from {val_file}...")
    queries, corpus, relevant_docs = {}, {}, {}
    for cid, text in chunks_map.items():
        corpus[cid] = f"passage: {text}"
    with open(val_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            query = ""
            for msg in data.get("messages", []):
                if msg["role"] == "user":
                    query = msg["content"]
                    break
            for sep in ["### Pertanyaan:", "Pertanyaan:", "pertanyaan:"]:
                if sep in query:
                    query = query.split(sep)[-1].strip()
                    break
            qid = f"q_{i}"
            queries[qid]       = f"query: {query}"
            relevant_docs[qid] = set()
            for g_id in data.get("metadata", {}).get("gold_chunk_ids", []):
                cid = str(g_id)
                if cid in corpus:
                    relevant_docs[qid].add(cid)
    return evaluation.InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name="val-retrieval", show_progress_bar=True,
        score_functions={"cos_sim": util.cos_sim}
    )


def train():
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

    if torch.cuda.is_available():
        free  = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.mem_get_info()[1] / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {free:.1f}/{total:.1f} GB")

    chunks_map    = load_chunks_map(CHUNKS_FILE)
    train_dataset = prepare_examples(TRAIN_FILE, chunks_map)

    print(f"Loading base model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Hapus HF cache setelah model masuk VRAM
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("HF cache cleared.")

    # Hitung warmup_steps manual (fix deprecation warning warmup_ratio)
    steps_per_epoch = -(-len(train_dataset) // BATCH_SIZE)  # ceiling division
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = int(total_steps * 0.1)
    print(f"Steps: {steps_per_epoch}/epoch × {EPOCHS} = {total_steps} total | warmup: {warmup_steps}")

    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator  = build_ir_evaluator(VAL_FILE, chunks_map)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,          # ← fix deprecation warning
        fp16=True,
        gradient_checkpointing=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="no",                 # ← trainer tidak simpan apapun
        load_best_model_at_end=False,       # ← kita handle manual via callback
    )

    best_callback = SaveBestOnlyCallback(model, OUTPUT_MODEL_PATH, METRIC_NAME)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=[best_callback],          # ← callback custom
    )

    print(f"Starting fine-tuning...")
    trainer.train()

    print(f"\nTraining complete!")
    print(f"Best MRR@10: {best_callback.best:.4f}")
    print(f"Best model at: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    train()
