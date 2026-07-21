import os
import sys
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

# Gunakan JSON yang berisi 1901 chunk yang sudah jadi (termasuk QASiNa)
CHUNKS_JSON_PATH = os.path.join(BASE_DIR, "data", "vectordb", "chunks_toc_baseline_BACKUP_CHUNK1000.json")

# Model baru dan Lokasi Qdrant baru
EMBEDDING_MODEL = os.path.join(BASE_DIR, "models", "finetuned-e5-sirah")
QDRANT_PATH = os.path.join(BASE_DIR, "data", "vectordb", "qdrant_toc_finetuned")
COLLECTION_NAME = "sirah_nabawiyah_toc"
VECTOR_SIZE = 1024

def main():
    if not os.path.exists(EMBEDDING_MODEL):
        print(f"ERROR: Model hasil finetune tidak ditemukan di {EMBEDDING_MODEL}")
        print("Pastikan Anda sudah mengunduh file ZIP dari Jupyter dan mengekstraknya di folder models/")
        return

    print("1. Membaca JSON 1901 Chunk yang sudah ada...")
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Total Chunk Ditemukan: {len(chunks)} chunks (Tidak akan ada yang berubah)")

    print(f"\n2. Memuat Model Fine-Tuned dari: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\n3. Membuat Database Qdrant Baru di: {QDRANT_PATH}...")
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    # Buat ulang collection agar kosong dan bersih
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    print("\n4. Meng-embed dan Mengunggah vektor ke Qdrant Baru...")
    BATCH_SIZE = 32
    point_id = 0
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting"):
        batch = chunks[i : i + BATCH_SIZE]
        
        # Format prefix khusus E5 model
        texts = []
        for c in batch:
            meta = c.get("metadata", {})
            prefix = f"Bab {meta.get('bab_title', '')}. Subbab {meta.get('subbab_title', '')}. "
            texts.append(f"passage: {prefix}{c['text']}")

        # Generate Embedding dengan Model Baru
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        points = []
        for j, c in enumerate(batch):
            points.append(PointStruct(
                id=point_id + j,
                vector=embeddings[j],
                payload={**c.get("metadata", {}), "chunk_id": c["id"], "text": c["text"]}
            ))
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        point_id += len(batch)

    print("\n✅ SELESAI! Qdrant berhasil dibangun ulang dengan model finetune!")
    print(f"Total Points di Qdrant: {client.get_collection(COLLECTION_NAME).points_count}")

if __name__ == "__main__":
    main()
