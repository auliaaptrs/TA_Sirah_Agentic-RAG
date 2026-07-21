import json
import os
import sys
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Path Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.rag.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL

def main():
    # 1. Load Data
    qasina_path = os.path.join(BASE_DIR, "data", "QASiNa.json")
    if not os.path.exists(qasina_path):
        print(f"ERROR: File {qasina_path} tidak ditemukan!")
        return

    with open(qasina_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Init Clients
    print(f"INFO: Memuat model embedding: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"INFO: Menghubungkan ke Qdrant di {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)

    # 3. Prepare Points
    points = []
    seen_contexts = set()
    NAMESPACE_QASINA = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    
    print("INFO: Memproses konteks QASiNa...")
    for item in tqdm(data):
        context_text = item['context']
        if context_text in seen_contexts:
            continue
        seen_contexts.add(context_text)
        
        ctx_id = str(item['context_id'])
        title = item.get('context_title', 'Tanpa Judul')
        
        # Generate Embedding
        vector = model.encode(f"passage: {context_text}", normalize_embeddings=True).tolist()
        
        # FIX: Gunakan UUID agar tidak error di Qdrant Local
        point_id = str(uuid.uuid5(NAMESPACE_QASINA, f"qasina_{ctx_id}"))
        
        points.append(models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "chunk_id": f"qasina_{ctx_id}",
                "text": context_text,
                "bab_title": f"QASiNa: {title}",
                "subbab_title": "External Database",
                "page_start": "N/A"
            }
        ))

    # 4. Upsert
    print(f"INFO: Mengunggah {len(points)} konteks ke koleksi '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print("OK: Berhasil! Database Qdrant Anda sekarang lebih kaya dengan data QASiNa.")

if __name__ == "__main__":
    main()
