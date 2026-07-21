"""
build_vectordb.py  (Research-Grade v2)
=======================================
Membangun Qdrant vector database dari document_full_cleaned.json
dengan pipeline retrieval yang dioptimasi untuk domain sejarah naratif.

Improvement dari versi sebelumnya:
  [1] Pre-clean teks OCR sebelum embed (hapus artefak scan, footnote, dll.)
  [2] Paragraph-aware splitting (bukan hanya regex punctuation)
      → split by paragraph/discourse marker dulu, baru kalimat
  [3] Ukuran chunk lebih kecil: 1000 chars (~200-250 token)
      → meningkatkan retrieval precision
  [4] Overlap lebih kecil: 100 chars
  [5] Teks yang diembed = teks yang sudah dibersihkan (bukan raw OCR)

Catatan kompatibilitas:
  - Collection name & embedding model SAMA → agentic_rag.py tidak perlu diubah.
  - chunks_toc_baseline.json akan ditimpa dengan versi bersih.

Usage:
    python build_vectordb.py
"""

import json
import os
import re
import sys

# ── PATH CONFIG ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
INPUT_JSON = os.path.join(DATA_DIR, "preprocess_result", "toc_result", "document_full_cleaned.json")
QDRANT_PATH = os.path.join(DATA_DIR, "vectordb", "qdrant_toc_baseline")

# ── CHUNKING CONFIG ──────────────────────────────────────────────────
MAX_CHUNK_CHARS = 2000   # CONFIG D: Eksperimen 2000 chars
OVERLAP_CHARS   = 300    # CONFIG D: 300 chars overlap
MIN_CHUNK_CHARS = 80     # skip chunk terlalu pendek

# ── EMBEDDING CONFIG ─────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "sirah_nabawiyah_toc"   # SAMA agar agentic_rag.py kompatibel
VECTOR_SIZE     = 1024


# ═══════════════════════════════════════════════════════════════════════
#  [1] TEXT CLEANING — hapus artefak OCR sebelum embed
# ═══════════════════════════════════════════════════════════════════════

def clean_text(text: str, collapse_newlines: bool = False) -> str:
    """
    Membersihkan teks OCR secara AMAN untuk domain sejarah/sirah.

    Yang dibersihkan (aman):
    - Karakter non-printable / garbled scan
    - Baris kosong berlebih (max 2)
    - Spasi berlebih dalam satu baris
    - Tanda baca yang berulang (........, ,,,,)
    - Carriage return Windows (\r)

    Yang TIDAK dihapus (penting untuk fakta sejarah):
    - Angka berapa pun (3, 12, 40, 1400, dll.) → jumlah pasukan, tahun, umur
    - Angka romawi → nomor bab, urutan
    - Token numerik apapun

    Parameter:
    - collapse_newlines=True  → newline diganti spasi (dipakai SETELAH paragraph split)
    - collapse_newlines=False → newline dipertahankan (dipakai SEBELUM paragraph split)
    """
    # Hapus karakter kontrol non-printable (bukan \n, \t)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)

    # Normalisasi \r\n → \n
    text = re.sub(r'\r\n|\r', '\n', text)

    # Maksimal 2 newline berurutan (jaga paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Hapus spasi berlebih dalam satu baris (bukan newline)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Tanda baca berulang
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r',{2,}', ',', text)

    # Collapse newline → spasi (hanya jika diminta, yaitu setelah paragraph split)
    if collapse_newlines:
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════
#  [2] PARAGRAPH-AWARE SPLITTING
# ═══════════════════════════════════════════════════════════════════════

# Discourse markers yang sering muncul di teks sejarah Indonesia
# → dipakai sebagai boundary splitting sekunder
_DISCOURSE_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=(?:'
    r'Kemudian|Setelah\s+itu|Lalu|Selanjutnya|Adapun|Sementara\s+itu|'
    r'Di\s+sisi\s+lain|Pada\s+saat\s+itu|Akhirnya|Namun|Akan\s+tetapi|'
    r'Oleh\s+karena\s+itu|Dengan\s+demikian|Maka|Di\s+samping\s+itu'
    r'))',
    re.IGNORECASE
)

def split_into_paragraphs(text: str) -> list[str]:
    """
    Memecah teks menjadi unit semantik naratif.

    Strategi (dari kasar ke cukup — TIDAK sampai kalimat):
    1. Split by double newline  → paragraph break eksplisit
    2. Split by discourse markers (kemudian, setelah itu, dll.)

    Unit yang dihasilkan adalah klausa/paragraf mini, BUKAN kalimat individual.
    Teks sejarah naratif lebih coherent jika unit = discourse chunk, bukan kalimat.
    Merge greedy berikutnya yang mengontrol ukuran akhir chunk.
    """
    # Langkah 1: paragraph break eksplisit
    raw_paragraphs = re.split(r'\n{2,}', text)

    units: list[str] = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Collapse newline sisa dalam satu paragraf → spasi
        para = re.sub(r'\n+', ' ', para).strip()

        # Langkah 2: discourse markers → semantic boundary
        discourse_parts = _DISCOURSE_PATTERN.split(para)

        for part in discourse_parts:
            part = part.strip()
            # [FIX #1] Simpan discourse unit UTUH — jangan pecah jadi kalimat
            # Karena fakta sejarah sering tersebar dalam 3-6 kalimat berurutan
            if part:
                units.append(part)

    return units


def _sentence_boundary_overlap(prev_chunk: str, overlap: int) -> str:
    """
    Ambil overlap dari akhir chunk sebelumnya dengan menghormati batas kalimat.
    Mencegah kata terpotong di tengah.
    """
    # Ambil kandidat overlap (2x lebih besar untuk cari batas kalimat)
    window = prev_chunk[-(overlap * 2):]
    # Cari semua posisi awal kalimat baru dalam window
    matches = list(re.finditer(r'(?<=[.!?])\s+(?=\S)', window))
    if matches:
        # Pilih batas kalimat yang menghasilkan overlap paling dekat dengan target
        for m in reversed(matches):
            candidate = window[m.end():]
            if len(candidate) <= overlap:
                return candidate
    # Fallback: raw tail tapi trim ke batas kata (jangan potong kata)
    tail = prev_chunk[-overlap:]
    space_idx = tail.find(' ')
    return tail[space_idx + 1:] if space_idx != -1 else tail


def merge_units_into_chunks(units: list[str], max_chars: int, overlap: int) -> list[str]:
    """
    Gabungkan unit-unit kecil (kalimat/klausa) secara greedy menjadi chunk
    berukuran max_chars, dengan overlap by sentence boundary.
    """
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = (current + " " + unit).strip() if current else unit

        if len(candidate) <= max_chars:
            current = candidate
        else:
            # Simpan chunk saat ini
            if current:
                chunks.append(current)

            # Kalau unit tunggal melebihi max → hard split
            if len(unit) > max_chars:
                parts = _hard_split(unit, max_chars, overlap)
                chunks.extend(parts[:-1])
                current = parts[-1] if parts else ""
            else:
                # [FIX] Overlap by sentence boundary, bukan raw char tail
                if chunks and overlap > 0:
                    tail = _sentence_boundary_overlap(chunks[-1], overlap)
                    candidate_with_overlap = (tail + " " + unit).strip()
                    current = candidate_with_overlap if len(candidate_with_overlap) <= max_chars else unit
                else:
                    current = unit

    if current:
        chunks.append(current)

    return chunks


def _safe_overlap_start(text: str, pos: int) -> int:
    """
    Cari posisi awal overlap yang aman (tidak memotong kata).
    Mundur dari `pos` mencari spasi/whitespace terdekat.
    """
    window_start = max(0, pos - 30)
    snippet = text[window_start:pos]
    spaces = [m.start() for m in re.finditer(r'\s', snippet)]
    if spaces:
        return window_start + spaces[-1] + 1  # mulai setelah spasi terakhir
    return pos  # fallback: posisi asli


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Fallback split dengan mencari batas tanda baca terdekat.
    Overlap menggunakan word boundary (tidak potong kata).
    """
    result: list[str] = []
    start = 0
    while start < len(text):
        if start + max_chars >= len(text):
            result.append(text[start:])
            break
        # Cari tanda baca terdekat dalam window [max_chars, max_chars+120]
        window = text[start : start + max_chars + 120]
        cut_positions = [m.end() for m in re.finditer(r'[.!?;:]\s+', window)]
        
        # Hanya consider cut yang cukup besar, misal > overlap
        # agar tidak bikin tiny chunks dan terjebak infinite loop
        valid_cuts = [p for p in cut_positions if p > overlap and p <= max_chars + 120]
        if valid_cuts:
            best = max((p for p in valid_cuts if p <= max_chars), default=min(valid_cuts))
            end = start + best
        else:
            end = start + max_chars
            
        result.append(text[start:end].strip())
        
        # [FIX #2] Overlap start menggunakan word boundary (tidak potong kata)
        overlap_pos = end - overlap
        next_start = _safe_overlap_start(text, overlap_pos)
        
        # [CRITICAL FIX] Failsafe: start HARUS selalu maju
        # Jika mundur atau stuck, paksa maju minimal overlap_pos atau +1 char
        if next_start <= start:
            start = max(start + 1, overlap_pos)
        else:
            start = next_start
            
    return [r for r in result if r]


# ═══════════════════════════════════════════════════════════════════════
#  [3] CHUNK CREATION — baca JSON, bersihkan, split, assign metadata
# ═══════════════════════════════════════════════════════════════════════

# ── Filter chunk informatif rendah ──────────────────────────────────
_WEAK_INTRO_PATTERNS = [
    r'^setelah membahas',
    r'^adapun pembahasan',
    r'^kini kita akan membahas',
    r'^secara ringkas',
    r'^sebagaimana telah disebutkan',
    r'^dalam bab ini',
    r'^pada bagian ini',
]

def is_low_information(text: str) -> bool:
    """
    Deteksi chunk yang isinya intro/transisi lemah tanpa nilai informatif tinggi.
    Chunk seperti ini hanya menyumbang noise pada nearest-neighbor retrieval.
    """
    t = text.lower().strip()
    # Terlalu pendek = kurang informatif
    if len(t) < 120:
        return True
    # Pola kalimat transisi generik
    return any(re.search(p, t) for p in _WEAK_INTRO_PATTERNS)


def create_chunks(document: list) -> list[dict]:
    """
    Konversi document_full.json → list of chunks siap embed.

    Pipeline per subbab:
      raw text → clean_text() → split paragraphs → merge into chunks (~1000 chars)
    """
    chunks: list[dict] = []
    toc_order = 0

    for bab in document:
        bab_title = bab.get("bab_title", "UNKNOWN BAB")

        for subbab in bab.get("subbab", []):
            subbab_title = subbab.get("subbab_title", "UNLABELED SECTION")
            content_parts = subbab.get("content", [])
            pages = subbab.get("pages", [])

            # [FIX] Gabung dengan \n agar paragraph break tersimpan untuk split
            raw_text = "\n".join(content_parts).strip()

            # [FIX] Clean TANPA collapse newline dulu — newline dipakai paragraph split
            cleaned = clean_text(raw_text, collapse_newlines=False)

            if len(cleaned) < MIN_CHUNK_CHARS:
                toc_order += 1
                continue

            page_start = min(pages) if pages else 0
            page_end   = max(pages) if pages else 0
            num_pages  = len(pages)

            base_meta = {
                "bab_title"   : bab_title,
                "subbab_title": subbab_title,
                "toc_order"   : toc_order,
                "page_start"  : page_start,
                "page_end"    : page_end,
                "num_pages"   : num_pages,
                # [NEW] section_uid: identifier unik per subbab
                # semua chunk dalam subbab yang sama punya section_uid yang sama
                # berguna untuk contextual sibling retrieval di agentic_rag
                "section_uid" : f"sec_{toc_order:04d}",
            }

            # Paragraph-aware split + collapse newline setelah split
            if len(cleaned) <= MAX_CHUNK_CHARS:
                # Untuk chunk pendek: collapse newline langsung
                final_text = clean_text(cleaned, collapse_newlines=True)
                if len(final_text) >= MIN_CHUNK_CHARS and not is_low_information(final_text):
                    chunk_id = f"toc_{toc_order:04d}_0"
                    chunks.append({
                        "id"      : chunk_id,
                        "text"    : final_text,
                        "metadata": {**base_meta, "chunk_index": 0},
                    })
            else:
                units      = split_into_paragraphs(cleaned)
                sub_chunks = merge_units_into_chunks(units, MAX_CHUNK_CHARS, OVERLAP_CHARS)

                for i, sub_text in enumerate(sub_chunks):
                    # [BONUS] Skip chunk informatif rendah
                    if len(sub_text) < MIN_CHUNK_CHARS or is_low_information(sub_text):
                        continue
                    chunk_id = f"toc_{toc_order:04d}_{i}"
                    chunks.append({
                        "id"      : chunk_id,
                        "text"    : sub_text,
                        "metadata": {**base_meta, "chunk_index": i},
                    })

            toc_order += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════
#  [4] BUILD QDRANT — embed teks BERSIH lalu insert
# ═══════════════════════════════════════════════════════════════════════

def build_qdrant(chunks: list[dict], qdrant_path: str):
    """Embed chunks (yang sudah bersih) dan masukkan ke Qdrant."""
    try:
        from sentence_transformers import SentenceTransformer  # pyre-ignore[21]
        from qdrant_client import QdrantClient                  # pyre-ignore[21]
        from qdrant_client.models import Distance, VectorParams, PointStruct  # pyre-ignore[21]
    except ImportError:
        print("ERROR: Install dulu:\n  pip install qdrant-client sentence-transformers")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Building Qdrant (Research-Grade v2)")
    print(f"  Model     : {EMBEDDING_MODEL}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Output    : {qdrant_path}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Chunk size: {MAX_CHUNK_CHARS} chars max")
    print(f"{'='*60}\n")

    # Load model
    print("[1/3] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Init Qdrant
    os.makedirs(qdrant_path, exist_ok=True)
    client = QdrantClient(path=qdrant_path)

    # Recreate collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  ↳ Deleted existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  ↳ Created collection '{COLLECTION_NAME}' (cosine, dim={VECTOR_SIZE})")

    # Embed & insert
    BATCH_SIZE = 32
    print(f"\n[2/3] Embedding & inserting (batch={BATCH_SIZE})...")

    point_id = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        # [FIX #3] Prefix E5 hanya Bab + Subbab — NO page range
        # Page range di payload sudah cukup; di embedding justru dilute semantic content
        texts = []
        for c in batch:
            meta   = c["metadata"]
            prefix = f"Bab {meta['bab_title']}. Subbab {meta['subbab_title']}. "
            texts.append(f"passage: {prefix}{c['text']}")

        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        points = []
        for j, c in enumerate(batch):
            meta = c["metadata"]
            # [IMPROVEMENT 2] keyword_text untuk hybrid retrieval BM25 di masa depan
            keyword_text = (
                f"{meta['bab_title']} {meta['subbab_title']} {c['text']}"
            )
            payload = {
                **c["metadata"],
                "chunk_id"    : c["id"],
                "text"        : c["text"],
                "keyword_text": keyword_text,   # untuk BM25 hybrid retrieval
            }
            points.append(PointStruct(
                id     = point_id + j,
                vector = embeddings[j],
                payload= payload,
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        point_id += len(batch)

        done = min(i + BATCH_SIZE, len(chunks))
        print(f"  ↳ {done}/{len(chunks)} ({done * 100 // len(chunks)}%)")

    # Verify
    print(f"\n[3/3] Verifikasi...")
    count = client.get_collection(COLLECTION_NAME).points_count
    print(f"  ↳ Total points: {count}")

    # Quick test
    test_query = "Perang Badr"
    q_vec = model.encode([f"query: {test_query}"], normalize_embeddings=True).tolist()[0]
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=3,
    ).points

    print(f"\n{'='*60}")
    print(f"  Test Query: '{test_query}'")
    print(f"{'='*60}")
    for j, hit in enumerate(results):
        p = hit.payload
        print(f"\n  [{j+1}] Score : {hit.score:.4f}")
        print(f"       Bab   : {p.get('bab_title', '?')}")
        print(f"       Subbab: {p.get('subbab_title', '?')}")
        print(f"       Pages : {p.get('page_start', '?')}-{p.get('page_end', '?')}")
        print(f"       Chars : {len(p.get('text', ''))}")

    print(f"\n✅ Qdrant berhasil dibangun: {qdrant_path}")
    print(f"   Total chunks: {count}")
    return client


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def save_chunks_json(chunks: list[dict], output_path: str):
    """Simpan chunks ke JSON untuk inspeksi."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"📄 Chunks saved: {output_path}")


def print_chunk_stats(chunks: list[dict]):
    """Print statistik distribusi ukuran chunk."""
    lengths = [len(c["text"]) for c in chunks]
    buckets = {"<300": 0, "300-600": 0, "600-800": 0, "800-1000": 0, ">1000": 0}
    for l in lengths:
        if l < 300:       buckets["<300"] += 1
        elif l < 600:     buckets["300-600"] += 1
        elif l < 800:     buckets["600-800"] += 1
        elif l <= 1000:   buckets["800-1000"] += 1
        else:             buckets[">1000"] += 1

    print(f"\n  📊 Distribusi ukuran chunk:")
    for k, v in buckets.items():
        bar = "█" * (v * 30 // max(buckets.values()))
        print(f"     {k:>10}  {bar} {v}")
    print(f"\n  Avg: {sum(lengths)//len(lengths)} chars")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

# ── BENCHMARK QUERIES untuk sanity check retrieval ──────────────────
# 10 query yang jawabannya jelas ada di bab tertentu
BENCHMARK_QUERIES = [
    ("siapa paman Nabi Muhammad yang ikut hijrah",    "NASAB"),
    ("berapa jumlah tawanan Perang Badr",              "PERANG BADR KUBRA"),
    ("kapan Nabi menerima wahyu pertama di Gua Hira", "DI BAWAH LINDUNGAN NUBUWAH"),
    ("apa isi Perjanjian Hudaibiyah",                  "PERJANJIAN HUDAIBIYAH"),
    ("siapa yang menyebarkan berhala ke Makkah",      "AGAMA BANGSA ARAB"),
    ("berapa lama Nabi di Gua Tsur sebelum hijrah",   "RASULULLAH HIJRAH"),
    ("kapan Khadijah wafat",                           "TAHUN BERDUKA"),
    ("siapa komandan pasukan Islam di Perang Mu'tah", "PERANG MU'TAH"),
    ("apa yang dilakukan Nabi saat Penaklukan Makkah","PERANG DAN PENAKLUKAN MAKKAH"),
    ("berapa jumlah pasukan Ahzab",                    "PERANG AHZAB"),
]


def run_benchmark(client, model) -> None:
    """Jalankan benchmark 10 query — tampilkan Precision@1 dan Precision@3."""
    print(f"\n{'='*60}")
    print(f"  🧪 BENCHMARK RETRIEVAL — Precision@1 & Precision@3")
    print(f"{'='*60}")
    print(f"  {'Query':<45} {'P@1':^4} {'P@3':^4} {'Top1 Score':^10} {'Top1 Bab':<30}")
    print(f"  {'-'*45} {'-'*4} {'-'*4} {'-'*10} {'-'*30}")

    hit1 = 0
    hit3 = 0
    for query, expected_bab_keyword in BENCHMARK_QUERIES:
        q_vec = model.encode([f"query: {query}"], normalize_embeddings=True).tolist()[0]
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=3,
        ).points

        top_babs = [r.payload.get("bab_title", "").upper() for r in results]
        top1_bab = top_babs[0] if top_babs else ""
        top1_score = results[0].score if results else 0.0
        kw = expected_bab_keyword.upper()

        p1 = kw in top1_bab
        p3 = any(kw in b for b in top_babs)

        if p1: hit1 += 1
        if p3: hit3 += 1

        s1 = "✅" if p1 else "❌"
        s3 = "✅" if p3 else "❌"
        print(f"  {query[:45]:<45} {s1:^4} {s3:^4} {top1_score:^10.4f} {top1_bab[:30]:<30}")

    n = len(BENCHMARK_QUERIES)
    print(f"\n  {'─'*60}")
    print(f"  Precision@1 : {hit1}/{n} ({hit1*100//n}%)  ← top-1 chunk benar")
    print(f"  Precision@3 : {hit3}/{n} ({hit3*100//n}%)  ← salah satu dari top-3 benar")
    print(f"  {'─'*60}")
    if hit1 * 100 // n >= 70:
        print("  ✅ Retrieval layak untuk eksperimen skripsi.")
    else:
        print("  ⚠️  Precision@1 < 70% — pertimbangkan tuning chunk size atau prefix.")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("  build_vectordb.py  — Research-Grade v2")
    print("=" * 60)

    # Step 1: Load
    print(f"\n📖 Loading: {INPUT_JSON}")
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: File tidak ditemukan: {INPUT_JSON}")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        document = json.load(f)
    print(f"   Found {len(document)} bab(s)")

    # Step 2: Chunk
    print(f"\n✂️  Creating research-grade chunks...")
    chunks = create_chunks(document)
    print(f"   Created {len(chunks)} chunks")
    print_chunk_stats(chunks)

    # Step 3: Save JSON
    chunks_json_path = os.path.join(DATA_DIR, "vectordb", "chunks_toc_baseline.json")
    save_chunks_json(chunks, chunks_json_path)

    # Step 4: Build Qdrant
    qdrant_client = build_qdrant(chunks, QDRANT_PATH)

    # Step 5: Benchmark sanity check
    from sentence_transformers import SentenceTransformer  # pyre-ignore[21]
    print("\n🔄 Loading model untuk benchmark...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    run_benchmark(qdrant_client, emb_model)


if __name__ == "__main__":
    main()
