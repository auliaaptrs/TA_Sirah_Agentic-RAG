import json
import os
import re

# =========================
# PATH CONFIGURATION
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
INPUT_JSON = os.path.join(DATA_DIR, "preprocess_result", "toc_result", "document_full.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "preprocess_result", "toc_result", "document_full_cleaned.json")

def clean_text(text: str) -> str:
    if not text:
        return text
    
    # 1. Normalisasi spasi putih (termasuk \n, \t berlebih) menjadi spasi tunggal
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Fix Hyphenation Error akibat OCR
    # Jika ada huruf diikuti tanda strip '-' dan spasi, ubah jadi strip saja rata kiri-kanan
    # Mencegah: "kata- kata" menjadi "kata-kata"
    # Mencegah: "Muham- mad" menjadi "Muham-mad" (masih jauh lebih baik untuk pencarian dbi banding terpisah spasi)
    text = re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)
    
    # 3. Hapus Karakter Siluman (Retain AlphaNumeric, Spaces, & Standard Punctuation)
    # Ini melindungi kita dari 'charmap codec can't decode' error
    text = re.sub(r'[^\w\s.,!?:;()[\]{}"\'\-—@/]', '', text)
    
    # 4. Merapikan kekacauan spasi di sekitar tanda baca (Khas OCR)
    text = re.sub(r'\s+([.,!?:;])', r'\1', text) # "kata , kata" -> "kata, kata"
    text = re.sub(r'\(\s+', '(', text)          # "( SAW" -> "(SAW"
    text = re.sub(r'\s+\)', ')', text)          # "SAW )" -> "SAW)"
    text = re.sub(r'\s+-\s+', '-', text)        # "kata - kata" -> "kata-kata"
    
    return text.strip()

def main():
    print(f"📖 Membaca file mentah: {INPUT_JSON}")
    if not os.path.exists(INPUT_JSON):
        print(f"❌ ERROR: File tidak ditemukan! Pastikan toc_fulldocument.py sudah sukses dijalankan.")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        document = json.load(f)

    print("🧹 Memulai Preprocessing Text...")
    cleaned_count = 0
    
    for bab in document:
        for subbab in bab.get("subbab", []):
            cleaned_content = []
            for paragraph in subbab.get("content", []):
                cleaned_par = clean_text(paragraph)
                if cleaned_par: # Lewati jika ternyata jadi kosong setelah dibersihkan
                    cleaned_content.append(cleaned_par)
                    cleaned_count += 1
            
            # Update isi content dengan list versi bersih
            subbab["content"] = cleaned_content

    print(f"💾 Menyimpan {cleaned_count} paragraf bersih ke: {OUTPUT_JSON}")
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    print("✅ Preprocessing selesai! Silakan lanjut ke build_vectordb.py")

if __name__ == "__main__":
    main()
