# Agentic RAG: Sirah Nabawiyah 📚

Repositori ini berisi keseluruhan *pipeline* kode untuk Tugas Akhir (TA) mengenai implementasi **Agentic Retrieval-Augmented Generation (RAG)** pada domain sejarah Islam, khususnya mengacu pada buku Sirah Nabawiyah (Syaikh Shafiyyurrahman Al-Mubarakfuri).

Proyek ini mencakup mulai dari pra-pemrosesan dokumen (OCR, *chunking*), pembuatan dataset otomatis (*golden dataset*), *fine-tuning* model *embedding* dan *LLM-as-a-Judge*, hingga implementasi dan evaluasi komprehensif sistem RAG (Konvensional vs Agentik).

---

## 📂 Struktur Direktori

Berikut adalah penjelasan fungsi masing-masing direktori dalam repositori ini:

### 1. `data/` (Data & Database)
Direktori ini menyimpan seluruh data mentah, data hasil pemrosesan, dan *vector database* (beberapa file berukuran besar harus di-*download* secara terpisah atau tidak di-push ke GitHub).
- **`finetune_dataset/`**: Berisi dataset *Q&A* (Tanya Jawab) yang telah diproses untuk tahap pelatihan (*fine-tuning*) dan pengujian. File penting di sini meliputi `train_ok.jsonl`, `test_ok.jsonl` (sebagai *ground truth* utama), serta dataset hasil pelabelan untuk model *evaluator*.
- **`vectordb/`**: Menyimpan potongan dokumen (*chunks*) dalam bentuk `.json` dan *database* vektor lokal (menggunakan Qdrant). Terdapat versi *baseline* (model *embedding* asli) dan versi *finetuned* (model yang sudah dilatih).
- **`evaluation/`**: (Biasanya digenerate otomatis) Menyimpan hasil *log* evaluasi, transkrip percakapan, dan skor metrik dari setiap skenario uji coba.

### 2. `src/` (Source Code)
Direktori ini berisi seluruh skrip Python yang menjalankan *pipeline* penelitian. Dibagi menjadi beberapa sub-modul:

- **`build_knowledge/`**
  Tahap awal mempersiapkan *knowledge base* dari buku PDF hingga masuk ke *database*.
  - `ocr_paddle.py`: Mengubah halaman PDF menjadi teks (menggunakan PaddleOCR).
  - `build_vectordb.py`: Mengubah teks menjadi *chunk* dan menyimpannya ke Qdrant (*baseline*).
  - `reembed_vectordb.py`: Meng-update *vector database* menggunakan model *embedding* yang telah di-*finetune*.
  - `generate_goldendataset_new.py`: Membuat dataset Q&A secara otomatis dari *chunk* (*synthetic data generation*).
  - `labelling_evaluator_dataset.py`: Memberikan label otomatis (misalnya skor relevansi) pada dataset untuk pelatihan *Sufficiency Evaluator*.

- **`finetuning/`**
  Modul untuk melatih (*fine-tuning*) komponen kecerdasan buatan.
  - `finetune_embedding.py`: *Script* untuk melatih model *embedding* dasar agar lebih memahami konteks spesifik domain (Sirah Nabawiyah).
  - `finetune_evaluator.py`: *Script* (mendukung LoRA/QLoRA dengan Unsloth) untuk melatih *Base LLM* (seperti Llama-3/Qwen/Gemma) menjadi model *Sufficiency Evaluator* (Agen Penilai Kecukupan Konteks).

- **`rag/`**
  Modul inti implementasi sistem *Retrieval-Augmented Generation*.
  - `config.py`: File konfigurasi sentral (path model, path database, dll).
  - `conventional_rag.py`: Implementasi *pipeline* RAG konvensional (statik, 1 tahap pencarian).
  - `real_agentic.py`: Implementasi *pipeline* Agentic RAG dengan fitur evaluasi mandiri (*self-reflection*) dan interaksi jamak.
  - `utils.py`: Fungsi-fungsi utilitas pendukung RAG.

- **`evaluation/`**
  Modul untuk mengevaluasi performa sistem dari berbagai sudut pandang.
  - `eval_1_retrieval.py`: Mengevaluasi performa pencarian (*retrieval metrics* seperti MRR, NDCG).
  - `eval_generator.py`: Mengevaluasi kualitas jawaban dari model (*generation metrics*).
  - `eval_multiturn.py`: Mengevaluasi kemampuan sistem mempertahankan konteks pada percakapan berkelanjutan (*multi-turn*).
  - `eval_6_konsistensi.py`: Uji konsistensi jawaban sistem ketika diberikan prompt berulang.

- **`notebook/`**
  Kumpulan *Jupyter Notebook* untuk Exploratory Data Analysis (EDA).
  - `EDA_Dokumen_Sirah.ipynb`: Analisis karakteristik buku sumber (panjang bab, statistik kata).
  - `eda_dataset.ipynb`: Analisis statistik distribusi dari dataset Q&A yang dihasilkan.
  - `plot_*.py`: Skrip-skrip untuk memvisualisasikan data (grafik evaluasi, dll).

- **`utils/`**
  Skrip utilitas umum seperti format ulang dataset (`prepare_qasina.py`), dsb.

---

## 🚀 Cara Menjalankan

*(Anda dapat menambahkan instruksi spesifik di sini, misalnya cara setup environment virtual (venv), install requirements, dan urutan cara menjalankan skrip dari build knowledge hingga evaluasi.)*
