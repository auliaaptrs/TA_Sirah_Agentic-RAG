import streamlit as st
import os
import sys
import time

# -- Setup Path --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# -- Override Config for Baseline Embedding --
# Menggunakan Baseline sesuai permintaan (Skenario 1)
import src.rag.config as rag_config
rag_config.QDRANT_PATH = os.path.join(SCRIPT_DIR, "data", "vectordb", "qdrant_toc_baseline_BACKUP_CHUNK1000")
rag_config.EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

import src.rag.utils as rag_utils
rag_utils.QDRANT_PATH = rag_config.QDRANT_PATH
rag_utils.EMBEDDING_MODEL = rag_config.EMBEDDING_MODEL

# [NEW] Menggunakan GPU Kampus via Cloudflare Tunnel (Dinamis tanpa perlu commit)
rag_config.LLM_PROVIDER = "ollama"
# Mengambil link dari Streamlit Secrets (jika tidak ada, fallback ke localhost)
rag_config.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

from src.rag.utils import init_vectordb, init_llm
from src.rag.real_agentic import real_agentic_rag_query

def render_styled_answer(text: str, question: str = ""):
    import re
    
    # Cek apakah seluruh jawaban murni penolakan (fallback total)
    is_total_fallback = "tidak ditemukan dalam potongan kitab" in text.lower() and len(text) < 150
    
    # Deteksi apakah pertanyaan meminta kronologi/urutan
    is_timeline = bool(re.search(r'(kronologis|urutan|rangkaian|timeline)', question, re.IGNORECASE))
    
    if is_timeline:
        # Pre-processing: Jika LLM memakai "Pertama,", "Kedua,", dst, ubah paksa menjadi format angka
        text = re.sub(r'(?i)\bPertama,\s*', '\n1. ', text)
        text = re.sub(r'(?i)\bKedua,\s*', '\n2. ', text)
        text = re.sub(r'(?i)\bKetiga,\s*', '\n3. ', text)
        text = re.sub(r'(?i)\bKeempat,\s*', '\n4. ', text)
        text = re.sub(r'(?i)\bKelima,\s*', '\n5. ', text)
    
    lines = text.split("\n")
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        lower_line = line_strip.lower()
        
        # Filter kalimat penutup basa-basi dari LLM
        if "semoga" in lower_line and ("bermanfaat" in lower_line or "membantu" in lower_line or "jawaban" in lower_line):
            continue
        if lower_line.startswith("salam") or "ahli sejarah" in lower_line or "asisten ai" in lower_line:
            continue
            
        # Filter kalimat disclaimer tambahan (jika agen sebenarnya sudah menjawab inti)
        if not is_total_fallback:
            if "informasi" in lower_line and "tidak ditemukan" in lower_line and ("konteks" in lower_line or "kitab" in lower_line):
                continue
            
        # Cocokkan pola angka di awal HANYA jika pertanyaan meminta kronologi
        match = re.match(r"^(\d+)\.\s*(.*)", line_strip)
        if match and is_timeline:
            num = match.group(1)
            content = match.group(2)
            
            # Ubah markdown bold **text** menjadi HTML <b>text</b>
            content_cleaned = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", content)
            
            card_html = f"""
<div style="
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.12);
    border-left: 5px solid #d4af37;
    padding: 16px;
    margin: 10px 0;
    border-radius: 8px;
">
    <div style="font-weight: bold; color: #d4af37; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
        📜 KRONOLOGI {num}
    </div>
    <div style="font-size: 1.0em; line-height: 1.6; color: inherit;">
        {content_cleaned}
    </div>
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.markdown(line)


st.set_page_config(page_title="Sirah Nabawiyah AI", page_icon="🕌", layout="wide")

st.title("🕌 Agentic RAG: Sirah Nabawiyah")
st.markdown("Sistem Tanya Jawab Interaktif berbasis kitab *Ar-Raheeq Al-Makhtum* menggunakan arsitektur **Agentic RAG**.")

# -- Initialize Models (Cached) --
@st.cache_resource
def load_models():
    with st.spinner("Memuat Vector DB dan Model Language... Ini mungkin memakan waktu sebentar."):
        vdb = init_vectordb()
        llm = init_llm()
        return vdb, llm

vdb, llm = load_models()

# -- Initialize Session State --
if "messages" not in st.session_state:
    st.session_state.messages = []

# -- Sidebar: History & Debug Logs --
with st.sidebar:
    st.header("📊 Analisis Agentic RAG")
    if st.session_state.messages:
        # Mencari respon asisten terakhir
        last_meta = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "metadata" in msg:
                last_meta = msg["metadata"]
                break
                
        if last_meta:
            st.subheader("⏱️ Statistik Pemrosesan")
            st.write(f"**Iterasi Loop:** {last_meta['iterations']}")
            st.write(f"**LLM Calls:** {last_meta['llm_calls']}")
            st.write(f"**Waktu Eksekusi:** {last_meta['elapsed']:.1f} detik")
            st.write(f"**Risiko Halusinasi:** {last_meta['hallucination_risk']}")
            
            st.divider()
            st.subheader("🔍 Kueri Pencarian Agen (IRCoT)")
            for q in last_meta['search_history']:
                st.caption(f"👉 {q}")
            
            st.divider()
            st.subheader("📋 Log Agentic Loop")
            with st.expander("Lihat Proses Berpikir Agen"):
                for log in last_meta['logs']:
                    st.text(log)
                    
            st.divider()
            st.subheader("📚 Konteks dari Kitab")
            with st.expander("Lihat Referensi (Chunks)"):
                for i, chunk in enumerate(last_meta['chunks'], 1):
                    bab = chunk['metadata'].get('bab_title', '?')
                    start = chunk['metadata'].get('page_start', '?')
                    end = chunk['metadata'].get('page_end', '?')
                    if start == "N/A" or end == "N/A":
                        st.markdown(f"**[{i}] Bab: {bab}**")
                    else:
                        st.markdown(f"**[{i}] Bab: {bab}** (Hal. {start}-{end})")
                    st.info(chunk['text'])
    else:
        st.info("Kirim pertanyaan untuk melihat proses analisis agen di sini.")

# -- Chat Interface --
last_user_msg = ""
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            render_styled_answer(msg["content"], last_user_msg)

prompt = st.chat_input("Tanyakan sesuatu tentang Sirah Nabawiyah (misal: Apa penyebab Perang Badar?)...")

if prompt:
    # Tampilkan prompt user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tampilkan jawaban assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Agen sedang berpikir, mengevaluasi kecukupan info, dan mencari referensi..."):
            try:
                # Ekstrak riwayat obrolan untuk kemampuan Multi-Turn
                chat_history = []
                msgs = st.session_state.messages
                for i in range(len(msgs) - 1):
                    if msgs[i]["role"] == "user" and i+1 < len(msgs) and msgs[i+1]["role"] == "assistant":
                        chat_history.append({
                            "question": msgs[i]["content"],
                            "answer": msgs[i+1]["content"]
                        })

                # Panggil Real Agentic RAG
                result = real_agentic_rag_query(prompt, vdb, llm, chat_history=chat_history)
                
                # Tampilkan jawaban
                answer = result["answer"]
                message_placeholder.empty()
                render_styled_answer(answer, prompt)
                
                # Simpan ke session state beserta metadata untuk sidebar
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": result
                })
                # Refresh UI untuk memperbarui sidebar
                st.rerun()
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
