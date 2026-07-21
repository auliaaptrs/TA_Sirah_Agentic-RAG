import os
import json
import re
import unicodedata
from difflib import SequenceMatcher
from tqdm import tqdm

# =========================
# CONFIG (relative to project root)
# =========================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "data")

OCR_RAW_DIR = os.path.join(DATA_DIR, "preprocess_result", "ocr_txt")
OUT_DIR     = os.path.join(DATA_DIR, "preprocess_result", "toc_result")
TOC_PATH    = os.path.join(DATA_DIR, "toc_groundtruth.json")
OUT_NAME    = "document_full.json"

PAGE_OFFSET = 0
ENABLE_FUZZY = True
FUZZY_THRESHOLD = 0.92
FOOTER_LAST_K = 2

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# TEXT NORMALIZATION & SIMILARITY
# =========================
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def looks_like_title(raw: str) -> bool:
    r = raw.strip()
    if not r:
        return False
    if r.upper().startswith("BAB ") or r.upper().startswith("BAGIAN "):
        return True
    letters = [c for c in r if c.isalpha()]
    if len(letters) >= 5:
        upper = sum(1 for c in letters if c.isupper())
        if upper / max(len(letters), 1) >= 0.65:
            return True
    if 3 <= len(r) <= 80 and len(r.split()) <= 12:
        return True
    return False

def merge_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    out = lines[0].strip()
    for nxt in lines[1:]:
        nxt = nxt.strip()
        if not nxt:
            continue
        if out.endswith("-"):
            out = out[:-1] + nxt
        else:
            out = out + " " + nxt
    return out

# =========================
# FOOTER STRIPPER
# =========================
def strip_footer_lines(ocr_lines: list[str], last_k: int = 2) -> list[str]:
    if not ocr_lines:
        return ocr_lines
    head = ocr_lines[:-last_k] if len(ocr_lines) > last_k else []
    tail = ocr_lines[-last_k:] if len(ocr_lines) >= last_k else ocr_lines
    def is_digits_only(s: str) -> bool:
        n = norm(s)
        return re.fullmatch(r"\d{1,5}", n) is not None
    cleaned_tail = []
    for ln in tail:
        n = norm(ln)
        if "sirah nabawiyah" in n:
            continue
        if "halaman" in n:
            continue
        if is_digits_only(ln):
            continue
        cleaned_tail.append(ln)
    return head + cleaned_tail

# =========================
# TOC LOADING
# =========================
def load_toc_index(toc_path: str):
    with open(toc_path, "r", encoding="utf-8") as f:
        toc_data = json.load(f)
    toc_index = {}
    for bab in toc_data:
        page_start = int(bab["page_start"])
        title = (bab.get("title") or "").strip()
        level = (bab.get("level") or "BAB").strip().upper()
        bab_type = "BAB" if level in ("BAB", "BAGIAN") else "BAB"
        if title:
            toc_index.setdefault(page_start, {})[norm(title)] = (bab_type, title)
        for sub in bab.get("subbab", []):
            sub_page = int(sub.get("page_start", page_start))
            sub_title = (sub.get("title") or "").strip()
            if sub_title:
                toc_index.setdefault(sub_page, {})[norm(sub_title)] = ("SUBBAB", sub_title)
    return toc_index

def match_toc(page_number: int, candidate_raw: str, toc_index: dict):
    page_map = toc_index.get(page_number, {})
    if not page_map:
        return None
    c = norm(candidate_raw)
    if c in page_map:
        typ, original = page_map[c]
        return {"type": typ, "title": original, "score": 1.0, "mode": "exact"}
    if not ENABLE_FUZZY or not looks_like_title(candidate_raw):
        return None
    best = None
    for toc_norm_title, (typ, original) in page_map.items():
        score = sim(c, toc_norm_title)
        if best is None or score > best["score"]:
            best = {"type": typ, "title": original, "score": score, "mode": "fuzzy"}
    if best and best["score"] >= FUZZY_THRESHOLD:
        return best
    return None

# =========================
# OCR READING
# =========================
def parse_page_number(fname: str) -> int:
    base = os.path.splitext(fname)[0]
    return int(base.split("_")[-1])

def read_txt_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [ln.strip() for ln in f if ln.strip()]

# =========================
# MERGE CONTENT PER SUBBAB
# =========================
def merge_subbab_content(document: list):
    for bab in document:
        for sub in bab.get("subbab", []):
            if isinstance(sub.get("content"), list):
                merged = " ".join([ln.strip() for ln in sub["content"] if ln.strip()])
                sub["content"] = [merged] if merged else []
    return document

# =========================
# MAIN
# =========================
toc_index = load_toc_index(TOC_PATH)
all_pages = sorted([f for f in os.listdir(OCR_RAW_DIR) if f.lower().endswith(".txt")])

document = []
current_bab = None
current_subbab = None

for fname in tqdm(all_pages, desc="Processing pages"):
    try:
        page_number = parse_page_number(fname)
    except Exception:
        continue

    page_number_key = page_number + PAGE_OFFSET
    page_path = os.path.join(OCR_RAW_DIR, fname)
    ocr_lines = read_txt_lines(page_path)
    if not ocr_lines:
        continue

    ocr_lines = strip_footer_lines(ocr_lines, last_k=FOOTER_LAST_K)

    i = 0
    while i < len(ocr_lines):
        line = ocr_lines[i]
        cand1 = line
        cand2 = merge_lines([ocr_lines[i], ocr_lines[i + 1]]) if i + 1 < len(ocr_lines) else None
        cand3 = merge_lines([ocr_lines[i], ocr_lines[i + 1], ocr_lines[i + 2]]) if i + 2 < len(ocr_lines) else None
        m1 = match_toc(page_number_key, cand1, toc_index)
        m2 = match_toc(page_number_key, cand2, toc_index) if cand2 else None
        m3 = match_toc(page_number_key, cand3, toc_index) if cand3 else None

        best, take_n = None, 1
        candidates = []
        if m1: candidates.append((m1, 1))
        if m2: candidates.append((m2, 2))
        if m3: candidates.append((m3, 3))
        if candidates:
            candidates.sort(key=lambda x: (x[0]["score"], -x[1]), reverse=True)
            best, take_n = candidates[0]

        if best:
            typ = best["type"]
            title = best["title"]
            if typ == "BAB":
                current_bab = {"bab_title": title, "subbab": []}
                document.append(current_bab)
                current_subbab = None
                i += take_n
                continue
            if typ == "SUBBAB":
                if current_bab is None:
                    current_bab = {"bab_title": "UNKNOWN BAB", "subbab": []}
                    document.append(current_bab)
                current_subbab = {"subbab_title": title, "content": [], "pages": [page_number]}
                current_bab["subbab"].append(current_subbab)
                i += take_n
                continue

        # PARAGRAF biasa
        if current_bab is None:
            current_bab = {"bab_title": "UNKNOWN BAB", "subbab": []}
            document.append(current_bab)
        if current_subbab is None:
            current_subbab = {"subbab_title": "UNLABELED SECTION", "content": [], "pages": [page_number]}
            current_bab["subbab"].append(current_subbab)

        current_subbab["content"].append(line)
        if page_number not in current_subbab["pages"]:
            current_subbab["pages"].append(page_number)

        i += 1

# =========================
# MERGE CONTENT PER SUBBAB
# =========================
document = merge_subbab_content(document)

# =========================
# SAVE JSON
# =========================
out_path = os.path.join(OUT_DIR, OUT_NAME)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(document, f, ensure_ascii=False, indent=2)

print(f"\n[DONE] Segmentation finished. Output saved: {out_path}")
