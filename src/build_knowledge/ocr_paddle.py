import os
import cv2
import json
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from tqdm import tqdm

# =========================
# PATH CONFIG (relative to project root)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

PDF_PATH = os.path.join(DATA_DIR, "Sirah Nabawiyah by Shafiyyur Rahman Mubarakfury (syaikh.), Kathur Suhardi.pdf")
PAGE_DIR = os.path.join(DATA_DIR, "pages")
OCR_TXT  = os.path.join(DATA_DIR, "preprocess_result", "ocr_txt")

os.makedirs(PAGE_DIR, exist_ok=True)
os.makedirs(OCR_TXT, exist_ok=True)

# =========================
# PDF → IMAGE
# =========================
print("[INFO] Converting PDF to images...")
pages = convert_from_path(
    PDF_PATH,
    dpi=300,
    use_cropbox=True,
    strict=False
)

# =========================
# PREPROCESS PAGE
# =========================
def preprocess_page(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1500:
        scale = 1500 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray

# =========================
# INIT OCR
# =========================
ocr = PaddleOCR(
    lang="id",
    use_gpu=False,
    use_angle_cls=False,
    show_log=False
)

# =========================
# OCR LOOP
# =========================
print("[INFO] Running OCR...")
for idx, pil_img in tqdm(enumerate(pages, start=1), total=len(pages)):
    page_id = f"page_{idx:03}"

    img_path = os.path.join(PAGE_DIR, f"{page_id}.png")
    txt_path = os.path.join(OCR_TXT, f"{page_id}.txt")

    if os.path.exists(txt_path):
        continue

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    processed = preprocess_page(img_bgr)
    cv2.imwrite(img_path, processed)

    result = ocr.ocr(img_path)

    lines = []

    if result and result[0]:
        for line in result[0]:
            text = line[1][0].strip()
            if text:
                lines.append(text)

    # SAVE TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("[DONE] OCR finished (TXT only).")