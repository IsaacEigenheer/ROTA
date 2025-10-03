import os
import re
import csv
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ====== Optional OCR engines ======
try:
    from paddleocr import PaddleOCR  # type: ignore
    _HAS_PADDLE = True
except Exception:
    _HAS_PADDLE = False
    PaddleOCR = None  # type: ignore

try:
    import easyocr  # type: ignore
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False
    easyocr = None  # type: ignore

# ====== Globals (instances cached across calls) ======
_PADDLE_INSTANCE: Any = None
_EASYOCR_INSTANCE: Any = None

# ====== Regex for "C<num> - value" (same as before) ======
PATTERN_C_VAL = re.compile(
r'\bC\s*(\d+)\s*(?:[-–—:])\s*([0-9]+(?:[.,][0-9]+)?)\b',
    flags=re.IGNORECASE
)

# ---------- Color filter: black→white, green→black ----------
def apply_green_black_filter(bgr: np.ndarray) -> np.ndarray:
    assert bgr.ndim == 3 and bgr.shape[2] == 3, "Esperado BGR"
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    mask_black = V <= 40
    mask_green = (H >= 35) & (H <= 95) & (S >= 40) & (V >= 40)
    out = np.full_like(bgr, 255, dtype=np.uint8)
    out[mask_green] = (0, 0, 0)
    out[mask_black] = (255, 255, 255)
    return out

# ---------- Preprocessing ----------
def _unsharp_mask(x: np.ndarray, sigma: float = 0.8, amount: float = 0.6) -> np.ndarray:
    blur = cv2.GaussianBlur(x, (0, 0), sigma)
    return cv2.addWeighted(x, 1 + amount, blur, -amount, 0)

def _clahe(gray: np.ndarray, clip: float = 2.0, tiles: Tuple[int, int] = (8, 8)) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    return c.apply(gray)

def preprocess_image(img: np.ndarray, method: str = "thresh") -> np.ndarray:
    """
    Retorna imagem binária uint8 com texto preferencialmente preto (0) em branco (255).
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Esperado BGR"
    if method == "none":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "thresh":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = 255 - th
        return inv if np.mean(inv) > np.mean(th) else th

    if method == "adaptive":
        eq = _clahe(gray, clip=2.0, tiles=(8, 8))
        sharp = _unsharp_mask(eq, sigma=0.8, amount=0.6)
        th = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        th = cv2.morphologyEx(
            th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1
        )
        inv = 255 - th
        return inv if np.mean(inv) > np.mean(th) else th

    # fallback (igual thresh)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th
    return inv if np.mean(inv) > np.mean(th) else th

# ---------- Normalização de linhas OCR ----------
def normalize_ocr_line(s: str) -> str:
    s = s.replace('\u00ad', '').replace('°', '').replace('?', '')

    def _fix_after_C(m):
        tail = m.group(1)
        tail = re.sub(r'^\s*[Aa]', '4', tail)
        tail = re.sub(r'^\s*[Ss]', '5', tail)
        tail = re.sub(r'^\s*[Oo]', '0', tail)
        tail = re.sub(r'^\s*[Il]', '1', tail)
        return 'C' + tail
    s = re.sub(r'C([^\W_]{0,2})', _fix_after_C, s)

    # Ex.: "C1/7-120" -> "C17-120"
    s = re.sub(r'C(\d)\s*/\s*(\d)', r'C\1\2', s)

    def _fix_num(tok: str) -> str:
        t = list(tok)
        for i, ch in enumerate(t):
            if ch in 'Oo': t[i] = '0'
            elif ch in 'Il': t[i] = '1'
        return ''.join(t)
    s = re.sub(r'\b[0-9OIl]+(?:[.,][0-9OIl]+)?\b', lambda m: _fix_num(m.group(0)), s)

    s = re.sub(r'\s*([\-–—:])\s*', r' \1 ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

# ---------- Engine selection & instances ----------
def choose_engine(ocr_engine: str) -> str:
    if ocr_engine == "tesseract":
        return "tesseract"
    if ocr_engine == "paddle":
        return "paddle" if _HAS_PADDLE else ("easyocr" if _HAS_EASYOCR else "tesseract")
    if ocr_engine == "easyocr":
        return "easyocr" if _HAS_EASYOCR else ("paddle" if _HAS_PADDLE else "tesseract")
    # auto
    if _HAS_PADDLE: return "paddle"
    if _HAS_EASYOCR: return "easyocr"
    return "tesseract"

def get_paddle(project_root: str) -> Any:
    global _PADDLE_INSTANCE
    if _PADDLE_INSTANCE is None and _HAS_PADDLE:
        det_path = os.path.join(project_root, "ch_ppocr_server_v2.0_det_infer")
        rec_path = os.path.join(project_root, "ch_ppocr_server_v2.0_rec_infer")
        kwargs = dict(use_angle_cls=True, lang='en', det=True, rec=True, show_log=False)
        if os.path.isdir(det_path): kwargs["det_model_dir"] = det_path
        if os.path.isdir(rec_path): kwargs["rec_model_dir"] = rec_path
        _PADDLE_INSTANCE = PaddleOCR(**kwargs)
    return _PADDLE_INSTANCE

def get_easyocr() -> Any:
    global _EASYOCR_INSTANCE
    if _EASYOCR_INSTANCE is None and _HAS_EASYOCR:
        _EASYOCR_INSTANCE = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_INSTANCE

# ---------- OCR backends ----------
def ocr_tesseract(bin_img: np.ndarray, psm: int) -> Dict[str, Any]:
    proc_up = cv2.resize(bin_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'
    raw_text = pytesseract.image_to_string(proc_up, config=config)
    data = pytesseract.image_to_data(proc_up, output_type=Output.DICT, config=config)
    n = len(data.get('text', []))
    byline: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        txt = (data['text'][i] or "").strip()
        if not txt: continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        byline.setdefault(key, []).append(i)
    lines = []
    for key, idxs in byline.items():
        idxs_sorted = sorted(idxs, key=lambda k: data['left'][k])
        words = [(data['text'][i] or "").strip() for i in idxs_sorted if (data['text'][i] or "").strip()]
        if not words: continue
        line_text = " ".join(words)
        lefts = [int(data['left'][i]) for i in idxs_sorted]
        tops  = [int(data['top'][i]) for i in idxs_sorted]
        rights= [int(data['left'][i]) + int(data['width'][i]) for i in idxs_sorted]
        bottoms=[int(data['top'][i]) + int(data['height'][i]) for i in idxs_sorted]
        confs = []
        for i in idxs_sorted:
            try:
                c = float(data['conf'][i])
                if c >= 0: confs.append(c)
            except Exception:
                pass
        l, t = min(lefts), min(tops)
        r, b = max(rights), max(bottoms)
        mean_conf = float(np.mean(confs)) if confs else -1.0
        lines.append({"text": line_text, "conf": mean_conf, "bbox": [l, t, r - l, b - t]})
    return {"lines": lines, "raw": raw_text}

def ocr_paddle(bgr_img: np.ndarray, project_root: str) -> Dict[str, Any]:
    ocr = get_paddle(project_root)
    if ocr is None: raise RuntimeError("PaddleOCR não disponível")
    res = ocr.ocr(bgr_img, cls=True)
    lines = []; raw_parts = []
    for page in res:
        for det in page:
            poly = det[0]
            txt, cf = det[1][0], float(det[1][1]) if det[1][1] is not None else -1.0
            if not txt: continue
            raw_parts.append(txt)
            xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
            l, t, r, b = min(xs), min(ys), max(xs), max(ys)
            lines.append({"text": txt, "conf": cf, "bbox": [l, t, r - l, b - t]})
    return {"lines": lines, "raw": "\n".join(raw_parts)}

def ocr_easyocr(bgr_img: np.ndarray) -> Dict[str, Any]:
    reader = get_easyocr()
    if reader is None: raise RuntimeError("EasyOCR não disponível")
    results = reader.readtext(bgr_img, detail=1, paragraph=False)
    lines = []; raw_parts = []
    for (bbox, txt, cf) in results:
        if not txt: continue
        raw_parts.append(txt)
        xs = [int(p[0]) for p in bbox]; ys = [int(p[1]) for p in bbox]
        l, t, r, b = min(xs), min(ys), max(xs), max(ys)
        lines.append({"text": txt, "conf": float(cf) if cf is not None else -1.0, "bbox": [l, t, r - l, b - t]})
    return {"lines": lines, "raw": "\n".join(raw_parts)}

def run_ocr(engine_name: str, bgr_img: np.ndarray, bin_img: np.ndarray, psm: int,
            project_root: str, debug: bool) -> Dict[str, Any]:
    if engine_name == "paddle":
        try: return ocr_paddle(bgr_img, project_root)
        except Exception as e:
            if debug: print("[WARN] PaddleOCR falhou:", e)
    if engine_name == "easyocr":
        try: return ocr_easyocr(bgr_img)
        except Exception as e:
            if debug: print("[WARN] EasyOCR falhou:", e)
    return ocr_tesseract(bin_img, psm)

def subbbox_for_match(line_text: str, bbox: List[int], match_span: Tuple[int, int]) -> List[int]:
    l, t, w, h = bbox
    start, end = match_span
    total = max(len(line_text), 1)
    x0 = l + int(round((start / total) * w))
    x1 = l + int(round((end   / total) * w))
    x0, x1 = sorted((x0, x1))
    x0 = max(x0, l); x1 = min(x1, l + w)
    return [x0, t, max(1, x1 - x0), h]

# ====== Class with macro pipeline only ======
class Extract:
    def __init__(self, id: int, project_root: str):
        self.id = id
        self.project_root = project_root

    def ocr_dir_to_csv(
        self,
        tesseract_cmd: Optional[str] = None,
        preprocess: str = "thresh",
        min_confidence: float = -1.0,
        psm: int = 6,
        debug: bool = True,
        ocr_engine: str = "paddle",  # "auto" | "paddle" | "easyocr" | "tesseract"
    ) -> str:
        """
        Lê imagens de temp/<id>/components_extracted/lenghts/*.png,
        aplica filtro (preto→branco, verde→preto), roda OCR (Paddle/Easy/Tesseract),
        extrai múltiplos pares 'C# - valor' por linha e escreve CSV no formato:

            Cx,valor
            C1,400
            C2,10
            ...

        Retorna o caminho do CSV.
        """
        # paths
        lengths_dir = os.path.join(self.project_root, "temp", str(self.id), "components_extracted", "lenghts")
        csv_path = os.path.join(lengths_dir, "file.csv")
        os.makedirs(lengths_dir, exist_ok=True)

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # lista de imagens
        images = [f for f in os.listdir(lengths_dir) if f.lower().endswith(".png")]
        images.sort()
        if debug:
            print("DEBUG: lengths_dir =", lengths_dir)
            print("DEBUG: found images:", images)
            print(f"DEBUG: OCR engine requested = {ocr_engine} (installed: paddle={_HAS_PADDLE}, easyocr={_HAS_EASYOCR})")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Cx", "valor"])

            chosen = choose_engine(ocr_engine)
            if debug:
                print(f"DEBUG: OCR engine used = {chosen}")

            for img_name in images:
                img_path = os.path.join(lengths_dir, img_name)
                bgr = cv2.imread(img_path)
                if bgr is None:
                    print(f"[WARNING] Could not open image: {img_path}")
                    continue

                # 1) filtro
                filtered = apply_green_black_filter(bgr)
                # 2) preprocess
                bin_img = preprocess_image(filtered, method=preprocess)
                # 3) OCR (engine + fallback handling)
                ocr_out = run_ocr(chosen, filtered, bin_img, psm, self.project_root, debug)

                raw_text = ocr_out.get("raw", "")
                lines_ocr: List[Dict[str, Any]] = ocr_out.get("lines", [])

                if debug:
                    print("----")
                    print("IMAGE:", img_name)
                    print("RAW OCR TEXT:")
                    print(raw_text)
                    print("----")
                    print(f"DEBUG: lines from {chosen}:", len(lines_ocr))

                total_matches_img = 0

                # Try main engine
                for ln in lines_ocr:
                    text_raw = (ln.get("text") or "").strip()
                    bbox = ln.get("bbox") or [0, 0, 0, 0]
                    mean_conf_line = float(ln.get("conf", -1.0))
                    if not text_raw:
                        continue

                    line_norm = normalize_ocr_line(text_raw)
                    if debug and line_norm != text_raw:
                        print("DEBUG line_text (raw):", text_raw)
                        print("DEBUG line_text (norm):", line_norm)

                    for m in PATTERN_C_VAL.finditer(line_norm):
                        if mean_conf_line < min_confidence:
                            if debug:
                                print(f"DEBUG: match skipped by confidence ({mean_conf_line} < {min_confidence})")
                            continue

                        try:
                            col_number = int(m.group(1))
                            raw_val = m.group(2)
                            if re.search(r'\d[.,]\d', raw_val):
                                value = float(raw_val.replace(',', '.'))
                            else:
                                value = int(raw_val.replace('.', '').replace(',', ''))
                        except Exception:
                            col_number, value = None, None

                        if col_number is not None and value is not None:
                            writer.writerow([f"C{col_number}", value])
                            total_matches_img += 1
                            if debug:
                                mstart, mend = m.span()
                                sub_bbox = subbbox_for_match(line_norm, bbox, (mstart, mend))
                                print("DEBUG: wrote row:", {"Cx": f"C{col_number}", "valor": value, "bbox": sub_bbox})

                # Fallback to Tesseract if needed
                if total_matches_img == 0 and chosen != "tesseract":
                    if debug:
                        print("DEBUG: no matches with strong engine; trying Tesseract fallback.")
                    ocr_out_fallback = ocr_tesseract(bin_img, psm)
                    for ln in ocr_out_fallback.get("lines", []):
                        text_raw = (ln.get("text") or "").strip()
                        if not text_raw:
                            continue
                        line_norm = normalize_ocr_line(text_raw)
                        mean_conf_line = float(ln.get("conf", -1.0))
                        for m in PATTERN_C_VAL.finditer(line_norm):
                            if mean_conf_line < min_confidence:
                                continue
                            try:
                                col_number = int(m.group(1))
                                raw_val = m.group(2)
                                if re.search(r'\d[.,]\d', raw_val):
                                    value = float(raw_val.replace(',', '.'))
                                else:
                                    value = int(raw_val.replace('.', '').replace(',', ''))
                            except Exception:
                                col_number, value = None, None
                            if col_number is not None and value is not None:
                                writer.writerow([f"C{col_number}", value])
                                total_matches_img += 1
                                if debug:
                                    print("DEBUG: wrote row (fallback):", {"Cx": f"C{col_number}", "valor": value})

                if debug:
                    print(f"DEBUG: total matches in {img_name} = {total_matches_img}")

        print(f"[DONE] OCR finalizado. Resultados gravados em: {csv_path}")
        return csv_path
