import os
import re
import csv
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Motores opcionais:
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

        # Regex para "C<num> - valor"
        pattern = re.compile(
            r'\bC\s*(\d+)\s*(?:[-–—:])\s*([0-9]+(?:[.,][0-9]+)?)\b',
            flags=re.IGNORECASE
        )

        # lista de imagens
        images = [f for f in os.listdir(lengths_dir) if f.lower().endswith(".png")]
        images.sort()
        if debug:
            print("DEBUG: lengths_dir =", lengths_dir)
            print("DEBUG: found images:", images)
            print(f"DEBUG: OCR engine requested = {ocr_engine} (installed: paddle={_HAS_PADDLE}, easyocr={_HAS_EASYOCR})")

        # ---------- Filtro: preto -> branco, verde -> preto ----------
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

        # ---------- Pré-processamento ----------
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

        # ---------- Normalização segura ----------
        def _normalize_ocr_line(s: str) -> str:
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

        # ---------- Motores OCR ----------
        _paddle: Any = None
        _easy: Any = None

        def _get_paddle() -> Any:
            nonlocal _paddle
            if _paddle is None and _HAS_PADDLE:
                # Se você tiver modelos locais "server v2.0", detecte e use:
                det_path = os.path.join(self.project_root, "ch_ppocr_server_v2.0_det_infer")
                rec_path = os.path.join(self.project_root, "ch_ppocr_server_v2.0_rec_infer")
                kwargs = dict(use_angle_cls=True, lang='en', det=True, rec=True, show_log=False)
                if os.path.isdir(det_path): kwargs["det_model_dir"] = det_path
                if os.path.isdir(rec_path): kwargs["rec_model_dir"] = rec_path
                _paddle = PaddleOCR(**kwargs)
            return _paddle

        def _get_easyocr() -> Any:
            nonlocal _easy
            if _easy is None and _HAS_EASYOCR:
                _easy = easyocr.Reader(['en'], gpu=False, verbose=False)
            return _easy

        def _choose_engine() -> str:
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

        def _ocr_tesseract(bin_img: np.ndarray) -> Dict[str, Any]:
            proc_up = cv2.resize(bin_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'
            raw_text = pytesseract.image_to_string(proc_up, config=config)
            data = pytesseract.image_to_data(proc_up, output_type=Output.DICT, config=config)
            n = len(data.get('text', []))
            byline: Dict[Tuple[int,int,int], List[int]] = {}
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

        def _ocr_paddle(bgr_img: np.ndarray) -> Dict[str, Any]:
            ocr = _get_paddle()
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

        def _ocr_easyocr(bgr_img: np.ndarray) -> Dict[str, Any]:
            reader = _get_easyocr()
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

        def _run_ocr(engine_name: str, bgr_img: np.ndarray, bin_img: np.ndarray) -> Dict[str, Any]:
            if engine_name == "paddle":
                try: return _ocr_paddle(bgr_img)
                except Exception as e:
                    if debug: print("[WARN] PaddleOCR falhou:", e)
            if engine_name == "easyocr":
                try: return _ocr_easyocr(bgr_img)
                except Exception as e:
                    if debug: print("[WARN] EasyOCR falhou:", e)
            return _ocr_tesseract(bin_img)

        def _subbbox_for_match(line_text: str, bbox: List[int], match_span: Tuple[int, int]) -> List[int]:
            l, t, w, h = bbox
            start, end = match_span
            total = max(len(line_text), 1)
            x0 = l + int(round((start / total) * w))
            x1 = l + int(round((end   / total) * w))
            x0, x1 = sorted((x0, x1))
            x0 = max(x0, l); x1 = min(x1, l + w)
            return [x0, t, max(1, x1 - x0), h]

        # ---------- Execução ----------
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            # >>> CSV SIMPLES: apenas "Cx,valor"
            writer = csv.writer(csvfile)
            writer.writerow(["Cx", "valor"])

            chosen = _choose_engine()
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
                # 3) OCR
                ocr_out = _run_ocr(chosen, filtered, bin_img)

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

                for ln in lines_ocr:
                    text_raw = (ln.get("text") or "").strip()
                    bbox = ln.get("bbox") or [0, 0, 0, 0]
                    mean_conf_line = float(ln.get("conf", -1.0))
                    if not text_raw:
                        continue

                    line_norm = _normalize_ocr_line(text_raw)
                    if debug and line_norm != text_raw:
                        print("DEBUG line_text (raw):", text_raw)
                        print("DEBUG line_text (norm):", line_norm)

                    for m in pattern.finditer(line_norm):
                        if mean_conf_line < min_confidence:
                            if debug:
                                print(f"DEBUG: match skipped by confidence ({mean_conf_line} < {min_confidence})")
                            continue

                        # Campos
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
                                # sub-bbox só para log/consistência visual (não salva no CSV)
                                mstart, mend = m.span()
                                sub_bbox = _subbbox_for_match(line_norm, bbox, (mstart, mend))
                                print("DEBUG: wrote row:", {"Cx": f"C{col_number}", "valor": value, "bbox": sub_bbox})

                # fallback Tesseract se nada encontrado
                if total_matches_img == 0 and chosen != "tesseract":
                    if debug:
                        print("DEBUG: no matches with strong engine; trying Tesseract fallback.")
                    ocr_out_fallback = _ocr_tesseract(bin_img)
                    for ln in ocr_out_fallback.get("lines", []):
                        text_raw = (ln.get("text") or "").strip()
                        if not text_raw:
                            continue
                        line_norm = _normalize_ocr_line(text_raw)
                        mean_conf_line = float(ln.get("conf", -1.0))
                        for m in pattern.finditer(line_norm):
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

    def ocr_components_to_csv(
        self,
        preprocess: str = "thresh",
        min_confidence: float = -1.0,
        psm: int = 7,
        debug: bool = True,
        ocr_engine: str = "paddle"
    ) -> Tuple[str, str]:
        """
        Roda OCR nos recortes de 'housing' (azul) e 'nodes' (vermelho) listados no detections.csv.
        Usa saved_crop como image_path. Segmenta texto (mais escuro) do fundo (um pouco mais claro)
        dentro da mesma faixa de matiz; fallback k-means se Otsu falhar.

        Gera:
        - components_extracted/housing.csv   (Hx, image_path, x, y, w, h)
        - components_extracted/nodes.csv     (Nx, image_path, x, y, w, h)

        Retorna (path_housing_csv, path_nodes_csv).
        """
        import csv

        base_out_root = os.path.join(self.project_root, "temp", self.id, "components_extracted")
        detections_csv = os.path.join(base_out_root, "detections.csv")
        housing_csv = os.path.join(base_out_root, "housing.csv")
        nodes_csv   = os.path.join(base_out_root, "nodes.csv")

        if not os.path.isfile(detections_csv):
            raise FileNotFoundError(f"Não encontrei {detections_csv}")

        # --------- helpers de OCR (mesmos motores do seu pipeline) ---------
        # (repetimos localmente para não depender de escopo interno do outro método)
        _HAS_PADDLE = False
        _HAS_EASYOCR = False
        try:
            from paddleocr import PaddleOCR  # type: ignore
            _HAS_PADDLE = True
        except Exception:
            PaddleOCR = None  # type: ignore

        try:
            import easyocr  # type: ignore
            _HAS_EASYOCR = True
        except Exception:
            easyocr = None  # type: ignore

        _paddle = None
        _easy = None

        def _get_paddle():
            nonlocal _paddle
            if _paddle is None and _HAS_PADDLE:
                det_path = os.path.join(self.project_root, "ch_ppocr_server_v2.0_det_infer")
                rec_path = os.path.join(self.project_root, "ch_ppocr_server_v2.0_rec_infer")
                kwargs = dict(use_angle_cls=True, lang='en', det=True, rec=True, show_log=False)
                if os.path.isdir(det_path): kwargs["det_model_dir"] = det_path
                if os.path.isdir(rec_path): kwargs["rec_model_dir"] = rec_path
                _paddle = PaddleOCR(**kwargs)
            return _paddle

        def _get_easy():
            nonlocal _easy
            if _easy is None and _HAS_EASYOCR:
                _easy = easyocr.Reader(['en'], gpu=False, verbose=False)
            return _easy

        def _choose_engine() -> str:
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

        def _ocr_tesseract(bin_img: np.ndarray) -> Dict[str, Any]:
            proc_up = cv2.resize(bin_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'
            raw_text = pytesseract.image_to_string(proc_up, config=config)
            data = pytesseract.image_to_data(proc_up, output_type=Output.DICT, config=config)
            n = len(data.get('text', []))
            lines = []
            byline = {}
            for i in range(n):
                txt = (data['text'][i] or "").strip()
                if not txt: continue
                key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                byline.setdefault(key, []).append(i)
            for key, idxs in byline.items():
                idxs_sorted = sorted(idxs, key=lambda k: data['left'][k])
                words = [(data['text'][i] or "").strip() for i in idxs_sorted if (data['text'][i] or "").strip()]
                if not words: continue
                line_text = " ".join(words)
                confs = []
                for i in idxs_sorted:
                    try:
                        c = float(data['conf'][i])
                        if c >= 0: confs.append(c)
                    except Exception:
                        pass
                mean_conf = float(np.mean(confs)) if confs else -1.0
                lines.append({"text": line_text, "conf": mean_conf})
            return {"lines": lines, "raw": raw_text}

        def _ocr_paddle(bgr_img: np.ndarray) -> Dict[str, Any]:
            ocr = _get_paddle()
            if ocr is None: raise RuntimeError("PaddleOCR não disponível")
            res = ocr.ocr(bgr_img, cls=True)
            lines = []; raw_parts = []
            if not res: return {"lines": [], "raw": ""}
            for page in res:
                if not page: continue
                for det in page:
                    if not det or len(det) < 2: continue
                    info = det[1]
                    txt = None; cf = -1.0
                    if isinstance(info,(list,tuple)):
                        if len(info)>=1 and info[0] is not None: txt = str(info[0])
                        if len(info)>=2 and info[1] is not None:
                            try: cf = float(info[1])
                            except: cf = -1.0
                    if not txt or not txt.strip(): continue
                    raw_parts.append(txt)
                    lines.append({"text": txt.strip(), "conf": cf})
            return {"lines": lines, "raw": "\n".join(raw_parts)}

        def _ocr_easy(bgr_img: np.ndarray) -> Dict[str, Any]:
            reader = _get_easy()
            if reader is None: raise RuntimeError("EasyOCR não disponível")
            results = reader.readtext(bgr_img, detail=0, paragraph=False)
            lines = []; raw_parts = []
            for txt in results:
                if not txt: continue
                raw_parts.append(txt)
                lines.append({"text": str(txt), "conf": -1.0})
            return {"lines": lines, "raw": "\n".join(raw_parts)}

        def _run_ocr(engine: str, bgr_or_bin: np.ndarray, use_bgr: bool = True) -> Dict[str, Any]:
            if engine == "paddle":
                try: return _ocr_paddle(bgr_or_bin if use_bgr else cv2.cvtColor(bgr_or_bin, cv2.COLOR_GRAY2BGR))
                except Exception as e:
                    if debug: print("[WARN] PaddleOCR falhou:", e)
            if engine == "easyocr":
                try: return _ocr_easy(bgr_or_bin if use_bgr else cv2.cvtColor(bgr_or_bin, cv2.COLOR_GRAY2BGR))
                except Exception as e:
                    if debug: print("[WARN] EasyOCR falhou:", e)
            return _ocr_tesseract(bgr_or_bin if not use_bgr else cv2.cvtColor(bgr_or_bin, cv2.COLOR_BGR2GRAY))

        # --------- segmentação por cor (texto escuro x fundo claro na MESMA matiz) ---------
        def _segment_text_by_hue(bgr: np.ndarray, color: str) -> np.ndarray:
            """
            Retorna imagem binária (uint8) com texto preto (0) em branco (255),
            isolando por faixa de matiz ('blue' ou 'red') e separando texto (mais escuro)
            do fundo (mais claro) por Otsu (fallback k-means).
            """
            assert color in ("blue","red")
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            H,S,V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

            if color == "blue":
                hue_mask = (H >= 90) & (H <= 140) & (S >= 30) & (V >= 30)
            else:
                # vermelho: duas faixas (wrap)
                hue_mask = ((H <= 10) | (H >= 170)) & (S >= 30) & (V >= 30)

            # pega somente os pixels na faixa da cor alvo
            mask = hue_mask.astype(np.uint8)*255
            if np.count_nonzero(mask) < 10:
                # fallback: usa o cinza inteiro se a máscara ficou vazia
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                inv = 255 - th
                return inv if np.mean(inv) > np.mean(th) else th

            # aplica Otsu no brilho (V) APENAS dentro da máscara
            v_vals = V[mask==255].astype(np.uint8)
            try:
                # Otsu precisa de histograma; aplicamos no vetor v_vals
                # construção de limiar global mas aplicado só na máscara
                th_val, _ = cv2.threshold(v_vals, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # texto é a classe mais escura → pixels com V < th_val
                text_bin_full = (V < th_val).astype(np.uint8)*255
            except Exception:
                # fallback k-means (k=2) no vetor v_vals
                vs = v_vals.reshape(-1,1).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
                _,labels,centers = cv2.kmeans(vs, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                c0,c1 = centers.flatten().tolist()
                dark_label = int(0 if c0<=c1 else 1)
                # reconstrói máscara global
                tmp = np.zeros_like(V, dtype=np.uint8)
                tmp[mask==255] = (labels.flatten()==dark_label).astype(np.uint8)*255
                text_bin_full = tmp

            # aplica a máscara de cor (garante que fora dela vira fundo)
            text_bin_full[mask==0] = 0  # nada fora da cor
            # queremos texto PRETO em fundo BRANCO para Tesseract
            # então invertendo se necessário:
            inv = 255 - text_bin_full
            return inv if np.mean(inv) > np.mean(text_bin_full) else text_bin_full

        # --------- Normalização de rótulos ---------
        re_housing = re.compile(r'\b[HB]\s*-?\s*(\d+)\b', flags=re.IGNORECASE)
        re_nodes   = re.compile(r'\bN\s*-?\s*(\d+)\b', flags=re.IGNORECASE)

        def _fix_digits(s: str) -> str:
            t = []
            for ch in s:
                if ch in 'Oo': t.append('0')
                elif ch in 'Il': t.append('1')
                else: t.append(ch)
            return ''.join(t)

        def _parse_housing(text: str) -> Optional[str]:
            m = re_housing.search(text)
            if not m: return None
            num = _fix_digits(m.group(1))
            return f"H{num}"

        def _parse_node(text: str) -> Optional[str]:
            m = re_nodes.search(text)
            if not m: return None
            num = _fix_digits(m.group(1))
            return f"N{num}"

        # --------- Carrega detections, filtra e processa ---------
        rows = []
        with open(detections_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("component") in ("housing", "nodes"):
                    rows.append(r)

        if debug:
            print(f"[OCR-COMP] total recortes a processar: {len(rows)}")

        # CSVs de saída
        with open(housing_csv, "w", newline="", encoding="utf-8") as fh, \
            open(nodes_csv,   "w", newline="", encoding="utf-8") as fn:

            wh = csv.writer(fh); wh.writerow(["Hx", "image_path", "x", "y", "w", "h"])
            wn = csv.writer(fn); wn.writerow(["Nx", "image_path", "x", "y", "w", "h"])

            engine = _choose_engine()
            if debug: print(f"[OCR-COMP] OCR engine used = {engine}")

            for r in rows:
                comp = r["component"]
                img_path = r["saved_crop"]
                x = int(float(r["x"])); y = int(float(r["y"]))
                w = int(float(r["w"])); h = int(float(r["h"]))

                bgr = cv2.imread(img_path)
                if bgr is None:
                    print(f"[WARN] Falha ao abrir recorte: {img_path}")
                    continue

                # segmenta por cor e cria binário (texto preto em branco)
                color = "blue" if comp == "housing" else "red"
                bin_img = _segment_text_by_hue(bgr, color=color)

                # roda OCR (engines que preferem BGR recebem conversão)
                ocr_out = _run_ocr(engine, bgr if engine != "tesseract" else bin_img, use_bgr=(engine!="tesseract"))
                lines = ocr_out.get("lines", [])
                if debug:
                    print(f"[OCR-COMP] {comp} -> {os.path.basename(img_path)}; candidates={len(lines)}")

                # pega a melhor linha (maior conf; se -1.0, usa a primeira não vazia)
                best_id = None
                best_conf = -2.0
                for ln in lines:
                    txt = (ln.get("text") or "").strip()
                    cf  = float(ln.get("conf", -1.0))
                    if not txt: continue
                    if cf > best_conf:
                        best_conf = cf
                        best_id = txt

                if not best_id:
                    print(f"[WARN] OCR vazio em {comp}: {img_path}")
                    continue

                if comp == "housing":
                    ident = _parse_housing(best_id)
                    if not ident:
                        print(f"[WARN] Falha em parse housing: '{best_id}' ({img_path})")
                        continue
                    wh.writerow([ident, os.path.abspath(img_path), x, y, w, h])

                else:  # nodes
                    ident = _parse_node(best_id)
                    if not ident:
                        print(f"[WARN] Falha em parse node: '{best_id}' ({img_path})")
                        continue
                    wn.writerow([ident, os.path.abspath(img_path), x, y, w, h])

        if debug:
            print(f"[OCR-COMP] CSVs gerados: {housing_csv} | {nodes_csv}")

        return housing_csv, nodes_csv

    def find_line_connections(self, debug: bool = True, point_radius: int = 3) -> str:
        """
        Lê detections.csv + housing.csv + nodes.csv e gera connections.csv
        vinculando cada linha (L1, L2, ...) a dois componentes (Hx/Nx) por
        containment de endpoints. StartSide/EndSide = L/R (esq/dir).
        """
        import csv

        base = os.path.join(self.project_root, "temp", self.id, "components_extracted")
        det_csv = os.path.join(base, "detections.csv")
        house_csv = os.path.join(base, "housing.csv")
        nodes_csv = os.path.join(base, "nodes.csv")
        out_csv = os.path.join(base, "connections.csv")

        if not (os.path.isfile(det_csv) and os.path.isfile(house_csv) and os.path.isfile(nodes_csv)):
            raise FileNotFoundError("detections.csv / housing.csv / nodes.csv ausentes")

        # carrega housing/nodes -> lista de dicts {id,label,x,y,w,h,image_path}
        def _load_comp(path: str, label_col: str) -> List[Dict[str,Any]]:
            out = []
            with open(path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    try:
                        label = r[label_col]
                        x = int(float(r["x"])); y = int(float(r["y"]))
                        w = int(float(r["w"])); h = int(float(r["h"]))
                        out.append({
                            "label": label,
                            "x": x, "y": y, "w": w, "h": h,
                            "image_path": r.get("image_path","")
                        })
                    except Exception:
                        pass
            return out

        housings = _load_comp(house_csv, "Hx")
        nodes    = _load_comp(nodes_csv,  "Nx")
        comps = housings + nodes

        # carrega linhas do detections
        det_rows = []
        with open(det_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("component") == "lines":
                    det_rows.append(r)

        def _contains(ptx:int, pty:int, bx:int, by:int, bw:int, bh:int) -> bool:
            return (ptx >= bx) and (ptx <= bx + bw) and (pty >= by) and (pty <= by + bh)

        def _square_intersection_area(px:int, py:int, bx:int, by:int, bw:int, bh:int, r:int) -> int:
            # interseção entre quadrado centrado no ponto (lado=2r) e o bbox
            sx0, sy0 = px - r, py - r
            sx1, sy1 = px + r, py + r
            bx0, by0 = bx, by
            bx1, by1 = bx + bw, by + bh
            ix0, iy0 = max(sx0, bx0), max(sy0, by0)
            ix1, iy1 = min(sx1, bx1), min(sy1, by1)
            iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
            return iw * ih

        def _parse_float_or_blank(s: str) -> Optional[float]:
            s = (s or "").strip()
            if s == "": return None
            try: return float(s)
            except: return None

        # poderá recalcular endpoints se necessário
        def _recalc_endpoints_if_needed(row: Dict[str,str]) -> Optional[Tuple[int,int,int,int]]:
            gx1 = _parse_float_or_blank(row.get("x1")); gy1 = _parse_float_or_blank(row.get("y1"))
            gx2 = _parse_float_or_blank(row.get("x2")); gy2 = _parse_float_or_blank(row.get("y2"))
            if None not in (gx1,gy1,gx2,gy2):
                return (int(gx1),int(gy1),int(gx2),int(gy2))
            # recalcula via recorte salvo
            crop_path = row.get("saved_crop")
            if not crop_path or not os.path.isfile(crop_path):
                return None
            bgr = cv2.imread(crop_path)
            if bgr is None: return None
            # estima no recorte local
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 50, 150, apertureSize=3, L2gradient=True)
            h, w = gray.shape[:2]
            linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(20,int(0.02*max(h,w))),
                                    minLineLength=max(10,int(0.25*min(h,w))), maxLineGap=max(5,int(0.02*max(h,w))))
            best = None; max_len = 0.0
            if linesP is not None:
                for seg in linesP.reshape(-1,4):
                    x1,y1,x2,y2 = map(int, seg)
                    L = float(np.hypot(x2-x1, y2-y1))
                    if L > max_len:
                        max_len = L; best = (x1,y1,x2,y2)
            if best is None:
                # fallback: extremos horizontais do bbox local
                yc = h//2; best = (0,yc, max(1,w-1), yc)
            # converte p/ global
            bx, by = int(float(row["x"])), int(float(row["y"]))
            return (bx+best[0], by+best[1], bx+best[2], by+best[3])

        # escreve connections.csv
        with open(out_csv, "w", newline="", encoding="utf-8") as fo:
            wcsv = csv.writer(fo)
            wcsv.writerow(["LineID", "StartComponent", "EndComponent", "StartSide", "EndSide", "x1", "y1", "x2", "y2"])

            for i, r in enumerate(det_rows, start=1):
                LID = f"L{i}"

                # endpoints globais
                eps = _recalc_endpoints_if_needed(r)
                if eps is None:
                    if debug: print(f"[WARN] Sem endpoints para {LID}, pulando.")
                    continue
                x1,y1,x2,y2 = eps

                # define L/R (esq/dir) de forma consistente
                # regra: menor x = Start(L), maior x = End(R); empate quebra por menor y
                start = (x1,y1); end = (x2,y2)
                if (x2 < x1) or (x2 == x1 and y2 < y1):
                    start, end = (x2,y2), (x1,y1)

                # encontra componente contendo cada endpoint
                def _match(pt):
                    px,py = pt
                    candidates = []
                    for c in comps:
                        bx,by,bw,bh = c["x"],c["y"],c["w"],c["h"]
                        if _contains(px,py,bx,by,bw,bh):
                            inter = _square_intersection_area(px,py,bx,by,bw,bh, point_radius)
                            candidates.append((inter, c))
                    if not candidates:
                        return None
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    return candidates[0][1]

                c_start = _match(start)
                c_end   = _match(end)

                if c_start is None or c_end is None:
                    if debug:
                        print(f"[WARN] Endpoint sem componente em {LID}: start_in={f'{c_start is not None}, {c_start}'}, end_in={f'{c_end is not None}, {c_end}'}")
                    # conforme combinado: registra aviso e pula
                    continue

                lab_start = c_start["label"]  # Hx ou Nx
                lab_end   = c_end["label"]

                wcsv.writerow([LID, lab_start, lab_end, "L", "R", start[0], start[1], end[0], end[1]])

        if debug:
            print(f"[CONN] connections.csv gerado em {out_csv}")

        return out_csv
