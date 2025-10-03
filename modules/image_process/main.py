#######################################################################
# Image processing module (organizational refactor + X1,Y1,X2,Y2 boxes)
#######################################################################

import os
import csv
from typing import Optional, Tuple, Dict, List

import fitz  # PyMuPDF
import cv2
import numpy as np

# =========================
# Config / Constants
# =========================
MIN_AREA_DEFAULT = 50
PAD_DEFAULT = 2
UPSCALE_FACTOR = 2.0
BORDER_PX = 4
SHARPEN_AMOUNT = 1.5
SHARPEN_NEG = -0.5


# =========================
# Common utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def join_project(*parts) -> str:
    return os.path.join(*[f"{p}" for p in parts])


def read_image_any(image_path: str) -> Optional[np.ndarray]:
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


def split_bgr_alpha(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    if img is None:
        return img, None, False  # type: ignore
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, False
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3], True
    return img, None, False


def build_foreground_mask(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    if alpha is not None:
        mask = (alpha > 0).astype("uint8") * 255
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def reinforce_horizontal_for_lengths(mask: np.ndarray) -> np.ndarray:
    h_img, w_img = mask.shape[:2]
    kx = max(15, int(max(15, w_img * 0.002)))
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    detect_mask = cv2.dilate(mask, kernel_horiz, iterations=2)
    detect_mask = cv2.morphologyEx(detect_mask, cv2.MORPH_CLOSE, kernel_horiz)
    return detect_mask


def connected_components(mask: np.ndarray):
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def clip_box_to_x1y1x2y2(
    x: int, y: int, w: int, h: int, pad: int, w_img: int, h_img: int
) -> Tuple[int, int, int, int]:
    """
    Apply padding to the (x,y,w,h) box and clip to image bounds.
    Returns (x1, y1, x2, y2) with x2/y2 exclusive clamped to image size.
    """
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, w_img)
    y2 = min(y + h + pad, h_img)
    return x1, y1, x2, y2


def upscale_for_saving(crop: np.ndarray) -> np.ndarray:
    interp = cv2.INTER_CUBIC
    up = cv2.resize(crop, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=interp)

    if up.ndim == 3 and up.shape[2] == 4:
        up_bgr = up[:, :, :3]
        up_alpha = up[:, :, 3]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        up_alpha = cv2.morphologyEx(up_alpha, cv2.MORPH_CLOSE, k, iterations=1)
        up_alpha = cv2.GaussianBlur(up_alpha, (3, 3), 0)
        up = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2BGRA)
        up[:, :, 3] = up_alpha
        up = cv2.copyMakeBorder(
            up, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
        )
    else:
        blur = cv2.GaussianBlur(up, (0, 0), 1.0)
        up = cv2.addWeighted(up, SHARPEN_AMOUNT, blur, SHARPEN_NEG, 0)
        up = cv2.copyMakeBorder(
            up, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
            borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    return up


def estimate_line_endpoints_from_crop(
    crop_bgr: np.ndarray,
    debug: Optional[Dict] = None,
) -> Optional[Tuple[int, int, int, int, float, float]]:
    """
    (INALTERADA na funcionalidade): detecta o maior segmento via Canny+Hough e retorna
    (x1,y1,x2,y2, angle_deg, length_px) em COORDENADAS LOCAIS do recorte.

    (NOVO) Se `debug` estiver habilitado, salva:
      - edges_<tag>.png    : mapa de bordas (Canny)
      - hough_<tag>.png    : overlay com todos os segmentos e o escolhido em destaque
      - line_debug.csv     : linha por recorte com parâmetros e métricas
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    # ---- Pré-processamento (igual ao atual) ----
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3, L2gradient=True)

    h, w = gray.shape[:2]
    max_len = 0.0
    best = None

    # Parâmetros relativos ao tamanho do recorte (iguais aos atuais)
    thr   = max(20, int(0.02 * max(h, w)))
    minL  = max(10, int(0.25 * min(h, w)))
    maxGap= max(5,  int(0.02 * max(h, w)))

    linesP = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=thr,
        minLineLength=minL,
        maxLineGap=maxGap,
    )

    candidates: List[Tuple[int,int,int,int,float]] = []  # (x1,y1,x2,y2,L)
    if linesP is not None:
        for seg in linesP.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, seg)
            L = float(np.hypot(x2 - x1, y2 - y1))
            candidates.append((x1, y1, x2, y2, L))
            if L > max_len:
                max_len = L
                best = (x1, y1, x2, y2)

    fallback_used = False
    if best is not None:
        x1, y1, x2, y2 = best
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        chosen = (x1, y1, x2, y2, angle, max_len)
    else:
        # Fallback horizontal no meio
        yc = h // 2
        x1, y1, x2, y2 = 0, yc, max(1, w - 1), yc
        angle = 0.0
        max_len = float(w)
        chosen = (x1, y1, x2, y2, angle, max_len)
        fallback_used = True

    # ---------------- DEBUG MODE ----------------
    if debug and debug.get("enabled", False):
        out_dir = debug.get("out_dir")
        if out_dir:
            try:
                ensure_dir(out_dir)
            except Exception:
                pass

            # Tag única por amostra
            gx1, gy1, gx2, gy2 = debug.get("bbox", (0, 0, w, h))
            seq = debug.get("seq", 0)
            src_name = debug.get("src_name", "src")
            tag = f"{seq:05d}_{src_name}_bbox_{gx1}_{gy1}_{gx2}_{gy2}_crop_{w}x{h}"

            # 1) Salva edges
            try:
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(out_dir, f"edges_{tag}.png"), edges_bgr)
            except Exception:
                pass

            # 2) Overlay com candidatos + escolhido
            try:
                overlay = crop_bgr.copy()
                # Todos os candidatos (amarelo/ciano claro)
                for (cx1, cy1, cx2, cy2, _) in candidates:
                    cv2.line(overlay, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1, cv2.LINE_AA)
                # Escolhido (vermelho grosso)
                bx1, by1, bx2, by2, ang, L = chosen
                cv2.line(overlay, (bx1, by1), (bx2, by2), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, (bx1, by1), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (bx2, by2), 4, (0, 0, 255), -1, cv2.LINE_AA)
                # Cantos locais do crop (verde) apenas para referência
                cv2.circle(overlay, (0, 0), 3, (0, 255, 0), -1)
                cv2.circle(overlay, (w - 1, h - 1), 3, (0, 255, 0), -1)
                # Texto com params
                txt = f"thr={thr} minL={minL} gap={maxGap} cand={len(candidates)} ang={ang:.1f} len={L:.1f} fb={int(fallback_used)}"
                cv2.putText(overlay, txt, (5, max(15, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 2, cv2.LINE_AA)
                cv2.putText(overlay, txt, (5, max(15, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imwrite(os.path.join(out_dir, f"hough_{tag}.png"), overlay)
            except Exception:
                pass

            # 3) Log CSV com parâmetros/medidas
            try:
                debug_csv = os.path.join(out_dir, "line_debug.csv")
                write_header = not os.path.isfile(debug_csv)
                with open(debug_csv, "a", newline="", encoding="utf-8") as df:
                    wtr = csv.writer(df)
                    if write_header:
                        wtr.writerow([
                            "seq","src_name","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                            "crop_w","crop_h","hough_threshold","minLineLength","maxLineGap",
                            "num_candidates","chosen_x1","chosen_y1","chosen_x2","chosen_y2",
                            "angle_deg","length_px","fallback"
                        ])
                    wtr.writerow([
                        seq, src_name, gx1, gy1, gx2, gy2,
                        w, h, thr, minL, maxGap,
                        len(candidates), bx1, by1, bx2, by2,
                        f"{ang:.4f}", f"{L:.2f}", int(fallback_used)
                    ])
            except Exception:
                pass

            # 4) (Opcional) Mostrar janelas
            if debug.get("show", False):
                try:
                    cv2.imshow(f"[edges] {tag}", edges)
                    cv2.imshow(f"[hough] {tag}", overlay)
                    key = cv2.waitKey(debug.get("wait_ms", 0))
                    if key == 27:
                        cv2.destroyAllWindows()
                except Exception:
                    pass
    # ---------------- FIM DEBUG ----------------

    return int(chosen[0]), int(chosen[1]), int(chosen[2]), int(chosen[3]), float(chosen[4]), float(chosen[5])



def write_detections_header(csv_path: str):
    """
    Create CSV and write header. Returns (writer, file_handle).
    Boxes are now X1,Y1,X2,Y2; line endpoints use prefixed names to avoid conflicts.
    """
    header = [
        "component", "x1", "y1", "x2", "y2", "area", "src_image", "saved_crop",
        "line_x1", "line_y1", "line_x2", "line_y2", "angle_deg", "length_px"
    ]
    f = open(csv_path, mode="w", newline="", encoding="utf-8")
    wtr = csv.writer(f)
    wtr.writerow(header)
    return wtr, f


def write_detection_row(
    wtr,
    component: str,
    bbox_x1y1x2y2: Tuple[int, int, int, int],
    area: int,
    src_image: str,
    saved_crop: str,
    line_endpoints_global: Optional[Tuple[int, int, int, int, float, float]] = None,
) -> None:
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if component == "lines" and line_endpoints_global is not None:
        lx1, ly1, lx2, ly2, ang, L = line_endpoints_global
        wtr.writerow([
            component, int(x1), int(y1), int(x2), int(y2), int(area),
            os.path.abspath(src_image), os.path.abspath(saved_crop),
            int(lx1), int(ly1), int(lx2), int(ly2), float(ang), float(L)
        ])
    else:
        wtr.writerow([
            component, int(x1), int(y1), int(x2), int(y2), int(area),
            os.path.abspath(src_image), os.path.abspath(saved_crop),
            "", "", "", "", "", ""
        ])


# =========================
# Class (pipelines only)
# =========================
class Image:
    def __init__(self, path: str, dpi: int, page: int, id, project_root: str):
        self.path = os.path.abspath(path)
        self.dpi = dpi
        self.page = page - 1
        self.id = id
        self.project_root = project_root

    def convert_image_to_pdf(self) -> Optional[str]:
        """
        Renders the selected PDF page to PNG at temp/<id>/images/<id>.png.
        Returns self.id on success.
        """
        output_dir = join_project(self.project_root, "temp", self.id, "images")
        ensure_dir(output_dir)

        doc = None
        try:
            doc = fitz.open(self.path)
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            page_obj = doc.load_page(self.page)
            pix = page_obj.get_pixmap(matrix=matrix)

            filename = f"{self.id}.png"
            output_path = os.path.join(output_dir, filename)
            pix.save(output_path)

            print(f"Conversão concluída. Imagem salva em: {output_path}")
            return self.id
        except Exception as e:
            print(f"Erro ao processar o arquivo: {str(e)}")
            return None
        finally:
            if doc is not None:
                doc.close()

    def apply_mask(self):
        """
        Keeps the original logic; only path helpers applied.
        """
        image_path = join_project(self.project_root, "temp", self.id, "images", f"{self.id}.png")

        img = read_image_any(image_path)
        if img is None:
            print(f"Falha ao ler a imagem: {image_path}")
            return None

        has_alpha = (img.ndim == 3 and img.shape[2] == 4)
        if has_alpha:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            if img.ndim == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 60, 40]);   red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 60, 40]); red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([15, 60, 40]); yellow_upper = np.array([35, 255, 255])
        green_lower = np.array([36, 60, 40]);  green_upper = np.array([85, 255, 255])
        blue_lower = np.array([90, 60, 40]);   blue_upper = np.array([140, 255, 255])

        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # preserve green text tweak (original behavior)
        kx = 1; ky = 1
        kernel_green_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_green_close)
        kernel_dilate_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        mask_green = cv2.dilate(mask_green, kernel_dilate_h, iterations=1)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_green_close)

        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        mask_full = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_yellow),
                                   cv2.bitwise_or(mask_green, mask_blue))

        def make_result_from_mask(mask_single: np.ndarray) -> np.ndarray:
            result_bgr = cv2.bitwise_and(bgr, bgr, mask=mask_single)
            if has_alpha:
                new_alpha = np.where(mask_single > 0, alpha, 0).astype(alpha.dtype)
                result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = new_alpha
            else:
                result = result_bgr
            return result

        result_full = make_result_from_mask(mask_full)
        result_red = make_result_from_mask(mask_red)
        result_yellow = make_result_from_mask(mask_yellow)
        result_green = make_result_from_mask(mask_green)
        result_blue = make_result_from_mask(mask_blue)

        output_dir = join_project(self.project_root, "temp", self.id, "images")
        ensure_dir(output_dir)

        out_full = os.path.join(output_dir, f"{self.id}_full.png")
        out_blue = os.path.join(output_dir, f"{self.id}_blue.png")
        out_red = os.path.join(output_dir,  f"{self.id}_red.png")
        out_yellow = os.path.join(output_dir, f"{self.id}_yellow.png")
        out_green = os.path.join(output_dir, f"{self.id}_green.png")

        saved_states = {
            "full":   cv2.imwrite(out_full, result_full),
            "blue":   cv2.imwrite(out_blue, result_blue),
            "red":    cv2.imwrite(out_red, result_red),
            "yellow": cv2.imwrite(out_yellow, result_yellow),
            "green":  cv2.imwrite(out_green, result_green),
        }

        for name, ok in saved_states.items():
            if not ok:
                path = out_full if name == "full" else locals()[f"out_{name}"]
                print(f"Falha ao salvar {name} em {path}")

        kept_pixels = int(np.count_nonzero(mask_full))
        total_pixels = mask_full.shape[0] * mask_full.shape[1]
        pct = kept_pixels / total_pixels * 100
        print(f"Imagem processada: {image_path}")
        print(f"Arquivos salvos em: {output_dir}")
        print(f"Pixels mantidos (full): {kept_pixels}/{total_pixels} ({pct:.2f}%)")

    def detect_components(
        self,
        debug_lines: bool = False,
        debug_out_dir: Optional[str] = None,
        debug_show: bool = False,
        debug_wait_ms: int = 0,
        debug_limit: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """
        (igual à sua versão, mas com 'debug_lines' para inspecionar somente 'lines').

        Gera artefatos em: temp/<id>/debug/line_detect/ (ou no 'debug_out_dir' informado):
        - edges_*.png, hough_*.png por recorte
        - line_debug.csv (um CSV de diagnóstico consolidado)
        """
        base_images_dir = join_project(self.project_root, "temp", self.id, "images")

        images = {
            "housing": os.path.join(base_images_dir, f"{self.id}_blue.png"),
            "lenghts": os.path.join(base_images_dir, f"{self.id}_green.png"),  # mantém 'lenghts'
            "lines":   os.path.join(base_images_dir, f"{self.id}_yellow.png"),
            "nodes":   os.path.join(base_images_dir, f"{self.id}_red.png"),
        }

        base_out_root = join_project(self.project_root, "temp", self.id, "components_extracted")
        out_dirs = {
            "housing": os.path.join(base_out_root, "housing"),
            "lenghts": os.path.join(base_out_root, "lenghts"),
            "lines":   os.path.join(base_out_root, "lines"),
            "nodes":   os.path.join(base_out_root, "nodes"),
        }
        ensure_dir(base_out_root)
        for d in out_dirs.values():
            ensure_dir(d)

        detections_csv = os.path.join(base_out_root, "detections.csv")
        csv_writer, csv_file = write_detections_header(detections_csv)

        # ----- setup de debug -----
        if debug_lines:
            if debug_out_dir is None:
                debug_out_dir = join_project(self.project_root, "temp", self.id, "debug", "line_detect")
            ensure_dir(debug_out_dir)
        debug_seq = 0

        results: Dict[str, List[str]] = {}
        min_area = MIN_AREA_DEFAULT
        pad = PAD_DEFAULT

        try:
            for component, image_path in images.items():
                saved_paths: List[str] = []
                out_dir = out_dirs[component]

                if not os.path.isfile(image_path):
                    print(f"[detect_components] Imagem não encontrada para '{component}': {image_path}")
                    results[component] = saved_paths
                    continue

                img = read_image_any(image_path)
                if img is None:
                    print(f"[detect_components] Falha ao ler: {image_path}")
                    results[component] = saved_paths
                    continue

                bgr, alpha, has_alpha = split_bgr_alpha(img)
                mask = build_foreground_mask(bgr, alpha)
                work_mask = reinforce_horizontal_for_lengths(mask) if component == "lenghts" else mask

                num_labels, labels, stats, _ = connected_components(work_mask)
                h_img, w_img = mask.shape[:2]
                saved_count = 0

                for lbl in range(1, num_labels):
                    x, y, w, h, area = stats[lbl]
                    if area < min_area:
                        continue

                    bx1, by1, bx2, by2 = clip_box_to_x1y1x2y2(x, y, w, h, pad, w_img, h_img)

                    # Recorte original (preserva alpha se existir)
                    crop_orig = img[by1:by2, bx1:bx2]
                    crop_for_line = crop_orig[:, :, :3] if (crop_orig.ndim == 3 and crop_orig.shape[2] == 4) else crop_orig

                    up = upscale_for_saving(crop_orig)

                    out_name = f"{self.id}_{component}_{lbl}.png"
                    out_path = os.path.join(out_dir, out_name)
                    ok = cv2.imwrite(out_path, up)
                    if not ok:
                        print(f"[detect_components] Falha ao salvar recorte: {out_path}")
                        continue

                    saved_paths.append(out_path)
                    saved_count += 1

                    bbox_x1y1x2y2 = (bx1, by1, bx2, by2)

                    if component == "lines":
                        # Monta contexto de debug por recorte
                        dbg: Optional[Dict] = None
                        if debug_lines:
                            if (debug_limit is None) or (debug_seq < debug_limit):
                                dbg = {
                                    "enabled": True,
                                    "out_dir": debug_out_dir,
                                    "seq": debug_seq,
                                    "bbox": bbox_x1y1x2y2,          # bbox GLOBAL do recorte
                                    "src_name": os.path.basename(image_path),
                                    "show": debug_show,
                                    "wait_ms": debug_wait_ms,
                                }
                                debug_seq += 1

                        ep = estimate_line_endpoints_from_crop(crop_for_line, debug=dbg)
                        if ep is not None:
                            lx1, ly1, lx2, ly2, ang, L = ep
                            gx1, gy1 = bx1 + int(lx1), by1 + int(ly1)
                            gx2, gy2 = bx1 + int(lx2), by1 + int(ly2)
                            write_detection_row(
                                csv_writer, component, bbox_x1y1x2y2, int(area), image_path, out_path,
                                line_endpoints_global=(gx1, gy1, gx2, gy2, float(ang), float(L))
                            )
                        else:
                            write_detection_row(csv_writer, component, bbox_x1y1x2y2, int(area), image_path, out_path, None)
                    else:
                        write_detection_row(csv_writer, component, bbox_x1y1x2y2, int(area), image_path, out_path, None)

                print(f"[detect_components] '{component}': detectadas {num_labels-1} regiões, "
                    f"salvas {saved_count} recortes em {out_dir}")
                results[component] = saved_paths

        finally:
            try:
                csv_file.close()
            except Exception:
                pass

        print(f"[detect_components] CSV de detecções salvo em: {detections_csv}")
        if debug_lines:
            print(f"[detect_components] Debug de linhas salvo em: {debug_out_dir}")
        return results


    def marcar_pontos_nas_imagens(
        self,
        salvar_base: bool = True,
        salvar_componentes: bool = True,
        raio: int = 6,
        espessura: int = -1,
        exibir_crops_linhas: bool = True,
        wait_ms: int = 0,
        limite_crops: Optional[int] = None,
    ) -> dict:
        """
        Lê temp/<id>/components_extracted/detections.csv e:
        1) DESENHA e SALVA overlays:
            - bolinhas VERDES em (x1,y1) e (x2-1,y2-1) da bbox (x2,y2 são exclusivos);
            - bolinhas VERMELHAS em (line_x1,line_y1) e (line_x2,line_y2), quando existirem.
            (gera imagens em temp/<id>/debug/points/)
        2) (NOVO) EXIBE cada RECORTE de 'lines' com os pontos LOCAIS (antes da transformação
            para coordenadas globais): desenha os endpoints da linha em coordenadas do recorte.

        Parâmetros:
        - exibir_crops_linhas: se True, mostra um cv2.imshow para cada recorte de 'lines'
        - wait_ms: tempo de espera para cada janela (0 = aguarda tecla; ESC interrompe)
        - limite_crops: máximo de recortes exibidos (None = todos)

        Retorna: dict com caminhos salvos {"base": <path>, "<nome_src>": <path>, ...}
        """
        # paths
        detections_csv = os.path.join(self.project_root, "temp", f"{self.id}", "components_extracted", "detections.csv")
        base_png_path = os.path.join(self.project_root, "temp", f"{self.id}", "images", f"{self.id}.png")
        out_dir = os.path.join(self.project_root, "temp", f"{self.id}", "debug", "points")
        ensure_dir(out_dir)

        if not os.path.isfile(detections_csv):
            print(f"[marcar_pontos_nas_imagens] CSV não encontrado: {detections_csv}")
            return {}

        # garante render base
        if salvar_base and not os.path.isfile(base_png_path):
            self.convert_image_to_pdf()

        # helpers
        def _to_bgr(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.ndim == 3 and img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

        def _clamp(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
            return max(0, min(x, w - 1)), max(0, min(y, h - 1))

        GREEN = (0, 255, 0)
        RED   = (0, 0, 255)

        # Carrega base (opcional)
        saved: dict = {}
        base_img = None
        if salvar_base and os.path.isfile(base_png_path):
            tmp = read_image_any(base_png_path)
            if tmp is not None:
                base_img = _to_bgr(tmp).copy()

        # Agrupa por src_image para gerar uma imagem com todos os pontos por cor (opcional)
        src_groups: Dict[str, List[dict]] = {}

        # Contador para exibição de crops
        exibidos = 0

        with open(detections_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                try:
                    x1 = int(float(row.get("x1", 0))); y1 = int(float(row.get("y1", 0)))
                    x2 = int(float(row.get("x2", 0))); y2 = int(float(row.get("y2", 0)))
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                component = row.get("component", "")
                src_path = row.get("src_image", "")

                # endpoints globais da linha (podem estar vazios)
                line_pts_global = None
                try:
                    lx1g = row.get("line_x1", "").strip(); ly1g = row.get("line_y1", "").strip()
                    lx2g = row.get("line_x2", "").strip(); ly2g = row.get("line_y2", "").strip()
                    if lx1g and ly1g and lx2g and ly2g:
                        line_pts_global = (int(float(lx1g)), int(float(ly1g)),
                                        int(float(lx2g)), int(float(ly2g)))
                except Exception:
                    line_pts_global = None

                # ---------- SALVAR OVERLAYS (base e src) ----------
                # Desenha na base (global)
                if base_img is not None:
                    h0, w0 = base_img.shape[:2]
                    gx1, gy1 = _clamp(x1, y1, w0, h0)
                    gx2, gy2 = _clamp(x2 - 1, y2 - 1, w0, h0)  # ajustar exclusivos
                    cv2.circle(base_img, (gx1, gy1), raio, GREEN, espessura)
                    cv2.circle(base_img, (gx2, gy2), raio, GREEN, espessura)
                    if line_pts_global is not None:
                        lx1, ly1_, lx2, ly2_ = line_pts_global
                        lx1, ly1_ = _clamp(lx1, ly1_, w0, h0)
                        lx2, ly2_ = _clamp(lx2, ly2_, w0, h0)
                        cv2.circle(base_img, (lx1, ly1_), raio, RED, espessura)
                        cv2.circle(base_img, (lx2, ly2_), raio, RED, espessura)

                # Armazena itens por src para salvar depois
                if salvar_componentes and os.path.isfile(src_path):
                    src_groups.setdefault(src_path, []).append(
                        {"bbox": (x1, y1, x2, y2), "line_global": line_pts_global}
                    )

                # ---------- (NOVO) EXIBIR CROP LOCAL DAS LINHAS ----------
                if exibir_crops_linhas and component == "lines" and os.path.isfile(src_path):
                    src_img = read_image_any(src_path)
                    if src_img is not None:
                        vis_src = _to_bgr(src_img)
                        h, w = vis_src.shape[:2]
                        # recorte do bbox (x2,y2 exclusivos)
                        cx1, cy1 = _clamp(x1, y1, w, h)
                        cx2, cy2 = _clamp(x2 - 1, y2 - 1, w, h)
                        # transformar cx2,cy2 de inclusivo para slicing exclusivo
                        crop = vis_src[cy1:cy2 + 1, cx1:cx2 + 1].copy()
                        if crop.size > 0:
                            # desenhar pontos LOCAIS (derivados do global - origem do crop)
                            if line_pts_global is not None:
                                lx1g, ly1g, lx2g, ly2g = line_pts_global
                                # locais = globais - (x1,y1)
                                lx1_local = int(lx1g - x1)
                                ly1_local = int(ly1g - y1)
                                lx2_local = int(lx2g - x1)
                                ly2_local = int(ly2g - y1)

                                ch, cw = crop.shape[:2]
                                # clamp local
                                lx1_local, ly1_local = _clamp(lx1_local, ly1_local, cw, ch)
                                lx2_local, ly2_local = _clamp(lx2_local, ly2_local, cw, ch)

                                # desenha linha e pontos locais (ANTES da transformação global)
                                cv2.line(crop, (lx1_local, ly1_local), (lx2_local, ly2_local), RED, 1)
                                cv2.circle(crop, (lx1_local, ly1_local), max(3, raio-2), RED, -1)
                                cv2.circle(crop, (lx2_local, ly2_local), max(3, raio-2), RED, -1)

                            # opcional: indicar cantos locais do crop (verde)
                            cv2.circle(crop, (0, 0), max(2, raio-3), GREEN, -1)
                            cv2.circle(crop, (crop.shape[1]-1, crop.shape[0]-1), max(2, raio-3), GREEN, -1)

                            # exibir
                            cv2.imshow(f"[crop linha #{i}] ({x1},{y1})-({x2},{y2})", crop)
                            key = cv2.waitKey(wait_ms)
                            if key == 27:  # ESC
                                cv2.destroyAllWindows()
                                print("[marcar_pontos_nas_imagens] Exibição interrompida pelo usuário (ESC).")
                                exibir_crops_linhas = False  # para não abrir mais janelas
                            else:
                                cv2.destroyAllWindows()

                            exibidos += 1
                            if limite_crops is not None and exibidos >= limite_crops:
                                exibir_crops_linhas = False  # parar de exibir mais

        # Salva base (se houver)
        if base_img is not None:
            base_out = os.path.join(out_dir, "base_points.png")
            if cv2.imwrite(base_out, base_img):
                saved["base"] = base_out
            else:
                print(f"[marcar_pontos_nas_imagens] Falha ao salvar {base_out}")

        # Salva overlays por src_image
        if salvar_componentes:
            for src_path, items in src_groups.items():
                img = read_image_any(src_path)
                if img is None:
                    continue
                vis = _to_bgr(img).copy()
                h, w = vis.shape[:2]
                for it in items:
                    x1, y1, x2, y2 = it["bbox"]
                    sx1, sy1 = _clamp(x1, y1, w, h)
                    sx2, sy2 = _clamp(x2 - 1, y2 - 1, w, h)  # ajustar exclusivos
                    cv2.circle(vis, (sx1, sy1), raio, GREEN, espessura)
                    cv2.circle(vis, (sx2, sy2), raio, GREEN, espessura)
                    if it["line_global"] is not None:
                        lx1, ly1, lx2, ly2 = it["line_global"]
                        lx1, ly1 = _clamp(lx1, ly1, w, h)
                        lx2, ly2 = _clamp(lx2, ly2, w, h)
                        cv2.circle(vis, (lx1, ly1), raio, RED, espessura)
                        cv2.circle(vis, (lx2, ly2), raio, RED, espessura)

                base_name = os.path.splitext(os.path.basename(src_path))[0]
                out_path = os.path.join(out_dir, f"{base_name}__points.png")
                if cv2.imwrite(out_path, vis):
                    saved[base_name] = out_path
                else:
                    print(f"[marcar_pontos_nas_imagens] Falha ao salvar {out_path}")

        print(f"[marcar_pontos_nas_imagens] Overlays salvos em: {out_dir} | crops exibidos: {exibidos}")
        return saved

