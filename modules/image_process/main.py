#######################################################################
# Módulo responsável por processar imagens
#######################################################################

import os
import fitz
from typing import Optional
import cv2
import numpy as np
import csv
from typing import Tuple

class Image:
    def __init__(self, path: str, dpi: int, page: int, id: int, project_root: str):
        self.path = os.path.abspath(path)
        self.dpi = dpi
        self.page = page - 1
        self.id = id
        self.project_root = project_root
        

    def convert_image_to_pdf(self) -> Optional[str]:
        #######################################################################
        # Converte a página especificada do PDF em imagem PNG, salva em
        # self.project_root/temp/images e retorna um id gerado a partir do datetime
        # (ou None em caso de erro).
        #######################################################################
        # Cria o diretório de destino se não existir
        output_dir = os.path.join(self.project_root, "temp", self.id, "images")
        os.makedirs(output_dir, exist_ok=True)

        doc = None
        try:
            # Abre o documento PDF
            doc = fitz.open(self.path)

            # Calcula o fator de escala baseado no DPI
            zoom = self.dpi / 72  # 72 DPI é o padrão do PDF
            matrix = fitz.Matrix(zoom, zoom)

            # Carrega a página específica
            page_obj = doc.load_page(self.page)

            # Renderiza a página como imagem
            pix = page_obj.get_pixmap(matrix=matrix)

            # Gera um id com datetime (formato: YYYYMMDDHHMMSSffffff)
            filename = f"{self.id}.png"
            output_path = os.path.join(output_dir, filename)

            # Salva a imagem
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
        image_path = os.path.join(self.project_root, "temp", self.id, "images", f"{self.id}.png")

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Falha ao ler a imagem: {image_path}")
            return None

        # Se a imagem tiver canal alpha, separa
        has_alpha = (img.ndim == 3 and img.shape[2] == 4)
        if has_alpha:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            # Se for grayscale, converte para BGR
            if img.ndim == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img

        # Converte para HSV para facilitar seleção por cor
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Definições de ranges HSV (valores aproximados — podem ser afinados)
        red_lower1 = np.array([0, 60, 40])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 60, 40])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([15, 60, 40])
        yellow_upper = np.array([35, 255, 255])

        green_lower = np.array([36, 60, 40])
        green_upper = np.array([85, 255, 255])

        blue_lower = np.array([90, 60, 40])
        blue_upper = np.array([140, 255, 255])

        # Cria máscaras (não combinadas)
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

        # Suaviza cada máscara individualmente para reduzir ruído (morphology)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # ===== alteração apenas aqui para preservar texto verde =====
        kx = 1
        ky = 1
        kernel_green_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_green_close)
        kernel_dilate_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        mask_green = cv2.dilate(mask_green, kernel_dilate_h, iterations=1)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_green_close)
        # ============================================================

        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        # Máscara combinada (full)
        mask_full = cv2.bitwise_or(cv2.bitwise_or(mask_red, mask_yellow),
                                cv2.bitwise_or(mask_green, mask_blue))

        # Função auxiliar para aplicar máscara e recompor alpha se necessário
        def make_result_from_mask(mask_single):
            result_bgr = cv2.bitwise_and(bgr, bgr, mask=mask_single)
            if has_alpha:
                new_alpha = np.where(mask_single > 0, alpha, 0).astype(alpha.dtype)
                result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = new_alpha
            else:
                result = result_bgr
            return result

        # Gera resultados individuais
        result_full = make_result_from_mask(mask_full)
        result_red = make_result_from_mask(mask_red)
        result_yellow = make_result_from_mask(mask_yellow)
        result_green = make_result_from_mask(mask_green)
        result_blue = make_result_from_mask(mask_blue)

        # Garante que pasta de saída existe
        output_dir = os.path.join(self.project_root, "temp", self.id, "images")
        os.makedirs(output_dir, exist_ok=True)

        # Caminhos de saída
        out_full = os.path.join(output_dir, f"{self.id}_full.png")
        out_blue = os.path.join(output_dir, f"{self.id}_blue.png")
        out_red = os.path.join(output_dir,  f"{self.id}_red.png")
        out_yellow = os.path.join(output_dir, f"{self.id}_yellow.png")
        out_green = os.path.join(output_dir, f"{self.id}_green.png")

        # Salva todos
        saved_full = cv2.imwrite(out_full, result_full)
        saved_blue = cv2.imwrite(out_blue, result_blue)
        saved_red = cv2.imwrite(out_red, result_red)
        saved_yellow = cv2.imwrite(out_yellow, result_yellow)
        saved_green = cv2.imwrite(out_green, result_green)

        saved_states = {
            "full": saved_full,
            "blue": saved_blue,
            "red": saved_red,
            "yellow": saved_yellow,
            "green": saved_green,
        }

        for name, ok in saved_states.items():
            if not ok:
                print(f"Falha ao salvar {name} em {locals()['out_' + name] if name!='full' else out_full}")

        # Estatísticas (opcional)
        kept_pixels = int(np.count_nonzero(mask_full))
        total_pixels = mask_full.shape[0] * mask_full.shape[1]
        pct = kept_pixels / total_pixels * 100
        print(f"Imagem processada: {image_path}")
        print(f"Arquivos salvos em: {output_dir}")
        print(f"Pixels mantidos (full): {kept_pixels}/{total_pixels} ({pct:.2f}%)")

    def detect_components(self):
        """
        Detecta regiões nas imagens já filtradas por cor e salva os recortes em
        self.project_root/temp/{id}/components_extracted/{component}/.
        Além disso, salva um CSV com as coordenadas (x,y,w,h) de cada detecção,
        e para 'lines' também salva x1,y1,x2,y2, angle_deg, length_px.
        Retorna um dict { component: [lista_de_caminhos_salvos] }.
        """

        base_images_dir = os.path.join(self.project_root, "temp", self.id, "images")

        images = {
            "housing": os.path.join(base_images_dir, f"{self.id}_blue.png"),
            "lenghts": os.path.join(base_images_dir, f"{self.id}_green.png"),   # mantém 'lenghts' por compatibilidade
            "lines":   os.path.join(base_images_dir, f"{self.id}_yellow.png"),
            "nodes":   os.path.join(base_images_dir, f"{self.id}_red.png"),
        }

        base_out_root = os.path.join(self.project_root, "temp", self.id, "components_extracted")
        images_save_dir = {
            "housing": os.path.join(base_out_root, "housing"),
            "lenghts": os.path.join(base_out_root, "lenghts"),
            "lines":   os.path.join(base_out_root, "lines"),
            "nodes":   os.path.join(base_out_root, "nodes"),
        }
        os.makedirs(base_out_root, exist_ok=True)

        # ===== CSV de detecções (enriquecido) =====
        detections_csv = os.path.join(base_out_root, "detections.csv")
        csv_header = [
            "component", "x", "y", "w", "h", "area", "src_image", "saved_crop",
            "x1", "y1", "x2", "y2", "angle_deg", "length_px"
        ]
        csv_file = open(detections_csv, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        # =========================================

        results = {}
        min_area = 50
        pad = 2

        UPSCALE_FACTOR = 2.0
        BORDER_PX = 4
        SHARPEN_AMOUNT = 1.5
        SHARPEN_NEG = -0.5

        def _estimate_line_endpoints_from_crop(crop_bgr: np.ndarray) -> Optional[Tuple[int,int,int,int,float,float]]:
            """
            Estima endpoints reais da linha em um recorte (sem upscale/borda).
            Retorna (x1,y1,x2,y2,angle_deg,length_px) nas coordenadas do recorte.
            Estratégia: Canny + Probabilistic HoughLinesP -> maior segmento.
            Fallback: extremos do retângulo central.
            """
            if crop_bgr is None or crop_bgr.size == 0:
                return None
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            # binariza "qualquer coisa" amarela/linha para realçar contorno
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            edges = cv2.Canny(blur, 50, 150, apertureSize=3, L2gradient=True)

            h, w = gray.shape[:2]
            max_len = 0
            best = None

            # Parâmetros razoáveis para linhas finas
            linesP = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180.0,
                threshold=max(20, int(0.02*max(h,w))),
                minLineLength=max(10, int(0.25*min(h,w))),
                maxLineGap=max(5, int(0.02*max(h,w))),
            )

            if linesP is not None:
                for seg in linesP.reshape(-1,4):
                    x1,y1,x2,y2 = map(int, seg)
                    L = float(np.hypot(x2-x1, y2-y1))
                    if L > max_len:
                        max_len = L
                        best = (x1,y1,x2,y2)

            if best is not None:
                x1,y1,x2,y2 = best
                angle = float(np.degrees(np.arctan2(y2-y1, x2-x1)))
                return (x1,y1,x2,y2, angle, max_len)

            # Fallback bem simples se Hough não achou nada: extremos horizontais no meio de y
            yc = h//2
            x1,y1,x2,y2 = 0, yc, max(1,w-1), yc
            angle = 0.0
            L = float(w)
            return (x1,y1,x2,y2, angle, L)

        try:
            for component, image_path in images.items():
                saved_paths = []
                out_dir = images_save_dir[component]
                os.makedirs(out_dir, exist_ok=True)

                if not os.path.isfile(image_path):
                    print(f"[detect_components] Imagem não encontrada para '{component}': {image_path}")
                    results[component] = saved_paths
                    continue

                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[detect_components] Falha ao ler: {image_path}")
                    results[component] = saved_paths
                    continue

                has_alpha = (img.ndim == 3 and img.shape[2] == 4)
                if has_alpha:
                    bgr = img[:, :, :3]
                    alpha = img[:, :, 3]
                else:
                    if img.ndim == 2:
                        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        bgr = img

                if has_alpha:
                    mask = (alpha > 0).astype("uint8") * 255
                else:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # reforço horizontal para 'lenghts'
                if component == "lenghts":
                    h_img, w_img = mask.shape[:2]
                    kx = max(15, int(max(15, w_img * 0.002)))
                    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
                    detect_mask = cv2.dilate(mask, kernel_horiz, iterations=2)
                    detect_mask = cv2.morphologyEx(detect_mask, cv2.MORPH_CLOSE, kernel_horiz)
                    work_mask = detect_mask
                else:
                    work_mask = mask

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work_mask, connectivity=8)

                saved_count = 0
                h_img, w_img = mask.shape[:2]

                for lbl in range(1, num_labels):
                    x, y, w, h, area = stats[lbl]
                    if area < min_area:
                        continue

                    x0 = max(x - pad, 0)
                    y0 = max(y - pad, 0)
                    x1 = min(x + w + pad, w_img)
                    y1 = min(y + h + pad, h_img)

                    # recorte ORIGINAL (sem upscale) para análises geométricas
                    if has_alpha:
                        crop_orig = img[y0:y1, x0:x1]
                    else:
                        crop_orig = bgr[y0:y1, x0:x1]

                    # Upscale e save visual (como você já fazia)
                    interp = cv2.INTER_CUBIC
                    up = cv2.resize(crop_orig, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=interp)

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

                    out_name = f"{self.id}_{component}_{lbl}.png"
                    out_path = os.path.join(out_dir, out_name)
                    ok = cv2.imwrite(out_path, up)
                    if ok:
                        saved_paths.append(out_path)
                        saved_count += 1

                        # ===== grava no CSV de detecções =====
                        if component == "lines":
                            ep = _estimate_line_endpoints_from_crop(
                                crop_bgr=crop_orig[:, :, :3] if (crop_orig.ndim==3 and crop_orig.shape[2]==4) else crop_orig
                            )
                            if ep is not None:
                                lx1,ly1,lx2,ly2,ang,L = ep
                                gx1, gy1 = x0 + int(lx1), y0 + int(ly1)
                                gx2, gy2 = x0 + int(lx2), y0 + int(ly2)
                                csv_writer.writerow([
                                    component, int(x0), int(y0), int(x1-x0), int(y1-y0), int(area),
                                    os.path.abspath(image_path), os.path.abspath(out_path),
                                    int(gx1), int(gy1), int(gx2), int(gy2), float(ang), float(L)
                                ])
                            else:
                                # se falhar, salva sem endpoints
                                csv_writer.writerow([
                                    component, int(x0), int(y0), int(x1-x0), int(y1-y0), int(area),
                                    os.path.abspath(image_path), os.path.abspath(out_path),
                                    "", "", "", "", "", ""
                                ])
                        else:
                            csv_writer.writerow([
                                component, int(x0), int(y0), int(x1-x0), int(y1-y0), int(area),
                                os.path.abspath(image_path), os.path.abspath(out_path),
                                "", "", "", "", "", ""
                            ])
                    else:
                        print(f"[detect_components] Falha ao salvar recorte: {out_path}")

                print(f"[detect_components] '{component}': detectadas {num_labels-1} regiões, salvas {saved_count} recortes em {out_dir}")
                results[component] = saved_paths

        finally:
            try:
                csv_file.close()
            except Exception:
                pass

        print(f"[detect_components] CSV de detecções salvo em: {detections_csv}")
        return results


