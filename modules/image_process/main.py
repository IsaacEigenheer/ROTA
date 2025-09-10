#######################################################################
# Módulo responsável por processar imagens
#######################################################################

import os
import fitz
from typing import Optional
import cv2
import numpy as np

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
        Retorna um dict { component: [lista_de_caminhos_salvos] }.
        """

        base_images_dir = os.path.join(self.project_root, "temp", self.id, "images")

        images = {
            "housing": os.path.join(base_images_dir, f"{self.id}_blue.png"),
            "lenghts": os.path.join(base_images_dir, f"{self.id}_green.png"),
            "lines":   os.path.join(base_images_dir, f"{self.id}_yellow.png"),
            "nodes":   os.path.join(base_images_dir, f"{self.id}_red.png"),
        }

        images_save_dir = {
            "housing": os.path.join(self.project_root, "temp", self.id, "components_extracted", "housing"),
            "lenghts": os.path.join(self.project_root, "temp", self.id, "components_extracted", "lenghts"),
            "lines":   os.path.join(self.project_root, "temp", self.id, "components_extracted", "lines"),
            "nodes":   os.path.join(self.project_root, "temp", self.id, "components_extracted", "nodes"),
        }

        results = {}
        min_area = 50       # filtra pequenos ruídos; ajuste conforme necessário
        pad = 2             # padding ao redor do bbox cortado

        # =================== [NOVO] parâmetros de upscale e borda ===================
        UPSCALE_FACTOR = 2.0                 # fator de ampliação
        BORDER_PX = 4                        # pixels de borda a adicionar após o upscale
        SHARPEN_AMOUNT = 1.5                 # nitidez para texto/linhas (sem alpha)
        SHARPEN_NEG = -0.5                   # peso do blur na máscara de nitidez
        # ===========================================================================

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

            # separa canais
            has_alpha = (img.ndim == 3 and img.shape[2] == 4)
            if has_alpha:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
            else:
                if img.ndim == 2:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = img

            # cria máscara binária: se tiver alpha, use alpha; senão threshold no cinza
            if has_alpha:
                mask = (alpha > 0).astype("uint8") * 255
            else:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # limpa ruído
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # ---------- ALTERAÇÃO pré-existente: máscara horizontal para 'lenghts' ----------
            if component == "lenghts":
                h_img, w_img = mask.shape[:2]
                kx = max(15, int(max(15, w_img * 0.002)))  # 2% da largura ou 15 px
                kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
                detect_mask = cv2.dilate(mask, kernel_horiz, iterations=2)
                detect_mask = cv2.morphologyEx(detect_mask, cv2.MORPH_CLOSE, kernel_horiz)
                work_mask = detect_mask
            else:
                work_mask = mask
            # -------------------------------------------------------------------------------

            # detecta componentes conectados (usa work_mask)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(work_mask, connectivity=8)

            saved_count = 0
            h_img, w_img = mask.shape[:2]  # dimensões da máscara original

            for lbl in range(1, num_labels):  # pula o label 0 (background)
                x, y, w, h, area = stats[lbl]
                if area < min_area:
                    continue

                # aplica padding e mantém dentro da imagem
                x0 = max(x - pad, 0)
                y0 = max(y - pad, 0)
                x1 = min(x + w + pad, w_img)
                y1 = min(y + h + pad, h_img)

                # corta a região da imagem original (preservando alpha se houver)
                if has_alpha:
                    crop = img[y0:y1, x0:x1]  # BGRA
                else:
                    crop = bgr[y0:y1, x0:x1]  # BGR

                # ====================== [NOVO] UPSCALE + TRATAMENTO DE BORDAS ======================
                # 1) Upscale (aumenta a resolução do recorte antes de salvar)
                #    Usa INTER_CUBIC para melhor qualidade em texto/linhas finas.
                interp = cv2.INTER_CUBIC
                up = cv2.resize(crop, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=interp)

                # 2) Tratamento de bordas
                if up.ndim == 3 and up.shape[2] == 4:
                    # a) Refina e suaviza levemente o canal alpha para remover "serrilhados"
                    up_bgr = up[:, :, :3]
                    up_alpha = up[:, :, 3]

                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    up_alpha = cv2.morphologyEx(up_alpha, cv2.MORPH_CLOSE, k, iterations=1)
                    up_alpha = cv2.GaussianBlur(up_alpha, (3, 3), 0)

                    up = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2BGRA)
                    up[:, :, 3] = up_alpha

                    # b) Adiciona uma borda transparente para evitar cortes no anti-aliasing
                    up = cv2.copyMakeBorder(
                        up, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
                    )
                else:
                    # a) Realça nitidez para melhorar legibilidade de texto/linhas
                    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
                    up = cv2.addWeighted(up, SHARPEN_AMOUNT, blur, SHARPEN_NEG, 0)

                    # b) Adiciona borda branca para manter afastamento visual
                    up = cv2.copyMakeBorder(
                        up, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
                        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
                    )
                # =====================================================================================

                out_name = f"{self.id}_{component}_{lbl}.png"
                out_path = os.path.join(out_dir, out_name)
                ok = cv2.imwrite(out_path, up)  # <-- salva a versão tratada e ampliada
                if ok:
                    saved_paths.append(out_path)
                    saved_count += 1
                else:
                    print(f"[detect_components] Falha ao salvar recorte: {out_path}")

            print(f"[detect_components] '{component}': detectadas {num_labels-1} regiões, salvas {saved_count} recortes em {out_dir}")
            results[component] = saved_paths

        return results
