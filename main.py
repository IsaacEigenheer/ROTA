#######################################################################

# Módulo principal, responsável por integrar todas as etapas.

#######################################################################

from modules.image_process import Image
from modules.utils import Instance
from modules.extract_data import Extract
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def main():
    instance = Instance()
    instance.create_folders(instance.id, PROJECT_ROOT)

    image = Image(
         "desenhos/whirlpool/w11340214.pdf", 300, 2, instance.id, PROJECT_ROOT
    )
    image.convert_image_to_pdf()
    image.apply_mask()
    image.detect_components()
    
    extract = Extract(instance.id, PROJECT_ROOT)
    

        # 3a) OCR dos "lenghts" (verde) -> file.csv    (formato: Cx, valor)
    #     Usa o mesmo motor configurado (paddle por padrão; cai em fallback se não disponível)
    file_csv = extract.ocr_dir_to_csv(
        ocr_engine="paddle",     # "paddle" | "easyocr" | "tesseract" | "auto"
        preprocess="thresh",
        psm=6,
        debug=True
    )
    print(f"[MAIN] lengths CSV: {file_csv}")

    # 3b) OCR de Housing (azul) e Nodes (vermelho) a partir do detections.csv (saved_crop)
    #     Saídas:
    #       - housing.csv: Hx, image_path, x, y, w, h   (Bx normalizado -> Hx)
    #       - nodes.csv:   Nx, image_path, x, y, w, h
    housing_csv, nodes_csv = extract.ocr_components_to_csv(
        ocr_engine="paddle",
        preprocess="thresh",
        psm=7,          # ligeiramente mais "linha única" para rótulos curtos
        debug=True
    )
    print(f"[MAIN] housing CSV: {housing_csv}")
    print(f"[MAIN] nodes   CSV: {nodes_csv}")

    # 4) Associação de linhas a componentes:
    #    Lê detections.csv (com x1,y1,x2,y2), housing.csv e nodes.csv.
    #    Gera connections.csv: LineID, StartComponent, EndComponent, StartSide, EndSide, x1,y1,x2,y2
    connections_csv = extract.find_line_connections(
        debug=True,
        point_radius=3   # raio do quadradinho para desempate por maior interseção
    )
    print(f"[MAIN] connections CSV: {connections_csv}")

    print("[DONE] Pipeline completo.")

if __name__ == "__main__":
    main()