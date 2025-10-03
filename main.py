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
    image.detect_components(
    debug_lines=True,
    debug_out_dir=None,   # usa temp/<id>/debug/line_detect
    debug_show=False,     # True para abrir janelas do OpenCV
    debug_wait_ms=0,      # tempo de espera nas janelas (se debug_show=True)
    debug_limit=None      # ou um int para limitar quantos recortes debugar
)
    
    extract = Extract(instance.id, PROJECT_ROOT)
    
    file_csv = extract.ocr_dir_to_csv(
        ocr_engine="paddle",     
        preprocess="thresh",
        psm=6,
        debug=True
    )
    print(f"[MAIN] lengths CSV: {file_csv}")

if __name__ == "__main__":
    main()