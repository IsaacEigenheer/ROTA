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
    extract.ocr_dir_to_csv()

if __name__ == "__main__":
    main()