from datetime import datetime
import os

class Instance:
    def __init__(self):
        self.id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    def create_folders(self, id: int, project_root: str):
        outputs = {
            "root_folder": os.path.join(project_root, "temp", str(id)),
            "images": os.path.join(project_root, "temp", str(id), "images"),
            "components": os.path.join(project_root, "temp", str(id), "components_extracted"),
            "components_housing": os.path.join(project_root, "temp", str(id), "components_extracted", "housing"),
            "components_lenghts": os.path.join(project_root, "temp", str(id), "components_extracted", "lenghts"),
            "components_lines": os.path.join(project_root, "temp", str(id), "components_extracted", "lines"),
            "components_nodes": os.path.join(project_root, "temp", str(id), "components_extracted", "nodes"),
        }
        for dir in outputs.values():
            os.makedirs(dir)