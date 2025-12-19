import os
import json

path_parts = __file__.split(os.path.sep)
base_dir = os.path.sep.join(path_parts[:-4])
default_materials = os.path.join(base_dir, 'data', 'materials.json')

class Materials:
    @staticmethod
    def load(material_name):
        with open(default_materials, 'r') as f:
            materials = json.load(f)['materials']
            return next(m for m in materials if m['name'] == material_name)