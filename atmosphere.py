import json

from surface import  Surface
from air_data import AirData
from thermodynamics import Thermodynamics
from adjacency_manager import AdjacencyManager

class Atmosphere:
    def __init__(self, surface: Surface, planet_mass: float, atmosphere_data: dict):
        self.surface = surface
        self.planet_mass = planet_mass
        self.atmosphere_data = atmosphere_data

        material_name = self.atmosphere_data['material_name']
        self.material = self.load_material(material_name)

        # Delegate layer setup to LayerManager
        self.layer_manager = AirData(self.surface, self.planet_mass, self.material, self.atmosphere_data)

        # Delegate adjacency handling
        self.adjacency_manager = AdjacencyManager(self.layer_manager, self.surface.adjacency_matrix)
        self.laplacian_matrix = self.adjacency_manager.laplacian_matrix

        # Delegate heat dynamics to Thermodynamics
        self.thermodynamics = Thermodynamics(self.layer_manager, self.adjacency_manager, self.surface, self.material)


    @staticmethod
    def load_material(material_name):
        with open('materials.json', 'r') as f:
            materials = json.load(f)['materials']
            return next(m for m in materials if m['name'] == material_name)

    def update(self, delta_t):
        self.thermodynamics.exchange_heat_with_surface(delta_t)
        self.thermodynamics.conduct_heat(delta_t)
