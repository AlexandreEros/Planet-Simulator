from .surface import  Surface
from .air_data import AirData
from .thermodynamics import Thermodynamics
from .adjacency_manager import AdjacencyManager
from .materials import Materials

class Atmosphere:
    def __init__(self, surface: Surface, planet_mass: float, atmosphere_data: dict):
        self.surface = surface
        self.planet_mass = planet_mass
        self.atmosphere_data = atmosphere_data

        self.material = Materials.load(self.atmosphere_data['material_name'])

        self.air_data = AirData(self.surface, self.planet_mass, self.material, self.atmosphere_data)

        self.adjacency_manager = AdjacencyManager(self.air_data, self.surface.adjacency_matrix)
        self.laplacian_matrix = self.adjacency_manager.laplacian_matrix

        self.thermodynamics = Thermodynamics(self.air_data, self.adjacency_manager, self.surface, self.material)


    def update(self, delta_t):
        self.thermodynamics.exchange_heat_with_surface(delta_t)
        self.thermodynamics.conduct_heat(delta_t)
