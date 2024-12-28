import numpy as np

from src.stellar_system.planet.surface import  Surface
from .air_data import AirData
from .thermodynamics import Thermodynamics
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.materials import Materials

class Atmosphere:
    def __init__(self, surface: Surface, planet_mass: float, omega: np.ndarray, atmosphere_data: dict):
        self.surface = surface
        self.planet_mass = planet_mass
        self.omega = omega  # Planetary angular velocity vector
        self.atmosphere_data = atmosphere_data

        self.material = Materials.load(self.atmosphere_data['material_name'])

        self.air_data = AirData(self.surface, self.planet_mass, self.material, self.atmosphere_data)

        self.adjacency_manager = AdjacencyManager(self.air_data, self.surface.adjacency_matrix)
        self.laplacian_matrix = self.adjacency_manager.laplacian_matrix

        self.thermodynamics = Thermodynamics(self.air_data, self.adjacency_manager, self.surface, self.material)


    def update(self, delta_t):
        self.thermodynamics.exchange_heat_with_surface(delta_t)
        self.thermodynamics.conduct_heat(delta_t)
        self.air_data.update()
        for layer_idx in range(self.air_data.n_layers):
            # self.air_data.pressure_gradient[layer_idx,:,0] = self.surface.zonal_derivative.dot(self.air_data.pressure[layer_idx])
            # self.air_data.pressure_gradient[layer_idx,:,1] = self.surface.meridional_derivative.dot(self.air_data.pressure[layer_idx])
            self.air_data.pressure_gradient[layer_idx] = self.surface.vector_operators.calculate_gradient(self.air_data.pressure[layer_idx])