import numpy as np

from src.stellar_system.planet.surface import  Surface
from .reference_atmosphere import ReferenceAtmosphere
from .thermodynamics import Thermodynamics
from .adjacency_manager import AdjacencyManager
from .air_flow import AirFlow
from src.stellar_system.planet.materials import Materials

class Atmosphere:
    def __init__(self, surface: Surface, planet_mass: float, omega: np.ndarray, atmosphere_data: dict):
        self.surface = surface
        self.planet_mass = planet_mass
        self.omega = omega  # Planetary angular velocity vector
        self.atmosphere_data = atmosphere_data

        self.material = Materials.load(self.atmosphere_data['material_name'])

        self.ref = ReferenceAtmosphere(self.surface, self.planet_mass, self.material, self.atmosphere_data)
        self.adjacency_manager = AdjacencyManager(self.ref, self.surface.adjacency_matrix)
        self.air_flow = AirFlow(self.ref, self.surface, self.adjacency_manager, self.omega)

        self.thermodynamics = Thermodynamics(self.air_flow, self.ref, self.adjacency_manager, self.surface, self.material)


    def update(self, delta_t):
        self.thermodynamics.exchange_heat_with_surface(delta_t)
        self.thermodynamics.conduct_heat(delta_t)
        self.air_flow.rk4_step(delta_t)