from src.stellar_system.planet.surface import  Surface
from .air_data import AirData
from .thermodynamics import Thermodynamics
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.materials import Materials

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
        self.air_data.update()
        #pressure_gradient = self.thermodynamics.calculate_pressure_gradient()
        #self.air_data.pressure_gradient = self.thermodynamics.cartesian_gradient_to_spherical(pressure_gradient)
        for layer_idx in range(self.air_data.n_layers):
            self.air_data.pressure_gradient[layer_idx,:,0] = self.surface.grad_lambda.dot(self.air_data.pressure[layer_idx])
            self.air_data.pressure_gradient[layer_idx,:,1] = self.surface.grad_phi.dot(self.air_data.pressure[layer_idx])