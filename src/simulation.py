import cupy as cp
import numpy as np
from scipy import constants

from .stellar_system import StellarSystem

class Simulation:
    def __init__(self, plot_type: str, planet_name: str, body_file: str):
        self.plot_type = plot_type

        self.G = constants.G

        self.stellar_system = StellarSystem(planet_name, body_file, self.G)
        self.planet = self.stellar_system.planet

        self.time = 0.0


    def run(self, duration, delta_t, time_between_snapshots):
        n_snapshots = int(np.ceil(duration / time_between_snapshots))

        self.position_history = {body.name: np.ndarray((n_snapshots, 3), dtype=np.float64)
                                 for body in self.stellar_system.bodies}

        if self.plot_type=='irradiance':
            self.irradiance_history = np.ndarray((n_snapshots,len(self.planet.surface.irradiance)), dtype=np.float64)
        if self.plot_type=='temperature':
            self.temperature_history = np.ndarray((n_snapshots,len(self.planet.surface.temperature)), dtype=np.float64)
        if self.plot_type=='heat':
            self.heat_history = np.ndarray((n_snapshots,len(self.planet.surface.temperature)), dtype=np.float64)
        if self.plot_type=='air_temperature':
            self.air_temperature_history = np.ndarray((n_snapshots,)+self.planet.atmosphere.adjacency_manager.atmosphere_shape, dtype=np.float64)
        if self.plot_type=='pressure':
            self.pressure_history = np.ndarray((n_snapshots,)+self.planet.atmosphere.adjacency_manager.atmosphere_shape, dtype=np.float64)
        if self.plot_type=='density':
            self.density_history = np.ndarray((n_snapshots,)+self.planet.atmosphere.adjacency_manager.atmosphere_shape, dtype=np.float64)

        time_since_snapshot = 0
        i_snapshot = 0
        while i_snapshot < n_snapshots:
            if time_since_snapshot >= time_between_snapshots:
                time_since_snapshot = 0
                for body in self.stellar_system.bodies:
                    if self.plot_type=='orbits':
                        self.position_history[body.name][i_snapshot] = body.position.asnumpy()

                if self.plot_type=='irradiance':
                    self.irradiance_history[i_snapshot] = self.planet.surface.irradiance.asnumpy()
                if self.plot_type=='temperature':
                    self.temperature_history[i_snapshot] = self.planet.surface.temperature.asnumpy()
                if self.plot_type=='heat':
                    self.heat_history[i_snapshot] = self.planet.surface.surface_heat_flux().asnumpy()
                if self.plot_type=='air_temperature':
                    self.air_temperature_history[i_snapshot] = self.planet.atmosphere.air_data.temperature.asnumpy()
                if self.plot_type=='pressure':
                    self.pressure_history[i_snapshot] = self.planet.atmosphere.air_data.pressure.asnumpy()
                if self.plot_type=='density':
                    self.density_history[i_snapshot] = self.planet.atmosphere.air_data.density.asnumpy()

                i_snapshot += 1

            self.stellar_system.update(delta_t)

            self.time += delta_t
            time_since_snapshot += delta_t
