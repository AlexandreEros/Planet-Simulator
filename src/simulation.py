import numpy as np
from scipy import constants

from .models.stellar_system import StellarSystem

class Simulation:
    def __init__(self, plot_type: str, planet_name: str, duration_sec: int, timestep_sec: float, time_between_snapshots_sec: float,
                 body_file: str):
        self.plot_type = plot_type
        self.planet_name = planet_name
        self.delta_t = timestep_sec
        self.duration = duration_sec
        self.time_between_snapshots = time_between_snapshots_sec

        self.n_snapshots = int(np.ceil(self.duration / self.time_between_snapshots))

        self.G = constants.G

        self.stellar_system = StellarSystem(self.planet_name, body_file, self.G)
        self.planet = self.stellar_system.planet
        if self.planet is None and self.plot_type != 'orbits':
            raise Exception(f"A valid celestial body must be given for the plot_type {self.plot_type}, and "
                            f"'{self.planet_name}' is not one.")

        self.time = 0.0

        if self.plot_type=='orbits':
            self.position_history = {body.name: np.ndarray((self.n_snapshots, 3), dtype=np.float64)
                                     for body in self.stellar_system.bodies}
        self.sunlight_vector_history = {body.name: np.ndarray((self.n_snapshots, 3), dtype=np.float64)
                                 for body in self.stellar_system.bodies if body.body_type=='planet'}
        if self.plot_type=='irradiance':
            self.irradiance_history = np.ndarray((self.n_snapshots,len(self.planet.surface.irradiance)), dtype=np.float64)
        if self.plot_type=='temperature':
            self.temperature_history = np.ndarray((self.n_snapshots,len(self.planet.surface.temperature)), dtype=np.float64)
        if self.plot_type=='heat':
            self.heat_history = np.ndarray((self.n_snapshots,len(self.planet.surface.temperature)), dtype=np.float64)


    def run(self):
        time_since_snapshot = 0
        i_snapshot = 0
        while i_snapshot < self.n_snapshots:
            if time_since_snapshot >= self.time_between_snapshots:
                time_since_snapshot = 0
                for body in self.stellar_system.bodies:
                    if self.plot_type=='orbits':
                        self.position_history[body.name][i_snapshot] = body.position
                    if body.body_type == 'planet':
                        self.sunlight_vector_history[body.name][i_snapshot] = body.sunlight

                if self.plot_type=='irradiance':
                    self.irradiance_history[i_snapshot] = self.planet.surface.irradiance
                if self.plot_type=='temperature':
                    self.temperature_history[i_snapshot] = self.planet.surface.temperature
                if self.plot_type=='heat':
                    self.heat_history[i_snapshot] = self.planet.surface.surface_heat_flux()

                i_snapshot += 1

            self.stellar_system.update(self.delta_t)

            self.time += self.delta_t
            time_since_snapshot += self.delta_t
