import numpy as np
from scipy import constants
import json
import os

from .core.stellar_system import StellarSystem

class Simulation:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_bodies = os.path.join(base_dir, 'data', 'bodies.json')

    def __init__(self, plot_type: str, planet_name: str, timestep: float, n_steps: int, steps_between_snapshots: int = 1,
                 body_file = default_bodies):
        self.delta_t = timestep
        self.n_steps = n_steps
        self.steps_between_snapshots = steps_between_snapshots
        self.n_snapshots = int(np.ceil(self.n_steps / self.steps_between_snapshots))

        self.plot_type = plot_type
        self.planet_name = planet_name

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
        for i_step in range(self.n_steps):
            if i_step % self.steps_between_snapshots == 0:
                i_snapshot = i_step // self.steps_between_snapshots

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

            self.time += self.delta_t
            self.stellar_system.update(self.delta_t)
