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
        self.history = None



    def run(self, duration, delta_t, time_between_snapshots):
        n_snapshots = int(np.ceil(duration / time_between_snapshots))
        if self.plot_type == 'orbits':
            self.history = {body.name: np.zeros((n_snapshots,)+body.position.shape, dtype=np.float64)
                            for body in self.stellar_system.bodies}
        else:
            self.history = np.zeros((n_snapshots,)+self.planet.variables[self.plot_type].shape, dtype=np.float64)
        time_since_snapshot = 0
        i_snapshot = 0
        while i_snapshot < n_snapshots:
            if time_since_snapshot >= time_between_snapshots:
                time_since_snapshot = 0
                for body in self.stellar_system.bodies:
                    if self.plot_type=='orbits':
                        self.history[body.name][i_snapshot] = body.position

                if self.plot_type != 'orbits':
                    self.history[i_snapshot] = self.planet.variables[self.plot_type]

                i_snapshot += 1

            self.stellar_system.update(delta_t)

            self.time += delta_t
            time_since_snapshot += delta_t
