import numpy as np
import pandas as pd
from scipy import constants
import matplotlib.pyplot as plt
from stellar_system import StellarSystem

class Simulation:
    def __init__(self, timestep: float = 3600.0, n_steps: int = 2400.0, steps_between_snapshots: int = 8):
        self.delta_t = timestep
        self.n_steps = n_steps
        self.steps_between_snapshots = steps_between_snapshots
        self.n_snapshots = int(np.ceil(self.n_steps / self.steps_between_snapshots))

        self.time = 0.0

        self.G = constants.G
        self.Boltzmann = constants.Boltzmann

        self.stellar_system = StellarSystem(self.G)

        sun_dict = {
            'name': "Sun",
            'mass': 1.989e30,
            'position': np.array([0, 0, 0]).astype(np.float64),
            'velocity': np.array([0, 0, 0]).astype(np.float64),
            'color': 'gold'
        }
        earth_dict = {
            'name': "Earth",
            'mass': 5.972e24,
            'position': np.array([1.496e11, 0, 0]).astype(np.float64),
            'velocity': np.array([0, 29780, 0]).astype(np.float64),
            'color': 'blue'
        }
        self.stellar_system.add_body(**sun_dict)
        self.stellar_system.add_body(**earth_dict)

        self.position_history = {body.name: np.ndarray((self.n_snapshots, 3), dtype=np.float64)
                                 for body in self.stellar_system.bodies}


    def run(self):
        for i_step in range(self.n_steps):
            self.time += self.delta_t
            self.stellar_system.update(self.delta_t)
            if i_step % self.steps_between_snapshots == 0:
                i_snapshot = i_step // self.steps_between_snapshots
                for body in self.stellar_system.bodies:
                    self.position_history[body.name][i_snapshot] = body.position
