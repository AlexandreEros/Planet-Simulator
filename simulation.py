import numpy as np
import pandas as pd
from scipy import constants
import matplotlib.pyplot as plt
from stellar_system import StellarSystem

class Simulation:
    def __init__(self, timestep: float = 3600.0, n_steps: int = 2400.0):
        self.delta_t = timestep
        self.total_t = self.delta_t * n_steps
        self.time = 0.0

        self.G = constants.G
        self.Boltzmann = constants.Boltzmann

        self.stellar_system = StellarSystem(self.G)

        sun_dict = {
            'name': "Sun",
            'mass': 1.989e30,
            'position': np.array([0, 0, 0]).astype(np.float64),
            'velocity': np.array([0, 0, 0]).astype(np.float64),
        }
        earth_dict = {
            'name': "Earth",
            'mass': 5.972e24,
            'position': np.array([1.496e11, 0, 0]).astype(np.float64),
            'velocity': np.array([0, 29780, 0]).astype(np.float64),
        }
        self.stellar_system.add_body(**sun_dict)
        self.stellar_system.add_body(**earth_dict)

        self.position_history: list[dict[str, list[float]]] = []


    def run(self):
        while self.time < self.total_t:
            self.time += self.delta_t
            self.stellar_system.update(self.delta_t)
            self.position_history.append(self.stellar_system.positions)


    def plot_orbits(self):
        # Create a figure for the plot
        plt.figure(figsize=(10, 6))

        # Extract positions for Sun and Earth from provided history
        sun_x, sun_y = [], []
        earth_x, earth_y = [], []

        for entry in self.position_history:
            sun_x.append(entry['Sun'][0])
            sun_y.append(entry['Sun'][1])
            earth_x.append(entry['Earth'][0])
            earth_y.append(entry['Earth'][1])

        # Plotting the Sun's trajectory
        plt.plot(sun_x, sun_y, 'o-', label='Sun', color='gold')

        # Plotting the Earth's trajectory
        plt.plot(earth_x, earth_y, 'o-', label='Earth', color='blue')

        # Setting labels and title
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Orbital Trajectories")
        plt.legend()
        plt.axis('equal')

        # Display the plot
        plt.show()