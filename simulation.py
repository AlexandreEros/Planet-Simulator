import numpy as np
import json
from scipy import constants
from stellar_system import StellarSystem

class Simulation:
    def __init__(self, timestep: float, n_steps: int, steps_between_snapshots: int = 1, body_file = 'bodies.json'):
        self.delta_t = timestep
        self.n_steps = n_steps
        self.steps_between_snapshots = steps_between_snapshots
        self.n_snapshots = int(np.ceil(self.n_steps / self.steps_between_snapshots))

        self.G = constants.G
        self.Boltzmann = constants.Boltzmann

        self.stellar_system = StellarSystem(self.G)
        self.load_bodies_from_file(body_file)

        self.time = 0.0
        self.position_history = {body.name: np.ndarray((self.n_snapshots, 3), dtype=np.float64)
                                 for body in self.stellar_system.bodies}
        self.velocity_history = {body.name: np.ndarray((self.n_snapshots, 3), dtype=np.float64)
                                 for body in self.stellar_system.bodies}
        self.total_angular_momentum_history = np.ndarray((self.n_snapshots, 3), dtype=np.float64)


    def load_bodies_from_file(self, body_file: str):
        """Load celestial bodies from a JSON file and add them to the system."""
        with open(body_file, 'r') as f:
            data = json.load(f)
            for body_data in data['bodies']:
                self.stellar_system.add_body(**body_data)


    def run(self):
        for i_step in range(self.n_steps):
            self.time += self.delta_t
            self.stellar_system.update(self.delta_t)
            if i_step % self.steps_between_snapshots == 0:
                i_snapshot = i_step // self.steps_between_snapshots
                for body in self.stellar_system.bodies:
                    self.position_history[body.name][i_snapshot] = body.position
                    self.velocity_history[body.name][i_snapshot] = body.velocity
                self.total_angular_momentum_history[i_snapshot] = self.stellar_system.current_total_angular_momentum

        total_angular_momentum_mag = np.linalg.norm(self.total_angular_momentum_history, axis=-1)
        min_angm = np.amin(total_angular_momentum_mag)
        max_angm = np.amax(total_angular_momentum_mag)
        print(f"Angular momentum: min = {min_angm},  max = {max_angm}  "
              f"({100.0 * (max_angm - min_angm) / min_angm}% error)")
