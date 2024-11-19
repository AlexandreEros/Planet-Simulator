import numpy as np
import json
from scipy import constants
from stellar_system import StellarSystem
from star import Star

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
        self.sunlight_vector_history = {body.name: np.ndarray((self.n_snapshots, 3), dtype=np.float64)
                                 for body in self.stellar_system.bodies if body.body_type=='planet'}
        self.angle_history = {body.name: np.ndarray((self.n_snapshots,), dtype=np.float64)
                                 for body in self.stellar_system.bodies if body.body_type=='planet'}

        self.irradiance_history = {body.name: np.ndarray((self.n_snapshots,len(body.surface.irradiance)), dtype=np.float64)
                                 for body in self.stellar_system.bodies if body.body_type=='planet'}
        self.temperature_history = {body.name: np.ndarray((self.n_snapshots,len(body.surface.temperature)), dtype=np.float64)
                                 for body in self.stellar_system.bodies if body.body_type=='planet'}


    def load_bodies_from_file(self, body_file: str):
        """Load celestial bodies from a JSON file and add them to the system."""
        with open(body_file, 'r') as f:
            data = json.load(f)
            for body_data in data['bodies']:
                try:
                    self.stellar_system.add_body(**body_data)
                except Exception as err:
                    raise Exception(f"Error creating the {body_data['body_type']} '{body_data['name']}':\n{err}")


    def run(self):
        for i_step in range(self.n_steps):
            if i_step % self.steps_between_snapshots == 0:
                i_snapshot = i_step // self.steps_between_snapshots
                for body in self.stellar_system.bodies:
                    self.position_history[body.name][i_snapshot] = body.position
                    self.velocity_history[body.name][i_snapshot] = body.velocity
                    if body.body_type == 'planet':
                        # self.sunlight_vector_history[body.name][i_snapshot] = body.sunlight
                        # self.angle_history[body.name][i_snapshot] = body.current_angle
                        # self.irradiance_history[body.name][i_snapshot] = body.surface.irradiance
                        self.temperature_history[body.name][i_snapshot] = body.surface.temperature
                self.total_angular_momentum_history[i_snapshot] = self.stellar_system.current_total_angular_momentum

            self.time += self.delta_t
            self.stellar_system.update(self.delta_t)

        # total_angular_momentum_mag = np.linalg.norm(self.total_angular_momentum_history, axis=-1)
        # min_angm = np.amin(total_angular_momentum_mag)
        # max_angm = np.amax(total_angular_momentum_mag)
        # print(f"Angular momentum: min = {min_angm},  max = {max_angm}  "
        #       f"({100.0 * (max_angm - min_angm) / min_angm}% error)")
        #
        # print("Sunlight vector for Earth's first day:")
        # for isnap in range(self.n_snapshots):
        #     coords = subsolar_point(-self.sunlight_vector_history['Earth'][isnap])
        #     print(f"{isnap+1} h: ({coords[0]:.2f}, {coords[1]:.2f})")



def subsolar_point(sunlight_vector: np.ndarray) -> tuple[float, float]:
    """
    Given a sunlight vector, return the latitude and longitude of the subsolar point.

    :param sunlight_vector: The sunlight vector (must be normalized, i.e., unit length)
    :return: (latitude in degrees, longitude in degrees)
    """
    sunlight_vector /= np.linalg.norm(sunlight_vector)
    x, y, z = tuple(sunlight_vector.tolist())

    # Latitude (in degrees)
    latitude = np.rad2deg(np.arcsin(z))  # arcsin(z) gives latitude in radians

    # Longitude (in degrees)
    longitude = np.rad2deg(np.arctan2(y, x))  # atan2(y, x) gives longitude in radians

    return float(latitude), float(longitude)
