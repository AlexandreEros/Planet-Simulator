import numpy as np
from scipy.spatial.transform import Rotation

class CelestialBody:
    def __init__(self, name: str, body_type: str, mass: float, color: str,
                 orbital_data: dict, parent_mass: float = 0.0, parent_position = np.zeros(3), parent_velocity = np.zeros(3)):
        self.name = name
        self.body_type = body_type
        self.mass = mass
        self.color = color

        self.orbital_data = orbital_data

        self.orbital_period = -1 if 'orbital_period' not in self.orbital_data else self.orbital_data['orbital_period']
        self.eccentricity = 0.0 if 'eccentricity' not in self.orbital_data else self.orbital_data['eccentricity']
        self.argument_of_perihelion = 0.0 if 'argument_of_perihelion_deg' not in self.orbital_data else np.radians(self.orbital_data['argument_of_perihelion_deg'])
        self.true_anomaly = 0.0 if 'true_anomaly_deg' not in self.orbital_data else np.radians(self.orbital_data['true_anomaly_deg'])
        self.inclination = 0.0 if 'inclination_deg' not in self.orbital_data else np.radians(self.orbital_data['inclination_deg'])
        self.lon_ascending_node = 0.0 if 'lon_ascending_node_deg' not in self.orbital_data else np.radians(self.orbital_data['lon_ascending_node_deg'])
        if 'position' in self.orbital_data and 'velocity' in self.orbital_data:
            self.position = np.array(self.orbital_data['position'], dtype=np.float64)
            self.velocity = np.array(self.orbital_data['velocity'], dtype=np.float64)
        else:
            self.position, self.velocity = self.get_start_vectors(self.orbital_period,
                    self.true_anomaly, self.eccentricity, self.argument_of_perihelion, self.inclination,
                    self.lon_ascending_node, parent_mass, parent_position, parent_velocity)

            self.semi_major_axis = self.get_semi_major_axis(self.orbital_period, parent_mass)

        ascending_node_vec = np.array([np.cos(self.lon_ascending_node), np.sin(self.lon_ascending_node), 0.0])
        self.inclination_matrix = Rotation.from_rotvec(ascending_node_vec * self.inclination).as_matrix()

        self.net_force = np.zeros((3,), dtype = np.float64)


    def apply_force(self, force):
        self.net_force = force

    def accelerate(self, delta_t):
        self.velocity += self.net_force * delta_t / self.mass

    def move(self, delta_t) -> None:
        self.position += self.velocity * delta_t

    @property
    def current_angular_momentum(self):
        return np.cross(self.position, self.mass * self.velocity)


    @staticmethod
    def get_specific_angular_momentum(T: float, e: float, M: float, G: float = 6.67430e-11) -> float:
        # From the vis-viva equation v = sqrt(G*M * (2/r - 1/a)):
        # Where 'r' is the distance at any point and 'a' is the semi-major axis - ie, mean distance
        try:
            mean_distance = CelestialBody.get_semi_major_axis(T, M, G)
            apoapsis = (1+e) * mean_distance
            # Vis-viva equation:
            speed_at_apoapsis = np.sqrt(G*M * (2*mean_distance - apoapsis) / (apoapsis*mean_distance))
            specific_angular_momentum = float(apoapsis * speed_at_apoapsis)
            return specific_angular_momentum
        except Exception as err:
            raise Exception(f"Error in `get_specific_angular_momentum`: {err}")

    @staticmethod
    def get_semi_major_axis(T: float, M: float, G: float = 6.67430e-11):
        # From Kepler's Third law T**2 = (4 * pi**2 * rm**3) / (G * M), where 'rm' is the mean distance:
        try:
            mean_distance = np.cbrt(G * M / (2 * np.pi / T) ** 2)
            return float(mean_distance)
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Division by zero in `StellarSystem.get_semi_major_axis`: `T` (orbital period) is 0 or inf.")
        except ValueError as err: raise ValueError(f"Error in `StellarSystem.get_semi_major_axis`:\n{err}")


    @staticmethod
    def get_start_vectors(orbital_period: float, true_anomaly: float, eccentricity: float, argument_of_perihelion: float,
                          inclination: float, lon_ascending_node: float, parent_mass: float,
                          parent_position = np.zeros(3), parent_velocity = np.zeros(3), G = 6.67430e-11)\
            -> (np.ndarray, np.ndarray):
        """
        Given orbital parameters, return the celestial body's initial position and velocity

        :param orbital_period: Orbital period in seconds
        :param true_anomaly: True anomaly (angular position in the ecliptic coordinate system) in radians
        :param eccentricity: Orbit eccentricity
        :param argument_of_perihelion: Argument of the perihelion in radians
        :param inclination: Orbit inclination in radians
        :param lon_ascending_node: Longitude of the ascending node in radians
        :param parent_mass: Mass of the object assumed to be at the focus of this orbit, in kg
        :param parent_position: (optional) np.ndarray of shape (3) with the position of the parent object
        :param parent_velocity: (optional) np.ndarray of shape (3) with the velocity of the parent object
        :param G: (optional) The Newtonian gravity constant
        :return: position, velocity
        """

        mean_distance = CelestialBody.get_semi_major_axis(orbital_period, parent_mass, G)
        distance = mean_distance * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))

        radial_velocity_mag = np.sqrt(G*parent_mass/mean_distance) * (eccentricity*np.sin(true_anomaly)) / np.sqrt(1 - eccentricity**2)
        specific_angular_momentum = CelestialBody.get_specific_angular_momentum(orbital_period, eccentricity, parent_mass, G)
        transverse_velocity_mag = specific_angular_momentum / distance
        # speed = float(np.sqrt(G*parent_mass * (2*mean_distance - distance) / (distance*mean_distance)))

        radial_vec = np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.], dtype=np.float64)
        transverse_vec = np.array([-np.sin(true_anomaly), np.cos(true_anomaly), 0.], dtype=np.float64)

        z_axis = np.array([0, 0, 1], dtype = np.float64)

        position = distance * radial_vec
        velocity = radial_velocity_mag * radial_vec + transverse_velocity_mag * transverse_vec
        position = Rotation.from_rotvec(z_axis * argument_of_perihelion).apply(position)
        velocity = Rotation.from_rotvec(z_axis * argument_of_perihelion).apply(velocity)

        ascending_node_axis = np.array([np.cos(lon_ascending_node), np.sin(lon_ascending_node), 0.0])
        position = Rotation.from_rotvec(ascending_node_axis * inclination).apply(position)
        velocity = Rotation.from_rotvec(ascending_node_axis * inclination).apply(velocity)

        position += parent_position
        velocity += parent_velocity
        return position, velocity