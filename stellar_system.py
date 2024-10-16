import numpy as np
import scipy as sp
from celestial_body import CelestialBody

class StellarSystem:
    def __init__(self, G: float):
        self.G = G
        self.bodies: list[CelestialBody] = []


    @property
    def positions(self) -> dict[str, list[float]]:
        return {body.name: body.position.tolist() for body in self.bodies}

    @property
    def velocities(self) -> dict[str, list[float]]:
        return {body.name: body.velocity.tolist() for body in self.bodies}


    def add_body(self, **kwargs) -> None:
        kws = kwargs.keys()
        if 'position' in kws and 'velocity' in kws:
            self.bodies.append(CelestialBody(**kwargs))
        elif 'orbital_period' in kws:
            if 'eccentricity' not in kws: kwargs['eccentricity'] = 0.0
            if 'year_percentage' not in kws: kwargs['year_percentage'] = 0.0
            kwargs['position'], kwargs['velocity'] = \
                self.get_start_vectors(kwargs['orbital_period'], 2*np.pi*kwargs['year_percentage'], kwargs['eccentricity'],
                                       self.bodies[0].mass, self.G)
                # self.get_periapsis_vectors(kwargs['orbital_period'], kwargs['eccentricity'], self.bodies[0].mass, self.G)
            self.bodies.append(CelestialBody(**kwargs))
        else:
            raise Exception(f"Either 'position' and 'velocity' or 'orbital_period' must be given for "
                            f"each planet, but {kwargs['name']} does not meet these conditions.")

    @staticmethod
    def get_periapsis_vectors(T: float, e: float, M: float, G: float = 6.67430e-11) -> (np.ndarray, np.ndarray):
        # T: Orbital period
        # e: Orbit eccentricity

        # From Kepler's Third law T**2 = (4 * pi**2 * a**3) / (G * M),
        # Where 'a' is the semi-major axis, or apoapsis:
        apoapsis = float(np.cbrt((G * M) / (2 * np.pi / T)**2))
        periapsis = (1 - e) * apoapsis

        # From the vis-viva equation v = sqrt(G*M * (2/r - 1/a)):
        max_speed = float(np.sqrt(G*M * (2/periapsis - 1/apoapsis)))

        # Let's assume the object always starts at periapsis, which is always on the positive x-axis:
        initial_position = np.array([periapsis, 0., 0.], dtype=np.float64)
        initial_velocity = np.array([0., max_speed, 0.], dtype=np.float64)
        return initial_position, initial_velocity

    @staticmethod
    def get_semimajor_axis(T: float, M: float, G: float = 6.67430e-11):
        # From Kepler's Third law T**2 = (4 * pi**2 * a**3) / (G * M),
        # Where 'a' is the semi-major axis, or apoapsis:
        apoapsis = float(np.cbrt((G * M) / (2 * np.pi / T)**2))
        return apoapsis

    @staticmethod
    def get_start_vectors(T, mean_anomaly, e, mass, G = 6.67430e-11) -> (np.ndarray, np.ndarray):
        a = StellarSystem.get_semimajor_axis(T, mass, G)

        eccentric_anomaly = sp.optimize.newton(lambda E: mean_anomaly - E + e * np.sin(E), 0)
        true_anomaly = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(eccentric_anomaly/2))
        distance = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))

        radial_velocity_mag = np.sqrt(G*mass/a) * (e*np.sin(true_anomaly)) / np.sqrt(1 - e**2)
        speed = float(np.sqrt(G*mass * (2/distance - 1/a)))
        transverse_velocity_mag = np.sqrt(speed**2 - radial_velocity_mag**2)
        # = np.sqrt((G*mass/a) * (1 + e*np.cos(true_anomaly)) / (1 - e**2))

        radial_vec = np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.], dtype=np.float64)
        transverse_vec = np.array([-np.sin(true_anomaly), np.cos(true_anomaly), 0.], dtype=np.float64)
        position = distance * radial_vec
        velocity = radial_velocity_mag * radial_vec + transverse_velocity_mag * transverse_vec
        return position, velocity

    @staticmethod
    def get_gravity(body: CelestialBody, all_bodies: list[CelestialBody], G: float = 6.67430e-11) -> np.ndarray[np.float64]:
        """
        Calculate the net gravitational acceleration acting on a celestial body.

        :param body: The celestial body for which to compute the gravitational force.
        :param all_bodies: A list of all celestial bodies in the system.
        :param G: Gravitational constant (default is the standard value in m^3 kg^-1 s^-2).
        :return: The net gravitational acceleration vector (in m/s^2).
        :raises ValueError: If a collision (zero distance) is detected between two bodies.
        """
        if body not in all_bodies:
            raise ValueError(f"{body.name} is not in the list passed as the argument all_bodies")

        other_positions = np.array([bd.position for bd in all_bodies if bd is not body]) # shape=(n_bodies-1, 3)
        other_masses = np.array([bd.mass for bd in all_bodies if bd is not body])
        if len(other_positions) == len(all_bodies):
            other_names = np.array([bd.name for bd in all_bodies if bd is not body])
            raise Exception(f"The list of other bodies is as long as all_bodies: {other_names}")

        r_vecs = other_positions - body.position
        distances = np.linalg.norm(r_vecs, axis=-1)

        # Check for zero distances (collision detection)
        zero_distance_idx = np.where(distances == 0)[0]
        if zero_distance_idx.size > 0:
            colliding_names = [all_bodies[i].name for i in zero_distance_idx]
            raise ValueError(f"Collision detected between {body.name} and {colliding_names}")

        u_vecs = r_vecs / distances[:, None]  # Normalize direction vectors

        gravities = (G * other_masses / distances ** 2)[:, None] * u_vecs
        return np.sum(gravities, axis=0)  # Net gravity


    def update(self, timestep: float) -> None:
        # Leapfrog integration
        for body in self.bodies:
            # Step 1: Update velocity by half a time step
            acc = self.get_gravity(body, self.bodies, self.G)
            body.accelerate(0.5 * acc * timestep)

            # Step 2: Update position by a full time step
            body.move(body.velocity * timestep)

        # Step 3: Update velocity by half a time step again
        for body in self.bodies:
            acc = self.get_gravity(body, self.bodies, self.G)
            body.accelerate(0.5 * acc * timestep)
