import numpy as np
import scipy as sp
from celestial_body import CelestialBody
from planet import Planet
from star import Star
from vector_utils import rotate_vector, deg2rad

class StellarSystem:
    def __init__(self, planet_name: str, G: float):
        self.G = G
        self.planet_name = planet_name
        self.planet: Planet
        self.bodies: list[Star | Planet] = []


    @property
    def positions(self) -> dict[str, list[float]]:
        return {body.name: body.position.tolist() for body in self.bodies}

    @property
    def velocities(self) -> dict[str, list[float]]:
        return {body.name: body.velocity.tolist() for body in self.bodies}


    def add_body(self, **kwargs) -> None:
        kws = kwargs.keys()
        for kw in kws:
            if isinstance(kwargs[kw], list):
                kwargs[kw] = np.array(kwargs[kw], dtype = np.float64)
        if 'position' in kws and 'velocity' in kws:
            pass
        elif 'orbital_period' in kws:
            if 'eccentricity' not in kws: kwargs['eccentricity'] = 0.0
            if 'year_percentage' not in kws: kwargs['year_percentage'] = 0.0
            if 'argument_of_perihelion_deg' not in kws: kwargs['argument_of_perihelion_deg'] = 0.0
            if 'inclination_deg' not in kws: kwargs['inclination_deg'] = 0.0
            if 'lon_ascending_node_deg' not in kws: kwargs['lon_ascending_node_deg'] = 0.0
            kwargs['position'], kwargs['velocity'] = \
                self.get_start_vectors(kwargs['orbital_period'], kwargs['year_percentage'], kwargs['eccentricity'],
                                       kwargs['argument_of_perihelion_deg'], kwargs['inclination_deg'], kwargs['lon_ascending_node_deg'],
                                       self.bodies[0].mass, self.G)

        else:
            raise Exception(f"Either 'position' and 'velocity' or 'orbital_period' must be given for "
                            f"each planet, but {kwargs['name']} does not meet these conditions.")

        if kwargs['body_type']=='star':
            self.bodies.append(Star(**kwargs))
        elif kwargs['body_type']=='planet':
            self.bodies.append(Planet(**kwargs))
        else:
            raise KeyError(f"Unsupported body_type: {kwargs['body_type']}")

        if kwargs['name']==self.planet_name:
            self.planet = self.bodies[-1]



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
    def get_specific_angular_momentum(T: float, e: float, M: float, G: float = 6.67430e-11) -> float:
        # From the vis-viva equation v = sqrt(G*M * (2/r - 1/a)):
        # Where 'r' is the distance at any point and 'a' is the semi-major axis - ie, mean distance
        try:
            mean_distance = StellarSystem.get_semi_major_axis(T, M, G)
            apoapsis = (1+e) * mean_distance
            # Vis-viva equation:
            speed_at_apoapsis = np.sqrt(G*M * (2*mean_distance - apoapsis) / (apoapsis*mean_distance))
            specific_angular_momentum = float(apoapsis * speed_at_apoapsis)
            return specific_angular_momentum
        except Exception as err:
            raise Exception(f"Error in `get_specific_angular_momentum`: {err}")

    @property
    def current_total_angular_momentum(self) -> np.ndarray:
        return np.sum([body.current_angular_momentum for body in self.bodies], axis=0)


    @staticmethod
    def get_true_anomaly(year_percentage, e):
        try:
            mean_anomaly = 2 * np.pi * year_percentage
            eccentric_anomaly = sp.optimize.newton(lambda E: mean_anomaly - E + e * np.sin(E), 0)
            true_anomaly = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(eccentric_anomaly/2))
            return true_anomaly
        except Exception as err:
            raise Exception(f"Error in `StellarSystem.get_true_anomaly`:\n{err}")


    @staticmethod
    def get_start_vectors(T, year_percentage, e, argument_of_perihelion_deg, inclination_deg, lon_ascending_node_deg,
                          mass, G = 6.67430e-11) -> (np.ndarray, np.ndarray):
        true_anomaly = StellarSystem.get_true_anomaly(year_percentage, e)

        mean_distance = StellarSystem.get_semi_major_axis(T, mass, G)
        distance = mean_distance * (1 - e**2) / (1 + e * np.cos(true_anomaly))

        radial_velocity_mag = np.sqrt(G*mass/mean_distance) * (e*np.sin(true_anomaly)) / np.sqrt(1 - e**2)
        specific_angular_momentum = StellarSystem.get_specific_angular_momentum(T, e, mass, G)
        transverse_velocity_mag = specific_angular_momentum / distance
        # speed = float(np.sqrt(G*mass * (2*mean_distance - distance) / (distance*mean_distance)))

        radial_vec = np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.], dtype=np.float64)
        transverse_vec = np.array([-np.sin(true_anomaly), np.cos(true_anomaly), 0.], dtype=np.float64)

        z_axis = np.array([0, 0, 1], dtype = np.float64)
        arg_rad = deg2rad(argument_of_perihelion_deg)

        position = distance * radial_vec
        velocity = radial_velocity_mag * radial_vec + transverse_velocity_mag * transverse_vec
        position, velocity = rotate_vector(position, z_axis, arg_rad), rotate_vector(velocity, z_axis, arg_rad)

        inclination = deg2rad(inclination_deg)
        lon_ascending_node = deg2rad(lon_ascending_node_deg)
        ascending_node_vec = np.array([np.cos(lon_ascending_node), np.sin(lon_ascending_node), 0.0])
        position, velocity = rotate_vector(position, ascending_node_vec, inclination), rotate_vector(velocity, ascending_node_vec, inclination)
        return position, velocity


    def get_gravitational_forces(self) -> np.ndarray:
        """
        Compute and return the gravitational accelerations for all bodies in the system.

        :return: A numpy array of shape (n_bodies, 3) containing the net force vector (in kg*m/s^2) acting on each body.
        :raises ValueError: If a collision (distance shorter than 10,000 km) is detected between two bodies.
        """
        n_bodies = len(self.bodies)
        grav_forces = np.zeros((n_bodies, 3), dtype = np.float64)

        # Loop over all unique body pairs
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                body_i = self.bodies[i]
                body_j = self.bodies[j]

                # Calculate force between body_i and body_j
                r_vec = body_j.position - body_i.position
                distance = np.linalg.norm(r_vec)

                if distance < 1e7:
                    raise ValueError(f"Collision detected between {body_i.name} and {body_j.name}")

                # Gravitational force magnitude
                force_mag = self.G * body_i.mass * body_j.mass / distance ** 2

                # Vectorized force calculation
                force_vec = force_mag * (r_vec / distance)

                # Apply equal and opposite forces
                grav_forces[i] += force_vec #/ body_i.mass
                grav_forces[j] -= force_vec #/ body_j.mass

        # assert np.allclose(np.sum(grav_forces, axis=0), np.zeros(3,), atol=1e3), (f"Newton's Third Law violated! "
        #     f"The sum of all gravitational forces in the system is {np.sum(grav_forces, axis=0)} N.")
        return grav_forces


    def update(self, delta_t: float) -> None:
        # Leapfrog integration
        grav_forces = self.get_gravitational_forces()
        for ib, body in enumerate(self.bodies):
            # Step 1: Update velocity by half a time step
            body.apply_force(grav_forces[ib])
            body.accelerate(0.5 * delta_t)

            # Step 2: Update position by a full time step
            body.move(delta_t)

        # Step 3: Update velocity by half a time step again
        grav_forces = self.get_gravitational_forces()
        for ib, body in enumerate(self.bodies):
            body.apply_force(grav_forces[ib])
            body.accelerate(0.5 * delta_t)

        # While we're at it, update:
        power_output = self.bodies[0].power
        r = np.linalg.norm(self.planet.position)
        irradiance = power_output / (4*np.pi*r**2)
        self.planet.update_sunlight(delta_t, irradiance)
        self.planet.update_temperature(delta_t)