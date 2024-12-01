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
        self.planet = None
        self.bodies: list[Star | Planet] = []


    @property
    def positions(self) -> dict[str, list[float]]:
        return {body.name: body.position.tolist() for body in self.bodies}

    @property
    def velocities(self) -> dict[str, list[float]]:
        return {body.name: body.velocity.tolist() for body in self.bodies}

    @property
    def idx(self) -> dict[str, int]:
        return {self.bodies[i].name: i for i in range(len(self.bodies))}


    def add_body(self, **kwargs) -> None:
        kws = kwargs.keys()
        for kw in kws:
            if isinstance(kwargs[kw], list):
                kwargs[kw] = np.array(kwargs[kw], dtype = np.float64)

        if kwargs['body_type']=='star':
            self.bodies.append(Star(**kwargs))
            self.star = self.bodies[-1]
        elif kwargs['body_type']=='planet':
            parent = self.star if 'parent' not in kwargs else self.bodies[self.idx[kwargs['parent']]]
            kwargs['parent_mass'] = parent.mass
            kwargs['parent_position'] = parent.position
            kwargs['parent_velocity'] = parent.velocity
            self.bodies.append(Planet(**kwargs))
        else:
            raise KeyError(f"Unsupported body_type: {kwargs['body_type']}")

        if kwargs['name']==self.planet_name:
            self.planet = self.bodies[-1]


    @property
    def current_total_angular_momentum(self) -> np.ndarray:
        return np.sum([body.current_angular_momentum for body in self.bodies], axis=0)


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

        if isinstance(self.planet, Planet):
        # While we're at it, update:
            self.planet.update_sunlight(delta_t, self.star)
            self.planet.update_temperature(delta_t)