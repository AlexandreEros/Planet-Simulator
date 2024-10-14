import numpy as np
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
        self.bodies.append(CelestialBody(**kwargs))


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
