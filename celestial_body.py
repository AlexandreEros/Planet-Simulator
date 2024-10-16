import numpy as np

class CelestialBody:
    def __init__(self, name: str, position: np.ndarray[np.float64], velocity: np.ndarray[np.float64], mass: float,
                 color: str, orbital_period: float = None, eccentricity: float = 0.0, year_percentage: float = 0.0):
        self.name = name
        self.position: np.ndarray[np.float64] = position
        self.velocity: np.ndarray[np.float64] = velocity
        self.mass = mass
        self.color = color
        self.orbital_period = orbital_period
        self.eccentricity = eccentricity
        self.initial_year_percentage = year_percentage

    def accelerate(self, delta_v: np.ndarray[np.float64]) -> None:
        self.velocity += delta_v

    def move(self, delta_x: np.ndarray[np.float64]) -> None:
        self.position += delta_x
