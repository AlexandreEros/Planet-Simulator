import sys
import json

from src.plot import Plot
from src.simulation import Simulation
from src.core.components.surface import Surface

def view():
    try:
        if len(sys.argv) < 2:
            raise TypeError(f"Not enough inputs; please enter the name of a celestial body described in bodies.json")
        planet_name = sys.argv[1]
        plot_type = 'elevation'
        idx: int = 0
        timestep: float = 1.0
        n_steps: int = 100

        if len(sys.argv) > 2:
            plot_type = sys.argv[2]
        if len(sys.argv) > 3:
            idx = int(sys.argv[3])
        if len(sys.argv) > 4:
            timestep = float(sys.argv[4])
        if len(sys.argv) > 5:
            n_steps = int(sys.argv[5])
        if len(sys.argv) > 6:
            raise TypeError(f"Surface.__init__() takes from 2 to 6 positional arguments but {len(sys.argv)} were given.")

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)  # Exit with an error code

    # Create
    if plot_type in ('mesh', 'elevation'):
        with open(Simulation.default_bodies, 'r') as f:
            data = json.load(f)
            bodies: list[dict] = data['bodies']
            planet_data = [body for body in bodies if body['name']==planet_name][0]
            surface_data = planet_data['surface_data']
            surface_data['blackbody_temperature'] = 123 # Who cares
        args=(Surface(**surface_data),)
    elif plot_type in ('atmosphere', 'pressure', 'density', 'air_temperature'):
        sim = Simulation(plot_type, planet_name, timestep, n_steps)
        args = (sim.planet, idx)

    Plot(plot_type, *args)

    sys.exit(0)



view()
