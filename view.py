import sys
import numpy as np
from surface import Surface
from planet import Planet
from simulation import Simulation
from plot import Plot
import json


def view():
    try:
        if len(sys.argv) < 2:
            raise TypeError(f"Not enough inputs; please enter the name of a celestial body described in bodies.json")
        planet_name = sys.argv[1]
        plot_type = 'elevation'
        idx: int = 0
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]
        if len(sys.argv) > 3:
            idx = int(sys.argv[3])
        if len(sys.argv) > 4:
            raise TypeError(f"Surface.__init__() takes from 2 to 4 positional arguments but {len(sys.argv)} were given.")

        with open('bodies.json', 'r') as f:
            data = json.load(f)
            bodies: list[dict] = data['bodies']
            planet_data = [body for body in bodies if body['name']==planet_name][0]
            surface_data = planet_data['surface_data']
            surface_data['blackbody_temperature'] = 123 # Who cares

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)  # Exit with an error code

    # Create
    try:
        if plot_type in ('mesh', 'elevation'):
            args=(Surface(**surface_data),)
        elif plot_type in ('atmosphere', 'density', 'air_temperature'):
            sim = Simulation(plot_type, planet_name, 1.0, 1)
            args = (sim.planet, idx)

    except Exception as e:
        print(f"Error while generating surface:\n{e}")
        sys.exit(1)  # Exit with an error code


    Plot(plot_type, *args)

    sys.exit(0)



view()
