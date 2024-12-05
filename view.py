import sys
import numpy as np
from surface import Surface
from planet import Planet
from plot import Plot
import json


def view():
    try:
        if len(sys.argv) < 2:
            raise TypeError(f"Not enough inputs; please enter the name of a celestial body described in bodies.json")
        planet_name = sys.argv[1]
        plot_type = 'elevation'
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]
        if len(sys.argv) > 3:
            raise TypeError(f"Surface.__init__() takes from 2 to 3 positional arguments but {len(sys.argv)} were given.")

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
        # grid = GeodesicGrid(resolution)
        # surf = Surface(radius, resolution, noise_scale, noise_octaves, noise_amplitude, noise_bias, noise_offset)
        surf = Surface(**surface_data)

    except Exception as e:
        print(f"Error while generating surface:\n{e}")
        sys.exit(1)  # Exit with an error code

    # Plot the geodesic grid
    # try:
        # plot_mesh(grid.vertices, grid.faces)
        # Plot(surf)
        # vmax = min(abs(np.amax(surf.elevation)), abs(np.amin(surf.elevation)))
        # vmin=-vmax
        # Plot.worldmap(surf.coordinates, surf.elevation, 540, 'Elevation (m)', vmin=vmin, vmax=vmax)
    Plot(plot_type, surf)

    sys.exit(0)



view()
