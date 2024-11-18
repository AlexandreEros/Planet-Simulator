import sys
import numpy as np
from surface import Surface
from planet import Planet
from plot import Plot
import json


def view():
    try:
        # radius = 1.0
        # resolution = 0
        # noise_scale = 1.0
        # noise_octaves = 4
        # noise_amplitude = 0.05
        # noise_bias = 0.0
        # noise_offset = (0.0, 0.0, 0.0)

        plot_type = 'elevation'
        if len(sys.argv) > 1:
            planet_name = sys.argv[1]
            # radius = int(sys.argv[1])
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]
            # resolution = int(sys.argv[2])
        # if len(sys.argv) > 3:
        #     noise_scale = float(sys.argv[3])
        # if len(sys.argv) > 4:
        #     noise_octaves = int(sys.argv[4])
        # if len(sys.argv) > 5:
        #     noise_amplitude = float(sys.argv[5])
        # if len(sys.argv) > 6:
        #     noise_bias = float(sys.argv[6])
        # if len(sys.argv) > 7:
        #     noise_offset = tuple([float(n.strip(' ')) for n in sys.argv[7][1:-1].split(',')])
        if len(sys.argv) > 3:
            raise TypeError(f"Surface.__init__() takes from 2 to 3 positional arguments but {len(sys.argv)} were given.")

        with open('bodies.json', 'r') as f:
            data = json.load(f)
            bodies: list[dict] = data['bodies']
            planet_data = [body for body in bodies if body['name']==planet_name][0]
            radius = planet_data['radius']
            surface_data = planet_data['surface_data']

        # if not (0 <= resolution < 8):
        #     raise ValueError("Resolution must be a non-negative integer smaller than 8.")

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)  # Exit with an error code

    # Create
    try:
        # grid = GeodesicGrid(resolution)
        # surf = Surface(radius, resolution, noise_scale, noise_octaves, noise_amplitude, noise_bias, noise_offset)
        surf = Surface(radius, **surface_data)

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
    # except Exception as e:
    #     print(f"Error while displaying geodesic grid:\n{e}")
    #     sys.exit(1)  # Exit with an error code

    sys.exit(0)



view()
