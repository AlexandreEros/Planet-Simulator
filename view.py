import sys
import numpy as np
from surface import Surface
from plot import Plot

def main():
    try:
        radius = 1.0
        resolution = 0.0
        noise_amplitude = 0.2
        noise_scale = 1.0
        noise_octaves = 4
        if len(sys.argv) > 1:
            radius = int(sys.argv[1])
        if len(sys.argv) > 2:
            resolution = int(sys.argv[2])
        if len(sys.argv) > 3:
            noise_scale = float(sys.argv[3])
        if len(sys.argv) > 4:
            noise_octaves = int(sys.argv[4])
        if len(sys.argv) > 5:
            noise_amplitude = float(sys.argv[5])
        if len(sys.argv) > 6:
            raise TypeError(f"Surface.__init__() takes from 2 to 6 positional arguments but {len(sys.argv)} were given.")

        if not (0 <= resolution < 8):
            raise ValueError("Resolution must be a non-negative integer smaller than 8.")

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)  # Exit with an error code

    # Create the geodesic grid
    try:
        # grid = GeodesicGrid(resolution)
        surf = Surface(radius, resolution, noise_scale, noise_octaves, noise_amplitude)
    except Exception as e:
        print(f"Error while generating surface:\n{e}")
        sys.exit(1)  # Exit with an error code

    # Plot the geodesic grid
    try:
        # plot_mesh(grid.vertices, grid.faces)
        # Plot(surf)
        # vmax = min(abs(np.amax(surf.elevation)), abs(np.amin(surf.elevation)))
        # vmin=-vmax
        # Plot.worldmap(surf.coordinates, surf.elevation, 540, 'Elevation (m)', vmin=vmin, vmax=vmax)
        Plot('elevation', surf.coordinates, surf.elevation, resolution=540)
    except Exception as e:
        print(f"Error while displaying geodesic grid:\n{e}")
        sys.exit(1)  # Exit with an error code

    sys.exit(0)



if __name__ == "__main__":
    main()
