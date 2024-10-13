import sys
from geodesic_grid import GeodesicGrid
from plot import plot_mesh

def main():
    try:
        if len(sys.argv) < 2:
            raise ValueError("Please provide a resolution as a command-line argument.")
        resolution_str = sys.argv[1]
        if not resolution_str.isdigit():
            raise ValueError("Resolution must be a non-negative integer.")
        resolution = int(resolution_str)
        if not (0 <= resolution < 8):
            raise ValueError("Resolution must be a non-negative integer smaller than 8.")

    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)  # Exit with an error code

    # Create the geodesic grid
    try:
        grid = GeodesicGrid(resolution)
    except Exception as e:
        print(f"Error while generating geodesic grid: {e}")
        sys.exit(1)  # Exit with an error code

    # Plot the geodesic grid
    try:
        plot_mesh(grid.vertices, grid.faces)
    except Exception as e:
        print(f"Error while displaying geodesic grid: {e}")
        sys.exit(1)  # Exit with an error code

    sys.exit(0)



if __name__ == "__main__":
    main()
