import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata

from celestial_body import CelestialBody
from geodesic_grid import GeodesicGrid
from simulation import Simulation


class Plot:
    def __init__(self, plot_type, *args, **kwargs):
        if plot_type=='mesh':
            self.func = self.mesh
        elif plot_type=='orbits':
            self.func = self.orbits
        elif plot_type=='elevation':
            self.func = self.worldmap
            elevation = args[0]
            kwargs['title'] = 'Elevation (m)'
            kwargs['vmax'] = min(abs(np.amax(elevation)), abs(np.amin(elevation)))
            kwargs['vmin']= -kwargs['vmax']

        self.func(*args, **kwargs)

    @staticmethod
    def mesh(grid: GeodesicGrid) -> None:
        try:
            vertices = grid.vertices
            faces = grid.faces

            # Check if vertices and faces are valid numpy arrays
            if not isinstance(vertices, np.ndarray) or not isinstance(faces, np.ndarray):
                raise TypeError("Vertices and faces must be numpy arrays.")
            # Ensure vertices is 2D and faces is 2D
            if vertices.ndim != 2 or faces.ndim != 2:
                raise ValueError("Vertices and faces must be 2D arrays.")
            # Ensure vertices have three columns (X, Y, Z coordinates)
            if vertices.shape[1] != 3:
                raise ValueError(f"Vertices must have shape (n, 3), got {vertices.shape}")
            # Ensure faces contain valid vertex indices
            if np.any(faces >= len(vertices)) or np.any(faces < 0):
                raise IndexError("Faces reference invalid vertex indices.")

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

            # Plot each triangular face as a wireframe
            for face in faces:
                try:
                    # face: {NDArray[int]: (3,)} are the indices of the triangles that make up the corresponding face
                    triangle_coords = vertices[face]   # vertices[face] uses fancy indexing; array of shape (3, 3)
                    # Close the triangle by appending the first point at the end
                    closed_triangle = np.vstack([triangle_coords, triangle_coords[0]])  # shape (4,3)
                    x = closed_triangle[:, 0]  # Arrays of shape (4,)
                    y = closed_triangle[:, 1]
                    z = closed_triangle[:, 2]
                    ax.plot(xs=x, ys=y, zs=z, color='c')
                except IndexError as e:
                    raise IndexError(f"Error while accessing vertices for face {face}: {e}")
                except Exception as e:
                    raise RuntimeError(f"Unexpected error while plotting face {face}: {e}")

            # Set labels and view angle for better visualization
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20, azim=30)
            ax.set_title(f"Geodesic Grid ({len(faces)} triangles)")

            # Show the plot
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"An unexpected error occurred: {e}")



    @staticmethod
    def orbits(sim: Simulation):
        position_history: dict[str, np.ndarray] = sim.position_history
        bodies: list[CelestialBody] = sim.stellar_system.bodies

        # Create a figure for the plot
        plt.figure(figsize=(10, 6))

        for body in bodies:
            # Extract positions for Sun and Earth from provided history
            x = position_history[body.name][:, 0]
            y = position_history[body.name][:, 1]
            plt.plot(x, y, 'o-', label=body.name, color=body.color)

        # Setting labels and title
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Orbital Trajectories")
        plt.legend()
        plt.axis('equal')

        # Display the plot
        plt.show()



    @staticmethod
    def worldmap(coordinates: np.ndarray, variable: np.ndarray, resolution: int = 360, title='', vmin=None, vmax=None):
        """
        Plot the equirectangular projection of the terrain.

        Parameters:
        - vertices: (n, 3) array of Cartesian coordinates representing the surface points.
        - variable: (n) array of values corresponding to each vertex.
        - resolution: Integer defining the resolution of the grid for plotting (default 360).
        """

        coordinates = np.array(coordinates)

        # Step 2: Create a 2D grid for the equirectangular projection
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-180, 180, resolution),  # Longitude from -180 to 180 degrees
            np.linspace(-90, 90, resolution//2)  # Latitude from -90 to 90 degrees
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        # Step 3: Interpolate elevation data onto the grid
        grid_values = griddata(
            points=coordinates[:,:2],  # Points at which we have data
            values=variable,  # Elevation data values
            xi=meshgrid,  # Points to interpolate at
            method='cubic'  # 'cubic', 'linear', or 'nearest'
        )

        # Step 4: Define a custom colormap with a sharp transition at sea level (elevation = 0)
        colors_undersea = plt.cm.Blues_r(np.linspace(0, 0.25, 128))
        colors_land = plt.cm.terrain(np.linspace(0.25, 1, 128))
        all_colors = np.vstack((colors_undersea, colors_land))
        world_cmap = mcolors.LinearSegmentedColormap.from_list('world_cmap', all_colors)

        # Step 5: Plot the elevation data on the equirectangular projection
        plt.figure(figsize=(12, 6))
        plt.imshow(grid_values, extent=(-180, 180, -90, 90), origin='lower', cmap=world_cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label=title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.tight_layout()
        plt.show()

