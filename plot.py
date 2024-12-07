import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime

from celestial_body import CelestialBody
from planet import Planet
from geodesic_grid import GeodesicGrid
from simulation import Simulation
from atmosphere import Atmosphere


class Plot:
    def __init__(self, plot_type, *args, **kwargs):
        if plot_type=='none':
            self.func = self.nop
        if plot_type=='mesh':
            self.func = self.mesh
        elif plot_type=='orbits':
            self.func = self.orbits
            args = (args[0],)

        elif plot_type=='atmosphere':
            planet, vertex = args
            args = (planet.atmosphere, vertex)
            self.func = self.atmosphere
        elif plot_type=='density':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            density = planet.atmosphere.density[layer_idx]
            args = (coordinates, density)
            altitude = planet.atmosphere.altitudes[layer_idx]
            kwargs['title'] = f'Air density (kg/m³) at {altitude/1000:.2f} km high'
            self.func = self.worldmap
        elif plot_type=='air_temperature':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            temperature = planet.atmosphere.temperature[layer_idx] - 273.15
            args = (coordinates, temperature)
            altitude = planet.atmosphere.altitudes[layer_idx]
            kwargs['title'] = f'Temperature (ºC) at {altitude/1000:.2f} km high'
            kwargs['cmap'] = 'plasma'
            self.func = self.worldmap

        elif plot_type=='elevation':
            self.func = self.worldmap
            surf = args[0]
            kwargs['title'] = 'Elevation (m)'
            kwargs['cmap'] = 'terrain' #surf.cmap
            kwargs['resolution'] = int(np.ceil(0.03 * len(surf.vertices)))
            args = (surf.coordinates, surf.elevation)
        elif plot_type=='albedo':
            self.func = self.worldmap
            surf = args[0]
            kwargs['title'] = 'Albedo'
            kwargs['resolution'] = int(np.ceil(0.03 * len(surf.vertices)))
            kwargs['vmax'] = 1
            kwargs['vmin']= 0
            args = (surf.coordinates, surf.albedo)
        elif plot_type=='heat_capacity':
            self.func = self.worldmap
            surf = args[0]
            coordinates = surf.coordinates
            kwargs['title'] = 'Heat capacity (J/m²·K)'
            kwargs['resolution'] = int(np.ceil(0.03 * max(coordinates.shape)))
            kwargs['vmax'] = np.amax(surf.heat_capacity)
            kwargs['vmin'] = np.amin(surf.heat_capacity)
            args = (coordinates, surf.heat_capacity)

        elif plot_type=='irradiance':
            self.func = self.animate
            sim = args[0]
            irradiance = sim.irradiance_history
            args = (sim.planet.surface.coordinates, irradiance,)
            kwargs['title'] = 'Irradiance (W/m²)'
            kwargs['vmax'] = np.amax(irradiance)
            kwargs['vmin'] = np.amin(irradiance)
        elif plot_type=='temperature':
            self.func = self.animate
            sim = args[0]
            temperature = sim.temperature_history - 273.15
            # temperature = temperature[len(temperature)//2:]
            coordinates = sim.planet.surface.coordinates
            is_equatorial = np.abs(coordinates[...,0]) < 10
            args = (coordinates, temperature)
            kwargs['title'] = 'Temperature (ºC)'
            kwargs['vmax'] = np.amax(temperature)
            kwargs['vmin'] = np.amin(temperature)
        elif plot_type=='heat':
            self.func = self.animate
            sim = args[0]
            heat = sim.heat_history
            args = (sim.planet.surface.coordinates, heat,)
            kwargs['title'] = 'Heat Flux (W/m²)'
            kwargs['vmax'] = np.amax(heat)
            kwargs['vmin'] = np.amin(heat)

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
        bodies: list[CelestialBody] = [body for body in sim.stellar_system.bodies if body.name in position_history]

        if sim.planet is not None:
            target_planet_position = sim.position_history[sim.planet.name].copy()
            for body in bodies:
                position_history[body.name] -= target_planet_position

        # Create a figure for the plot
        plt.figure(figsize=(10, 6))

        masses = np.array([body.mass for body in bodies])
        minmass = np.amin(masses)
        for body in bodies:
            # Extract positions for Sun and Earth from provided history
            x = position_history[body.name][:, 0]
            y = position_history[body.name][:, 1]
            plt.plot(x, y, 'o-', label=body.name, color=body.color, markersize=2+np.log(body.mass/minmass))

        # Setting labels and title
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Orbital Trajectories")
        plt.legend()
        plt.axis('equal')

        # Display the plot
        plt.show()



    @staticmethod
    def worldmap(coordinates: np.ndarray, variable: np.ndarray,
                 resolution: int = 360, title='', **kwargs):
        """
        Plot the equirectangular projection of the terrain.

        Parameters:
        - vertices: (n, 3) array of Cartesian coordinates representing the surface points.
        - variable: (n) array of values corresponding to each vertex.
        - resolution: Integer defining the resolution of the grid for plotting (default 360).
        """

        coordinates = np.array(coordinates)

        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-180, 180, resolution),  # Longitude from -180 to 180 degrees
            np.linspace(-90, 90, resolution//2)  # Latitude from -90 to 90 degrees
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        grid_values = griddata(
            points=coordinates[:,:2],  # Points at which we have data
            values=variable,  # Data values
            xi=meshgrid,  # Points to interpolate at
            method='cubic'  # 'cubic', 'linear', or 'nearest'
        )

        plt.figure(figsize=(12, 6))
        plt.imshow(grid_values, extent=(-180, 180, -90, 90), origin='lower', **kwargs)
        plt.colorbar(label=title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.tight_layout()
        plt.show()

        nowstr = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        simplified_title = "".join(c for c in title if c.isalnum())
        plt.savefig(f"temp/{simplified_title}_map_{nowstr}.png")



    @staticmethod
    def animate(coordinates, variable_history, title: str = '', resolution: int = 360, vmin=None, vmax=None):
        """
        Create an animation to visualize the variable changes over time.

        Parameters:
        - lat_grid, lon_grid: Latitude and Longitude grids from meshgrid.
        - variable_history: List of 2D arrays, each representing a value at one time step.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-180, 180, resolution),  # Longitude from -180 to 180 degrees
            np.linspace(-90, 90, resolution//2)  # Latitude from -90 to 90 degrees
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        # Create initial plot with first frame
        datagrid_values = griddata(
                        points=coordinates[:,:2],  # Points at which we have data
                        values=variable_history[0],  # Data values
                        xi=meshgrid,  # Points to interpolate at
                        method='cubic'  # 'cubic', 'linear', or 'nearest'
                    )
        img = ax.imshow(datagrid_values, extent=(-180, 180, -90, 90), origin='lower', cmap='plasma',
                        vmin=vmin, vmax=vmax)
        fig.colorbar(img, label=title)

        # Define the animation update function
        def update(frame):
            grid_values = griddata(
                points=coordinates[:,:2],  # Points at which we have data
                values=variable_history[frame],  # Data values
                xi=meshgrid,  # Points to interpolate at
                method='cubic'  # 'cubic', 'linear', or 'nearest'
            )
            img.set_array(grid_values)
            ax.set_title(f"Frame {frame}")
            return [img]

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(variable_history), blit=True)

        # Save animation as a GIF or mp4
        nowstr = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        simplified_title = "".join(c for c in title if c.isalnum())
        ani.save(f"temp/{simplified_title}_history_{nowstr}.gif", writer='pillow', fps=4)

        # Show the animation in the notebook or console
        plt.show()


    @staticmethod
    def atmosphere(atmosphere: Atmosphere, vertex: int = 0):
        plt.plot(atmosphere.temperature[:,vertex]-273.15, atmosphere.altitudes / 1000)
        plt.title("Temperature vs Altitude")
        plt.xlabel("Temperature (ºC)")
        plt.ylabel("Altitude (km)")
        plt.tight_layout()
        plt.show()

        plt.plot(atmosphere.pressure[:,vertex], atmosphere.altitudes / 1000)
        plt.title("Pressure vs Altitude")
        plt.xlabel("Pressure (Pa)")
        plt.ylabel("Altitude (km)")
        plt.tight_layout()
        plt.show()

        plt.plot(atmosphere.density[:,vertex], atmosphere.altitudes / 1000)
        plt.title("Density vs Altitude")
        plt.xlabel("Density (kg/m³)")
        plt.ylabel("Altitude (km)")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def nop(*args, **kwargs):
        pass