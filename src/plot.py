# matplotlib.use('Agg')
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime

from .simulation import Simulation
from .stellar_system.celestial_body import CelestialBody
from .stellar_system.planet.atmosphere import Atmosphere
from .math_utils import GeodesicGrid


class Plot:
    def __init__(self, plot_type, *args, **kwargs):
        if plot_type == 'none':
            self.func = self.nop
        if plot_type == 'mesh':
            self.func = self.mesh
        elif plot_type == 'orbits':
            self.func = self.orbits
            args = (args[0],)

        elif plot_type == 'atmosphere':
            planet, vertex = args
            args = (planet.atmosphere, vertex)
            self.func = self.atmosphere
        elif plot_type == 'pressure':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            pressure = planet.atmosphere.ref.pressure[layer_idx]
            args = (coordinates, pressure)
            min_altitude = np.amin(planet.atmosphere.ref.altitude[layer_idx])
            max_altitude = np.amax(planet.atmosphere.ref.altitude[layer_idx])
            kwargs[
                'title'] = f'Air pressure (Pa) between {min_altitude / 1000:.2f} and {max_altitude / 1000:.2f} km high'
            self.func = self.worldmap
        elif plot_type == 'density':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            density = planet.atmosphere.ref.density[layer_idx]
            args = (coordinates, density)
            min_altitude = np.amin(planet.atmosphere.ref.altitude[layer_idx])
            max_altitude = np.amax(planet.atmosphere.ref.altitude[layer_idx])
            kwargs[
                'title'] = f'Air density (kg/m³) between {min_altitude / 1000:.2f} and {max_altitude / 1000:.2f} km high'
            self.func = self.worldmap
        elif plot_type == 'air_temperature':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            temperature = planet.atmosphere.ref.temperature[layer_idx] - 273.15
            args = (coordinates, temperature)
            min_altitude = np.amin(planet.atmosphere.ref.altitude[layer_idx])
            max_altitude = np.amax(planet.atmosphere.ref.altitude[layer_idx])
            kwargs[
                'title'] = f'Temperature (ºC) between {min_altitude / 1000:.2f} and {max_altitude / 1000:.2f} km high'
            kwargs['cmap'] = 'plasma'
            self.func = self.worldmap

        elif plot_type == 'pressure_gradient':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            pressure = planet.atmosphere.ref.pressure[layer_idx]
            pressure_gradient = planet.atmosphere.air_flow.pressure_gradient[layer_idx]
            args = (coordinates, pressure, pressure_gradient)
            altitude = planet.atmosphere.ref.altitude[layer_idx]
            kwargs['title'] = f'Pressure Gradient at {altitude / 1000:.2f} km high'
            self.func = self.gradient

        elif plot_type == 'elevation':
            self.func = self.worldmap
            surf = args[0]
            coordinates = surf.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            kwargs['title'] = 'Elevation (m)'
            kwargs['cmap'] = 'terrain'  # surf.cmap
            kwargs['resolution'] = int(np.ceil(0.03 * len(surf.vertices)))
            args = (coordinates, surf.elevation)
        elif plot_type == 'albedo':
            self.func = self.worldmap
            surf = args[0]
            coordinates = surf.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            kwargs['title'] = 'Albedo'
            kwargs['resolution'] = int(np.ceil(0.03 * len(surf.vertices)))
            kwargs['vmax'] = 1
            kwargs['vmin'] = 0
            args = (coordinates, surf.albedo)
        elif plot_type == 'heat_capacity':
            self.func = self.worldmap
            surf = args[0]
            coordinates = surf.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            kwargs['title'] = 'Heat capacity (J/m²·K)'
            kwargs['resolution'] = int(np.ceil(0.03 * max(coordinates.shape)))
            kwargs['vmax'] = np.amax(surf.heat_capacity)
            kwargs['vmin'] = np.amin(surf.heat_capacity)
            args = (coordinates, surf.heat_capacity)

        elif plot_type == 'irradiance':
            self.func = self.animate
            sim = args[0]
            coordinates = sim.planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            irradiance = sim.history
            args = (coordinates, irradiance,)
            kwargs['title'] = 'Irradiance (W/m²)'
            kwargs['vmax'] = np.amax(irradiance)
            kwargs['vmin'] = np.amin(irradiance)
        elif plot_type == 'temperature':
            self.func = self.animate
            sim = args[0]
            temperature = sim.history - 273.15
            # temperature = temperature[len(temperature)//2:]
            coordinates = sim.planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            is_equatorial = np.abs(coordinates[..., 0]) < 10
            args = (coordinates, temperature)
            kwargs['title'] = 'Temperature (ºC)'
            kwargs['vmax'] = np.amax(temperature)
            kwargs['vmin'] = np.amin(temperature)
        elif plot_type == 'heat':
            self.func = self.animate
            sim = args[0]
            heat = sim.history
            coordinates = sim.planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            args = (coordinates, heat,)
            kwargs['title'] = 'Heat Flux (W/m²)'
            kwargs['vmax'] = np.amax(heat)
            kwargs['vmin'] = np.amin(heat)


        elif plot_type == 'velocity':
            planet, layer_idx = args
            coordinates = planet.surface.coordinates
            latitude = coordinates[..., 1]
            longitude = coordinates[..., 0]
            coordinates = np.stack([latitude, longitude], axis=-1)
            velocity = planet.atmosphere.air_flow.velocity[layer_idx]
            args = (coordinates, velocity)
            min_altitude = np.amin(planet.atmosphere.ref.altitude[layer_idx])
            max_altitude = np.amax(planet.atmosphere.ref.altitude[layer_idx])
            kwargs[
                'title'] = f'Streamlines of air flow between {min_altitude / 1000:.2f} and {max_altitude / 1000:.2f} km high'
            self.func = self.stream

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
                    triangle_coords = vertices[face]  # vertices[face] uses fancy indexing; array of shape (3, 3)
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
        position_history: dict[str, np.ndarray] = sim.history
        bodies: list[CelestialBody] = [body for body in sim.stellar_system.bodies if body.name in position_history]

        if sim.planet is not None:
            target_planet_position = sim.history[sim.planet.name].copy()
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
            plt.plot(x, y, 'o-', label=body.name, color=body.color, markersize=2 + np.log(body.mass / minmass))

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
            np.linspace(-90, 90, resolution // 2)  # Latitude from -90 to 90 degrees
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        grid_values = griddata(
            points=coordinates[:, :2],  # Points at which we have data
            values=variable,  # Data values
            xi=meshgrid,  # Points to interpolate at
            method='cubic'  # 'cubic', 'linear', or 'nearest'
        )

        fig = plt.figure(figsize=(12, 6))
        plt.imshow(grid_values, extent=(-180, 180, -90, 90), origin='lower', **kwargs)
        plt.colorbar(label=title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.tight_layout()
        plt.show()

        nowstr = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        simplified_title = "".join(c for c in title if c.isalnum())
        fig.savefig(f"temp/{simplified_title}_map_{nowstr}.png")

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
            np.linspace(-90, 90, resolution // 2)  # Latitude from -90 to 90 degrees
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        # Create initial plot with first frame
        datagrid_values = griddata(
            points=coordinates[:, :2],  # Points at which we have data
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
                points=coordinates[:, :2],  # Points at which we have data
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
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey='row')
        fig.set_figwidth(12.0)

        ax0.plot(atmosphere.air_data.temperature[:, vertex] - 273.15, atmosphere.air_data.altitude / 1000)
        ax0.set_ylabel("Altitude (km)")
        ax0.set_xlabel("Temperature (ºC)")

        ax1.plot(atmosphere.air_data.pressure[:, vertex], atmosphere.air_data.altitude / 1000)
        ax1.set_xlabel("Pressure (Pa)")

        ax2.plot(atmosphere.air_data.density[:, vertex], atmosphere.air_data.altitude / 1000)
        ax2.set_xlabel("Density (kg/m³)")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def nop(*args, **kwargs):
        pass

    @staticmethod
    def gradient(coordinates: np.ndarray, pressure: np.ndarray, gradients: np.ndarray,
                 resolution: int = 360, title: str = 'Pressure Gradient',
                 cmap='viridis', **kwargs):
        """
        Plots the pressure as a background with pressure gradient vectors superimposed
        on an equirectangular projection of a sphere.

        Parameters:
        - coordinates: (n, 3) array of Cartesian coordinates.
        - pressure: (n,) array of pressure values corresponding to each vertex.
        - gradients: (n, 2) array of gradients (dPressure/dLon, dPressure/dLat).
        - resolution: Integer defining the grid resolution for plotting (default 360).
        - title: Title of the plot.
        - cmap: Colormap for the pressure background.
        - kwargs: Additional argument settings for matplotlib's quiver function.
        """

        # Prepare coordinate arrays and gradients
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-180, 180, resolution // 4),
            np.linspace(-90, 90, resolution // 8)
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        # Interpolate pressure values to grid for background
        pressure_grid = griddata(
            points=coordinates[:, :2],
            values=pressure,
            xi=meshgrid,
            method='cubic'
        )

        # Interpolate gradients to the grid
        gradient_lon = griddata(
            points=coordinates[::4, :2],
            values=gradients[::4, 0],
            xi=meshgrid,
            method='cubic'
        )
        gradient_lat = griddata(
            points=coordinates[::4, :2],
            values=gradients[::4, 1],
            xi=meshgrid,
            method='cubic'
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Plot the pressure as the background
        img = ax.imshow(pressure_grid, extent=(-180, 180, -90, 90), origin='lower', cmap=cmap)
        plt.colorbar(img, ax=ax, label='Pressure')

        # Plot the gradients as quivers
        ax.quiver(lon_grid, lat_grid, gradient_lon, gradient_lat, **kwargs)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def stream(coordinates: np.ndarray, velocity: np.ndarray, resolution: int = 360, **kwargs):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(kwargs.get('title', "Streamlines of Air Flow"))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Create a meshgrid for the streamline plot
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-180, 180, resolution),  # Longitude from -180 to 180 degrees
            np.linspace(-90, 90, resolution // 2)  # Latitude from -90 to 90 degrees
        )
        meshgrid = np.stack((lat_grid, lon_grid), axis=-1)

        # Interpolate velocity components to the grid
        velocity_u = griddata(
            points=coordinates[:, :2],
            values=velocity[:, 0],  # Zonal velocity
            xi=meshgrid,
            method='cubic'
        )
        velocity_v = griddata(
            points=coordinates[:, :2],
            values=velocity[:, 1],  # Meridional velocity
            xi=meshgrid,
            method='cubic'
        )

        # Calculate the speed for the background
        speed = np.sqrt(velocity_u ** 2 + velocity_v ** 2)

        # Plot the background speed
        img = ax.imshow(speed, extent=(-180, 180, -90, 90), origin='lower', cmap='viridis')
        plt.colorbar(img, ax=ax, label='Speed')

        # Add streamlines for the airflow velocity
        ax.streamplot(
            lon_grid, lat_grid, velocity_u, velocity_v, color='white', density=1.5, linewidth=0.5
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def interactive_map(data, coordinates, plot_type):
        """
        Display a time-varying, layer-varying quantity on an equirectangular map
        with two sliders: one for time index, one for layer index.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_snapshots, n_layers, n_columns).
            data[t, l, :] gives the quantity for time t, layer l, and all columns.
        coordinates : np.ndarray
            Array of shape (n_columns,2) where each row contains the longitude and latitude of an air column.
        plot_type : str
            The name of the quantity described by `data`, used for the title and descriptive purposes.
            (e.g. 'pressure', 'density', 'air_temperature')
        """
        title = plot_type.replace('_', ' ').title()

        longitude = coordinates[..., 0]
        latitude = coordinates[..., 1]

        n_snapshots = data.shape[0]
        n_columns = data.shape[-1]
        data = data.reshape(n_snapshots, -1, n_columns)
        n_layers = data.shape[1]
        assert latitude.shape == (n_columns,)
        assert longitude.shape == (n_columns,)

        # Create a regular lat-lon grid for displaying the data
        # Adjust resolution as needed.
        lat_res = 180
        lon_res = 360
        lat_lin = np.linspace(-90, 90, lat_res)
        lon_lin = np.linspace(-180, 180, lon_res)
        LON_grid, LAT_grid = np.meshgrid(lon_lin, lat_lin)

        # Initial indices for time and layer
        time_idx = 0
        layer_idx = 0

        # Interpolate the data onto the regular grid
        # data_slice shape is (n_columns,) for the chosen time/layer
        data_slice = data[time_idx, layer_idx, :]
        grid_vals = griddata((longitude, latitude),
                             data_slice,
                             (LON_grid, LAT_grid),
                             method='linear')

        # Create figure and main axis
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.subplots_adjust(left=0.25, bottom=0.25)  # Make space for sliders

        # Plot the initial data using an equirectangular projection
        # Here we simply use ax.imshow() with an "extent" that spans lon/lat
        # from -180 to 180 and -90 to 90. Alternatively, you could use:
        #   ax.pcolormesh(LON_grid, LAT_grid, grid_vals, cmap='viridis', shading='auto')
        # Just be mindful of the coordinate orientation.

        im = ax.imshow(grid_vals,
                       extent=(-180, 180, -90, 90),
                       origin='lower',
                       cmap='viridis',
                       aspect='auto',
                       vmin=np.amin(data), vmax=np.amax(data))
        ax.set_title(f"{title} (Time=0, Layer=0)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label(title)

        # Define slider axes for Time and Layer
        ax_time = plt.axes((0.25, 0.1, 0.65, 0.03))
        ax_layer = plt.axes((0.25, 0.05, 0.65, 0.03))

        # Create the sliders
        slider_time = Slider(ax_time, 'Time', 0, n_snapshots - 1,
                             valinit=time_idx, valstep=1)
        if n_layers > 1:
            slider_layer = Slider(ax_layer, 'Layer', 0, n_layers - 1,
                                  valinit=layer_idx, valstep=1)

        def update(_):
            # Fetch current slider values
            t_idx = int(slider_time.val)
            l_idx = int(slider_layer.val) if n_layers > 1 else 0

            # Interpolate new data for the selected time/layer
            new_data_slice = data[t_idx, l_idx, :]
            new_grid_vals = griddata((longitude, latitude),
                                     new_data_slice,
                                     (LON_grid, LAT_grid),
                                     method='linear')

            # Update the image
            im.set_data(new_grid_vals)
            ax.set_title(f"{title} (Time={t_idx}, Layer={l_idx})")

            # Force redraw
            fig.canvas.draw_idle()

        # Connect sliders to update function
        slider_time.on_changed(update)
        if n_layers > 1:
            slider_layer.on_changed(update)

        # Show the interactive plot
        plt.show()

    @staticmethod
    def interactive_stream(vector_field: np.ndarray, coordinates: np.ndarray, resolution: int = 360):
        """
        Create an interactive plot to visualize vector fields as stream plots.
        Includes sliders for time and layer selection.

        Parameters
        ----------
        vector_field : np.ndarray
            Array of shape (n_snapshots, n_layers, n_columns, 3) where:
            velocity[t, l, :, 0] gives the zonal component for the atmospheric node (t, l),
            velocity[t, l, :, 1] gives the meridional component,
            velocity[t, l, :, 2] gives the vertical component.
        coordinates : np.ndarray
            Array of shape (n_columns, 2) where each row contains the longitude and latitude of an air column.
        resolution : int, optional
            The resolution of the grid for interpolation. Default is 360.
        """
        n_snapshots = vector_field.shape[0]
        n_layers = vector_field.shape[1]
        n_columns = vector_field.shape[2]

        longitude = coordinates[:, 0]
        latitude = coordinates[:, 1]

        assert longitude.shape == (n_columns,)
        assert latitude.shape == (n_columns,)

        # Initial indices for time and layer
        time_idx = 0
        layer_idx = 0

        # Create a regular lat-lon grid for displaying the data
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(-180, 180, resolution),
            np.linspace(-90, 90, resolution // 2)
        )
        meshgrid = np.stack((lon_grid, lat_grid), axis=-1)

        # Interpolate velocity components to the grid for initial indices
        u = griddata(
            points=coordinates[..., :2],
            values=vector_field[time_idx, layer_idx, :, 0],
            xi=meshgrid,
            method='linear'
        )
        v = griddata(
            points=coordinates[..., :2],
            values=vector_field[time_idx, layer_idx, :, 1],
            xi=meshgrid,
            method='linear'
        )
        w = griddata(
            points=coordinates[..., :2],
            values=vector_field[time_idx, layer_idx, :, 2],
            xi=meshgrid,
            method='nearest'
        )
        vmin, vmax = np.amin(vector_field[..., 2]), np.amax(vector_field[..., 2])

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(left=0.25, bottom=0.25)  # Make space for sliders

        # Plot the initial speed as background
        img = ax.imshow(w, extent=(-180, 180, -90, 90), origin='lower', cmap='viridis', aspect='auto', vmin=vmin,
                        vmax=vmax)
        plt.colorbar(img, ax=ax, label='Vertical Component')

        # Add initial streamlines
        stream = ax.streamplot(
            lon_grid, lat_grid, u, v,
            color='white', density=1.5, linewidth=0.5
        )
        ax.set_title(f"Streamlines (Time={time_idx}, Layer={layer_idx})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Define slider axes for Time and Layer
        ax_time = plt.axes((0.25, 0.1, 0.65, 0.03))
        ax_layer = plt.axes((0.25, 0.05, 0.65, 0.03))

        # Create sliders
        slider_time = Slider(ax_time, 'Time', 0, n_snapshots - 1, valinit=time_idx, valstep=1)
        if n_layers > 1:
            slider_layer = Slider(ax_layer, 'Layer', 0, n_layers - 1, valinit=layer_idx, valstep=1)

        def update(_):
            nonlocal stream
            # Fetch current slider values
            t_idx = int(slider_time.val)
            l_idx = int(slider_layer.val) if n_layers > 1 else 0

            # Interpolate velocity components to the grid for selected indices
            u = griddata(
                points=coordinates[..., :2],
                values=vector_field[t_idx, l_idx, :, 0],
                xi=meshgrid,
                method='linear'
            )
            v = griddata(
                points=coordinates[..., :2],
                values=vector_field[t_idx, l_idx, :, 1],
                xi=meshgrid,
                method='linear'
            )
            # Corrected line: use t_idx and l_idx for the vertical component
            w = griddata(
                points=coordinates[..., :2],
                values=vector_field[t_idx, l_idx, :, 2],
                xi=meshgrid,
                method='nearest'
            )

            # Update background
            img.set_data(w)

            # Remove and redraw streamlines
            stream.lines.remove()
            for art in ax.get_children():
                if isinstance(art, matplotlib.patches.FancyArrowPatch):
                    art.remove()
            stream = ax.streamplot(
                lon_grid, lat_grid, u, v,
                color='white', density=1.5, linewidth=0.5
            )

            # Update title
            ax.set_title(f"Streamlines (Time={t_idx}, Layer={l_idx})")

            # Force redraw
            fig.canvas.draw_idle()

        # Connect sliders to update function
        slider_time.on_changed(update)
        if n_layers > 1:
            slider_layer.on_changed(update)

        # Show interactive plot
        plt.show()