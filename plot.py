import matplotlib.pyplot as plt
import numpy as np



def plot_mesh(vertices, faces):
    try:
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
                triangle = vertices[face]
                # Close the triangle by appending the first point at the end
                triangle = np.vstack([triangle, triangle[0]])
                ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='c')
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

    except TypeError as e:
        print(f"Type error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")