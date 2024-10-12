import matplotlib.pyplot as plt
import numpy as np



def plot_mesh(vertices, faces):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Plot each triangular face as a wireframe
    for face in faces:
        triangle = vertices[face]
        # Close the triangle by appending the first point at the end
        triangle = np.vstack([triangle, triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='c')

    # Set labels and view angle for better visualization
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)
    ax.set_title(f"Geodesic Grid ({len(faces)} triangles)")

    # Show the plot
    plt.tight_layout()
    plt.show()