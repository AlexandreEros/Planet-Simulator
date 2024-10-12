import sys
import geodesic_grid as grd
import plot



def main(resolution):
    # Generate and plot the corrected geodesic grid
    geo_grid = grd.GeodesicGrid(resolution)
    vertices, faces = geo_grid.vertices, geo_grid.faces
    plot.plot_mesh(vertices, faces)


if __name__ == "__main__":
    depth = int(sys.argv[1])
    main(depth)
