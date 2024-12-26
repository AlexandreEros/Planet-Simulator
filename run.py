import os
import sys

from src.simulation import Simulation
from src.plot import Plot

base_dir = os.path.dirname(os.path.abspath(__file__))
default_bodies = os.path.join(base_dir, 'data', 'bodies.json')

def run():
    plot_type = 'orbits'
    planet='Mars'
    duration_sec = 86400
    timestep_sec = 600
    time_between_snapshots_sec = 0.0
    bodies_file = default_bodies
    if len(sys.argv) > 1:
        plot_type = sys.argv[1]
    if len(sys.argv) > 2:
        planet = sys.argv[2]
    if len(sys.argv) > 3:
        duration_sec = float(sys.argv[3])
    if len(sys.argv) > 4:
        timestep_sec = int(sys.argv[4])
        time_between_snapshots_sec = timestep_sec
    if len(sys.argv) > 5:
        time_between_snapshots_sec = float(sys.argv[5])
    if len(sys.argv) > 6:
        bodies_file = sys.argv[6]
    if len(sys.argv) > 7:
        raise TypeError(f"run() takes from 0 to 7 positional arguments but {len(sys.argv)} were given.")

    sim = Simulation(plot_type, planet, bodies_file)
    sim.run(duration_sec, timestep_sec, time_between_snapshots_sec)

    Plot(plot_type, sim, planet)

if __name__ == '__main__':
    run()
    sys.exit(0)
