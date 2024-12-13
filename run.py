import sys

from src.simulation import Simulation
from src.plot import Plot

def run():
    plot_type = 'orbits'
    planet='Earth'
    timestep = 3600
    n_steps = 8766
    steps_between_snapshots = 8
    if len(sys.argv) > 1:
        plot_type = str(sys.argv[1])
    if len(sys.argv) > 2:
        planet = sys.argv[2]
    if len(sys.argv) > 3:
        timestep = float(sys.argv[3])
    if len(sys.argv) > 4:
        n_steps = int(sys.argv[4])
    if len(sys.argv) > 5:
        steps_between_snapshots = int(sys.argv[5])
    if len(sys.argv) > 6:
        raise TypeError(f"run() takes from 0 to 6 positional arguments but {len(sys.argv)} were given.")

    sim = Simulation(plot_type, planet, timestep, n_steps, steps_between_snapshots)
    sim.run()

    Plot(plot_type, sim, planet)

run()
sys.exit(0)
