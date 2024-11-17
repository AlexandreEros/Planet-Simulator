import sys

from simulation import Simulation
# import plot
from plot import Plot

def run():
    plot_type = 'orbits'
    timestep = 3600
    n_steps = 8766
    steps_between_snapshots = 8
    if len(sys.argv) > 1:
        plot_type = str(sys.argv[1])
    if len(sys.argv) > 2:
        timestep = float(sys.argv[2])
    if len(sys.argv) > 3:
        n_steps = int(sys.argv[3])
    if len(sys.argv) > 4:
        steps_between_snapshots = int(sys.argv[4])
    if len(sys.argv) > 5:
        raise TypeError(f"run() takes from 0 to 5 positional arguments but {len(sys.argv)} were given.")

    try:
        sim = Simulation(timestep, n_steps, steps_between_snapshots)
    except Exception as err:
        print(f"Error setting up simulation:\n{err}")
        sys.exit(1)

    try:
        sim.run()
    except Exception as err:
        print(f"Error running simulation at {sim.time} seconds:\n{err}")
        sys.exit(1)

    try:
        Plot(plot_type, sim)
    except Exception as err:
        print(f"Error plotting:\n{err}")
        sys.exit(1)


run()
sys.exit(0)
