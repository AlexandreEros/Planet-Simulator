import sys

from simulation import Simulation
from plot import plot_orbits

def test_orbit():
    timestep = 3600
    n_steps = 8766
    steps_between_snapshots = 8
    if len(sys.argv) > 1:
        timestep = float(sys.argv[1])
    if len(sys.argv) > 2:
        n_steps = int(sys.argv[2])
    if len(sys.argv) > 3:
        steps_between_snapshots = int(sys.argv[3])
    if len(sys.argv) > 4:
        print("Too many arguments; only 'timestep' and 'n_steps' are required.")
        sys.exit(1)

    try:
        sim = Simulation(timestep, n_steps, steps_between_snapshots)
    except Exception as err:
        print(f"Error setting up simulation: {err}")
        sys.exit(1)

    try:
        sim.run()
    except Exception as err:
        print(f"Error running simulation at {sim.time} seconds: {err}")
        sys.exit(1)

    try:
        plot_orbits(sim)
    except Exception as err:
        print(f"Error plotting orbits: {err}")
        sys.exit(1)


test_orbit()
sys.exit(0)
