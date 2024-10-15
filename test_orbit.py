from argparse import ArgumentError

from simulation import Simulation
import sys

def test_orbit():
    timestep = None
    n_steps = None
    steps_between_snapshots = None
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
        sim.plot_orbits()
    except Exception as err:
        print(f"Error plotting orbits: {err}")
        sys.exit(1)


test_orbit()
sys.exit(0)
