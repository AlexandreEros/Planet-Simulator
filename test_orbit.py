from argparse import ArgumentError

from simulation import Simulation
import sys

def test_orbit():
    if len(sys.argv) == 1:
        timestep = 3600.0
        n_steps = 2400
    elif len(sys.argv) == 2:
        timestep = float(sys.argv[1])
        n_steps = 2400
    elif len(sys.argv) == 3:
        timestep = float(sys.argv[1])
        n_steps = int(sys.argv[2])
    else:
        print("Too many arguments; only 'timestep' and 'n_steps' are required.")
        sys.exit(1)

    try:
        sim = Simulation(timestep, n_steps)
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
