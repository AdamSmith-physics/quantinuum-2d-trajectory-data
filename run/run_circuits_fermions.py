from fix_pathing import root_dir
from src.circuits.fermion_circuits import trajectory_density, trajectory_current, density_readout, current_readout
from src.circuits.timer import Timer
from src.circuits.common_circuits import run_local
from src.io import save_to_hdf5

import sys
import os
import numpy as np

#PARAMETERS
V = 0.0
phi = 0.0 # Peierls phase in units of pi
dt = 0.21
p = 2*dt
steps = 5
shots = 10
start = 'random'
n_init = [] 

print("Running with parameters:")
print(f"V = {V}")
print(f"phi = {phi} (in units of pi)")
print(f"dt = {dt}")
print(f"p = {p}")
print(f"steps = {steps}")
print(f"shots = {shots}")
print(f"start = {start}")
if start == 'custom':
    print(f"n_init = {n_init}\n")

N = 16 # Number of sites
J = 1
dephasing = False


sector_bonds = [[[4, 8], [7, 11], [1, 2], [5, 6], [9, 10], [13, 14]]]  # sector 1
sector_bonds.append([[1, 5], [2, 6], [9, 13], [10, 14]])  # sector 2
sector_bonds.append([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]])  # sector 3
sector_bonds.append([[5, 9], [6, 10], [0, 4], [3, 7], [8, 12], [11, 15]])  # sector 4

#LOAD AND RUN CIRCUIT 
print("Building circuits...")
circ_density = trajectory_density(J, V, N=N, dt=dt, p=p, steps=steps, start=start, n_init=n_init, phi=phi, name="2D Trajectory Fermions", dephasing = dephasing)
circ_current_1 = trajectory_current(J, V, N=N, dt=dt, sector='sector1', p=p, steps=steps, start=start, n_init=n_init, phi=phi, name="2D Trajectory Fermions - Current sector1", dephasing = dephasing)
circ_current_2 = trajectory_current(J, V, N=N, dt=dt, sector='sector2', p=p, steps=steps, start=start, n_init=n_init,phi=phi, name="2D Trajectory Fermions - Current sector2", dephasing = dephasing)
circ_current_3 = trajectory_current(J, V, N=N, dt=dt, sector='sector3', p=p, steps=steps, start=start, n_init=n_init,phi=phi, name="2D Trajectory Fermions - Current sector3", dephasing = dephasing)
circ_current_4 = trajectory_current(J, V, N=N, dt=dt, sector='sector4', p=p, steps=steps, start=start, n_init=n_init,phi=phi, name="2D Trajectory Fermions - Current sector4", dephasing = dephasing)

circuits = [circ_density, circ_current_1, circ_current_2, circ_current_3, circ_current_4]
print(f"Circuits built.\n")


print("Running..."); timer = Timer()
results_local = run_local(circuits, shots=shots)
print(f"Finished running! Time elapsed is {timer}.\n")


print("Reading results...")
data = dict()
data["parameters"] = {
    "N": N,
    "J": J,
    "V": V,
    "dt": dt,
    "p": p,
    "steps": steps,
    "shots": shots,
    "dephasing": dephasing,
    "n_init": n_init,
    "sector_bond_1": sector_bonds[0],
    "sector_bond_2": sector_bonds[1],
    "sector_bond_3": sector_bonds[2],
    "sector_bond_4": sector_bonds[3],
}


coin1, coin2, trajectory_source, trajectory_sink, c_init, densities, stabilizer = density_readout(results_local[0], N=N, shots=shots, steps=steps)

data[f"density_circuit"] = {
        "coin1": coin1,
        "coin2": coin2,
        "trajectory_source": trajectory_source,
        "trajectory_sink": trajectory_sink,
        "c_init": c_init,
        "densities": densities,
        "stabilizer": stabilizer
    }

print(f"Stabilizer density = {stabilizer}")


for ii in range(4):

    coin1, coin2, out1, out2, den_currents, den_ancillas, c_init, ancilla_c_init, trajectory_source, trajectory_sink, stabilizer = current_readout(sector_bonds[ii], results_local[ii+1], N, shots, steps)

    print(f"Stabilizer current {ii+1} = {stabilizer}")

    data[f"current_circuit_{ii+1}"] = {
        "coin1": coin1,
        "coin2": coin2,
        "out1": out1,
        "out2": out2,
        "trajectory_source": trajectory_source,
        "trajectory_sink": trajectory_sink,
        "den_currents": den_currents,
        "den_ancillas": den_ancillas,
        "c_init": c_init,
        "ancilla_c_init": ancilla_c_init,
        "stabilizer": stabilizer
    }

print(f"Results read.\n")

print("Saving results...")

#SAVE THE OUTPUTS
# Create data directory if it doesn't exist
if not os.path.exists(f'data_local/fermions'):
    os.makedirs(f'data_local/fermions')

save_to_hdf5(data, f'data_local/fermions/{start}_V{V}_phi{phi}_dt{dt}_p{p}_steps{steps}_shots{shots}.h5')

print("Finished!")
