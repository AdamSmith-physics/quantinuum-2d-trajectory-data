from fix_pathing import root_dir

from src.circuits.boson_circuits import trajectory_density, trajectory_current, density_readout, current_readout, commuting_bonds
from src.circuits.timer import Timer
from src.circuits.common_circuits import run_local

from src.io import save_to_hdf5

import sys
import os
import numpy as np


#PARAMETERS
V = 0.0
phi = 0.0 # Peierls phase in units of pi
dt = 0.31
p = 2*dt
steps = 2
shots = 2
start = 'random'
n_init = [] 
trotter_order = [0,1,2,3]  #Default sector order
trotter_staggered = False  # Whether to use staggered sectors or not

print("Running with parameters:")
print(f"V = {V}")
print(f"phi = {phi} (in units of pi)")
print(f"dt = {dt}")
print(f"p = {p}")
print(f"steps = {steps}")
print(f"shots = {shots}")
print(f"trotter_order = {trotter_order}")
print(f"trotter_staggered = {trotter_staggered}\n")
print(f"start = {start}")
if start == 'custom':
    print(f"n_init = {n_init}\n")


N = 16 # Number of sites
J = 1
dephasing = False

# Create the list of bonds for the sectors
sectors_list = commuting_bonds(N, staggered=trotter_staggered)
sectors_list = [sectors_list[i] for i in trotter_order]


#LOAD AND RUN CIRCUIT 
print("Building circuits...")
circuits = [trajectory_density(J, V, N=N, dt=dt, p=p, steps=steps, start=start, n_init = n_init, sector_list=sectors_list, phi=phi, name="2D Trajectory Bosons", dephasing = dephasing)]
for ii in range(4):
    circuits.append(trajectory_current(J, V, N=N, dt=dt, sector=sectors_list[ii], p=p, steps=steps, start=start, n_init = n_init, sector_list=sectors_list, phi=phi, name=f"2D Trajectory Bosons - Current {ii+1}", dephasing = dephasing))

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
    "sector_bond_1": sectors_list[0],
    "sector_bond_2": sectors_list[1],
    "sector_bond_3": sectors_list[2],
    "sector_bond_4": sectors_list[3]
}

coin1, coin2, out1, out2, trajectory_source, trajectory_sink, c_init, densities = density_readout(results_local[0], N=N, shots=shots, steps=steps)

data[f"density_circuit"] = {
        "coin1": coin1,
        "coin2": coin2,
        "trajectory_source": trajectory_source,
        "trajectory_sink": trajectory_sink,
        "c_init": c_init,
        "densities": densities
    }

for ii in range(4):

    coin1, coin2, out1, out2, trajectory_source, trajectory_sink, c_init, den_currents = current_readout(sectors_list[ii], results_local[ii+1], N=N, shots=shots, steps=steps)

    data[f"current_circuit_{ii+1}"] = {
            "coin1": coin1,
            "coin2": coin2,
            "out1": out1,
            "out2": out2,
            "den_currents": den_currents,
            "c_init": c_init,
            "trajectory_source": trajectory_source,
            "trajectory_sink": trajectory_sink
        }


print(f"Results read.\n")

print("Saving results...")



#SAVE THE OUTPUTS
# Create data directory if it doesn't exist
if not os.path.exists(f'data_local'):
    os.makedirs(f'data_local')

trotter_order_string = "_order" + str(trotter_order).replace(' ', '').replace('[','').replace(']','').replace(',','') if trotter_order != [0,1,2,3] else ""  # Format for filename
staggered_string = "_staggered" if trotter_staggered else ""
save_to_hdf5(data, f'data_local/bosons_{start}_V{V}_phi{phi}_dt{dt}_p{p}_steps{steps}_shots{shots}{trotter_order_string}{staggered_string}.h5')

print("Finished!")