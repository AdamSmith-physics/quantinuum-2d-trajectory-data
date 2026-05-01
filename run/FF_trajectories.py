from fix_pathing import root_dir

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.linalg import expm
from multiprocessing import Manager, Pool, cpu_count
import time
import pickle

from src.initial_state import checkerboard_state, empty_state, random_state, product_state
from src.simulation import trajectory
from src.setup import construct_H, get_bonds
from src.parameter_dataclasses import SimulationParameters

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

num_processes = cpu_count() - 1  # Leave one core free for the OS
if num_processes < 1:
    raise ValueError("Not enough CPU cores available for multiprocessing.")

######################## SIMULATION PARAMETERS ########################
dt = 0.21
p = 2*dt
Nx = 7
Ny = 7
N = Nx*Ny
phi = 0.0 # 2/((Nx-1)*(Ny-1))   # Magnetic field strength
B = phi*np.pi                   # Magnetic field in units of flux quantum
t = 0.0                         # Diagonal hopping
num_iterations = 100
steps = 100
site_in = 0                     # Site where the current is injected (source)
site_out = N-1                  # Site where the current is extracted (drain)
drive_type = "current"          # "current", "dephasing"
corner_dephasing = False        # Whether to apply dephasing at the corners
initial_state = "random"        # "checkerboard", "empty", "random", "custom"

even_parity = False             # Only used for random state
occupation_list = [0,1,0,1,1,0,1,0,0]       # Only used for custom state
########################################################################

# Spread iterations across the available processes
batch_size = [num_iterations // num_processes]*num_processes
# Spread any remaining iterations across the first few processes
for i in range(num_iterations % num_processes):
    batch_size[i] += 1

t_list = np.linspace(0, steps*dt, steps+1)

sim_params = SimulationParameters(
    steps=steps,
    Nx=Nx,
    Ny=Ny,
    p=p,
    bonds=get_bonds(Nx, Ny, site_in, site_out, t=t),
    site_in=site_in,
    site_out=site_out,
    drive_type=drive_type,
    corner_dephasing=corner_dephasing,
    initial_state=initial_state
)

if not drive_type in ["current", "dephasing"]:
    raise ValueError(f"Invalid drive_type: {drive_type}")

if not initial_state in ["checkerboard", "empty", "random", "custom"]:
    raise ValueError(f"Invalid initial_state: {initial_state}")

# Needed for multiprocessing
if __name__ == "__main__":

    ######################## RUN SIMULATION #######################

    bonds = sim_params.to_dict()["bonds"]
    H = construct_H(Nx, Ny, B, t)
    U = expm(-1j*H*dt)

    alpha = None
    if initial_state == "checkerboard":
        alpha = checkerboard_state(Nx, Ny)
    elif initial_state == "empty":
        alpha = empty_state(Nx, Ny)
    elif initial_state == "random":
        alpha = random_state(Nx, Ny, even_parity=even_parity)
    elif initial_state == "custom":
        alpha = product_state(occupation_list, Nx, Ny)

    manager = Manager()
    data = manager.dict()
    data["H"] = H
    data["U"] = U
    data["alpha"] = alpha
    data["completed"] = 0

    print(f"CPU count: {cpu_count()}")

    K_avg = 0.
    n_avg = 0.
    n_sq_avg = 0.
    avg_currents = 0.
    currents_sq_avg = 0.
    avg_dd_correlations = 0.

    t1 = time.perf_counter()

    with Pool(processes=num_processes) as pool:
        for procid in range(num_processes):
            pool.apply_async(trajectory, args=(procid, data, batch_size[procid], steps, sim_params))
        pool.close()
        pool.join()

    # # For debugging the parallel execution, you can uncomment the following lines:
    # for procid in range(num_iterations):
    #     trajectory(procid, data, steps, batch_size[procid], Nx, Ny, p, bonds, site_in, site_out, drive_type, corner_dephasing, initial_state)
        
    for i in range(num_processes):
        res = data[i]
        K_avg += res["K_list"] / num_iterations
        n_avg += res["n_list"] / num_iterations
        n_sq_avg += res["n_list"]**2 / num_iterations
        avg_currents += res["currents_list"] / num_iterations
        currents_sq_avg += res["currents_list"]**2 / num_iterations
        avg_dd_correlations += res["density_correlations"] / num_iterations

    t2 = time.perf_counter()
    print(f"\n Finished all trajectories")
    print(f"Time taken (parallel): {t2 - t1} seconds")
    

    ################# PLOT THE FINAL TIME SNAPSHOT  #################

    plotting_threshold = 0.0  # Threshold for plotting currents
    marker_size = 750 * (3/Nx)**2  # Size of the markers for the density plot
    arrow_width = 0.035 * 3/Nx  # Width of the arrows in the quiver plot

    # single line definition of empty lists for X, Y, U, V, C
    X = []; Y = []; U = []; V = []; C = []
    for i, bond in enumerate(bonds):
        # convert back from n to x,y coordinates
        x1, y1 = bond[0] % Nx, bond[0] // Nx
        x2, y2 = bond[1] % Nx, bond[1] // Nx

        if np.abs(avg_currents[-1,i]) > plotting_threshold*np.abs(avg_currents[-1,:]).max():
            C.append(np.abs(avg_currents[-1,i]))

            if np.real(avg_currents[-1,i]) > 0:
                X.append(x1)
                Y.append(y1)
                U.append(x2-x1)
                V.append(y2-y1)
            else:
                X.append(x2)
                Y.append(y2)
                U.append(x1-x2)
                V.append(y1-y2)

    fig, ax = plt.subplots(figsize = (6,6))

    p1 = ax.quiver(X, Y, U, V, C, cmap='YlGn', angles='xy', scale_units='xy', scale=1, width=arrow_width)
    cb1 = plt.colorbar(p1, ax=ax, orientation='vertical', shrink=0.53, pad=0.1)
    cb1.ax.tick_params(labelsize=13)
    cb1.set_label('Current Magnitude', labelpad=1, fontsize = 15)

    X = []; Y = []; C = []
    for x in range(Nx):
        for y in range(Ny):
            n = x % Nx + y*Nx
            X.append(x)
            Y.append(y)
            C.append(n_avg[-1,n])

    p2 = ax.scatter(X, Y, c=C, cmap='RdBu_r', s=marker_size, edgecolors= "black", vmin=0, vmax=1)
    cb2 = plt.colorbar(p2, ax=ax, orientation='vertical', shrink=0.53, pad=0.1)
    cb2.set_ticks(np.arange(0, 1.1, 0.2))
    cb2.ax.tick_params(labelsize=13)
    cb2.set_label('Density', labelpad=1, fontsize = 15)

    # Add automatic padding to prevent cutoff
    ax.margins(0.1 * 3/Nx)  # Add padding around the data
    ax.set_axis_off()
    ax.set_aspect('equal')

    plt.savefig(f"figures/FF_{Nx}x{Ny}_phi{phi}_dt{dt}_p{p}_steps{steps}_tajectories{num_iterations}.pdf", bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.show()
