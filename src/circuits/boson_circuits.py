import numpy as np
from pytket import Circuit, OpType

from .common_circuits import unitary, initial_state, source, sink, two_qubit_rotation

np.set_printoptions(legacy='1.25')


def commuting_bonds(N, staggered = True):
    '''Creates a list of sites for the 4 sectors of commuting bonds.
    Default order is horizontal_even, vertical_even, horizontal_odd, vertical_odd.
    Here even refers to including site 0, odd to not including site 0.
    '''
    n = int(np.sqrt(N))

    sector_list = []

    for start_site in ["even", "odd"]:
        for direction in ['horizontal', 'vertical']:

            sector = []

            if direction == 'horizontal':
                grid = np.arange(n*n).reshape(n, n)
            if direction == 'vertical':
                grid = np.arange(n*n).reshape(n, n).T

            shift = 0
            if start_site == "odd":
                shift = 1

            for l in range(n):
                start = (staggered*l + shift) % 2

                for m in range(start, n-1, 2):
                    pair = [grid[l,m], grid[l,m+1]]
                    sector.append(pair)

            sector_list.append(sector)

    return sector_list


def trotter_step_bosons(circ, N, J, V, dt, sector_list, phi=0.0):
    """Full trotter Hamiltonian: series of unitary building blocks applied 
        to commuting bonds simultaneously. 4 Commuting sectors in total.

    Args:
        N (int): number of sites
        J (int): hopping strength.
        V (int): interaction strength.
        dt (int): time interval of evolution (T/M).
        phi (float, optional): Peierls phase (IN UNITS OF PI!!). Defaults to 0.0.

    """

    for sector in sector_list:
        for pair in sector: 
            # Dealing with magnetic field 
            bond_phi = 0.0
            row = pair[0] // 4
            if pair[1]//4 != row:
                pass
            else:
                bond_phi = row*phi

            unitary(circ, [pair[0], pair[1]], J, V, dt, phi=bond_phi)



def trajectory_density(J, V, N, dt, p, steps, start, n_init = [], sector_list=None, phi=0.0, name="2D Trajectory Bosons", dephasing = False):

    if sector_list is None:
        sector_list = commuting_bonds(N, staggered=True)  # Default sectors of commuting bonds

    print(f"Using sectors:\n{sector_list}")

    circ = Circuit(name=name)
    qr = circ.add_q_register("q", N)  # qubits for particle positions
    qcoin = circ.add_q_register("q_coin", 2)  # coin for source / sink

    coin = []
    out = []
    for step in range(steps):
        coin.append(circ.add_c_register(f"coin_{step}", 2))
        out.append(circ.add_c_register(f"out_{step}", 2))
    densities = circ.add_c_register("densities", N)  # record particle densities
    c_init = circ.add_c_register("c_init", N)  # initial condition for

    # Create random product state initial state
    initial_state(circ, N, filling = start, n_init= n_init)
    for ii in range(N):
        circ.Measure(qr[ii], c_init[ii])
        circ.Reset(qr[ii])
        circ.X(qr[ii], condition_bits=[c_init[ii]], condition_value=1)  # Flip back to measured value!

    for step in range(steps):
        # Apply Trotter step
        trotter_step_bosons(circ, N, J, V, dt, sector_list, phi=phi)

        # Source step
        source(circ, 0, qr, qcoin[0], coin[step][0], out[step][0], p, dephasing)

        # Sink step
        sink(circ, N-1, qr, qcoin[1], coin[step][1], out[step][1], p, dephasing)

    # Measure particle densities
    for ii in range(N):
        circ.Measure(qr[ii], densities[ii])

    return circ


def trajectory_current(J, V, N, sector, dt, p, steps, start, n_init = [], sector_list=None, phi=0.0, name="2D Trajectory Bosons", dephasing = False):

    if sector_list is None:
        sector_list = commuting_bonds(N, staggered=True)  # Default sectors of commuting bonds

    circ = Circuit(name=name)
    qr = circ.add_q_register("q", N)  # qubits for particle positions
    qcoin = circ.add_q_register("q_coin", 2)  # coin for source / sink

    coin = []
    out = []
    for step in range(steps):
        coin.append(circ.add_c_register(f"coin_{step}", 2))
        out.append(circ.add_c_register(f"out_{step}", 2))
    currents = circ.add_c_register("currents", N)  # record particle currents
    c_init = circ.add_c_register("c_init", N)  # initial condition for

    # Create random product state initial state
    initial_state(circ, N, filling = start, n_init = n_init)
    for ii in range(N):
        circ.Measure(qr[ii], c_init[ii])
        circ.Reset(qr[ii])
        circ.X(qr[ii], condition_bits=[c_init[ii]], condition_value=1)  # Flip back to measured value!

    for step in range(steps):
        # Apply Trotter step
        trotter_step_bosons(circ, N, J, V, dt, sector_list, phi=phi)

        # Source step
        source(circ, 0, qr, qcoin[0], coin[step][0], out[step][0], p, dephasing)

        # Sink step
        sink(circ, N-1, qr, qcoin[1], coin[step][1], out[step][1], p, dephasing)

    circ.add_barrier(range(N)) # add a barrier on all qubits and bits
    # Turn current into density
    for bond in sector:
        # Dealing with magnetic field 
        bond_phi = 0.0
        row = bond[0] // 4
        if bond[1]//4 != bond[0]//4:
            pass
        else:
            # Only apply magnetic field horizontally. Multiple by row number.
            bond_phi = row*phi

        two_qubit_rotation(circ, qr, bond[0], bond[1], phi=bond_phi)

    # Measure particle densities
    for ii in range(N):
        circ.Measure(qr[ii], currents[ii])

    return circ


def density_readout(results, N, shots, steps):
    lookup = {f"{key}": value for key, value in results.c_bits.items()}  # convert to a dictionary with string keys
    data = results.get_shots()

    coin1 = np.zeros((shots, steps), dtype=int)
    coin2 = np.zeros((shots, steps), dtype=int)
    out1 = np.zeros((shots, steps), dtype=int)
    out2 = np.zeros((shots, steps), dtype=int)
    densities = np.zeros((shots, N), dtype=int)
    c_init = np.zeros((shots, N), dtype=int)
    for step in range(steps):
        coin1[:, step] = data[:,lookup[f"coin_{step}[0]"]]
        coin2[:, step] = data[:,lookup[f"coin_{step}[1]"]]
        out1[:, step] = data[:,lookup[f"out_{step}[0]"]]
        out2[:, step] = data[:,lookup[f"out_{step}[1]"]]
    for ii in range(N):
        densities[:, ii] = data[:,lookup[f"densities[{ii}]"]]
        c_init[:, ii] = data[:,lookup[f"c_init[{ii}]"]]

    trajectory_source = np.zeros((shots, steps), dtype = int)
    trajectory_sink = np.zeros((shots, steps), dtype = int)

    for run in range(shots):
        for idx in range(steps):
            flip1 = coin1[run,idx]
            flip2 = coin2[run,idx]
            if flip1 == 1:
                if out1[run,idx] == 0:
                    trajectory_source[run, idx] = 1
                else:
                    trajectory_source[run, idx] = 2
            if flip2 == 1:
                if out2[run,idx] == 0:
                    trajectory_sink[run, idx] = 1
                else:
                    trajectory_sink[run, idx] = 2
    
    return coin1, coin2, out1, out2, trajectory_source, trajectory_sink, c_init, densities


def current_readout(sector, results, N, shots, steps):
    lookup = {f"{key}": value for key, value in results.c_bits.items()}  # convert to a dictionary with string keys
    data = results.get_shots()

    edges = [site for pairs in sector for site in pairs]

    coin1 = np.zeros((shots, steps), dtype=int)
    coin2 = np.zeros((shots, steps), dtype=int)
    out1 = np.zeros((shots, steps), dtype=int)
    out2 = np.zeros((shots, steps), dtype=int)
    den_currents = np.zeros((shots, N), dtype=int)          #for each sector
    c_init = np.zeros((shots, N), dtype=int)
    for step in range(steps):
        coin1[:, step] = data[:,lookup[f"coin_{step}[0]"]]
        coin2[:, step] = data[:,lookup[f"coin_{step}[1]"]]
        out1[:, step] = data[:,lookup[f"out_{step}[0]"]]
        out2[:, step] = data[:,lookup[f"out_{step}[1]"]]
    for ii in range(N):
        c_init[:, ii] = data[:,lookup[f"c_init[{ii}]"]]
    for ii in edges:
        den_currents[:, ii] = data[:, lookup[f"currents[{ii}]"]]

    trajectory_source = np.zeros((shots, steps), dtype = int)
    trajectory_sink = np.zeros((shots, steps), dtype = int)

    for run in range(shots):
        for idx in range(steps):
            flip1 = coin1[run,idx]
            flip2 = coin2[run,idx]
            if flip1 == 1:
                if out1[run,idx] == 0:
                    trajectory_source[run, idx] = 1
                else:
                    trajectory_source[run, idx] = 2
            if flip2 == 1:
                if out2[run,idx] == 0:
                    trajectory_sink[run, idx] = 1
                else:
                    trajectory_sink[run, idx] = 2
    
    return coin1, coin2, out1, out2, trajectory_source, trajectory_sink, c_init, den_currents