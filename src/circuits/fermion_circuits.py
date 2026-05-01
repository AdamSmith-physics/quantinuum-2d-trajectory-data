import numpy as np
from pytket import Circuit, OpType

from .common_circuits import unitary, initial_state, source, drain, two_qubit_rotation


def trotter_step_fermions(circ, qr, ancilla, N, J, V, dt, phi=0.0):
    """Full trotter Hamiltonian: series of unitary building blocks applied 
    to commuting bonds simultaneously. Note '-J' on bonds [5,6] and [13,14]!

    Args:
        circ (Circuit): input quantum circuit
        qr (qr): qubit on numbered site
        N (int): number of sites
        J (int): hopping strength.
        V (int): interaction strength.
        dt (int): time interval of evolution (T/m).
        phi (float, optional): Peierls phase (IN UNITS OF PI!!). Defaults to 0.0.

    """

    #Green sector (h + v):
    circ.Sdg(ancilla[0])
    circ.H(ancilla[0])  
    circ.CZ(qr[2], ancilla[0])
    circ.CZ(ancilla[0], qr[5])
    unitary(circ, [qr[1], qr[2]], J, V, dt, phi=0*phi)
    unitary(circ, [qr[5], qr[6]], -J, V, dt, phi=1*phi)
    circ.CZ(ancilla[0], qr[5])
    circ.H(ancilla[0])
    circ.ZZPhase(0.5, qr[2], ancilla[0])
    circ.H(ancilla[0])
    circ.CZ(ancilla[0], qr[5])
    unitary(circ, [qr[1], qr[5]], J, V, dt)
    unitary(circ, [qr[2], qr[6]], J, V, dt)     
    circ.CZ(ancilla[0], qr[5])
    circ.CZ(qr[2], ancilla[0])
    circ.H(ancilla[0]) 

    circ.Sdg(ancilla[1])
    circ.H(ancilla[1])    
    circ.CZ(qr[10], ancilla[1])
    circ.CZ(ancilla[1], qr[13])
    unitary(circ, [qr[9], qr[10]], J, V, dt, phi=2*phi)
    unitary(circ, [qr[13], qr[14]], -J, V, dt, phi=3*phi)
    circ.CZ(ancilla[1], qr[13])
    circ.H(ancilla[1])
    circ.ZZPhase(0.5, qr[10], ancilla[1])
    circ.H(ancilla[1])
    circ.CZ(ancilla[1], qr[13])
    unitary(circ, [qr[9], qr[13]], J, V, dt)
    unitary(circ, [qr[10], qr[14]], J, V, dt)     
    circ.CZ(ancilla[1], qr[13])
    circ.CZ(qr[10], ancilla[1])
    circ.H(ancilla[1]) 

    unitary(circ, [qr[4], qr[8]], J, V, dt)
    unitary(circ, [qr[7], qr[11]], J, V, dt) 

    #Light blue sector
    for j in range(0, N, 2):
        row = j//4
        unitary(circ, [qr[j], qr[j+1]], J, V, dt, phi=row*phi)
        circ.CZ(qr[j], qr[j+1])

    #Yellow Sector
    circ.H(ancilla[0])      
    circ.CZ(qr[4], ancilla[0])
    circ.CZ(ancilla[0], qr[3])
    unitary(circ, [qr[0], qr[4]], J, V, dt)
    unitary(circ, [qr[3], qr[7]], J, V, dt)
    circ.CZ(ancilla[0], qr[3])
    circ.CZ(qr[4], ancilla[0])
    circ.H(ancilla[0]) 

    circ.H(ancilla[1])      
    circ.CZ(qr[12], ancilla[1])
    circ.CZ(ancilla[1], qr[11])
    unitary(circ, [qr[8], qr[12]], J, V, dt)
    unitary(circ, [qr[11], qr[15]], J, V, dt)
    circ.CZ(ancilla[1], qr[11])
    circ.CZ(qr[12], ancilla[1])
    circ.H(ancilla[1]) 

    unitary(circ, [qr[5], qr[9]], J, V, dt)
    unitary(circ, [qr[6], qr[10]], J, V, dt)

    #Dark blue sector
    for j in range(0, N, 2):
        circ.CZ(qr[j], qr[j+1])


def trajectory_density(J, V, N, dt, p, steps, start, n_init = [], phi=0.0, name="2D Trajectory Fermions", dephasing = False):
    """Run one quantum trajectory given initial parameters and at the end, measure occupation on each site. """

    circ = Circuit(name=name)
    qr = circ.add_q_register("q", N)  # qubits for particle positions
    ancilla = circ.add_q_register("a", 2)  # ancillas "a" or "b"
    qcoin = circ.add_q_register("q_coin", 2)  # coin for source / drain

    coin = []
    out = []
    for step in range(steps):
        coin.append(circ.add_c_register(f"coin_{step}", 2))
        out.append(circ.add_c_register(f"out_{step}", 2))
    densities = circ.add_c_register("densities", N)  # record particle densities
    c_init = circ.add_c_register("c_init", N)  # initial condition for
    ancilla_c = circ.add_c_register("ancilla_c", 2)  # ancilla measurements for stabilizer
    stabilizer = circ.add_c_register("stabilizer", 1)  # stabilizer for fermionic occupation


    ### INITIAL STATE ###
    initial_state(circ, N, filling = start, n_init = n_init)

    circ.CX(qr[4], ancilla[0])
    circ.CX(qr[5], ancilla[0])
    circ.CX(qr[6], ancilla[0])
    circ.CX(qr[7], ancilla[0])
    circ.CX(qr[8], ancilla[1])
    circ.CX(qr[9], ancilla[1])
    circ.CX(qr[10], ancilla[1])
    circ.CX(qr[11], ancilla[1])

    circ.Measure(ancilla[0], ancilla_c[0])
    circ.Reset(ancilla[0])
    circ.Measure(ancilla[1], ancilla_c[1])
    circ.Reset(ancilla[1])

    circ.Rx(-0.5, ancilla[0], condition_bits=[ancilla_c[0]], condition_value=0)
    circ.Rx(0.5, ancilla[0], condition_bits=[ancilla_c[0]], condition_value=1)

    circ.Rx(-0.5, ancilla[1], condition_bits=[ancilla_c[1]], condition_value=0)
    circ.Rx(0.5, ancilla[1], condition_bits=[ancilla_c[1]], condition_value=1)

    for ii in range(N):
        circ.Measure(qr[ii], c_init[ii])
        circ.Reset(qr[ii])
        circ.X(qr[ii], condition_bits=[c_init[ii]], condition_value=1)  # Flip back to measured value!
    ####################

    for step in range(steps):
        # Apply Trotter step
        trotter_step_fermions(circ, qr, ancilla, N, J, V, dt, phi=phi)

        # Source step
        source(circ, 0, qr, qcoin[0], coin[step][0], out[step][0], p, dephasing)

        # Drain step
        drain(circ, N-1, qr, qcoin[1], coin[step][1], out[step][1], p, dephasing)


    # Apply circuit and measure stabilizer before density to reduce measurement error
    circ.Rx(0.5, ancilla[0])  # rotate ancilla back to Z basis
    circ.Rx(0.5, ancilla[1])

    circ.CX(qr[4], ancilla[0])
    circ.CX(qr[5], ancilla[0])
    circ.CX(qr[6], ancilla[0])
    circ.CX(qr[7], ancilla[0])
    circ.CX(qr[8], ancilla[1])
    circ.CX(qr[9], ancilla[1])
    circ.CX(qr[10], ancilla[1])
    circ.CX(qr[11], ancilla[1])
    circ.CX(ancilla[0], ancilla[1])
    circ.Measure(ancilla[1], stabilizer[0])

    # Measure particle densities
    for ii in range(N):
        circ.Measure(qr[ii], densities[ii])

    return circ


def density_readout(results, N, shots, steps):
    """Reads the result of measurements on classical bits and uses it to obtain the quantum trajectory data 
    and density measurements. The results are stored as arrays for future use.

    Args:
        results (d): measurement result from quantum simulation
        N (int): total number of qubits
        shots (int): number of trajectories/measurement shots (one shot per trajectory)
        steps (int): timesteps of the evolution

    Returns:
        coin1, coin2: results of coinflips for each timestep. 
        out1, out2: outcome of corner measurements for each timestep.
        trajectory_source, trajectory_drain: Kraus operator applied on source/drain at each timestep.
        c_init: initial occupations 
        densities: measured occupation pattern for all sites in each trajectory
        stabilizer: measured value of stabilizer plaquette at the end of each trajectory
    """
        
    lookup = {f"{key}": value for key, value in results.c_bits.items()}  # convert to a dictionary with string keys
    data = results.get_shots()

    coin1 = np.zeros((shots, steps), dtype=int)
    coin2 = np.zeros((shots, steps), dtype=int)
    out1 = np.zeros((shots, steps), dtype=int)
    out2 = np.zeros((shots, steps), dtype=int)
    densities = np.zeros((shots, N), dtype=int)
    c_init = np.zeros((shots, N), dtype=int)
    stabilizer = np.zeros((shots, ), dtype=int)
    for step in range(steps):
        coin1[:, step] = data[:,lookup[f"coin_{step}[0]"]]
        coin2[:, step] = data[:,lookup[f"coin_{step}[1]"]]
        out1[:, step] = data[:,lookup[f"out_{step}[0]"]]
        out2[:, step] = data[:,lookup[f"out_{step}[1]"]]
    for ii in range(N):
        densities[:, ii] = data[:,lookup[f"densities[{ii}]"]]
        c_init[:, ii] = data[:,lookup[f"c_init[{ii}]"]]
    stabilizer[:] = data[:,lookup["stabilizer[0]"]]

    trajectory_source = np.zeros((shots, steps), dtype = int)
    trajectory_drain = np.zeros((shots, steps), dtype = int)

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
                    trajectory_drain[run, idx] = 1
                else:
                    trajectory_drain[run, idx] = 2
    
    return coin1, coin2, trajectory_source, trajectory_drain, c_init, densities, stabilizer



def current_rotations(circuit, qr, N, ancilla, sector, phi=0.0):
    """ Makes two-qubit rotations in given sector on both qubits and ancillas involved. 
    This is necessary to compute currents from density measurements"""

    if sector == 'sector1':
        print("Running current rotations for sector 1") # red sector
        #Bonds (4-8), (7-11):
        two_qubit_rotation(circuit, qr, 4, 8)  
        two_qubit_rotation(circuit, qr, 7, 11) 
        
        #Bonds (1-2), (13-14), (5-6), (9-10):
        two_qubit_rotation(circuit, qr, 1, 2, phi=0*phi)
        two_qubit_rotation(circuit, qr, 5, 6, phi=1*phi)
        circuit.Rx(0.5, ancilla[0])

        two_qubit_rotation(circuit, qr, 9, 10, phi=2*phi)
        two_qubit_rotation(circuit, qr, 13, 14, phi=3*phi)
        circuit.Rx(0.5, ancilla[1])

    elif sector == 'sector2':
        print("Running current rotations for sector 2")     # green sector
        #Bonds (1,5), (2, 6)
        circuit.Ry(-0.5, ancilla[0])
        circuit.CZ(qr[5], ancilla[0])
        circuit.CZ(ancilla[0], qr[6])
        two_qubit_rotation(circuit, qr, 1, 5)  
        two_qubit_rotation(circuit, qr, 2, 6) 
        circuit.Rx(0.5, ancilla[0])

        #Bonds (9,13), (10, 14)
        circuit.Ry(-0.5, ancilla[1])
        circuit.CZ(qr[9], ancilla[1])
        circuit.CZ(ancilla[1], qr[10])
        two_qubit_rotation(circuit, qr, 9, 13)  
        two_qubit_rotation(circuit, qr, 10, 14) 
        circuit.Rx(0.5, ancilla[1])

    elif sector == 'sector3':
        print("Running current rotations for sector 3")     # blue sector
        for site in range(0, N, 2):
            row = site//4
            two_qubit_rotation(circuit, qr, site, site+1, phi=row*phi)

        circuit.Rx(0.5, ancilla[0])
        circuit.Rx(0.5, ancilla[1])

    elif sector == 'sector4':
        print("Running current rotations for sector 4")    # yellow sector
        
        for site in range(0, N, 2):
            circuit.CZ(qr[site], qr[site+1])

        two_qubit_rotation(circuit, qr, 5, 9)
        two_qubit_rotation(circuit, qr, 6, 10)

        circuit.Ry(-0.5, ancilla[0])
        circuit.CZ(qr[4], ancilla[0])
        circuit.CZ(ancilla[0], qr[7])
        two_qubit_rotation(circuit, qr, 0, 4)
        two_qubit_rotation(circuit, qr, 3, 7)
        circuit.Rx(0.5, ancilla[0])

        circuit.Ry(-0.5, ancilla[1])
        circuit.CZ(qr[8], ancilla[1])
        circuit.CZ(ancilla[1], qr[11])
        two_qubit_rotation(circuit, qr, 8, 12)
        two_qubit_rotation(circuit, qr, 11, 15)
        circuit.Rx(0.5, ancilla[1])

    else:
        raise ValueError("Invalid sector specified in current_rotations. Choose from 'sector1', 'sector2', 'sector3', or 'sector4'.")



def trajectory_current(J, V, N, sector, dt, p, steps, start, n_init = [], phi=0.0, name="1D Trajectory Fermions", dephasing = False):
    """Run one quantum trajectory given initial parameters and at the end measure densities on the sites in the selected non-overlapping sector of bonds.
     From these measurements instantaneous currents will be computed. """
    
    circ = Circuit(name=name)
    qr = circ.add_q_register("q", N)  # qubits for particle positions
    ancilla = circ.add_q_register("a", 2)  # ancillas "a/b"
    qcoin = circ.add_q_register("q_coin", 2)  # coin for source / drain

    coin = []
    out = []
    for step in range(steps):
        coin.append(circ.add_c_register(f"coin_{step}", 2))
        out.append(circ.add_c_register(f"out_{step}", 2))
    c_init = circ.add_c_register("c_init", N)  # initial condition
    densities = circ.add_c_register("densities", N)  # record particle densities
    ancilla_c_init = circ.add_c_register("ancilla_c_init", 2)  # record particle densities
    den_ancillas = circ.add_c_register("den_ancillas", 2)  # record ancilla measurements for current
    stabilizer = circ.add_c_register("stabilizer", 1)

    # Create random product state initial state
    initial_state(circ, N, filling = start, n_init = n_init)

    for ii in range(N):
        circ.Measure(qr[ii], c_init[ii])
        circ.Reset(qr[ii])
        circ.X(qr[ii], condition_bits=[c_init[ii]], condition_value=1)  # Flip back to measured value!


    # Alternative circuit for state initialization to reduce measurement error
    circ.CX(qr[4], ancilla[0])
    circ.CX(qr[5], ancilla[0])
    circ.CX(qr[6], ancilla[0])
    circ.CX(qr[7], ancilla[0])
    circ.CX(qr[8], ancilla[1])
    circ.CX(qr[9], ancilla[1])
    circ.CX(qr[10], ancilla[1])
    circ.CX(qr[11], ancilla[1])

    circ.Measure(ancilla[0], ancilla_c_init[0])
    circ.Reset(ancilla[0])
    circ.Measure(ancilla[1], ancilla_c_init[1])
    circ.Reset(ancilla[1])

    circ.Rx(-0.5, ancilla[0], condition_bits=[ancilla_c_init[0]], condition_value=0)
    circ.Rx(0.5, ancilla[0], condition_bits=[ancilla_c_init[0]], condition_value=1)

    circ.Rx(-0.5, ancilla[1], condition_bits=[ancilla_c_init[1]], condition_value=0)
    circ.Rx(0.5, ancilla[1], condition_bits=[ancilla_c_init[1]], condition_value=1)


    for step in range(steps):
        # Apply Trotter step
        trotter_step_fermions(circ, qr, ancilla, N, J, V, dt, phi=phi)

        # Source step
        source(circ, 0, qr, qcoin[0], coin[step][0], out[step][0], p, dephasing)

        # drain step
        drain(circ, N-1, qr, qcoin[1], coin[step][1], out[step][1], p, dephasing)

    # Measure particle densities
    current_rotations(circ, qr, N, ancilla, sector, phi=phi)

    circ.Measure(ancilla[0], den_ancillas[0])
    circ.Measure(ancilla[1], den_ancillas[1])

    # Density measurements
    for ii in range(N):
        circ.Measure(qr[ii], densities[ii])


    ### Classical post-processing to get stabilizer value ###
    if sector == "sector1":
        print("Running measurements for sector 1")
        circ.add_clexpr_from_logicexp(den_ancillas[0] ^ den_ancillas[1] ^ densities[4] ^ densities[5] ^ densities[6] ^ densities[7] ^ densities[8] ^ densities[9] ^ densities[10] ^ densities[11], stabilizer)
    elif sector == "sector2":
        print("Running measurements for sector 2")
        circ.add_clexpr_from_logicexp(den_ancillas[0] ^ den_ancillas[1] ^ densities[4] ^ densities[7] ^ densities[8] ^ densities[11], stabilizer)
    elif sector == "sector3":
        print("Running measurements for sector 3")
        circ.add_clexpr_from_logicexp(den_ancillas[0] ^ den_ancillas[1] ^ densities[4] ^ densities[5] ^ densities[6] ^ densities[7] ^ densities[8] ^ densities[9] ^ densities[10] ^ densities[11], stabilizer)
    elif sector == "sector4":
        print("Running measurements for sector 4")
        circ.add_clexpr_from_logicexp(den_ancillas[0] ^ den_ancillas[1] ^ densities[5] ^ densities[6] ^ densities[9] ^ densities[10], stabilizer)
    else:
        raise ValueError("Invalid sector specified in trajectory_current. Choose from 'sector1', 'sector2', 'sector3', or 'sector4'.")
    ###################


    return circ


def current_readout(sector, results, N, shots, steps):
    """Reads the result of measurements on classical bits and uses it to obtain the quantum trajectory data 
    and density measurements on sites included in current sector. The results are stored as arrays for future use.

    Args:
        results (d): measurement result from quantum simulation
        N (int): total number of qubits
        shots (int): number of trajectories/measurement shots (one shot per trajectory)
        steps (int): timesteps of the evolution

    Returns:
        coin1, coin2: results of coinflips for each timestep. 
        out1, out2: outcome of corner measurements for each timestep.
        trajectory_source, trajectory_drain: Kraus operator applied on source/drain at each timestep.
        c_init, ancilla_c_init: initial occupations on sites and ancillas
        den_currents, den_ancillas: measured occupation pattern for sites and ancillas in given current sector for each trajectory.
        stabilizer: measured value of stabilizer plaquette at the end of each trajectory
    """

    lookup = {f"{key}": value for key, value in results.c_bits.items()}  # convert to a dictionary with string keys
    data = results.get_shots()

    edges = [site for pairs in sector for site in pairs]

    coin1 = np.zeros((shots, steps), dtype=int)
    coin2 = np.zeros((shots, steps), dtype=int)
    out1 = np.zeros((shots, steps), dtype=int)
    out2 = np.zeros((shots, steps), dtype=int)
    den_currents = np.zeros((shots, N), dtype = int)          #for each sector
    den_ancillas = np.zeros((shots, 2), dtype = int)
    stabilizer = np.zeros((shots, ), dtype=int)
    c_init = np.zeros((shots, N), dtype=int)
    ancilla_c_init = np.zeros((shots, 2), dtype = int)
    for step in range(steps):
        coin1[:, step] = data[:,lookup[f"coin_{step}[0]"]]
        coin2[:, step] = data[:,lookup[f"coin_{step}[1]"]]
        out1[:, step] = data[:,lookup[f"out_{step}[0]"]]
        out2[:, step] = data[:,lookup[f"out_{step}[1]"]]
    for ii in range(N):
        c_init[:, ii] = data[:,lookup[f"c_init[{ii}]"]]
    for ii in edges:
        den_currents[:, ii] = data[:, lookup[f"densities[{ii}]"]]
    ancilla_c_init[:, 0] = data[:, lookup["ancilla_c_init[0]"]]
    ancilla_c_init[:, 1] = data[:, lookup["ancilla_c_init[1]"]]
    den_ancillas[:, 0] = data[:, lookup["den_ancillas[0]"]]
    den_ancillas[:, 1] = data[:, lookup["den_ancillas[1]"]]
    stabilizer[:] = data[:,lookup["stabilizer[0]"]]

    trajectory_source = np.zeros((shots, steps), dtype = int)
    trajectory_drain = np.zeros((shots, steps), dtype = int)

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
                    trajectory_drain[run, idx] = 1
                else:
                    trajectory_drain[run, idx] = 2
    
    return coin1, coin2, out1, out2, den_currents, den_ancillas, c_init, ancilla_c_init, trajectory_source, trajectory_drain, stabilizer
