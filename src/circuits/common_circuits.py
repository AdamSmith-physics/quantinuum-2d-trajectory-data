import numpy as np
from pytket import Circuit, OpType
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline


def run_local(circuits, shots):
    # Initialise the local Quantinuum backend
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)      # run on local noiseless emulator "H1-1LE"

    compiled_circuits = [backend.get_compiled_circuit(circuit)for circuit in circuits]
    results = backend.run_circuits(compiled_circuits, n_shots=shots)

    return results


def initial_state(circ, num_qubits, filling = 'random', n_init =[]):
    """ Circuit to create an initial density pattern. Defaults to 'random'.
    """

    if filling == 'checkerboard':
        n = int(np.sqrt(num_qubits))
        # create a n * n matrix
        x = np.zeros((n, n), dtype=int)
        # fill with 1 the alternate cells in rows and columns
        x[1::2, ::2] = 1
        x[::2, 1::2] = 1
        x = x.flatten()
        ones = [i for i, e in enumerate(x) if e != 0]
        for i in ones:
            circ.X(i)

    elif filling == 'full':
        for qubit in range(num_qubits):
            circ.X(qubit)

    elif filling == 'random':
        # random initial state with source filled and sink empty
        circ.X(0)
        for qubit in range(1, num_qubits-1):
            circ.H(qubit)

    elif filling == 'custom':
        for qubit in range(num_qubits):
            theta = 2/np.pi * np.arcsin(np.sqrt(n_init[qubit]))
            circ.Ry(theta, qubit)


def unitary(circ, wires, J, V, dt, phi=0.0): 
    """Two-qubit building block of the trotterized Hamiltonian.

    Args:
        circ (Circuit): input quantum circuit
        wires (list): sites on a chosen bond. For magnetic field this assumes wires[0] < wires[1]
        J (int): hopping strength.
        V (int): interaction strength.
        dt (int): time interval of evolution (T/m).
        phi (float, optional): Peierls phase (IN UNITS OF PI!!). Defaults to 0.0.
    """

    # Rotation angles (I would account for the factor of 2 and 4 here instead. So J always means J)
    J_dt = J*dt/2
    V_dt = V*dt/4 

    if phi != 0.0:
        circ.Rz(-phi/2, wires[0])
        circ.Rz(phi/2, wires[1])

    # Implement exp^[i(JXX + JYY - VZZ)] for two neighbouring sites
    circ.TK2(-2*J_dt/np.pi, -2*J_dt/np.pi, 2*V_dt/np.pi, wires[0], wires[1])

    if phi != 0.0:
        circ.Rz(phi/2, wires[0])
        circ.Rz(-phi/2, wires[1])

    # Implement exp^[i(VZ_i + VZ_j)] for two neighbouring sites
    circ.Rz(-2*V_dt/np.pi, wires[1])
    circ.Rz(-2*V_dt/np.pi, wires[0])


def drain(circ, site, qr, qcoin, coin, out, p, dephasing = False):
    """ Non-unitary evolution on the drain conditioned on quantum coin flip.
    """
    # rotation angl for quantum coin 
    theta = 2/np.pi * np.arcsin(np.sqrt(p))
    
    circ.Ry(theta, qcoin)  # coin flip
    circ.Measure(qcoin, coin)  # measure coin
    circ.Reset(qcoin)  # reset coin and keep in zero

    # mid-circuit measurement and reset (should only happen if coin[step] == 1)
    circ.Measure(qr[site], out, condition_bits=[coin], condition_value=1)  # measure particle position
    circ.Reset(qr[site], condition_bits=[coin], condition_value=1)
    if dephasing:
        circ.X(qr[site], condition_bits=[out], condition_value=1)  # re-use measurement 


def source(circ, site, qr, qcoin, coin, out, p,dephasing = False):
    """  Non-unitary evolution on the source conditioned on quantum coin flip.
    """
    if dephasing:
        drain(circ, site, qr, qcoin, coin, out, p, dephasing)
    else:
        drain(circ, site, qr, qcoin, coin, out, p, dephasing)
        circ.X(qr[site], condition_bits=[coin], condition_value=1)  # create particle at source site


def two_qubit_rotation(circuit, qr, qubit1, qubit2, phi=0.0):
    """ Makes the two-qubit rotation: 
            M = <R_z(-phi/2) XY R_z(phi/2)> -> <1Z>,  <R_z(phi/2) YX R_z(-phi/2)> -> <Z1>
        necessary to measure currents on a given bond.
    """

    if phi != 0.0:
        circuit.Rz(-phi/2, qr[qubit1])
        circuit.Rz(phi/2, qr[qubit2])

    circuit.Ry(-0.5, qr[qubit1])
    circuit.Ry(-0.5, qr[qubit2])
    circuit.CZ(qr[qubit1], qr[qubit2])
    circuit.Sdg(qr[qubit1]) 
    circuit.Sdg(qr[qubit2])
    circuit.H(qr[qubit1])
    circuit.H(qr[qubit2])
