from src.io import load_from_hdf5, load_key_from_hdf5
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def get_device_observables(path, filename, statistics, shots):
    """Calculates average densities and currents with corresponding error 
        from the simulated data on quantinuum's platforms (H1 or Emulator)"

    Args:
        path (str): folder cooresponding to used device data (H1-1 or Emulator)
        filename (str): name of .hd5 file
        statistics (str): 'fermions' or 'bosons'
        shots (int): number of shots/trajectories we want to sample

    Returns:
        n_avg: densities averaged over trajectories
        n_sem: statistical error of the mean for densities
        currents:  current magnitudes averaged over trajectories
        currents_sem: statistical error of the mean for currents
        bonds: list of bonds in circuit order
    """

    parameters = load_key_from_hdf5(path+filename, 'parameters')
    densities = load_key_from_hdf5(path+filename, "density_circuit/densities")[:shots]
    den_currents1 = load_key_from_hdf5(path+filename, f"current_circuit_1/den_currents")[:shots]    # density measurements for current sector 1
    den_currents2 = load_key_from_hdf5(path+filename, f"current_circuit_2/den_currents")[:shots]    # density measurements for current sector 2
    den_currents3 = load_key_from_hdf5(path+filename, f"current_circuit_3/den_currents")[:shots]    # density measurements for current sector 3
    den_currents4 = load_key_from_hdf5(path+filename, f"current_circuit_4/den_currents")[:shots]    # density measurements for current sector 4

    bonds = np.concatenate([parameters['sector_bond_1'], parameters['sector_bond_2'], parameters['sector_bond_3'],parameters['sector_bond_4']])
    if statistics == 'fermions':
        negative_bonds = [[5,6], [13,14]]

    n_avg = np.mean(densities, axis=0)      
    n_sem = np.std(densities, axis = 0)/np.sqrt(shots) 

    currents = []
    currents_std = []   # standard deviation
    currents_sem = []  

    avg_den_currents1 = np.mean(den_currents1, axis=0)
    avg_den_currents2 = np.mean(den_currents2, axis=0)
    avg_den_currents3 = np.mean(den_currents3, axis=0)
    avg_den_currents4 = np.mean(den_currents4, axis=0)

    bonds1 = parameters['sector_bond_1']
    bonds2 = parameters['sector_bond_2']
    bonds3 = parameters['sector_bond_3']
    bonds4 = parameters['sector_bond_4']
    bonds = np.concatenate([bonds1, bonds2, bonds3, bonds4])

    if statistics == 'bosons':

        # current calculated as : <1Z> - <Z1>  for all bonds within each sector
        for ii, (i, j) in enumerate(bonds1):
            currents.append((avg_den_currents1[i] - avg_den_currents1[j]))
            currents_std.append(np.std(den_currents1[:,i] - den_currents1[:,j], axis=0))
            currents_sem.append(np.std(den_currents1[:,i] - den_currents1[:,j], axis=0)/np.sqrt(den_currents1.shape[0]))

        for ii, (i, j) in enumerate(bonds2):
            currents.append((avg_den_currents2[i] - avg_den_currents2[j]))
            currents_std.append(np.std(den_currents2[:,i] - den_currents2[:,j], axis=0))
            currents_sem.append(np.std(den_currents2[:,i] - den_currents2[:,j], axis=0)/np.sqrt(den_currents2.shape[0]))

        for ii, (i, j) in enumerate(bonds3):
            currents.append((avg_den_currents3[i] - avg_den_currents3[j]))
            currents_std.append(np.std(den_currents3[:,i] - den_currents3[:,j], axis=0))
            currents_sem.append(np.std(den_currents3[:,i] - den_currents3[:,j], axis=0)/np.sqrt(den_currents3.shape[0]))

        for ii, (i, j) in enumerate(bonds4):
            currents.append((avg_den_currents4[i] - avg_den_currents4[j]))
            currents_std.append(np.std(den_currents4[:,i] - den_currents4[:,j], axis=0))
            currents_sem.append(np.std(den_currents4[:,i] - den_currents4[:,j], axis=0)/np.sqrt(den_currents4.shape[0]))

    elif statistics == 'fermions':

        # calculate currents from density measurements for specific bonds i-j
        den_ancillas1 = load_key_from_hdf5(path+filename, f"current_circuit_1/den_ancillas")[:shots]
        den_ancillas2 = load_key_from_hdf5(path+filename, f"current_circuit_2/den_ancillas")[:shots]
        den_ancillas3 = load_key_from_hdf5(path+filename, f"current_circuit_3/den_ancillas")[:shots]
        den_ancillas4 = load_key_from_hdf5(path+filename, f"current_circuit_4/den_ancillas")[:shots]

        # for 4-8, 7-11
        for (i, j) in [(4,8), (7,11)]:
            currents.append((avg_den_currents1[i] - avg_den_currents1[j]))
            currents_std.append(np.std(den_currents1[:,i] - den_currents1[:,j], axis=0))
            currents_sem.append(np.std(den_currents1[:,i] - den_currents1[:,j], axis=0)/np.sqrt(den_currents1.shape[0]))

        # 1-2, 5-6
        for (i, j) in [(1,2), (5,6)]:

            Z_i_shots = 1-2*den_currents1[:,i]
            Z_j_shots = 1-2*den_currents1[:,j]
            Z_a_shots = 1-2*den_ancillas1[:,0]

            temp_current = 1/2 * (Z_j_shots * Z_a_shots - Z_i_shots * Z_a_shots)

            sign = 1
            if [i, j] in negative_bonds:
                sign = -1

            currents.append(sign * np.mean(temp_current, axis=0))
            currents_std.append(np.std(temp_current, axis=0))
            currents_sem.append(np.std(temp_current, axis=0)/np.sqrt(temp_current.shape[0]))

        # 9-10, 13-14
        for (i, j) in [(9,10), (13,14)]:

            Z_i_shots = 1-2*den_currents1[:,i]
            Z_j_shots = 1-2*den_currents1[:,j]
            Z_b_shots = 1-2*den_ancillas1[:,1]

            temp_current = 1/2 * (Z_j_shots * Z_b_shots - Z_i_shots * Z_b_shots)

            sign = 1
            if [i, j] in negative_bonds:
                sign = -1

            currents.append(sign * np.mean(temp_current, axis=0))
            currents_std.append(np.std(temp_current, axis=0))
            currents_sem.append(np.std(temp_current, axis=0)/np.sqrt(temp_current.shape[0]))

        # 1-5, 2-6, 9-13, 10-14
        for (i, j) in [(1,5), (2,6), (9,13), (10,14)]:
            currents.append((avg_den_currents2[i] - avg_den_currents2[j]))
            currents_std.append(np.std(den_currents2[:,i] - den_currents2[:,j], axis=0))
            currents_sem.append(np.std(den_currents2[:,i] - den_currents2[:,j], axis=0)/np.sqrt(den_currents2.shape[0]))

        # 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13, 14-15
        for (i, j) in [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]:
            currents.append((avg_den_currents3[i] - avg_den_currents3[j]))
            currents_std.append(np.std(den_currents3[:,i] - den_currents3[:,j], axis=0))
            currents_sem.append(np.std(den_currents3[:,i] - den_currents3[:,j], axis=0)/np.sqrt(den_currents3.shape[0]))

        # 5-9, 6-10, 0-4, 3-7, 8-12, 11-15
        for (i, j) in [(5,9), (6,10), (0,4), (3,7), (8,12), (11,15)]:
            currents.append((avg_den_currents4[i] - avg_den_currents4[j]))
            currents_std.append(np.std(den_currents4[:,i] - den_currents4[:,j], axis=0))
            currents_sem.append(np.std(den_currents4[:,i] - den_currents4[:,j], axis=0)/np.sqrt(den_currents4.shape[0]))

    currents = np.array(currents)

    return n_avg, n_sem, currents, currents_sem, bonds


def get_numerics_observables(path, filename, shots):
    """Calculates average densities and currents with corresponding error
        from ideal numerics (trotter circuit and Lindblad limit) ran with Julia code"

    Args:
        path (str): folder with numerical data
        filename (str): name of .hd5 file
        shots (int): number of trajectories we want to sample

    Returns:
        n_avg: densities averaged over trajectories
        n_sem: statistical error of the mean for densities
        currents: current magnitudes averaged over trajectories
        currents_sem: statistical error of the mean for current
        bonds: list of bonds in circuit order
    """    
    
    # renumber bonds from 0-15 instead of 1-16 
    init_bonds = load_key_from_hdf5(path+filename, 'params/bonds')
    bonds = []
    for i, bond in enumerate(init_bonds):
        new_label = [int(bond[i]-1) for i in range(len(bond))]
        bonds.append(new_label)

    # densities 
    n_avg = load_key_from_hdf5(path+filename, "n_avg").T
    n_avg = n_avg[-1,:]
    n_sq_avg = load_key_from_hdf5(path+filename, "n_sq_avg").T
    n_sq_avg = n_sq_avg[-1,:]

    # currents
    currents = load_key_from_hdf5(path+filename, "avg_currents").T
    currents = currents[-1,:]
    currents_sq_avg = load_key_from_hdf5(path+filename, "currents_sq_avg").T
    currents_sq_avg = currents_sq_avg[-1,:]

    # errors:
    n_sem = np.sqrt(n_sq_avg - n_avg**2)/np.sqrt(shots)
    currents_sem = np.sqrt(currents_sq_avg - currents**2)/np.sqrt(shots)

    return n_avg, n_sem, currents, currents_sem, bonds


def kraus_current(path, filename, shots):
    """ Calculates current obtained from Kraus operators.

    Args:
        path (str): folder cooresponding to used device
        filename (str): name of .hd5 file
        shots (int): number of shots/trajectories ran

    Returns:
        driving_current: average Kraus current per timestep
        inst_current: Kraus current per timestep for each trajectory
    """
    
    trajectory_source = load_key_from_hdf5(path+filename, "density_circuit/trajectory_source")[:shots] 
    trajectory_sink = load_key_from_hdf5(path+filename, "density_circuit/trajectory_sink")[:shots]
    
    # combined kraus operator for each circuit are now labelled between 0 and 8 (instead of 0,1,2 for each sink and source separately)
    trajectory = trajectory_source + 3*trajectory_sink 
    
    # Kraus current contrib and SEM for each timestep (for each trajectories)
    inst_current = (trajectory==1)/2 + (trajectory==4)/2 + (trajectory==6)/2 + (trajectory==8)/2 + (trajectory==7) #(shots, steps) ---> 0=0, 7=1, else=1/2 

    # average current per timestep
    K = [np.mean(trajectory==element, axis=0) for element in range(9)] 
    driving_current = (K[1] + K[4] + K[6] + K[8] + 2*K[7])/2  

    return driving_current, inst_current


def current_cut_average(path, filename, statistics, shots, Nx=4, Ny=4):
    """ Takes the average instantaneous currents and calculates the toatl average currents with respect to
    the taxicab distance from the source r = (x+y)"

    Args:
        path (str): path to device data (H1-1 or Emulator)
        filename (str): name of file 
        statistics (str): 'fermions' or 'bosons'

    Returns:
        average_j_cut: current averaged over all currents at the same distance from source.
        j_cut_sem: average_j_cut error (statistical error of the mean)
    """

    _, _, currents, currents_sem, bonds = get_device_observables(path, filename, statistics, shots)

    j_cut = [0]* Nx         # current average per cut
    j_cut_var = [0]* Nx     #variance of current cut average
    for j, bond, sem in zip(currents, bonds, currents_sem):
        # skip bonds connected to source / drain
        if (0 in bond) or ((Nx*Ny)-1 in bond):
            continue

        # convert back from n to x,y coordinates
        x1, y1 = bond[0] % Nx, bond[0] // Nx
        x2, y2 = bond[1] % Nx, bond[1] // Nx

        # calculate taxicab distance d_T between source and the midpoint of each bond
        midpoint = [(x1+x2)/2 , (y1+y2)/2]
        d_T = abs(midpoint[0]) + abs(midpoint[1])
        
        idx = int(d_T - 1.5)        #1.5 == smallest distance 
        j_cut[idx] += j
        j_cut_var[idx] += sem**2

    average_j_cut = np.mean(j_cut)                      # sum over elements in j_cut and divide by number of cuts
    j_cut_sem = np.sqrt(np.sum(j_cut_var)/len(j_cut_var))   # error of the current average

    return average_j_cut, j_cut_sem


def density_imbalance(n_avg, sem, Nx=4, Ny=4):
    """ Calculates the average density difference between the sites below and above the diagonal  

    Args:
        n_avg (list): average density per site
        sem (list): error in the average density (Statistical Error of the Mean)
        Nx (int, optional): number of sites along x direction. Defaults to 4.
        Ny (int, optional): number of sites along y direction. Defaults to 4.

    Returns:
        tot_density_imbalance: total density imbalance
        std_tot: standard deviation of density imbalance
    """

    X = []; Y = []; C = []
    for x in range(Nx):
        for y in range(Ny):
            n = x % Nx + y*Nx           # n= 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 4, 7, 11, 15
            X.append(x)
            Y.append(y)
            C.append(n_avg[n])

    n_above, n_below = [0]*Nx, [0]*Nx
    var = [0]* Nx
    var_tot = 0
    count = [0]*Nx
    
    for xval, yval, n_i, sigma in zip(X, Y, C, sem):        
        # distance from the diagonal:
        d = xval-yval

        if d > 0:               # below the diagonal
            n_below[abs(d)] += n_i 
            count[abs(d)] += 1                   
        elif d < 0:             # above the diagonal
            n_above[abs(d)] += n_i 
            count[abs(d)] += 1
        elif d==0:              # on the diagonal ()
            n_below[abs(d)] += n_i
            count[abs(d)] += 1      

        var[abs(d)] += sigma**2
        var_tot += sigma**2

    # density imbalace
    tot_density_imbalance = np.sum(np.subtract(n_below[1:], n_above[1:]))
    # standard deviation
    std_tot = np.sqrt(var_tot)

    return tot_density_imbalance, std_tot


def current_imbalance_edges(edge_bonds, bonds, currents, currents_sem, Nx=4, Ny=4):
    """ Calculates the average current difference (imbalance) between the edge 
    bonds below and above the diagonal. 
    

    Args:
        bonds (array): bonds of neighbouring qubits in sector order
        currents (list): average instantaneous currents
        currents_sem (list): error in average instantaneous currents
        Nx (int, optional): number of sites along x direction. Defaults to 4.
        Ny (int, optional): number of sites along y direction. Defaults to 4.

    Returns:
        tot_j_imb: total current imbalance 
        std_tot: standard deviation of the current imbalance
    """

    j_below, j_above = [0]*(Nx-1), [0]*(Nx-1)
    var = [0]*(Nx-1)
    var_tot = 0
    
    for bond in edge_bonds:
        
        i = bonds.index(bond)
        # skip bonds connected to source / drain
        if (0 in bond) or ((Nx*Ny)-1 in bond):
            continue
        
        # convert back from n to x,y coordinates
        x1, y1 = bond[0] % Nx, bond[0] // Nx
        x2, y2 = bond[1] % Nx, bond[1] // Nx

        # distance from the diagonal:
        d1, d2 = (x1 - y1)/np.sqrt(2), (x2 - y2)/np.sqrt(2)
        d = (d1 + d2)/np.sqrt(2)
        idx = int(np.round(abs(d)))

        if d > 0:
            j_below[idx] += currents[i]     
        elif d < 0:
            j_above[idx] += currents[i]

        var[idx] += currents_sem[i]**2
        var_tot += currents_sem[i]**2


    # current imbalance
    tot_j_imb = np.sum(np.subtract(j_below[1:], j_above[1:]))

    # total standard deviation
    std_tot = np.sqrt(var_tot)

    return tot_j_imb, std_tot


def all_cases_imbalances(simulation, shots_list):
    """ Loads the data for each simulation and calculate density and current imbalances for all experimental setups. 

    Args:
        simulation (str): what data to use (H1, Noisy Sim, Ideal Sim)
        shots_list (list): number of shots/trajectories for each experimental setup

    Returns:
        density_imbalances: density imbalance and associated error for all setups
        current_imbalances: current imbalance and associated error for all setups
    """

    stats = ['bosons','fermions',  'bosons', 'fermions', 'fermions', 'fermions']
    edge_bonds = [[1, 2], [2, 3], [12, 13], [13, 14], [3, 7], [4, 8], [7, 11], [8, 12]]     # the current imbalance is measured only along the edges of teh system

    density_imbalances = np.zeros((2,5))
    current_imbalances = np.zeros((2,5))

    if simulation == "Ideal":
        path = '../data/data_numerics/'
        filenames = [f'bosons_random_V0.0_phi0.0_dt0.31_p0.62_steps10_shots10000_trotter.h5',
                    f'fermions_random_V0.0_phi0.0_dt0.21_p0.42_steps14_shots10000_trotter.h5',
                    f'bosons_custom_V1.5_phi0.0_dt0.31_p0.62_steps14_shots10000_trotter.h5',
                    f'fermions_random_V0.0_phi0.5_dt0.27_p0.54_steps16_shots10000_trotter.h5',
                    f'fermions_random_V1.0_phi0.5_dt0.29_p0.58_steps18_shots10000_trotter.h5']

        for idx in range(len(filenames)):
            n_avg, n_sem, currents, currents_sem, bonds = get_numerics_observables(path, filenames[idx], shots_list[idx])
            density_imbalances[0, idx], density_imbalances[1, idx] = density_imbalance(n_avg, n_sem)
            current_imbalances[0, idx], current_imbalances[1, idx] = current_imbalance_edges(edge_bonds, bonds, currents, currents_sem)
        return density_imbalances, current_imbalances

    elif simulation == 'H1':
        path = '../data/data_H1/'
        filenames = [f'bosons_random_V0.0_phi0.0_dt0.31_p0.62_steps10_shots1280.h5',
                f'fermions_random_V0.0_phi0.0_dt0.21_p0.42_steps14_shots1480.h5',
                f'bosons_custom_V1.5_phi0.0_dt0.31_p0.62_steps14_shots1280.h5',
                f'fermions_random_V0.0_phi0.5_dt0.27_p0.54_steps16_shots1480.h5',
                f'fermions_random_V1.0_phi0.5_dt0.29_p0.58_steps18_shots1480.h5']
            
    elif simulation == 'Noisy':
        path = '../data/data_Emulator/'
        filenames = [f'bosons_random_V0.0_phi0.0_dt0.31_p0.62_steps10_shots15000.h5',
                        f'fermions_random_V0.0_phi0.0_dt0.21_p0.42_steps14_shots6500.h5',
                        f'bosons_custom_V1.5_phi0.0_dt0.31_p0.62_steps14_shots15000.h5',
                        f'fermions_random_V0.0_phi0.5_dt0.27_p0.54_steps16_shots6250.h5',
                        f'fermions_random_V1.0_phi0.5_dt0.29_p0.58_steps18_shots6500.h5']
        
    for idx in range(len(filenames)):
        n_avg, n_sem, currents, currents_sem, bonds = get_device_observables(path, filenames[idx], stats[idx], shots_list[idx])
        if type(bonds) != list:
            bonds = bonds.tolist()
        density_imbalances[0, idx], density_imbalances[1, idx] = density_imbalance(n_avg, n_sem)
        current_imbalances[0, idx], current_imbalances[1, idx] = current_imbalance_edges(edge_bonds, bonds, currents, currents_sem)
    
    return density_imbalances, current_imbalances
