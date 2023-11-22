#%% Imports
import os, sys, math, torch
from mendeleev import element
from mendeleev.fetch import fetch_table
import numpy as np
from tqdm.auto import tqdm
import h5py
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx.algorithms.components import is_connected
from ase.io import read
from ase.build import make_supercell
from ase.spacegroup import get_spacegroup
# from Code.simScatteringPatterns import simPDFs, cif_to_NP_GPU
from debyecalculator import DebyeCalculator

#%% Graph construction

class h5Constructor():
    def __init__(
        self,
        save_dir,
        cif_dir=None,
        cif_paths=None,
    ):
        
        # Find cif files
        if isinstance(cif_paths, list):
            self.cifs = cif_paths
        elif isinstance(cif_dir, str):
            self.cif_dir = Path(cif_dir)
            self.cifs = [str(x) for x in self.cif_dir.glob('*.cif')]
            self.cif_dir = str(self.cif_dir)
        else:
            raise ValueError('Either cif_dir or cif_paths must be specified')

        # Save directory
        self.save_dir = Path(save_dir)

        #Create save directory if it doesn't exist
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
            #print("Save directory doesn't exist.\nCreated the save directory at " + str(self.save_dir))
        
        self.save_dir = str(self.save_dir)

    def repeating_unit_mask_attempt2(self, cell_pos, cell_dist, cell_lens, cell_angles):
        mask = np.zeros((cell_pos.shape[0], cell_pos.shape[0]))
        cell_angles = [math.radians(angle) for angle in cell_angles]
        twice_distance = cell_dist * 2
        for i, cell_len in enumerate(cell_lens):
            max_distance = cell_len * math.sin(cell_angles[i])
            mask += twice_distance >= max_distance

        return mask >= 0

    def repeating_unit_mask_attempt1(self, pos): # TODO vectorize this function
        p_out = np.zeros((pos.shape[0], pos.shape[0]), dtype='int')
        for cpos in pos.T:
            for i in range(len(cpos)):
                for j in range(len(cpos)):
                    add = 2 * abs(cpos[i] - cpos[j])
                    p_out[i,j] += int(add >= 1)
                    
        return p_out == 1

    def gen_single_h5(self, input_tuple, override=False, verbose=False, check_connectivity=False, check_periodic=False, save_discrete_nps=True):
        cif, np_radii, device, node_feature_table = input_tuple
        cif_name = cif.split('/')[-1].split('.')[0]
        print(cif_name, flush=True)
        # Check if graph has already been made
        if os.path.isfile(f'{self.save_dir}/{cif_name}.h5') and not override:
            if verbose:
                print(f'{cif_name} h5 file already exists, returning\n', flush=True)
            return None

        # Load cif
        unit_cell = read(cif)
        
        # Find space group
        space_group = get_spacegroup(unit_cell)
        
        # Find crystal type
        cif_name_split = cif_name.split('_')
        if len(cif_name_split) > 1:
            crystal_type = cif_name_split[0]
        elif len(cif_name_split) == 1:
            crystal_type = None
        
        # Assert if pbc is true
        if check_periodic:
            if not np.any(unit_cell.pbc):
                # TODO figure out what to do with unit cells that are not periodic or partly periodic
                print(f'{cif_name} is not periodic, returning', flush=True)
                # unit_cell.pbc = (1,1,1)
                return

        # Get distances with MIC (NOTE I don't think this makes a difference as long as pbc=True in the unit cell)
        unit_cell_dist = unit_cell.get_all_distances(mic=True)
        unit_cell_atoms = unit_cell.get_atomic_numbers().reshape(-1, 1)

        # Make supercell to get Lattice constant
        supercell = make_supercell(unit_cell, np.diag([2,2,2]))
        metal_distances = supercell[supercell.get_atomic_numbers() != 8].get_all_distances()
        lc = np.amin(metal_distances[metal_distances > 0.])
        
        # Create edges and node features
        lc_mask = (unit_cell_dist > 0) & (unit_cell_dist < lc)
        oxy_mask = np.outer(unit_cell_atoms == 8, unit_cell_atoms == 8)
        metal_mask = np.outer(unit_cell_atoms != 8, unit_cell_atoms !=8)
        direction = np.argwhere(lc_mask & ~oxy_mask & ~metal_mask).T
        
        # Figure out where there should be double bonds because of very small unit cells
        if False:
            lens_and_angles = unit_cell.get_cell_lengths_and_angles()
            unit_cell_lens = lens_and_angles[:3]
            unit_cell_angles = lens_and_angles[3:]
            double_bond_mask = self.repeating_unit_mask(unit_cell_pos, unit_cell_dist, unit_cell_lens, unit_cell_angles) & ~oxy_mask & ~metal_mask & lc_mask
            double_bond_direction = np.argwhere(double_bond_mask).T

            direction = np.concatenate((direction, double_bond_direction), axis=1)
            edge_features = np.concatenate((unit_cell_dist[lc_mask],unit_cell_dist[double_bond_mask]), axis=0)
            node_features = np.concatenate((unit_cell_pos, unit_cell_atoms), axis=1)
        else:
            edge_features = unit_cell_dist[lc_mask]
            node_features = np.array([
                node_feature_table.loc[atom[0]].values
                for atom in unit_cell_atoms
                ], dtype='float')
            node_pos_real = unit_cell.get_positions()
            node_pos_relative = unit_cell.get_scaled_positions()

        # Construct cell parameter matrix
        cell_parameters = unit_cell.cell.cellpar()
        
        #Create graph to check for connectivity
        node_features[node_features == None] = 0.0
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(direction, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        if check_connectivity:
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            g = to_networkx(graph, to_undirected=True)
            if not is_connected(g):
                print(f'{cif_name} is not connected, returning', flush=True)
                return None
        
        ## Setup
        # Create an instance of DebyeCalculator
        xray_calculator = DebyeCalculator(device=device, qmin=1, qmax=30, qstep=0.1, biso=0.3, rmin=0.0, rmax=55.0, rstep=0.01, radiation_type='xray')
        neutron_calculator = DebyeCalculator(device=device, qmin=1, qmax=30, qstep=0.1, biso=0.3, rmin=0.0, rmax=55.0, rstep=0.01, radiation_type='neutron')
        
        # Construct discrete particles for simulation of spectra
        struc_list, size_list = xray_calculator.generate_nanoparticles(structure_path=cif, radii=np_radii, sort_atoms=False, disable_pbar=True)
        
        # Calculate scattering for large Q-range
        x_r, x_q, x_iq, _, _, x_gr = xray_calculator._get_all(struc_list)
        n_r, n_q, n_iq, _, _, n_gr = neutron_calculator._get_all(struc_list)
        
        # Simulation parameters for small Q-range
        xray_calculator.update_parameters(qmin=0, qmax=3.0, qstep=0.01)
        neutron_calculator.update_parameters(qmin=0, qmax=3.0, qstep=0.01)
        
        # Calculate SAS
        saxs_q, saxs_iq = xray_calculator.iq(struc_list)
        sans_q, sans_iq = neutron_calculator.iq(struc_list)
        
        # Construct .h5 file
        with h5py.File(f'{self.save_dir}/{cif_name}.h5', 'w') as h5_file:
            # Save global labels for the graph
            params_h5 = h5_file.require_group('GlobalLabels')
            params_h5.create_dataset('CellParameters', data=cell_parameters)
            params_h5.create_dataset('CrystalType', data=crystal_type)
            params_h5.create_dataset('SpaceGroupSymbol', data=space_group.symbol)
            params_h5.create_dataset('SpaceGroupNumber', data=space_group.no)
            params_h5.create_dataset('ElementsPresent', data=np.unique(node_features[:,0]))

            # Save unit cell graph
            graph_h5 = h5_file.require_group('UnitCellGraph')
            graph_h5.create_dataset('NodeFeatures', data=node_features)
            graph_h5.create_dataset('EdgeFeatures', data=edge_features)
            graph_h5.create_dataset('EdgeDirections', data=direction)
            graph_h5.create_dataset('FractionalCoordinates', data=node_pos_relative)
            graph_h5.create_dataset('AbsoluteCoordinates', data=node_pos_real)

            if save_discrete_nps:
                # Save discrete particle graphs
                npgraphs_h5 = h5_file.require_group('DiscreteParticleGraphs')

                for i, discrete_np in enumerate(struc_list):
                    # Differentiate graphs by NP size
                    npgraph_size_h5 = npgraphs_h5.require_group(f'{size_list[i]:.2f}Å')
                    npgraph_size_h5.create_dataset('NP size (Å)', data=size_list[i])

                    np_dists = discrete_np.get_all_distances(mic=False)
                    np_atoms = discrete_np.get_atomic_numbers().reshape(-1, 1)

                    # Find lattice constant
                    metal_distances = discrete_np[discrete_np.get_atomic_numbers() != 8].get_all_distances()
                    lc = np.amin(metal_distances[metal_distances > 0.])

                    # Create edges and node features
                    lc_mask = (np_dists > 0) & (np_dists < lc)
                    oxy_mask = np.outer(np_atoms == 8, np_atoms == 8)
                    metal_mask = np.outer(np_atoms != 8, np_atoms !=8)
                    direction = np.argwhere(lc_mask & ~oxy_mask & ~metal_mask).T

                    edge_features = np_dists[lc_mask]
                    node_features = np.array([
                        node_feature_table.loc[atom[0]].values
                        for atom in np_atoms
                        ], dtype='float')
                    node_pos_real = discrete_np.get_positions()
                    node_pos_relative = discrete_np.get_scaled_positions()

                    # Convert to torch tensors
                    node_features[node_features == None] = 0.0
                    x = torch.tensor(node_features, dtype=torch.float32)
                    edge_index = torch.tensor(direction, dtype=torch.long)
                    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

                    npgraph_size_h5.create_dataset('NodeFeatures', data=node_features)
                    npgraph_size_h5.create_dataset('EdgeFeatures', data=edge_features)
                    npgraph_size_h5.create_dataset('EdgeDirections', data=direction)
                    npgraph_size_h5.create_dataset('FractionalCoordinates', data=node_pos_relative)
                    npgraph_size_h5.create_dataset('AbsoluteCoordinates', data=node_pos_real)
            
            # Save scattering data
            scattering_h5 = h5_file.require_group('ScatteringData')

            for i, np_size in enumerate(size_list):
                # Differentiate scattering data by NP size
                scattering_size_h5 = scattering_h5.require_group(f'{np_size:.2f}Å')
                scattering_size_h5.create_dataset('NP size (Å)', data=np_size)
                
                # XRD
                scattering_size_h5.create_dataset('XRD', data=np.vstack((x_q, x_iq[i])))
                
                # X-ray PDF
                scattering_size_h5.create_dataset('xPDF', data=np.vstack((x_r, x_gr[i])))
                
                # ND
                scattering_size_h5.create_dataset('ND', data=np.vstack((n_q, n_iq[i])))
                
                # Neutron PDF
                scattering_size_h5.create_dataset('nPDF', data=np.vstack((n_r, n_gr[i])))
                
                # SAXS
                scattering_size_h5.create_dataset('SAXS', data=np.vstack((saxs_q, saxs_iq[i])))
                
                # SANS
                scattering_size_h5.create_dataset('SANS', data=np.vstack((sans_q, sans_iq[i])))

        return None
    
    def gen_h5s(self, np_radii=[5., 10., 15., 20., 25.], parallelize=True, num_processes=cpu_count() - 1, device=None):
        #Initialize the number of workers you want to work in parallel. Default is the number of cores -1 to not freeze your pc.
        if device == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Fetch node features and replace NaNs with 0.0
        node_feature_table = fetch_table('elements')[['atomic_number', 'atomic_radius', 'atomic_weight', 'electron_affinity']]
        node_feature_table['electron_affinity'].fillna(0.0, inplace=True)
        
        inputs = zip(self.cifs, repeat(np_radii), repeat(device), repeat(node_feature_table))
        #print('\nConstructing graphs from cif files:')

        if parallelize:
            with Pool(processes=num_processes) as pool:
                #Run the parallized process
                with tqdm(total=len(self.cifs), disable=True) as pbar:
                    for _ in pool.imap_unordered(self.gen_single_h5, inputs):
                        pbar.update()
        else:
            for input_tuple in tqdm(inputs, disable=True):
                self.gen_single_h5(input_tuple)
        return None

def calc_dist(position_0, position_1):
    """ Returns the distance between vectors position_0 and position_1 """
    return np.sqrt((position_0[0] - position_1[0]) ** 2 + (position_0[1] - position_1[1]) ** 2 + (
            position_0[2] - position_1[2]) ** 2)

def main():
    h5C = h5Constructor('../Dataset/CIFs/Train', 'Test_out')
    h5C.gen_h5s()


if __name__ == "__main__":
    main()

