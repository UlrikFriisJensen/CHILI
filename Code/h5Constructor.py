#%% Imports
import os, sys, math, torch
from mendeleev import element
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
        cif_paths,
        save_dir
    ):
        # Cif file paths
        self.cifs = cif_paths

        self.save_dir = Path(save_dir)

        #Create save directory if it doesn't exist
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
            print("Save directory doesn't exist.\nCreated the save directory at " + str(self.save_dir))
        
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

    def gen_single_h5(self, input_tuple, override=False, verbose=False):
        cif, np_radii, device = input_tuple
        print('\n', flush=True)
        print(cif, flush=True)
        cif_name = cif.split('/')[-1].split('.')[0]
        # Check if graph has already been made
        if os.path.isfile(f'{self.save_dir}/graph_{cif_name}.h5') and not override:
            if verbose:
                print(f'{cif_name} h5 file already exists')
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
        if not np.any(unit_cell.pbc):
            # TODO figure out what to do with unit cells that are not periodic or partly periodic
            print('Not periodic')
            # unit_cell.pbc = (1,1,1)
            return

        # Get distances with MIC (NOTE I don't think this makes a difference as long as pbc=True in the unit cell)
        unit_cell_dist = unit_cell.get_all_distances(mic=True)
        unit_cell_atoms = unit_cell.get_atomic_numbers().reshape(-1, 1)
        unit_cell_pos = unit_cell.get_scaled_positions()

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
                [
                    element(int(atom[0])).atomic_number, 
                    element(int(atom[0])).atomic_radius, 
                    element(int(atom[0])).atomic_weight, 
                    element(int(atom[0])).electron_affinity
                ] 
                for atom in unit_cell_atoms
                ], dtype='float')
            node_pos_real = unit_cell.get_positions()
            node_pos_relative = unit_cell_pos

        # Construct cell parameter matrix
        cell_parameters = unit_cell.cell.cellpar()
        
        #Create graph to check for connectivity
        node_features[node_features == None] = 0.0
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(direction, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g = to_networkx(graph, to_undirected=True)

        if not is_connected(g):
            print('\n'+cif_name + '\n')
            return
        
        ## Setup
        # Create an instance of DebyeCalculator
        xray_calculator = DebyeCalculator(device=device, qmin=1, qmax=30, qstep=0.1, biso=0.3, rmin=0.0, rmax=55.0, rstep=0.01, radiation_type='xray')
        neutron_calculator = DebyeCalculator(device=device, qmin=1, qmax=30, qstep=0.1, biso=0.3, rmin=0.0, rmax=55.0, rstep=0.01, radiation_type='neutron')
        
        # Construct discrete particles for simulation of spectra
        # struc_list, size_list = cif_to_NP_GPU(self.cif_dir + '/' + cif, np_radii)
        #struc_list, size_list = xray_calculator.generate_nanoparticles(structure_path=self.cif_dir + '/' + cif, radii=np_radii, sort_atoms=False, disable_pbar=True)
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
            # Save elements for the graph
            graph_h5 = h5_file.require_group('LocalLabels')
            graph_h5.create_dataset('NodeFeatures', data=node_features)
            graph_h5.create_dataset('EdgeFeatures', data=edge_features)
            graph_h5.create_dataset('EdgeDirections', data=direction)
            graph_h5.create_dataset('FractionalCoordinates', data=node_pos_relative)
            graph_h5.create_dataset('Coordinates', data=node_pos_real)
            
            params_h5 = h5_file.require_group('GlobalLabels')
            params_h5.create_dataset('CellParameters', data=cell_parameters)
            params_h5.create_dataset('CrystalType', data=crystal_type)
            params_h5.create_dataset('SpaceGroupSymbol', data=space_group.symbol)
            params_h5.create_dataset('SpaceGroupNumber', data=space_group.no)
            params_h5.create_dataset('ElementsPresent', data=np.unique(node_features[:,0]))

            # Save scattering data
            scattering_h5 = h5_file.require_group('ScatteringData')
            
            ## Simulate scattering data
            for i, np_size in enumerate(size_list):
                # Differentiate spectra by size
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
    
    def gen_h5s(self, np_radii=[5., 10., 15., 20., 25.], parallelize=True, num_processes=cpu_count() - 1, device=None):
        #Initialize the number of workers you want to work in parallel. Default is the number of cores -1 to not freeze your pc.
        if device == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        inputs = zip(self.cifs, repeat(np_radii), repeat(device))
        print('\nConstructing graphs from cif files:')
        if parallelize:
            with Pool(processes=num_processes) as pool:
                #Run the parallized process
                with tqdm(total=len(self.cifs)) as pbar:
                    for _ in pool.imap_unordered(self.gen_single_h5, inputs):
                        pbar.update()
        else:
            for input_tuple in tqdm(inputs):
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

