#%% Imports
import os, sys, math, torch
from tkinter.filedialog import LoadFileDialog

from diffpy.Structure import loadStructure, Lattice
from mendeleev import element
import numpy as np
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
from multiprocessing import Pool, cpu_count
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx.algorithms.components import is_connected
from networkx import draw
from ase.io import write, read
from ase.build import make_supercell
from ase.neighborlist import neighbor_list, natural_cutoffs
from Code.simScatteringPatterns import simPDFs, cif_to_NP, Debye_Calculator_GPU_bins

#%% Graph construction

class h5Constructor():
    def __init__(self, cif_dir, save_dir):
        self.cif_dir = Path(cif_dir)
        self.save_dir = Path(save_dir)

        #Create save directory if it doesn't exist
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
            print("Save directory doesn't exist.\nCreated the save directory at " + str(self.save_dir))
        
        self.cifs = [str(x.name) for x in self.cif_dir.glob('*.cif')]
        
        self.cifs = sorted(self.cifs, key=lambda x: (x.split('_')[1], x.split('_')[0]))
        
        self.cif_dir = str(self.cif_dir)
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

    def gen_single_h5(self, cif, override=False):
        # Check if graph has already been made
        if os.path.isfile(f'{self.save_dir}/graph_{cif[:-4]}.h5') and not override:
            print(f'{cif} h5 file already exists')
            return None

        # Load cif
        unit_cell = read(f'{self.cif_dir}/{cif}')
        
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
            node_features = np.concatenate((unit_cell_pos, unit_cell_atoms), axis=1)

        # Construct cell parameter matrix
        cell_parameters = unit_cell.cell.cellpar()
        
        #Create graph to check for connectivity
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(direction, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g = to_networkx(graph, to_undirected=True)

        if not is_connected(g):
            print('\n'+cif + '\n')
            return
        
        # Construct discrete particles for simulation of spectra
        radii = [5, 10, 15, 20, 25] # Å
        
        struc_list, size_list = cif_to_NP(self.cif_dir + '/' + cif, radii)
        
        ### Simulate spectra
        ## Setup
        # X-ray PDF
        pdf_xray_generator = simPDFs()
        pdf_xray_generator.genPDFs(self.cif_dir + '/' + cif)
        
        # Neutron PDF
        pdf_neutron_generator = simPDFs()
        pdf_neutron_generator.set_parameters(radiationType='N')
        pdf_neutron_generator.genPDFs(self.cif_dir + '/' + cif)
        
        # Small Angle Scattering
        q_sas = np.arange(0, 3, 0.01)
        
        # Diffraction
        q_diff = np.arange(1, 30, 0.05)
        
        # Construct .h5 file
        with h5py.File(f'{self.save_dir}/{cif[:-4]}.h5', 'w') as h5_file:
            # Save elements for the graph
            graph_h5 = h5_file.require_group('GraphElements')
            graph_h5.create_dataset('NodeFeatures', data=node_features)
            graph_h5.create_dataset('EdgeFeatures', data=edge_features)
            graph_h5.create_dataset('EdgeDirections', data=direction)

            # Save spectra
            spectra_h5 = h5_file.require_group('Spectra')
            
            ## Simulate spectra
            for i, np_size in tqdm(enumerate(size_list), total=len(size_list), desc='Simulating spectra', leave=False):
                # Differentiate spectra by size
                spectra_size_h5 = spectra_h5.require_group(f'{size_list[i]:.2f}Å')
                
                # X-ray PDF
                pdf_xray = pdf_xray_generator.getPDF(psize=np_size)
                # Save spectra
                spectra_size_h5.create_dataset('PDF (X-ray)', data=pdf_xray[1])
                
                # Neutron PDF
                pdf_neutron = pdf_neutron_generator.getPDF(psize=np_size)
                # Save spectra
                spectra_size_h5.create_dataset('PDF (Neutron)', data=pdf_neutron[1])
                
                # SAXS
                saxs = Debye_Calculator_GPU_bins(
                    struc_list[i].get_chemical_symbols(), 
                    np.float16(struc_list[i].get_positions()), 
                    q_sas, 
                    n_bins=10000, 
                    radiationType='X'
                )
                # Save spectra
                spectra_size_h5.create_dataset('SAXS', data=saxs)
                
                # SANS
                sans = Debye_Calculator_GPU_bins(
                    struc_list[i].get_chemical_symbols(), 
                    np.float16(struc_list[i].get_positions()), 
                    q_sas, 
                    n_bins=10000, 
                    radiationType='N'
                )
                # Save spectra
                spectra_size_h5.create_dataset('SANS', data=sans)
                
                # XRD
                xrd = Debye_Calculator_GPU_bins(
                    struc_list[i].get_chemical_symbols(), 
                    np.float16(struc_list[i].get_positions()), 
                    q_diff, 
                    n_bins=10000, 
                    radiationType='X'
                )
                # Save spectra
                spectra_size_h5.create_dataset('XRD', data=xrd)
                                
                # ND
                nd = Debye_Calculator_GPU_bins(
                    struc_list[i].get_chemical_symbols(), 
                    np.float16(struc_list[i].get_positions()), 
                    q_diff, 
                    n_bins=10000, 
                    radiationType='N'
                )
                # Save spectra
                spectra_size_h5.create_dataset('ND', data=nd)
        
            # # Construct .h5 file
            # with h5py.File(f'{self.save_dir}/graph_{cif[:-4]}_{np_size:.2f}Å.h5', 'w') as h5_file: # TODO: Think about file naming
            #     h5_file.create_dataset('Edge Feature Matrix', data=edge_features)
            #     h5_file.create_dataset('Node Feature Matrix', data=node_features)
            #     h5_file.create_dataset('Edge Directions', data=direction)
            #     h5_file.create_dataset('Cell parameters', data=cell_parameters)
            #     h5_file.create_dataset('PDF (X-ray)', data=pdf_xray[1])
            #     h5_file.create_dataset('PDF (Neutron)', data=pdf_neutron[1])
            #     h5_file.create_dataset('SAXS', data=saxs)
            #     h5_file.create_dataset('SANS', data=sans)
            #     h5_file.create_dataset('XRD', data=xrd)
            #     h5_file.create_dataset('ND', data=nd)
    
    def gen_h5s(self, num_processes=cpu_count() - 1, parallelize=True):
        
        #Initialize the number of workers you want to work in parallel. Default is the number of cores -1 to not freeze your pc.
        print('\nConstructing graphs from cif files:')
        if parallelize:
            with Pool(processes=num_processes) as pool:
                #Run the parallized process
                with tqdm(total=len(self.cifs)) as pbar:
                    for _ in pool.imap_unordered(self.gen_single_h5, self.cifs):
                        pbar.update()
        else:
            for cif in tqdm(self.cifs):
                self.gen_single_h5(cif)
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

