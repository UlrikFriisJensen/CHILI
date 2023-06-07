#%% Imports
import os, sys, math, torch
from tkinter.filedialog import LoadFileDialog

from diffpy.Structure import loadStructure, Lattice
from mendeleev import element
import numpy as np
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    def gen_single_h5(self, cif):
        # Check if graph has already been made
        if os.path.isfile(f'{self.save_dir}/graph_{cif[:-4]}.h5'):
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
        direction = np.argwhere(lc_mask).T
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
        
        # Simulate spectra
        placeholder_spectra = torch.zeros([1,1500])
        
        # Construct .h5 file
        with h5py.File(f'{self.save_dir}/graph_{cif[:-4]}.h5', 'w') as h5_file: # TODO: Think about file naming
            h5_file.create_dataset('Edge Feature Matrix', data=edge_features)
            h5_file.create_dataset('Node Feature Matrix', data=node_features)
            h5_file.create_dataset('Edge Directions', data=direction)
            h5_file.create_dataset('Cell parameters', data=cell_parameters)
            h5_file.create_dataset('PDF (X-ray)', data=placeholder_spectra)
            h5_file.create_dataset('PDF (Neutron)', data=placeholder_spectra)
            h5_file.create_dataset('SAXS', data=placeholder_spectra)
            h5_file.create_dataset('SANS', data=placeholder_spectra)
            h5_file.create_dataset('XANES', data=placeholder_spectra)
    
    def gen_h5s(self, num_processes=cpu_count() - 1):
        
        #Initialize the number of workers you want to work in parallel. Default is the number of cores -1 to not freeze your pc.
        print('\nConstructing graphs from cif files:')
        with Pool(processes=num_processes) as pool:
            #Run the parallized process
            with tqdm(total=len(self.cifs)) as pbar:
                for _ in pool.imap_unordered(self.gen_single_h5, self.cifs):
                    pbar.update()

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

