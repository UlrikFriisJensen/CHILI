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
        
        # Load in cif file
        unit_cell = read(f'{self.cif_dir}/{cif}')
        
        # Construct graph
        node_matrix = []
        edge_index1 = []
        edge_index2 = []
        edge_features = []
        
        n_atoms = len(unit_cell)
        # positions_real = unit_cell.get_positions()
        positions_fractional = unit_cell.get_scaled_positions()
        # position_scale = np.diag(unit_cell.get_cell())
        atomic_number_matrix = unit_cell.get_atomic_numbers()
        
        metal_distances = make_supercell(unit_cell, np.diag([2,2,2])).get_all_distances()
        lc = np.amin(metal_distances[metal_distances > 0.]) 
        
        # try:
        #     metal_distances = unit_cell[atomic_number_matrix != 8].get_all_distances()
        #     lc = np.amin(metal_distances[metal_distances > 0.]) 
        # except:
        #     metal_distances = make_supercell(unit_cell, np.diag([2,2,2])).get_all_distances()
        #     lc = np.amin(metal_distances[metal_distances > 0.]) 
        for i in range(n_atoms):
            for j in range(i,n_atoms):
                if i == j: 
                    continue
                else:
                    dist = unit_cell.get_distance(i, j, mic=True) # TODO: Ensure periodic graph edges are actually created
                    print(dist)
                if (dist < lc):
                    edge_index1.append(i)
                    edge_index2.append(j)
                    edge_features.append(dist)
                    edge_index1.append(j)
                    edge_index2.append(i)
                    edge_features.append(dist)
            node_matrix.append([*positions_fractional[i], atomic_number_matrix[i]])
            
        edge_features = np.array(edge_features)

        node_matrix = np.array(node_matrix)
        direction = np.array([edge_index1, edge_index2])

        # Construct cell parameter matrix
        cell_parameters = unit_cell.cell.cellpar()
        
        #Create graph to check for connectivity
        x = torch.tensor(node_matrix, dtype=torch.float)
        edge_index = torch.tensor(direction, dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g = to_networkx(graph, to_undirected=True)

        if not is_connected(g):
            # print(lc)
            # print(metal_distances)
            print('\n'+cif + '\n')
            return None
        
        # Simulate spectra
        placeholder_spectra = torch.zeros([1,1500])
        
        # Construct .h5 file
        h5_file = h5py.File(f'{self.save_dir}/graph_{cif[:-4]}.h5', 'w') # TODO: Think about file naming
        h5_file.create_dataset('Edge Feature Matrix', data=edge_features)
        h5_file.create_dataset('Node Feature Matrix', data=node_matrix)
        h5_file.create_dataset('Edge Directions', data=direction)
        h5_file.create_dataset('Cell parameters', data=cell_parameters)
        h5_file.create_dataset('PDF (X-ray)', data=placeholder_spectra)
        h5_file.create_dataset('PDF (Neutron)', data=placeholder_spectra)
        h5_file.create_dataset('SAXS', data=placeholder_spectra)
        h5_file.create_dataset('SANS', data=placeholder_spectra)
        h5_file.create_dataset('XANES', data=placeholder_spectra)
        h5_file.close()
        return None
    
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