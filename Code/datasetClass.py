#%% Imports

# Standard imports
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import h5py
from collections import namedtuple
import pandas as pd

# Chemistry imports
from diffpy.Structure import loadStructure
from mendeleev import element
from ase.io import read

# Pytorch imports
import torch
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.utils import to_networkx
from networkx.algorithms.components import is_connected

#%% Dataset Class

class InOrgMatDatasets(Dataset):
    def __init__(self, dataset, root='./', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        root += self.dataset + '/'
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self):
        return self.update_file_names(self.raw_dir, file_extension='h5')
    
    @property
    def processed_file_names(self):
        return self.update_file_names(self.processed_dir, file_extension='pt')

    def update_file_names(self, folder_path, file_extension='*'):
        file_names = [str(filepath.relative_to(folder_path)) for filepath in Path(folder_path).glob(f'**/*.{file_extension}')]
        return file_names
    
    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(f'https://sid.erda.dk/share_redirect/h6ktCBGzPF/{self.dataset}.zip', self.raw_dir)
        extract_zip(path, self.raw_dir)
        Path(path).unlink()

    def process(self):  
        Path(self.processed_dir + '/Train/').mkdir(parents=True, exist_ok=True)   
        Path(self.processed_dir + '/Val/').mkdir(parents=True, exist_ok=True)   
        Path(self.processed_dir + '/Test/').mkdir(parents=True, exist_ok=True)          
        idx_train = 0
        idx_val = 0
        idx_test = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with h5py.File(raw_path, 'r') as h5f:
                # Read graph attributes
                node_feat = torch.tensor(h5f['LocalLabels']['NodeFeatures'][:], dtype=torch.float32)
                edge_index = torch.tensor(h5f['LocalLabels']['EdgeDirections'][:], dtype=torch.long)
                edge_feat = torch.tensor(h5f['LocalLabels']['EdgeFeatures'][:], dtype=torch.float32)
                pos_real = torch.tensor(h5f['LocalLabels']['Coordinates'][:], dtype=torch.float32)
                pos_frac = torch.tensor(h5f['LocalLabels']['FractionalCoordinates'][:], dtype=torch.float32)
                # Read other labels
                cell_params = torch.tensor(h5f['GlobalLabels']['CellParameters'][:], dtype=torch.float32)
                atomic_species = torch.tensor(h5f['GlobalLabels']['ElementsPresent'][:], dtype=torch.float32)
                crystal_type = h5f['GlobalLabels']['CrystalType'][()].decode()
                # Read scattering data
                for key in h5f['ScatteringData'].keys():
                    target_dict = dict(
                        crystal_type = crystal_type,
                        atomic_species = atomic_species,
                        n_atomic_species = len(atomic_species),
                        cell_params = cell_params,
                        np_size = h5f['ScatteringData'][key]['NP size (Å)'][()],
                        nd = torch.tensor(h5f['ScatteringData'][key]['ND'][:], dtype=torch.float32),
                        xrd = torch.tensor(h5f['ScatteringData'][key]['XRD'][:], dtype=torch.float32),
                        nPDF = torch.tensor(h5f['ScatteringData'][key]['nPDF'][:], dtype=torch.float32),
                        xPDF = torch.tensor(h5f['ScatteringData'][key]['xPDF'][:], dtype=torch.float32),
                        sans = torch.tensor(h5f['ScatteringData'][key]['SANS'][:], dtype=torch.float32),
                        saxs = torch.tensor(h5f['ScatteringData'][key]['SAXS'][:], dtype=torch.float32),
                    )
                    
                    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, pos=pos_frac, pos_real=pos_real, y=target_dict)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if 'Train' in raw_path:
                        torch.save(data, Path(self.processed_dir).joinpath(f'./Train/data_{idx_train}.pt'))
                        idx_train += 1
                    elif 'Val' in raw_path:
                        torch.save(data, Path(self.processed_dir).joinpath(f'./Val/data_{idx_val}.pt'))
                        idx_val += 1
                    elif 'Test' in raw_path:
                        torch.save(data, Path(self.processed_dir).joinpath(f'./Test/data_{idx_test}.pt'))
                        idx_test += 1
        return None          

    def len(self, split=None):
        if split:
            length = sum([split in file_path for file_path in self.processed_file_names])
        else:
            length = len(self.processed_file_names)
        return length

    def get(self, idx, data_split='train'):
        data = torch.load(Path(self.processed_dir).joinpath(f'./{data_split.capitalize()}/data_{idx}.pt'))
        return data
    
    def get_statistics(self, return_dataframe=False, verbose=True):
        stat_path = Path(self.processed_dir).joinpath('../datasetStatistics.pkl')
        if stat_path.exists():
            df_stats = pd.read_pickle(stat_path)
        else:
            df_stats = pd.DataFrame(columns=['# of nodes', '# of edges', '# of elements', 'Crystal type', 'NP size (Å)', 'Elements', 'Split'])
            for data_split in ['Train', 'Val', 'Test']:
                for idx in range(self.len(split=data_split)):
                    graph = self.get(idx=idx,data_split=data_split)
                    df_stats.loc[df_stats.shape[0]] = [float(graph.num_nodes), float(graph.num_edges), float(graph.y['n_atomic_species']), graph.y['crystal_type'], graph.y['np_size'], graph.y['atomic_species'], data_split]
        
        df_stats.to_pickle(stat_path)

        if return_dataframe:
            return df_stats
        else:
            return None
    
if __name__ == '__main__':
    InOrgMatDatasets('DatasetTest')
    