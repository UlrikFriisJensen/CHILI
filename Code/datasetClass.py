#%% Imports

# Standard imports
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import h5py
from collections import namedtuple

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
    def __init__(self, dataset, root='../Dataset/', transform=None, pre_transform=None, pre_filter=None, force_update=False):
        self.dataset = dataset
        root += self.dataset + '/'
        self.force_update = force_update
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if not self.force_update:
            raw_file_names = [str(filepath.relative_to(self.raw_dir)) for filepath in Path(self.raw_dir).glob('**/*.h5')]
        else:
            raw_file_names = []
        return raw_file_names

    @property
    def processed_file_names(self):
        if not self.force_update:
            processed_file_names = [str(filepath.relative_to(self.processed_dir)) for filepath in Path(self.processed_dir).glob('**/*.pt')]
        else:
            processed_file_names = []
        return processed_file_names

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(f'https://sid.erda.dk/share_redirect/HiFeydCIqA/{self.dataset}.zip', self.raw_dir)
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
                node_feat = torch.tensor(h5f['GraphElements']['NodeFeatures'][:], dtype=torch.float32)
                edge_index = torch.tensor(h5f['GraphElements']['EdgeDirections'][:], dtype=torch.long)
                edge_feat = torch.tensor(h5f['GraphElements']['EdgeFeatures'][:], dtype=torch.float32)
                pos_real = torch.tensor(h5f['GraphElements']['RealPositions'][:], dtype=torch.float32)
                pos_relative = torch.tensor(h5f['GraphElements']['ScaledPositions'][:], dtype=torch.float32)
                # Read other labels
                cell_params = torch.tensor(h5f['OtherLabels']['CellParameters'][:], dtype=torch.float32)
                atomic_species = torch.tensor(h5f['OtherLabels']['ElementsPresent'][:], dtype=torch.float32)
                crystal_type = h5f['OtherLabels']['CrystalType'][()].decode()
                # Read spectra
                for key in h5f['Spectra'].keys():
                    target_dict = dict(
                        crystal_type = crystal_type,
                        atomic_species = atomic_species,
                        cell_params = cell_params,
                        np_size = h5f['Spectra'][key]['NP size (Å)'][()],
                        nd = torch.tensor(h5f['Spectra'][key]['ND'][:], dtype=torch.float32),
                        xrd = torch.tensor(h5f['Spectra'][key]['XRD'][:], dtype=torch.float32),
                        nPDF = torch.tensor(h5f['Spectra'][key]['nPDF'][:], dtype=torch.float32),
                        xPDF = torch.tensor(h5f['Spectra'][key]['xPDF'][:], dtype=torch.float32),
                        sans = torch.tensor(h5f['Spectra'][key]['SANS'][:], dtype=torch.float32),
                        saxs = torch.tensor(h5f['Spectra'][key]['SAXS'][:], dtype=torch.float32),
                    )
                    
                    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, pos=pos_relative, pos_real=pos_real, y=target_dict) # TODO: Don't know if we should use pos or include positions in node features

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
                    

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx, data_split='train'):
        data = torch.load(Path(self.processed_dir).joinpath(f'./{data_split.capitalize()}/data_{idx}.pt'))
        return data
    
if __name__ == '__main__':
    InOrgMatDatasets('DatasetTest')
    