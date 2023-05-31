#%% Imports

# Standard imports
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np

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
    def __init__(self, dataset, root='../Dataset/', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        root += self.dataset + '/'
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [f'{self.dataset}.zip']

    @property
    def processed_file_names(self):
        return ['UnknownDataFile.txt']

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(f'https://sid.erda.dk/share_redirect/HiFeydCIqA/{self.dataset}.zip', self.raw_dir)
        extract_zip(path, self.raw_dir)
        Path(path).unlink()

    def process(self):            
        train_files = sorted([str(x.name) for x in Path(self.raw_dir + '/Train/').glob('*.cif')])
        # print(train_files)
        
        # TODO: Construct graphs
        # TODO: Calculate spectra
        # TODO: Normalize graphs
        # TODO: Save in a format that is easy to load using DataLoader
        
        val_files = sorted([str(x.name) for x in Path(self.raw_dir + '/Val/').glob('*.cif')])
        # print(val_files)
        
        test_files = sorted([str(x.name) for x in Path(self.raw_dir + '/Test/').glob('*.cif')])
        # print(test_files)
            # # Read data from `raw_path`.
            # data = Data(...)

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue

            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            # torch.save(data, Path(self.processed_dir).joinpath(f'data.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir).joinpath(f'data.pt'))
        return data
    
if __name__ == '__main__':
    InOrgMatDatasets('DatasetTest')
    