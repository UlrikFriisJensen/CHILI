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
    def __init__(self, dataset, root='../Dataset/', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        root += self.dataset + '/'
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        raw_file_names = [str(filepath.relative_to(self.raw_dir)) for filepath in Path(self.raw_dir).glob('**/*.h5')]
        return raw_file_names

    @property
    def processed_file_names(self):
        return ['UnknownDataFile.txt']

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(f'https://sid.erda.dk/share_redirect/HiFeydCIqA/{self.dataset}.zip', self.raw_dir)
        extract_zip(path, self.raw_dir)
        Path(path).unlink()

    def process(self):            
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with h5py.File(raw_path, 'r') as h5f:
                # Read graph attributes
                node_feat = torch.tensor(h5f['GraphElements']['NodeFeatures'][:], dtype=torch.float32)
                edge_index = torch.tensor(h5f['GraphElements']['EdgeDirections'][:], dtype=torch.long)
                edge_feat = torch.tensor(h5f['GraphElements']['EdgeFeatures'][:], dtype=torch.float32)
                # TODO: Figure out if coordinates should be placed in pos argument
                # Read spectra
                for key in h5f['Spectra'].keys():
                    # TODO: Get cell parameters here as well
                    target_dict = dict(
                        np_size = float(key[:-1]),
                        nd = torch.tensor(h5f['Spectra'][key]['ND'][:], dtype=torch.float32),
                        xrd = torch.tensor(h5f['Spectra'][key]['XRD'][:], dtype=torch.float32),
                        nPDF = torch.tensor(h5f['Spectra'][key]['PDF (Neutron)'][:], dtype=torch.float32),
                        xPDF = torch.tensor(h5f['Spectra'][key]['PDF (X-ray)'][:], dtype=torch.float32),
                        sans = torch.tensor(h5f['Spectra'][key]['SANS'][:], dtype=torch.float32),
                        saxs = torch.tensor(h5f['Spectra'][key]['SAXS'][:], dtype=torch.float32),
                    )
                    
                    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, y=target_dict) # TODO: Don't know if we should use pos or include positions in node features

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, Path(self.processed_dir).joinpath(f'data_{idx}.pt'))
                    idx += 1
                    

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir).joinpath(f'data_{idx}.pt'))
        return data
    
if __name__ == '__main__':
    InOrgMatDatasets('DatasetTest')
    