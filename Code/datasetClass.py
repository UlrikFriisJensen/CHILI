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
# from diffpy.Structure import loadStructure
# from mendeleev import element
# from ase.io import read

# Machine Learning imports
import torch
from torch.utils.data import Subset
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.utils import to_networkx
from networkx.algorithms.components import is_connected
from sklearn.model_selection import train_test_split

#%% Dataset Class

class InOrgMatDatasets(Dataset):
    def __init__(self, dataset, root='./', transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        root += self.dataset + '/'
        self.train_set = None
        self.validation_set = None
        self.test_set = None
        super().__init__(root, transform, pre_transform, pre_filter)       

    @property
    def raw_file_names(self):
        return self.update_file_names(self.raw_dir, file_extension='h5')
    
    @property
    def processed_file_names(self):
        return self.update_file_names(self.processed_dir, file_extension='pt')

    def update_file_names(self, folder_path, file_extension='*'):
        file_names = [str(filepath.relative_to(folder_path)) for filepath in Path(folder_path).glob(f'*.{file_extension}') if 'pre' not in str(filepath)]
        if len(file_names) == 0 and Path(folder_path).exists():
            for subfolder in Path(folder_path).iterdir():
                if subfolder.is_dir():
                    file_names += [str(filepath.relative_to(folder_path)) for filepath in Path(subfolder).glob(f'*.{file_extension}') if 'pre' not in str(filepath)]
        return file_names
    
    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(f'https://sid.erda.dk/share_redirect/h6ktCBGzPF/{self.dataset}.zip', self.raw_dir)
        extract_zip(path, self.raw_dir)
        Path(path).unlink()

    def process(self):  
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with h5py.File(raw_path, 'r') as h5f:
                # Read unit cell graph attributes
                unit_cell_node_feat = torch.tensor(h5f['UnitCellGraph']['NodeFeatures'][:], dtype=torch.float32)
                unit_cell_edge_index = torch.tensor(h5f['UnitCellGraph']['EdgeDirections'][:], dtype=torch.long)
                unit_cell_edge_feat = torch.tensor(h5f['UnitCellGraph']['EdgeFeatures'][:], dtype=torch.float32)
                unit_cell_pos_abs = torch.tensor(h5f['UnitCellGraph']['AbsoluteCoordinates'][:], dtype=torch.float32)
                unit_cell_pos_frac = torch.tensor(h5f['UnitCellGraph']['FractionalCoordinates'][:], dtype=torch.float32)
                # Read other labels
                cell_params = torch.tensor(h5f['GlobalLabels']['CellParameters'][:], dtype=torch.float32)
                atomic_species = torch.tensor(h5f['GlobalLabels']['ElementsPresent'][:], dtype=torch.float32)
                crystal_type = h5f['GlobalLabels']['CrystalType'][()].decode()
                space_group_symbol = h5f['GlobalLabels']['SpaceGroupSymbol'][()].decode()
                space_group_number = h5f['GlobalLabels']['SpaceGroupNumber'][()]
                # Read scattering data
                for key in h5f['DiscreteParticleGraphs'].keys():
                    # Read discrete particle graph attributes
                    node_feat = torch.tensor(h5f['DiscreteParticleGraphs'][key]['NodeFeatures'][:], dtype=torch.float32)
                    edge_index = torch.tensor(h5f['DiscreteParticleGraphs'][key]['EdgeDirections'][:], dtype=torch.long)
                    edge_feat = torch.tensor(h5f['DiscreteParticleGraphs'][key]['EdgeFeatures'][:], dtype=torch.float32)
                    pos_abs = torch.tensor(h5f['DiscreteParticleGraphs'][key]['AbsoluteCoordinates'][:], dtype=torch.float32)
                    pos_frac = torch.tensor(h5f['DiscreteParticleGraphs'][key]['FractionalCoordinates'][:], dtype=torch.float32)
                    
                    # Create target dictionary
                    target_dict = dict(
                        # Save global labels
                        crystal_type = crystal_type,
                        space_group_symbol = space_group_symbol,
                        space_group_number = space_group_number,
                        atomic_species = atomic_species,
                        n_atomic_species = len(atomic_species),
                        np_size = h5f['DiscreteParticleGraphs'][key]['NP size (Å)'][()],
                        n_atoms = node_feat.shape[0],
                        n_bonds = edge_index.shape[1],
                        # Save unit cell graph attributes
                        cell_params = cell_params,
                        unit_cell_node_feat = unit_cell_node_feat,
                        unit_cell_edge_index = unit_cell_edge_index,
                        unit_cell_edge_feat = unit_cell_edge_feat,
                        unit_cell_pos_abs = unit_cell_pos_abs,
                        unit_cell_pos_frac = unit_cell_pos_frac,
                        # Save scattering data
                        nd = torch.tensor(h5f['ScatteringData'][key]['ND'][:], dtype=torch.float32),
                        xrd = torch.tensor(h5f['ScatteringData'][key]['XRD'][:], dtype=torch.float32),
                        nPDF = torch.tensor(h5f['ScatteringData'][key]['nPDF'][:], dtype=torch.float32),
                        xPDF = torch.tensor(h5f['ScatteringData'][key]['xPDF'][:], dtype=torch.float32),
                        sans = torch.tensor(h5f['ScatteringData'][key]['SANS'][:], dtype=torch.float32),
                        saxs = torch.tensor(h5f['ScatteringData'][key]['SAXS'][:], dtype=torch.float32),
                    )
                    
                    # Create graph data object
                    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, pos_frac=pos_frac, pos_abs=pos_abs, y=target_dict)

                    # Apply filters
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    
                    # Apply transforms
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    
                    # Save to `self.processed_dir`.
                    torch.save(data, Path(self.processed_dir).joinpath(f'./data_{idx}.pt'))
                    
                    # Update index
                    idx += 1
        return None          

    def len(self, split=None):
        if split is None:
            length = len(self.processed_file_names)
        elif split.lower() == 'train':
            length = len(self.train_set)
        elif split.lower() in ['validation', 'val']:
            length = len(self.validation_set)
        elif split.lower() == 'test':
            length = len(self.test_set)
        else:
            raise ValueError('Split not recognized. Please use either "train", "validation" or "test"')
        return length

    def get(self, idx, split=None):
        if split is None:
            data = torch.load(Path(self.processed_dir).joinpath(f'./data_{idx}.pt'))
        elif split.lower() == 'train':
            data = self.train_set[idx]
        elif split.lower() in ['validation', 'val']:
            data = self.validation_set[idx]
        elif split.lower() == 'test':
            data = self.test_set[idx]
        else:
            raise ValueError('Split not recognized. Please use either "train", "validation" or "test"')
        return data
    
    def create_data_split(self, test_size=0.1, validation_size=None, split_strategy='random', stratify_on='Space group (Number)', random_state=42, return_idx=False):
        '''
        Split the dataset into train, validation and test sets. The indices of the split are saved to csv files in the processed directory.
        '''

        if validation_size is None:
            validation_size = test_size

        df_stats = self.get_statistics(return_dataframe=True)

        if split_strategy == 'random':
            # Split data into train, validation and test sets
            train_idx, test_idx = train_test_split(np.arange(self.len()), test_size=test_size, random_state=random_state)
            train_idx, validation_idx = train_test_split(train_idx, test_size=validation_size/(1-test_size), random_state=random_state)
            
            # Save indices to csv files
            np.savetxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_train.csv'), train_idx, delimiter=',', fmt='%i')
            np.savetxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_validation.csv'), validation_idx, delimiter=',', fmt='%i')
            np.savetxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_test.csv'), test_idx, delimiter=',', fmt='%i')
            
            # Update statistics dataframe
            df_stats[f'{split_strategy.capitalize()} data split'] = ''
            df_stats[f'{split_strategy.capitalize()} data split'].loc[train_idx] = 'Train'
            df_stats[f'{split_strategy.capitalize()} data split'].loc[validation_idx] = 'Validation'
            df_stats[f'{split_strategy.capitalize()} data split'].loc[test_idx] = 'Test'

        elif split_strategy == 'stratified':
            # Split data into train, validation and test sets
            train_idx, test_idx = train_test_split(np.arange(self.len()), test_size=test_size, random_state=random_state, stratify=df_stats[stratify_on])
            train_idx, validation_idx = train_test_split(train_idx, test_size=validation_size/(1-test_size), random_state=random_state, stratify=df_stats.loc[train_idx][stratify_on])
            
            # Save indices to csv files
            np.savetxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_{stratify_on.replace(" ","")}_train.csv'), train_idx, delimiter=',', fmt='%i')
            np.savetxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_{stratify_on.replace(" ","")}_validation.csv'), validation_idx, delimiter=',', fmt='%i')
            np.savetxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_{stratify_on.replace(" ","")}_test.csv'), test_idx, delimiter=',', fmt='%i')

            # Update statistics dataframe
            df_stats[f'{split_strategy.capitalize()} data split ({stratify_on})'] = ''
            df_stats[f'{split_strategy.capitalize()} data split ({stratify_on})'].loc[train_idx] = 'Train'
            df_stats[f'{split_strategy.capitalize()} data split ({stratify_on})'].loc[validation_idx] = 'Validation'
            df_stats[f'{split_strategy.capitalize()} data split ({stratify_on})'].loc[test_idx] = 'Test'   
        else:
            # Raise error if split strategy is not recognized
            raise ValueError('Split strategy not recognized. Please use either "random" or "stratified"')

        # Update statistics file
        df_stats.to_pickle(Path(self.processed_dir).joinpath('../datasetStatistics.pkl'))

        if return_idx:
            return train_idx, validation_idx, test_idx
        else:
            return None
        
    def load_data_split(self, split_strategy='random', stratify_on='Space group (Number)'):
        '''
        Load the indices of the train, validation and test sets from csv files in the processed directory.
        '''
        if split_strategy == 'random':
            # Load indices from csv files
            train_idx = np.loadtxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_train.csv'), delimiter=',', dtype=int)
            validation_idx = np.loadtxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_validation.csv'), delimiter=',', dtype=int)
            test_idx = np.loadtxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_test.csv'), delimiter=',', dtype=int)
            
            # Split the dataset into train, validation and test sets
            self.train_set = Subset(self, train_idx)
            self.validation_set = Subset(self, validation_idx)
            self.test_set = Subset(self, test_idx)
        elif split_strategy == 'stratified':
            # Load indices from csv files
            train_idx = np.loadtxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_{stratify_on.replace(" ","")}_train.csv'), delimiter=',', dtype=int)
            validation_idx = np.loadtxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_{stratify_on.replace(" ","")}_validation.csv'), delimiter=',', dtype=int)
            test_idx = np.loadtxt(Path(self.processed_dir).joinpath(f'../dataSplit_{split_strategy}_{stratify_on.replace(" ","")}_test.csv'), delimiter=',', dtype=int)
            
            # Split the dataset into train, validation and test sets
            self.train_set = Subset(self, train_idx)
            self.validation_set = Subset(self, validation_idx)
            self.test_set = Subset(self, test_idx)

        return None

    def get_statistics(self, return_dataframe=False):
        stat_path = Path(self.processed_dir).joinpath('../datasetStatistics.pkl')
        if stat_path.exists():
            df_stats = pd.read_pickle(stat_path)
        else:
            df_stats = pd.DataFrame(
                columns=[
                    'idx',
                    '# of nodes', 
                    '# of edges', 
                    '# of elements',
                    'Space group (Symbol)',
                    'Space group (Number)', 
                    'Crystal type', 
                    'NP size (Å)', 
                    'Elements', 
                ])
            
            for idx in range(self.len()):
                graph = self.get(idx=idx,)
                df_stats.loc[df_stats.shape[0]] = [
                    idx,
                    float(graph.num_nodes), 
                    float(graph.num_edges), 
                    float(graph.y['n_atomic_species']), 
                    graph.y['space_group_symbol'],
                    float(graph.y['space_group_number']),
                    graph.y['crystal_type'], 
                    graph.y['np_size'], 
                    graph.y['atomic_species'], 
                ]
        
        df_stats.to_pickle(stat_path)

        if return_dataframe:
            return df_stats
        else:
            return None
    
if __name__ == '__main__':
    InOrgMatDatasets('DatasetTest')
    