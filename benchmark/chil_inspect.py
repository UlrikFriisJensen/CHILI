#%%
!pip install torch torch-geometric h5py pandas
#%%
!pwd
# %%
from benchmark.dataset_class import CHILI
import torch

# Initialize dataset
root = "dataset"  # Directory where data will be downloaded
dataset = CHILI(root=root, dataset="CHILI-3K")  # or "CHILI-100K"

# Load a single data point
data = dataset[0]

# Print available attributes
print("\nAvailable attributes:")
print("Node features (x):", data.x.shape)
print("Edge indices (edge_index):", data.edge_index.shape)
print("Edge attributes (edge_attr):", data.edge_attr.shape)
print("Absolute positions (pos_abs):", data.pos_abs.shape)
print("Fractional positions (pos_frac):", data.pos_frac.shape)

# Print all available properties in y dictionary
print("\nProperties in y dictionary:")
for key, value in data.y.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")

# Example of accessing specific properties
print("\nExample property values:")
print("Crystal type:", data.y["crystal_type"])
print("Space group number:", data.y["space_group_number"])
print("Crystal system:", data.y["crystal_system"])
print("Number of atoms:", data.y["n_atoms"])
print("Cell parameters:", data.y["cell_params"])
# %%
import torch
from torch_geometric.data import Data


# Extract required information from your CHILI data
crystal_array_dict = {
    'frac_coords': data.pos_frac.numpy(),
    'atom_types': data.x[:, 0].long().numpy(),  # First column contains atomic numbers
    'lengths': data.y['cell_params'][:3].numpy(),  # a, b, c
    'angles': data.y['cell_params'][3:].numpy(),  # alpha, beta, gamma
}

# Create a list with single crystal
crystal_array_list = [crystal_array_dict]

#%%
import sys
sys.path.append('/Users/dmitriynielsen/GitHub/DiffCSP/New_try02/DiffCSP')

#%%
from diffcsp.pl_data.dataset import TensorCrystDataset
# Create TensorCrystDataset
dataset = TensorCrystDataset(
    crystal_array_list,
    niggli=True,
    primitive=False,
    graph_method='crystalnn',
    preprocess_workers=1,
    lattice_scale_method='scale_length'
)
# %%
import torch
import numpy as np
from torch_geometric.data import Data
from diffcsp.pl_data.dataset import TensorCrystDataset
from diffcsp.common.utils import load_model
from diffcsp.common.data_utils import get_scaler_from_data_list

# %%
