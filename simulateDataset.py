#%% Imports

# Standard imports
from pathlib import Path
from tqdm import tqdm

# Import local functions
from Code.cifSimulation import structureGenerator

#%% Setup

datasetPath = './Dataset'

#%% Simulate CIF files

# Initialize the CIF generator
generator = structureGenerator()

# Specify which metals to use
metals = ['Fe', 'Co', 'Mo', 'Ir', 'Pt']

# Simulate mono-metal oxides
generator.create_cif_dataset(
    n_species=2,
    required_atoms=['O'],
    optional_atoms=metals,
    from_table_values=False,
    save_folder=datasetPath + '/CIFs/',
)