#%% Imports

# Standard imports
from pathlib import Path
from tqdm import tqdm

# Chemistry imports
from elements import elements
from mendeleev import element

# Import local functions
from Code.cifSimulation import structureGenerator
from Code.h5Constructor import h5Constructor

#%% Setup

datasetPath = './Dataset'

#%% Simulate CIF files

# Initialize the CIF generator
generator = structureGenerator()

# Maximum number of atom species in generated structures
n_species = 2

# Choose required atoms to use
required_atoms = ['O']

# Choose optional atoms to use
metals = [atom.Symbol for atom in elements.Alkali_Metals] 
metals += [atom.Symbol for atom in elements.Alkaline_Earth_Metals] 
metals += [atom.Symbol for atom in elements.Transition_Metals] 
metals += [atom.Symbol for atom in elements.Metalloids] 
metals += [atom.Symbol for atom in elements.Others]

optional_atoms = []
for metal in metals:
    try:
        elm_data = element(metal)
        if elm_data.metallic_radius or elm_data.atomic_radius:
            optional_atoms.append(metal)
    except:
        print(f'Removed {metal} from dataset as no table values were available.')

# Simulate mono-metal oxides
generator.create_cif_dataset(
    n_species=required_atoms,
    required_atoms=required_atoms,
    optional_atoms=metals,
    from_table_values=False,
    save_folder=datasetPath + '/CIFs/Simulated/',
)

#%% Simulate scattering data
