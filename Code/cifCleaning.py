#%% Imports

import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import fileinput
import warnings
from ase.io import read
from traceback_with_variables import iter_exc_lines

#%% Functions

def fix_loop_error():
    return None

def fix_truncation_error():
    return None

def fix_parenthesis_error():
    return None

def clean_cif(cif_path):
    return None

def remove_duplicate_cifs(cif_folder):
    return None

def cif_cleaning_pipeline(cif_folder, remove_duplicates=True, inplace=False, save_folder=None):
    
    if not inplace:
        if isinstance(save_folder, None):
            save_folder = cif_folder + '_cleaned'
            
        if not Path(save_folder).exists():
            Path(save_folder).mkdir(parents=True)
    else:
        save_folder = cif_folder
    
    cifs = [str(file.name) for file in Path(cif_folder).glob('*.cif')]
    
    for cif in tqdm(cifs, desc='Cleaning CIFs'):
        clean_cif(cif)
    
    if remove_duplicates:
        remove_duplicate_cifs(save_folder)
    
    return None