#%% Imports

from pathlib import Path
import requests
import zipfile
import io
from itertools import islice
from tqdm.auto import tqdm

#%% Functions

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch
        
def queryCOD(save_folder, included_atoms=None, excluded_atoms=None, batch_size=800):
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)
    
    id_url = 'https://www.crystallography.net/cod/result?format=lst'
    if included_atoms:
        for i, included_atom in enumerate(included_atoms):
            id_url += f'&el{i+1}={included_atom}'
    if excluded_atoms:
        for i, excluded_atom in enumerate(excluded_atoms):
            id_url += f'&nel{i+1}={excluded_atom}'
            
    id_response = requests.get(id_url)
    
    with open(f'{save_folder}cif_IDs.txt', 'w') as file:
        file.write(id_response.text)
        
    with open(f'{save_folder}cif_IDs.txt', 'r') as file:
        ids = file.readlines()
        with tqdm(total=len(ids), desc='Downloading CIFs') as pbar:
            for id_batch in batched(ids, batch_size):
                try:
                    requested_ids = ''.join(id_batch)
                
                    # Request cif files
                    api_url = f'https://www.crystallography.net/cod/result?format=zip&id={requested_ids}'
                    response = requests.get(api_url)
                    # Extract requested cif files
                    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                    zip_file.extractall(save_folder)
                    
                    pbar.update(n=len(id_batch))
                except requests.exceptions.ConnectionError:
                    print('Exception caught')
                    for id_sub_batch in batched(id_batch, batch_size // 10):
                        requested_ids = ''.join(id_sub_batch)
                        
                        # Request cif files
                        api_url = f'https://www.crystallography.net/cod/result?format=zip&id={requested_ids}'
                        response = requests.get(api_url)
                        # Extract requested cif files
                        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                        zip_file.extractall(save_folder)
                        
                        pbar.update(n=len(id_sub_batch))
            
    Path(f'{save_folder}cif_IDs.txt').unlink()
    
    return None

if __name__ == '__main__':
    from cifCleaning import cif_cleaning_pipeline
    
    cif_folder = '../Dataset/CIFs/COD_subset/'
    included_atoms = None
    excluded_atoms = ['C']
    
    queryCOD(cif_folder, included_atoms=included_atoms, excluded_atoms=excluded_atoms)
    
    cif_cleaning_pipeline(cif_folder)