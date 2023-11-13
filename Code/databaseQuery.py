#%% Imports

from pathlib import Path
import requests
import zipfile
import io
from multiprocessing import Pool, cpu_count
from itertools import islice, repeat
from tqdm.auto import tqdm
import argparse

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

def downloadFromCOD(input_tuple):
    id_batch, save_folder, batch_size = input_tuple
    try:
        requested_ids = ''.join(id_batch)
    
        # Request cif files
        api_url = f'https://www.crystallography.net/cod/result?format=zip&id={requested_ids}'
        response = requests.get(api_url)
        # Extract requested cif files
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall(save_folder)
    except requests.exceptions.ConnectionError:
        for id_sub_batch in batched(id_batch, batch_size // 10):
            try:
                requested_ids = ''.join(id_sub_batch)
                
                # Request cif files
                api_url = f'https://www.crystallography.net/cod/result?format=zip&id={requested_ids}'
                response = requests.get(api_url)
                # Extract requested cif files
                zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                zip_file.extractall(save_folder)
            except requests.exceptions.ConnectionError:
                for id_sub_sub_batch in batched(id_sub_batch, batch_size // 100):
                    requested_ids = ''.join(id_sub_sub_batch)
                    
                    # Request cif files
                    api_url = f'https://www.crystallography.net/cod/result?format=zip&id={requested_ids}'
                    response = requests.get(api_url)
                    # Extract requested cif files
                    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                    zip_file.extractall(save_folder)
    return len(id_batch)

def queryCOD(save_folder, included_atoms=None, excluded_atoms=None, batch_size=800, n_processes=cpu_count() - 1):
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)
    
    id_url = 'https://www.crystallography.net/cod/result?format=lst'
    if included_atoms:
        for i, included_atom in enumerate(included_atoms):
            id_url += f'&el{i+1}={included_atom}'
    if excluded_atoms:
        for i, excluded_atom in enumerate(excluded_atoms):
            id_url += f'&nel{i+1}={excluded_atom}'
    print('Requesting CIF IDs')
    id_response = requests.get(id_url)
    
    with open(f'{save_folder}cif_IDs.txt', 'w') as file:
        file.write(id_response.text)
        
    with open(f'{save_folder}cif_IDs.txt', 'r') as file:
        ids = file.readlines()
    
    inputs = zip(batched(ids, batch_size), repeat(save_folder), repeat(batch_size))
    
    with Pool(processes=n_processes) as pool:
        with tqdm(total=len(ids), desc='Downloading CIFs') as pbar:
            for n_ids in pool.imap_unordered(downloadFromCOD, inputs, chunksize=1):
                pbar.update(n=n_ids)
            
    Path(f'{save_folder}cif_IDs.txt').unlink()
    
    return None

if __name__ == '__main__':
    from cifCleaning import cif_cleaning_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--include', nargs ='*', type=str)
    parser.add_argument('-e', '--exclude', nargs ='*', type=str)
    args = parser.parse_args()
    
    included_atoms = args.include #None if len(args.include) < 1 else args.include
    excluded_atoms = args.exclude #None if len(args.exclude) < 1 else args.exclude
    
    cif_folder = '../Dataset/CIFs/COD_subset/'
    #included_atoms = ['Pd'] #None
    #excluded_atoms = ['C']
    
    #queryCOD(cif_folder, included_atoms=included_atoms, excluded_atoms=excluded_atoms)
    
    cif_cleaning_pipeline(cif_folder, chunksize=100)
