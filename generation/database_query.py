# %% Imports

import argparse
import io
import zipfile
from itertools import islice, repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
import requests
from elements import elements
from tqdm.auto import tqdm

# %% Functions


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def queryCODIDs(input_tuple):
    (included_atoms, excluded_atoms), max_volume = input_tuple

    id_url = "https://www.crystallography.net/cod/result?format=lst"
    if included_atoms:
        for i, included_atom in enumerate(included_atoms):
            id_url += f"&el{i+1}={included_atom}"
    if excluded_atoms:
        for i, excluded_atom in enumerate(excluded_atoms):
            id_url += f"&nel{i+1}={excluded_atom}"
    if max_volume:
        id_url += f"&vmax={max_volume}"

    id_response = requests.get(id_url)

    return id_response.text.split("\n")


def getCODIDs(file_path="./COD_subset_IDs.csv"):
    # Define metals
    metals = [atom.Symbol for atom in elements.Alkali_Metals]
    metals += [atom.Symbol for atom in elements.Alkaline_Earth_Metals]
    metals += [atom.Symbol for atom in elements.Transition_Metals]
    metals += [atom.Symbol for atom in elements.Metalloids]
    metals += [atom.Symbol for atom in elements.Others]  # Post-transition metals
    metals += [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]  # Lanthanides

    # Remove elements that does not have a well defined radius or are rare in nanoparticles
    unwanted_elements = [
        "Fr",
        "Po",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Uub",
        "Uun",
        "Uuu",
    ]
    for elm in unwanted_elements:
        metals.remove(elm)

    # Define non-metals
    non_metals = [atom.Symbol for atom in elements.Non_Metals]
    non_metals += [atom.Symbol for atom in elements.Halogens]
    non_metals.remove("At")

    # Find all possible two element combinations of metals and non-metals
    combinations = []
    for metal in metals:
        for non_metal in non_metals:
            combinations.append(([metal, non_metal], []))
    # All metals without any non-metals
    for metal in metals:
        combinations.append(([metal], non_metals))

    # Query COD for IDs
    inputs = zip(combinations, repeat(1000))
    id_list = []
    with Pool(processes=cpu_count() - 1) as pool:
        with tqdm(total=len(combinations), desc="Querying COD") as pbar:
            for returned_ids in pool.imap_unordered(queryCODIDs, inputs, chunksize=1):
                id_list.extend(returned_ids)
                pbar.update()

    # Remove duplicate IDs
    df_ids = pd.DataFrame(id_list)
    df_ids = df_ids.drop_duplicates()
    # Save IDs to file
    df_ids.to_csv(file_path, index=False, header=False)


def downloadFromCOD(input_tuple):
    id_batch, save_folder, batch_size = input_tuple
    try:
        requested_ids = "".join(id_batch)

        # Request cif files
        api_url = (
            f"https://www.crystallography.net/cod/result?format=zip&id={requested_ids}"
        )
        response = requests.get(api_url)
        # Extract requested cif files
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall(save_folder)
    except requests.exceptions.ConnectionError:
        for id_sub_batch in batched(id_batch, batch_size // 10):
            try:
                requested_ids = "".join(id_sub_batch)

                # Request cif files
                api_url = f"https://www.crystallography.net/cod/result?format=zip&id={requested_ids}"
                response = requests.get(api_url)
                # Extract requested cif files
                zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                zip_file.extractall(save_folder)
            except requests.exceptions.ConnectionError:
                for id_sub_sub_batch in batched(id_sub_batch, batch_size // 100):
                    requested_ids = "".join(id_sub_sub_batch)

                    # Request cif files
                    api_url = f"https://www.crystallography.net/cod/result?format=zip&id={requested_ids}"
                    response = requests.get(api_url)
                    # Extract requested cif files
                    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                    zip_file.extractall(save_folder)
    return len(id_batch)


def queryCOD(
    save_folder,
    included_atoms=None,
    excluded_atoms=None,
    batch_size=800,
    n_processes=cpu_count() - 1,
    id_file=None,
):
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)

    if not id_file:
        id_url = "https://www.crystallography.net/cod/result?format=lst"
        if included_atoms:
            for i, included_atom in enumerate(included_atoms):
                id_url += f"&el{i+1}={included_atom}"
        if excluded_atoms:
            for i, excluded_atom in enumerate(excluded_atoms):
                id_url += f"&nel{i+1}={excluded_atom}"
        print("Requesting CIF IDs")
        id_response = requests.get(id_url)

        with open(f"{save_folder}cif_IDs.txt", "w") as file:
            file.write(id_response.text)

        with open(f"{save_folder}cif_IDs.txt", "r") as file:
            ids = file.readlines()
    else:
        with open(id_file, "r") as file:
            ids = file.readlines()

    inputs = zip(batched(ids, batch_size), repeat(save_folder), repeat(batch_size))

    with Pool(processes=n_processes) as pool:
        with tqdm(total=len(ids), desc="Downloading CIFs") as pbar:
            for n_ids in pool.imap_unordered(downloadFromCOD, inputs, chunksize=1):
                pbar.update(n=n_ids)

    if not id_file:
        Path(f"{save_folder}cif_IDs.txt").unlink()

    return None


if __name__ == "__main__":
    from cif_cleaning import cif_cleaning_pipeline

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--include', nargs ='*', type=str)
    # parser.add_argument('-e', '--exclude', nargs ='*', type=str)
    # args = parser.parse_args()

    # included_atoms = args.include
    # excluded_atoms = args.exclude

    cif_folder = "../Dataset/CIFs/COD_subset/"
    id_file = "./COD_subset_IDs.csv"

    # Get CIF IDs
    getCODIDs(file_path=id_file)

    # Download CIFs
    queryCOD(cif_folder, id_file=id_file)

    # Clean CIFs
    cif_cleaning_pipeline(cif_folder, chunksize=100)
