#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm.auto import tqdm
import fileinput
import warnings
from ase.io import read
from traceback_with_variables import iter_exc_lines
from fractions import Fraction

#%% Functions

def fix_loop_error(cif, err):
    lines = list(iter_exc_lines(err))

    token_found = False
    columns_found = False
    insert_found = False
    
    for line in lines:
        if ('tokens' in line) and ('more' not in line):
            array = ' '.join(line.split('=')[-1].split(' ')[1:])[1:-1].split(', ')
            n_entries = len(array)
            z_fill_len = sum(c.isdigit() for c in array[-1][1:-1].split('.')[-1])
            token_found = True
        if 'ncolumns' in line:
            n_columns = int(line.split(' ')[-1])
            columns_found = True

        if token_found and columns_found:
            n_missing_entries = n_columns - n_entries
            insert = ' 0.'.ljust(z_fill_len+3, '0') * n_missing_entries
            insert_found = True
            break

    if insert_found:
        for line in fileinput.input(cif, inplace=True):
            if np.all([a[1:-1] in line for a in array]):
                line = line[:-4]
                line += insert
                line += '\n'
            print('{}'.format(line), end='')
    return None

def fix_precision_errors(cif, precision=5):
    atom_positions = False
    error_count = 0
    for line in fileinput.input(cif, inplace=True):
        if atom_positions:
            word_list = []
            for word in line.split(' '):
                try:
                    number = float(word.replace('(', '').replace(')', ''))
                    # Test if the number is a numerical approximation of a fraction
                    fraction = Fraction(number).limit_denominator(max_denominator=1000)
                    if (fraction.denominator <=10) and fraction.denominator !=1:
                        number = np.around(float(fraction), precision)
                        word_list.append(str(number))
                        if len(word_list[-1]) != len(word):
                            error_count += 1
                    else:
                        word_list.append(word)
                except:
                    word_list.append(word)
                    continue
            line = ' '.join(word_list)
        else:
            if '_atom_site_label' in line:
                atom_positions = True
        print('{}'.format(line), end='')
    return error_count

def fix_parenthesis_error(cif, err):
    lines = list(iter_exc_lines(err))
    
    invalid_number = str(err).split(':')[1][2:-1]
    
    if '(' == invalid_number[-1]:
        valid_number = invalid_number[:-1]
    elif ('(' in invalid_number) and (')' not in invalid_number):
        valid_number = invalid_number + ')'

    for line in fileinput.input(cif, inplace=True):
        if invalid_number in line:
            line = line.replace(invalid_number, valid_number)
        print('{}'.format(line), end='')
    return None

def clean_cif(cif, unwanted_atoms=None, debug=False):
    clean = False
    error_report = dict(removed=0, loop_error=0, precision_error=0, parenthesis_error=0, unwanted_atom=0)
    cif_metadata = dict()
    while not clean:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                atoms = read(cif, format='cif')
                if unwanted_atoms:
                    if any([atom in unwanted_atoms for atom in atoms.get_atomic_numbers()]):
                        Path(cif).unlink()
                        error_report['unwanted_atom'] += 1
                        break
                error_report['precision_error'] = fix_precision_errors(cif)
                clean = True
                cif_metadata['filepath'] = cif
                cif_metadata['Spacegroup'] = [atoms.info['spacegroup'].no]
                for i, element in enumerate(np.unique(atoms.get_atomic_numbers())):
                    cif_metadata[f'Element{i}'] = element
        except AssertionError as err:
            lines = list(iter_exc_lines(err))
            if any("line.lower().startswith('data_')" in line for line in lines):
                Path(cif).unlink()
                error_report['removed'] += 1
                break
            else:
                if debug:
                    print(cif)
                    print('\n')
                    print(err)
                    return True, error_report, cif_metadata
                else:
                    Path(cif).unlink()
                    error_report['removed'] += 1
                    break
        except StopIteration as err:
            Path(cif).unlink()
            error_report['removed'] += 1
            break
        except IndexError as err:
            lines = list(iter_exc_lines(err))
            if any("pop from empty list" in line for line in lines):
                Path(cif).unlink()
                error_report['removed'] += 1
                break
            else:
                if debug:
                    print(cif)
                    print('\n')
                    print(err)
                    return True, error_report, cif_metadata
                else:
                    Path(cif).unlink()
                    error_report['removed'] += 1
                    break
        except RuntimeError as err:
            lines = list(iter_exc_lines(err))
            if any("CIF loop ended unexpectedly with incomplete row" in line for line in lines):
                fix_loop_error(cif, err)
                error_report['loop_error'] += 1
            else:
                if debug:
                    print(cif)
                    print('\n')
                    print(err)
                    return True, error_report, cif_metadata
                else:
                    Path(cif).unlink()
                    error_report['removed'] += 1
                    break
        except ValueError as err:
            lines = list(iter_exc_lines(err))
            if any("could not convert string to float" in line for line in lines):
                fix_parenthesis_error(cif, err)
                error_report['parenthesis_error'] += 1
            else:
                if debug:
                    print(cif)
                    print('\n')
                    print(err)
                    return True, error_report, cif_metadata
                else:
                    Path(cif).unlink()
                    error_report['removed'] += 1
                    break
        except Exception as err:
            if debug:
                lines = list(iter_exc_lines(err))
                print(cif)
                print('\n')
                print(err)
                print('\n')        
                for line in lines:
                    print(line)
                return True, error_report, cif_metadata
            else:
                Path(cif).unlink()
                error_report['removed'] += 1
                break
    
    return False, error_report, cif_metadata

def remove_duplicate_cifs(metadata):
    # Compare several attributes to determine whether duplicates are present
    
    # Easy parameters to check (Quick filter)
    # Atom species
    # Crystal system
    # Spacegroup
    metadata_noDuplicates = metadata.drop_duplicates(subset=metadata.columns[metadata.columns != 'filepath'], keep='first')
    
    # Detailed check but harder (Detailed filter)
    # Atom positions
    # Cell parameters
    
    # Remove files no longer in the dataset
    for cif in metadata['filepath']:
        if cif not in metadata_noDuplicates['filepath'].to_list():
            Path(cif).unlink()
    
    print(f'Duplicate CIFs removed!\n{len(metadata)} CIFs --> {len(metadata_noDuplicates)} CIFs')
    
    return None

def cif_cleaning_pipeline(cif_folder, save_folder=None, remove_duplicates=True, verbose=True, cod=False, unwanted_atoms=None):
    
    if save_folder is None:
        save_folder = cif_folder[:-1] + '_cleaned/'
        
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)

    if cod:
        cifs = [str(file)[len(cif_folder):] for file in Path(cif_folder).rglob('*.cif')]
    else:
        cifs = [str(file.name) for file in Path(cif_folder).glob('*.cif')]
    
    error_summary = dict(removed=0, loop_error=0, precision_error=0, parenthesis_error=0, unwanted_atom=0)
    
    df_metadata = pd.DataFrame()
    # parellelize
    for cif in tqdm(cifs, desc='Cleaning CIFs'):
        if cod:
            subfolder = '/'.join((save_folder + cif).split('/')[:-1]) + '/'
            if not Path(subfolder).exists():
                Path(subfolder).mkdir(parents=True)
        shutil.copy(cif_folder + cif, save_folder + cif)
        stop_loop, error_report, cif_metadata = clean_cif(save_folder + cif, unwanted_atoms)
        
        if remove_duplicates:
            df_metadata = pd.concat([df_metadata, pd.DataFrame.from_dict(cif_metadata)], ignore_index=True)

        for key in error_report:
            error_summary[key] += error_report[key]
        
        if stop_loop:
            break
    
    if verbose:
        print('Summary of corrected errors in cif dataset\n')
        for key in error_summary:
            print(f'\t{key.capitalize().replace("_", " ")}: {error_summary[key]}')
        print('\n')
    
    if remove_duplicates:
        remove_duplicate_cifs(df_metadata)
    
    return df_metadata