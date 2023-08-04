#%% Imports

import numpy as np
from pathlib import Path
import shutil
from tqdm.auto import tqdm
import fileinput
import warnings
from ase.io import read
from traceback_with_variables import iter_exc_lines

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

def fix_truncation_error():
    return None

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

def clean_cif(cif):
    cleaning = True
    error_report = dict(removed=0, loop_error=0, truncation_error=0, parenthesis_error=0)
    while cleaning:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                atoms = read(cif, format='cif')
                cleaning = False
        except AssertionError as err:
            lines = list(iter_exc_lines(err))
            if any("line.lower().startswith('data_')" in line for line in lines):
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
                print(cif)
                print('\n')
                print(err)
        except RuntimeError as err:
            lines = list(iter_exc_lines(err))
            if any("CIF loop ended unexpectedly with incomplete row" in line for line in lines):
                fix_loop_error(cif, err)
                error_report['loop_error'] += 1
        except ValueError as err:
            lines = list(iter_exc_lines(err))
            if any("could not convert string to float" in line for line in lines):
                fix_parenthesis_error(cif, err)
                error_report['parenthesis_error'] += 1
        except Exception as err:
            lines = list(iter_exc_lines(err))
            print(cif)
            print('\n')
            print(err)
            print('\n')        
            for line in lines:
                print(line)
            return True, error_report
    
    return False, error_report

def remove_duplicate_cifs(cif_folder):
    return None

def cif_cleaning_pipeline(cif_folder, save_folder=None, remove_duplicates=True, verbose=True):
    
    if save_folder is None:
        save_folder = cif_folder[:-1] + '_cleaned/'
        
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True)

    
    cifs = [str(file.name) for file in Path(cif_folder).glob('*.cif')]
    
    error_summary = dict(removed=0, loop_error=0, truncation_error=0, parenthesis_error=0)
    
    for cif in tqdm(cifs, desc='Cleaning CIFs'):
        shutil.copy(cif_folder + cif, save_folder + cif)
        stop_loop, error_report = clean_cif(save_folder + cif)
        
        for key in error_report:
            error_summary[key] += error_report[key]
        
        if stop_loop:
            break
    
    if verbose:
        print('Summary of corrected errors in cif dataset\n')
        for key in error_summary:
            print(f'\t{key}: {error_summary[key]}')
    
    if remove_duplicates:
        remove_duplicate_cifs(save_folder)
    
    
    return None