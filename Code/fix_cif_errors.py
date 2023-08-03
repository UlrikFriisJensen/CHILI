import numpy as np
import re, os, argparse
from glob import glob
import fileinput
from tqdm import tqdm
from ase.io import read
import warnings
import traceback
import fileinput

from traceback_with_variables import iter_exc_lines

def fix_loop_errors(
    folder_path,
    debug_print = False,
):
    def find_and_fix(
        path,
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                atoms = read(path, format='cif')
                return False
        except Exception as e:
            lines = list(iter_exc_lines(e))

            token_found = False
            columns_found = False
            insert_found = False
            
            for l,line in enumerate(lines):
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
                for i, line in enumerate(fileinput.input(path, inplace=True)):
                    if np.all([a[1:-1] in line for a in array]):
                        lineb = line
                        line = line[:-4]
                        line += insert
                        line += '\n'
                        linea = line
                    print('{}'.format(line), end='')
                if debug_print:
                    print('Error found')
                    print(lineb)
                    print(linea)
                return True
            else:
                return False
            
    # Glob the files
    try:
        paths = glob(os.path.join(folder_path, '*.cif'))
    except:
        print('Given directory does not exists')
        return
    if not len(paths):
        print('No files were found in the given directory')
        return

    # Run throuh files
    pbar = tqdm(total=len(paths), desc='Fixing "loop" errors in the .cif files...')
    count = 0
    for i, path in enumerate(paths):
        tries = 0
        found_path = False
        while find_and_fix(path):
            tries += 1
            found_path = True
            if tries < 5:
                continue
            else:
                break
        if found_path:
            count += 1
        pbar.update(1)
    pbar.close()

    print('Found (and fixed)', count, 'files with "loop" errors')

def fix_truncation_errors(
    folder_path,
    debug_print = False,
):
    paths = sorted(glob(os.path.join(folder_path, '*.cif')))
    pbar = tqdm(total=len(paths), desc='Fixing "truncation" errors in the .cif files...')
    count = 0
    for i, path in enumerate(paths):
        found_error = False
        for line in fileinput.input(path, inplace = True):
            if '0.1111 ' in line:
                found_error = True
                line = line.replace('0.1111', '0.11111')
            if '0.111111 ' in line:
                found_error = True
                line = line.replace('0.111111', '0.11111')
            if '0.8333 ' in line:
                found_error = True
                line = line.replace('0.8333', '0.83333')
            if '0.6667 ' in line:
                found_error = True
                line = line.replace('0.6667', '0.66667')
            if '0.6666 ' in line:
                found_error = True
                line = line.replace('0.6666', '0.66666')
            if '0.666666 ' in line:
                found_error = True
                line = line.replace('0.666666', '0.66666')
            if '0.3333 ' in line:
                found_error = True
                line = line.replace('0.3333', '0.33333')
            if '0.333333 ' in line:
                found_error = True
                line = line.replace('0.333333', '0.33333')
            print('{}'.format(line), end='')
        if found_error:
            count += 1
        pbar.update(1)
    pbar.close()

    print('Found (and fixed)', count, 'files with "truncation" errors')

def main(args):
    folder_path = args.path
    print()
    inp = input(f'Are you sure you want to change all the files in dir: "{folder_path}" ?\nThis action cannot be undone and will perminantly change the files in the directory!\n(y/n)')
    if (inp == 'y') or (inp == 'yes'):
        if (not args.fix_loop) and (not args.fix_trunction):
            print('No options selected, please use "-l" for loop/import fix and "-t" for truncation fix')
            return
        if args.fix_loop:
            fix_loop_errors(folder_path)
        if args.fix_trunction:
            fix_truncation_errors(folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', help='Path to folder containing the cif files')
    parser.add_argument('--fix_loop', '-l', action='store_true', help='Fix the Import/Loop errors')
    parser.add_argument('--fix_trunction', '-t', action='store_true', help='Fix the truncation errors')
    args = parser.parse_args()
    main(args)
