# %% Imports
import argparse
import os
from glob import glob

import numpy as np


# %% Main function
def main(args):
    files = glob(os.path.join(args.dataset, "*.cif"))
    files_split = np.array_split(np.array(files), args.batch_size)

    for i, f in enumerate(files_split):
        np.savetxt(f"batch_{i+1}.txt", f, fmt="%s", delimiter="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-b", "--batch_size", required=True, type=int)
    args = parser.parse_args()

    main(args)
