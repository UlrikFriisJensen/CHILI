import sys
from Code.h5Constructor import h5Constructor

# FILEPATH: /root/InOrgMatDataset/batched_data_generation.py

# Get the slurm id from the command line arguments
slurm_id = int(sys.argv[1])

# Load the batch of files corresponding to the slurm id
batch_file_path = f".Dataset/batch_files/batch_{slurm_id}.txt"
with open(batch_file_path, "r") as f:
    batch_files = f.read().splitlines()

constructor = h5Constructor('./Dataset/CIFs/Simulated/', './Dataset/h5/Simulated/')

# Construct the h5 files
constructor.gen_h5s(batch_files, parallelize=False, device='cuda')