import numpy as np
import pdb, glob
import matplotlib.pyplot as plt

def is_float(value):
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        delimiter = None
        # Read lines one by one, looking for the first line containing one of the delimiters
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue  # Skip lines starting with '#'
            if ',' in line:
                delimiter = ','
            elif ' ' in line:
                delimiter = ' '
            elif ';' in line:
                delimiter = ';'

            if delimiter:
                columns = line.split(delimiter)
                if len(columns) == 2 and is_float(columns[0]) and is_float(columns[1]):
                    data.append([float(columns[0]), float(columns[1])])

    return np.array(data)

# List of files to load
files = sorted(glob.glob("PDFs/*.gr"))

# Load data from all files
for iter, file_path in enumerate(files):
    data = load_data(file_path)
    data[:,1] /= max(data[:,1])
    plt.plot(data[:,0], data[:,1]+iter, label=file_path[5:])

plt.xlabel("r [Å]")
plt.ylabel("G(r) (a. u.)")
plt.legend()
plt.savefig("PDFs.png", dpi=300)
plt.show()
