import os, random, sys, glob, re
import numpy as np
import cupy as cp 
import matplotlib.pyplot as plt
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.structure.expansion.makeellipsoid import makeSphere

sys.path.append(os.getcwd())
random.seed(14)  # 'Random' numbers

class simPDFs:
    def __init__(self):

        # Parameters
        self._starting_parameters()  # Initiates starting parameters
        self.sim_para = ['xyz', 'Biso', 'rmin', 'rmax', 'rstep',
                         'qmin', 'qmax', 'qdamp', 'delta2']

        r = np.arange(self.rmin, self.rmax, self.rstep)  # Used to create header

    def _starting_parameters(self):
        
        self.qmin = 0 # Smallest Qrange included in the PDF generation
        self.qmax = 30  # Largest Qrange included in the PDF generation
        self.qdamp = 0.04  # Instrumental dampening
        self.rmin = 0  # Smallest r value
        self.rmax = 30.1  # Largest r value.
        self.rstep = 0.1  # Nyquist for qmax = 34.1 Å-1
        self.Biso = 0.3  # Atomic vibration
        self.delta2 = 2  # Corelated vibration
        self.psize = 1000000000 # Crystalline size of material

        return None

    def genPDFs(self, StructureFile):
        stru = loadStructure(StructureFile)

        stru.B11 = self.Biso
        stru.B22 = self.Biso
        stru.B33 = self.Biso
        stru.B12 = 0
        stru.B13 = 0
        stru.B23 = 0    

        PDFcalc = PDFCalculator(rmin=self.rmin, rmax=self.rmax, rstep=self.rstep,
                                qmin=self.qmin, qmax=self.qmax, qdamp=self.qdamp, delta2=self.delta2)
        
        PDFcalc.radiationType="N" # Does not work, WHY?
        r0, g0 = PDFcalc(stru)

        dampening = self.size_damp(r0, self.psize)
        g0 = g0 * dampening

        self.r = r0
        self.Gr = g0

        return None

    def size_damp(self, x, spdiameter):

        tau = x / spdiameter
        ph = 1 - 1.5 * tau + 0.5 * tau ** 3
        index_min = np.argmin(ph)
        ph[index_min + 1:] = 0

        return ph

    def set_parameters(self, rmin, rmax, rstep,  Qmin, Qmax, Qdamp, Biso, delta2, psize):
        # Add some random factor to the simulation parameters

        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        self.qmin = Qmin
        self.qmax = Qmax
        self.qdamp = Qdamp
        self.Biso = Biso
        self.delta2 = delta2
        self.psize = psize

        return None

    def getPDF(self):
        return self.r, self.Gr


def read_XYZ(structure_model):
    """
    Read an XYZ file and return an array of atomic symbols and positions.

    Parameters:
    - structure_model: name of the XYZ file

    Returns:
    - numpy array containing atomic symbols and positions
    """
    struct = []
    with open(structure_model, 'r') as fi:
        for iter, line in enumerate(fi.readlines()):
            sep_line = line.strip('{}\n\r ').split()
            if iter > 1:
                struct.append(sep_line)
    return np.array(struct)


def sinc(x):
    """
    Compute the sinc function, defined as sin(x) / x.

    Parameters:
    - x: numpy array or float

    Returns:
    - numpy array or float
    """
    return np.sin(x) / x


def retrieve_cromer_mann():
    """
    Read in the Cromer-Mann parameters from the flat text file 'cromer-mann.txt'.

    Returns:
    - cromer_mann_params : dict
    """
    with open('cromer-mann.txt', 'r') as f:
        cromer_mann_params = {}
        lines = f.readlines()

        for line in lines:
            if line[:2] == '#S':
                atomicZ = int(line[3:5].strip())
                symb = line[5:].strip()

                ion_state = 0  # default, not an ion

                g = re.search('(\d+)\+', symb)
                if g: ion_state = int(g.group()[:-1])  # cation

                g = re.search('(\d+)\-', symb)
                if g: ion_state = int(g.group()[:-1])  # anion

                g = re.search('\.', symb)
                if g: ion_state = '.'  # radical

            elif line[0] != '#':
                params = [float(p) for p in line.strip().split()]
                params.append(params.pop(4))  # parameter 'c' is in the middle -- move it to end
                cromer_mann_params[(atomicZ, ion_state)] = params
                cromer_mann_params[symb] = params

    return cromer_mann_params


def atomic_scattering_factors_from_atom(atom, q, cromer_mann_params):
    """
    Compute the atomic scattering factor for a given atom and array of q values.

    Parameters:
    - atom: atomic symbol
    - q: numpy array of q values
    - cromer_mann_params: Cromer-Mann parameters (dictionary)

    Returns:
    - numpy array of atomic scattering factors for each q
    """
    a1, a2, a3, a4, b1, b2, b3, b4, c = cromer_mann_params[atom]
    a = np.array([a1, a2, a3, a4])
    b = np.array([b1, b2, b3, b4])
    q = q.reshape(-1, 1)  # Reshape q to allow broadcasting
    return np.sum(a * np.exp(-b * (q / (4*np.pi))**2), axis=-1) + c


def get_atomic_formfactors_from_xyz(atom_list, xyz, q):
    """
    Compute atomic form factors for a list of atoms.

    Parameters:
    - atom_list: list of atomic symbols
    - xyz: atomic coordinates
    - q: numpy array of q values

    Returns:
    - numpy array of atomic form factors for each atom and q
    """
    cromer_mann_params = retrieve_cromer_mann()
    f_atoms = np.zeros((len(xyz), len(q)))  # Initialize scattering factors array

    # Find unique atom types
    unique_atoms = list(set(atom_list))

    # Pre-compute scattering factors for unique atom types
    unique_f_atoms = np.array([atomic_scattering_factors_from_atom(atom, q, cromer_mann_params) for atom in unique_atoms])

    # Create a mapping from atom type to its index in the unique list
    atom_to_index = {atom: index for index, atom in enumerate(unique_atoms)}

    # Fill in f_atoms for all atoms based on the pre-computed results
    f_atoms = unique_f_atoms[np.array([atom_to_index[atom] for atom in atom_list])]

    return f_atoms


def Debye_Calculator_GPU(atom_list, xyz, q):
    """
    Compute the Debye scattering equation on GPU.

    Parameters:
    - atom_list: list of atomic symbols
    - xyz: atomic coordinates
    - q: numpy array of q values

    Returns:
    - numpy array of intensities for each q
    """
    f_atoms = get_atomic_formfactors_from_xyz(atom_list, xyz, q)

    xyz_gpu = cp.asarray(xyz)
    q_gpu = cp.asarray(q)
    f_atoms_gpu = cp.asarray(f_atoms)

    dist_matrix_gpu = cp.sqrt(cp.sum((xyz_gpu[:, cp.newaxis, :] - xyz_gpu)**2, axis=-1))
    sinc_gpu = cp.sinc(cp.tensordot(q_gpu, dist_matrix_gpu, axes=0) / np.pi)
    f_outer = cp.einsum('ik,jk->ijk', f_atoms_gpu, f_atoms_gpu)
    
    intensity_gpu = cp.zeros(len(q))
    for i in range(len(q)):
        intensity_gpu[i] = cp.sum(f_outer[:, :, i] * sinc_gpu[i])

    return cp.asnumpy(intensity_gpu)  # Transfer result back to CPU

def Debye_Calculator_GPU_bins(atom_list, xyz, q, n_bins=1000):
    """
    Compute the Debye scattering equation on GPU with binned distances.

    Parameters:
    - atom_list: list of atomic symbols
    - xyz: atomic coordinates
    - q: numpy array of q values
    - n_bins: number of bins for the histogram of distances

    Returns:
    - numpy array of intensities for each q
    """
    # Retrieve atomic form factors
    f_atoms = get_atomic_formfactors_from_xyz(atom_list, xyz, q)
  
    # Transfer data to GPU
    xyz_gpu = cp.asarray(xyz)
    q_gpu = cp.asarray(q)
    f_atoms_gpu = cp.asarray(f_atoms)
    
    # Compute pairwise distances
    dist_matrix_gpu = cp.sqrt(cp.sum((xyz_gpu[:, cp.newaxis, :] - xyz_gpu)**2, axis=-1))
 
    # Compute distance histogram for each q value
    dist_hist = cp.zeros((len(q_gpu), n_bins))
    bin_edges = cp.zeros(n_bins + 1)  # we will store the bin edges here
    for i in range(len(q_gpu)):
        f_outer = cp.outer(f_atoms_gpu[:, i], f_atoms_gpu[:, i])
        dist_hist[i], bin_edges = cp.histogram(dist_matrix_gpu, bins=n_bins, weights=f_outer)
    #f_outer = cp.outer(f_atoms_gpu, f_atoms_gpu)
    #dist_hist, bin_edges = cp.histogram(dist_matrix_gpu, bins=n_bins, weights=f_outer)
    

    # Compute the bin centers from the edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute sinc functions for all q and r_ij
    sinc_gpu = cp.sinc(q_gpu[:, None] * bin_centers[None, :] / np.pi)

    # Initialize intensity array
    intensity_gpu = cp.zeros(len(q_gpu))

    # Compute the intensity for each q
    for i, q_value in enumerate(q_gpu):
        # Multiply sinc functions by the histogram and sum the result
        intensity_gpu[i] = cp.sum(dist_hist[i] * sinc_gpu[i])

    # Transfer result back to CPU
    return cp.asnumpy(intensity_gpu)

CIF_file = "XXX"
# Simulate a Pair Distribution Function - on CPU
generator_PDF = simPDFs()
generator_PDF.set_parameters(rmin=0, rmax=30, rstep=0.1, Qmin=0.1, Qmax=20, Qdamp=0.04, Biso=0.3, delta2=2, psize=10)
generator_PDF.genPDFs(CIF_file)
r_constructed, Gr_constructed = generator_PDF.getPDF()

# Simulate a Small-Angle Scattering pattern - on GPU
XYZ_file = CutOutXYZFile(CIF_file) # Ulrik has script
struct = read_XYZ(XYZ_file)
atom_list_small = struct[:,0]
xyz_small = np.float16(struct[:,1:])
q = np.arange(0, 3, 0.01)
intensity = Debye_Calculator_GPU_bins(atom_list_small, xyz_small, q, n_bins=10000)

# Simulate a Powder Diffraction pattern - on GPU
XYZ_file = CutOutXYZFile(CIF_file) # Ulrik has script
struct = read_XYZ(XYZ_file)
atom_list_small = struct[:,0]
xyz_small = np.float16(struct[:,1:])
q = np.arange(1, 30, 0.05)
intensity = Debye_Calculator_GPU_bins(atom_list_small, xyz_small, q, n_bins=10000)