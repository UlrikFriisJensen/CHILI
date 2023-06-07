import os, random, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
from SASCalculator import SASCalculator
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




CIF_path = "CIFs/"
CIFs = sorted(glob.glob(CIF_path + "*.xyz"))
for CIF in CIFs:
    generator_PDF = simPDFs()
    generator_PDF.set_parameters(rmin=0, rmax=30, rstep=0.1, Qmin=0.1, Qmax=20, Qdamp=0.04, Biso=0.3, delta2=2, psize=10)
    generator_PDF.genPDFs(CIF)
    r_constructed, Gr_constructed = generator_PDF.getPDF()

    plt.plot(r_constructed, Gr_constructed, label="non-normalised")
    Gr_constructed /= max(Gr_constructed)
    plt.plot(r_constructed, Gr_constructed, label="normalised")
    plt.legend()
    plt.show()
    plt.clf()

# np.savetxt("NbO3_100nm_PDF.gr", np.column_stack([r_constructed, Gr_constructed]))

