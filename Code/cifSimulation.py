#%% Imports

# Standard imports
from pathlib import Path
from itertools import repeat
from tqdm import tqdm
import re
from typing import Union, Tuple

#Multicore processing
from multiprocessing import Pool, cpu_count

# Math imports
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Periodic table imports
from mendeleev import element

# # Diffpy imports
# from diffpy.structure import Structure
# from diffpy.srreal.pdfcalculator import PDFCalculator

# ASE imports
from ase import Atoms
from ase.build import bulk, make_supercell
from ase.spacegroup import crystal # https://wiki.fysik.dtu.dk/ase/ase/spacegroup/spacegroup.html#ase.spacegroup.crystal
from ase.visualize import view
from ase.io import write, read
from ase.build.tools import sort as ase_sort
# https://databases.fysik.dtu.dk/ase/gettingstarted/tut04_bulk/bulk.html
# https://materialsproject.org/materials/ 

#%% Functions
class structureGenerator():
    def __init__(self) -> None:
        # Cell parameters from chapter 1 of Solid State Chemistry by West
        self.parameter_lookup = dict(
            NaCl = dict(
                MgO = dict(a=4.213),
                CaO = dict(a=4.8105),
                SrO = dict(a=5.160),
                BaO = dict(a=5.539),
                TiO = dict(a=4.177),
                MnO = dict(a=4.445),
                FeO = dict(a=4.307),
                CoO = dict(a=4.260),
                NiO = dict(a=4.1769),
                CdO = dict(a=4.6953),
                TiC = dict(a=4.3285),
                MgS = dict(a=5.200),
                CaS = dict(a=5.6948),
                SrS = dict(a=6.020),
                BaS = dict(a=6.386),
                alpha_MnS = dict(a=5.224),
                MgSe = dict(a=5.462),
                CaSe = dict(a=5.924),
                SrSe = dict(a=6.246),
                BaSe = dict(a=6.600),
                CaTe = dict(a=6.356),
                LaN = dict(a=5.30),
                LiF = dict(a=4.0270),
                LiCl = dict(a=5.1396),
                LiBr = dict(a=5.5013),
                LiI = dict(a=6.00),
                LiH = dict(a=4.083),
                NaF = dict(a=4.64),
                NaCl = dict(a=5.6402),
                NaBr = dict(a=5.9772),
                NaI = dict(a=6.473),
                TiN = dict(a=4.240),
                UN = dict(a=4.890),
                KF = dict(a=5.347),
                KCl = dict(a=6.2931),
                KBr = dict(a=6.5966),
                KI = dict(a=7.0655),
                RbF = dict(a=5.6516),
                RbCl = dict(a=6.5810),
                RbBr = dict(a=6.889),
                RbI = dict(a=7.342),
                AgF = dict(a=4.92),
                AgCl = dict(a=5.549),
                AgBr = dict(a=5.7745),
            ),
            ZincBlende = dict(
                CuF = dict(a=4.255),
                CuCl = dict(a=5.416),
                gamma_CuBr = dict(a=5.6905),
                gamma_CuI = dict(a=6.051),
                gamma_AgI = dict(a=6.495),
                beta_MnS = dict(a=5.600),
                diamond_CC = dict(a=3.5667),
                BeS = dict(a=4.8624),
                BeSe = dict(a=5.07),
                BeTe = dict(a=5.54),
                beta_ZnS = dict(a=5.4060),
                ZnSe = dict(a=5.667),
                beta_SiC = dict(a=4.358),
                SiSi = dict(a=5.4307),
                beta_CdS = dict(a=5.818),
                CdSe = dict(a=6.077),
                CdTe = dict(a=6.481),
                HgS = dict(a=5.8517),
                HgTe = dict(a=6.453),
                GeGe = dict(a=5.6574),
                BN = dict(a=3.616),
                BP = dict(a=4.538),
                BAs = dict(a=4.777),
                AlP = dict(a=5.451),
                AlAs = dict(a=5.662),
                AlSb = dict(a=6.1347),
                alpha_SnSn = dict(a=6.4912),
                GaP = dict(a=5.448),
                GaAs = dict(a=5.6534),
                GaSb = dict(a=6.095),
                InP = dict(a=5.869),
                InAs = dict(a=6.058),
                InSb = dict(a=6.4782),
            ),
            Fluorite = dict(
                CaF2 = dict(a=5.4626),
                SrF2 = dict(a=5.800),
                SrCl2 = dict(a=6.9767),
                BaF2 = dict(a=6.2001),
                CdF2 = dict(a=5.3895),
                beta_PbF2 = dict(a=5.940),
                PbO2 = dict(a=5.349),
                CeO2 = dict(a=5.4110),
                PrO2 = dict(a=5.392),
                ThO2 = dict(a=5.600),
                UO2 = dict(a=5.372),
                NpO2 = dict(a=5.4334),
            ),
            AntiFluorite = dict(
                Li2O = dict(a=4.6114),
                Li2S = dict(a=5.710),
                Li2Se = dict(a=6.002),
                Li2Te = dict(a=6.517),
                Na2O = dict(a=5.55),
                Na2S = dict(a=6.539),
                K2O = dict(a=6.449),
                K2S = dict(a=7.406),
                K2Se = dict(a=7.692),
                K2Te = dict(a=8.168),
                Rb2O = dict(a=6.74),
                Rb2S = dict(a=7.65),
            ),
            Wurtzite = dict(
                ZnO = dict(a=3.2495, c=5.2069, w=0.345),
                ZnS = dict(a=3.811, c=6.234, w=0.375),
                ZnSe = dict(a=3.98, c=6.53, w=0.375),
                ZnTe = dict(a=4.27, c=6.99, w=0.375),
                BeO = dict(a=2.698, c=4.380, w=0.378),
                CdS = dict(a=4.1348, c=6.7490, w=0.375),
                CdSe = dict(a=4.30, c=7.02, w=0.375),
                MnS = dict(a=3.976, c=6.432, w=0.375),
                AgI = dict(a=4.580, c=7.494, w=0.375),
                AlN = dict(a=3.111, c=4.978, w=0.385),
                GaN = dict(a=3.180, c=5.166, w=0.375),
                InN = dict(a=3.533, c=5.693, w=0.375),
                TaN = dict(a=3.05, c=4.94, w=0.375),
                NH4F = dict(a=4.39, c=7.02, w=0.365), # Big cell compared to radius
                SiC = dict(a=3.076, c=5.048, w=0.375),
                MnSe = dict(a=4.12, c=6.72, w=0.375),
            ),
            NiAs = dict(
                NiS = dict(a=3.4392, c=5.3484),
                NiAs = dict(a=3.602, c=5.009),
                NiSb = dict(a=3.94, c=5.14),
                NiSe = dict(a=3.6613, c=5.3562),
                NiSn = dict(a=4.048, c=5.123),
                NiTe = dict(a=3.957, c=5.354),
                FeS = dict(a=3.438, c=5.880),
                FeSe = dict(a=3.637, c=5.958),
                FeTe = dict(a=3.800, c=5.651),
                FeSb = dict(a=4.06, c=5.13),
                delta_NbN = dict(a=2.968, c=5.549),
                PtB = dict(a=3.358, c=4.058), # Anti structure
                PtSn = dict(a=4.103, c=5.428),
                CoS = dict(a=3.367, c=5.160),
                CoSe = dict(a=3.6294, c=5.3006),
                CoTe = dict(a=3.886, c=5.360),
                CoSb = dict(a=3.866, c=5.188),
                CrSe = dict(a=3.684, c=6.019),
                CrTe = dict(a=3.981, c=6.211),
                CrSb = dict(a=4.108, c=5.440),
                MnTe = dict(a=4.1429, c=6.7031),
                MnAs = dict(a=3.710, c=5.691),
                MnSb = dict(a=4.120, c=5.784),
                MnBi = dict(a=4.30, c=6.12),
                PtSb = dict(a=4.130, c=5.472),
                PtBi = dict(a=4.315, c=5.490),
            ),
            CsCl = {
                'CsCl': dict(a=4.123),
                'CsBr': dict(a=4.286),
                'CsI': dict(a=4.5667),
                'CsCN': dict(a=4.25),
                '(NH4)Cl': dict(a=3.8756), # Big cell compared to radius
                '(NH4)Br': dict(a=4.0594), # Big cell compared to radius
                'TlCl': dict(a=3.8340),
                'TlBr': dict(a=3.97),
                'TlI': dict(a=4.198),
                'CuZn': dict(a=2.945),
                'CuPd': dict(a=2.988),
                'AuMg': dict(a=3.259),
                'AuZn': dict(a=3.19),
                'AgZn': dict(a=3.156),
                'LiAg': dict(a=3.168),
                'AlNi': dict(a=2.881),
                'LiHg': dict(a=3.287),
                'MgSr': dict(a=3.900),
            },
            Rutile = dict(
                TiO2 = dict(a=4.5937, c=2.9581, x=0.305),
                CrO2 = dict(a=4.41, c=2.91, x=0.3),
                GeO2 = dict(a=4.395, c=2.859, x=0.307),
                IrO2 = dict(a=4.49, c=3.14, x=0.3),
                beta_MnO2 = dict(a=4.396, c=2.871, x=0.302),
                MoO2 = dict(a=4.86, c=2.79, x=0.3),
                NbO2 = dict(a=4.77, c=2.96, x=0.3),
                OsO2 = dict(a=4.51, c=3.19, x=0.3),
                PbO2 = dict(a=4.946, c=3.379, x=0.3),
                RuO2 = dict(a=4.51, c=3.11, x=0.3),
                CoF2 = dict(a=4.6951, c=3.1796, x=0.306),
                FeF2 = dict(a=4.6966, c=3.3091, x=0.300),
                MgF2 = dict(a=4.623, c=3.052, x=0.303),
                MnF2 = dict(a=4.8734, c=3.3099, x=0.305),
                NiF2 = dict(a=4.6506, c=3.0836, x=0.302),
                PdF2 = dict(a=4.931, c=3.367, x=0.3),
                ZnF2 = dict(a=4.7034, c=3.1864, x=0.303),
                SnO2 = dict(a=4.7373, c=3.1864, x=0.307),
                TaO2 = dict(a=4.709, c=3.065, x=0.3),
                WO2 = dict(a=4.86, c=2.77, x=0.3),
            ),
            CdI2 = {
                'CdI2': dict(a=4.24, c=6.84),
                'CaI2': dict(a=4.48, c=6.96),
                'CoI2': dict(a=3.96, c=6.65),
                'FeI2': dict(a=4.04, c=6.75),
                'MgI2': dict(a=4.14, c=6.88),
                'MnI2': dict(a=4.16, c=6.82),
                'PbI2': dict(a=4.555, c=6.977),
                'ThI2': dict(a=4.13, c=7.02),
                'TiI2': dict(a=4.110, c=6.820),
                'TmI2': dict(a=4.520, c=6.967),
                'VI2': dict(a=4.000, c=6.670),
                'YbI2': dict(a=4.503, c=6.972),
                'ZnI2': dict(a=4.25, c=6.54),
                'VBr2': dict(a=3.768, c=6.180),
                'TiBr2': dict(a=3.629, c=6.492),
                'MnBr2': dict(a=3.82, c=6.19),
                'FeBr2': dict(a=3.74, c=6.17),
                'CoBr2': dict(a=3.68, c=6.12),
                'TiCl2': dict(a=3.561, c=5.875),
                'VCl2': dict(a=3.601, c=5.835),
                'Mg(OH)2': dict(a=3.147, c=4.769),
                'Ca(OH)2': dict(a=3.584, c=4.896),
                'Fe(OH)2': dict(a=3.258, c=4.605),
                'Co(OH)2': dict(a=3.173, c=4.640),
                'Ni(OH)2': dict(a=3.117, c=4.595),
                'Cd(OH)2': dict(a=3.48, c=4.67),
            },
            CdCl2 = dict(
                CdCl2 = dict(a=3.854, c=17.457, anti=False),
                CdBr2 = dict(a=3.95, c=18.67, anti=False),
                CoCl2 = dict(a=3.544, c=17.430, anti=False),
                FeCl2 = dict(a=3.579, c=17.536, anti=False),
                MgCl2 = dict(a=3.596, c=17.589, anti=False),
                MnCl2 = dict(a=3.686, c=17.470, anti=False),
                NiCl2 = dict(a=3.543, c=17.335, anti=False),
                NiBr2 = dict(a=3.708, c=18.300, anti=False),
                NiI2 = dict(a=3.892, c=19.634, anti=False),
                ZnBr2 = dict(a=3.92, c=18.73, anti=False),
                ZnI2 = dict(a=4.25, c=21.5, anti=False),
                # Cs2O = dict(a=4.256, c=18.99, anti=True), #TODO: Find more anti structures and implement it in parameter estimation
            ),
            Perovskite = dict(
                KNbO3 = dict(a=4.007),
                KTaO3 = dict(a=3.9885),
                KIO3 = dict(a=4.410),
                NaNbO3 = dict(a=3.915),
                NaWO3 = dict(a=3.8622),
                LaCoO3 = dict(a=3.824),
                LaCrO3 = dict(a=3.874),
                LaFeO3 = dict(a=3.920),
                LaGaO3 = dict(a=3.875),
                LaVO3 = dict(a=3.99),
                SrTiO3 = dict(a=3.9051),
                SrZrO3 = dict(a=4.101),
                SrHfO3 = dict(a=4.069),
                SrSnO3 = dict(a=4.0334),
                CsCaF3 = dict(a=4.522),
                CsCdBr3 = dict(a=5.33),
                CsCdCl3 = dict(a=5.20),
                CsHgBr3 = dict(a=5.77),
                CsHgCl3 = dict(a=5.44),
            ),
            ReO3 = dict(
                ReO3 = dict(a=3.734),
                UO3 = dict(a=4.156),
                MoF3 = dict(a=3.8985),
                NbF3 = dict(a=3.903),
                TaF3 = dict(a=3.9012),
                NCu3 = dict(a=3.807),
            ),
            Spinel = dict(
                MgAl2O4 = dict(charges=(2,3), a=8.0800, structure='Normal'),
                CoAl2O4 = dict(charges=(2,3), a=8.1068, structure='Normal'),
                CuCr2S4 = dict(charges=(2,3), a=9.629, structure='Normal'),
                CuCr2Se4 = dict(charges=(2,3), a=10.357, structure='Normal'),
                CuCr2Te4 = dict(charges=(2,3), a=11.051, structure='Normal'),
                MgTi2O4 = dict(charges=(2,3), a=8.474, structure='Normal'),
                GeCo2O4 = dict(charges=(2,4), a=8.318, structure='Normal'),
                GeFe2O4 = dict(charges=(2,4), a=8.411, structure='Normal'),
                FeFe2O4 = dict(charges=(2,3), a=8.397, structure='Normal'),
                # MgFe2O4 = dict(charges=(2,3), a=8.389, structure='Inverse'),
                # NiFe2O4 = dict(charges=(2,3), a=8.3532, structure='Inverse'),
                # MgIn2O4 = dict(charges=(2,3), a=8.81, structure='Inverse'),
                # MgIn2S4 = dict(charges=(2,3), a=10.708, structure='Inverse'),
                # Mg2TiO4 = dict(charges=(2,4), a=8.44, structure='Inverse'),
                # Zn2SnO4 = dict(charges=(2,4), a=8.70, structure='Inverse'),
                # Zn2TiO4 = dict(charges=(2,4), a=8.467, structure='Inverse'),
                # LiAlTiO4 = dict(charges=(1,3,4), a=8.34, structure='Li in tet'),
                # LiMnTiO4 = dict(charges=(1,3,4), a=8.30, structure='Li in tet'),
                # LiZnSbO4 = dict(charges=(1,2,5), a=8.55, structure='Li in tet'),
                # LiCoSbO4 = dict(charges=(1,2,5), a=8.56, structure='Li in tet'),
            ),
            K2NiF4 = dict(
                K2NiF4 = dict(a=4.006, c=13.076, z_M_ion=0.352, z_anion=0.151),
                K2CuF4 = dict(a=4.155, c=12.74, z_M_ion=0.356, z_anion=0.153),
                Ba2SnO4 = dict(a=4.140, c=13.295, z_M_ion=0.355, z_anion=0.155),
                Ba2PbO4 = dict(a=4.305, c=13.273, z_M_ion=0.355, z_anion=0.155),
                Sr2SnO4 = dict(a=4.037, c=12.53, z_M_ion=0.353, z_anion=0.153),
                Sr2TiO4 = dict(a=3.884, c=12.60, z_M_ion=0.355, z_anion=0.152),
                La2NiO4 = dict(a=3.855, c=12.652, z_M_ion=0.360, z_anion=0.170),
                K2MgF4 = dict(a=3.955, c=13.706, z_M_ion=0.35, z_anion=0.15),
            ),
        )
        self.structure_types = list(self.parameter_lookup.keys())
        self.basic_structure_types = ['SC', 'FCC', 'BCC', 'SH', 'HCP', 'DIA']
        self.conversion_points = None
        self.conversion_fits = None
        return None
    
    def calculate_cell_parameter_approximations(self):
        all_point_dict = dict()
        all_fit_dict = dict()
        for structure_type in tqdm(self.structure_types, desc='Calculating cell parameter approximations'):
            structure_dict = self.parameter_lookup[structure_type]
            cell_params = list(structure_dict.items())[0][1].keys()
            point_dict = dict()
            fit_dict = dict()
            for cell_param in cell_params:
                if cell_param not in ['a', 'c']:
                    continue
                param_values = []
                atom_radii_sum = []
                for symbols, params in structure_dict.items():
                    if 'NH4' in symbols:
                        continue
                    group_multiplier = 1
                    radii_sum_list = []
                    symbols = symbols.split('_')[-1]
                    symbols = re.sub('H[1-9]+', '', symbols)
                    param_values.append(params[cell_param])
                    for atom_group in re.split('[()]', symbols)[::-1]:
                        atom_group *= group_multiplier
                        if atom_group.isdigit():
                            group_multiplier = int(atom_group)
                            continue
                        else:
                            group_multiplier = 1
                        atom_list = re.findall('[A-Z][^A-Z]*', atom_group)
                        for atom in atom_list:
                            elm_list = re.findall('(\d+|[A-Za-z]+)', atom)
                            for elm in elm_list:
                                if elm.isdigit():
                                    radii_sum_list[-1] *= int(elm)
                                else:
                                    radii_sum_list.append(self.get_radius(elm))
                    atom_radii_sum.append(np.sum(radii_sum_list))
                point_dict[cell_param] = [atom_radii_sum, param_values]
                fit_dict[cell_param] = np.polyfit(atom_radii_sum, param_values, 1)
            all_point_dict[structure_type] = point_dict
            all_fit_dict[structure_type] = fit_dict
        self.conversion_points = all_point_dict                  
        self.conversion_fits = all_fit_dict
        return None
    
    def visualize_cell_parameter_fits(self, structure_type):
        params = list(self.conversion_fits[structure_type].keys())
        n_params = len(params)
        
        fig, ax = plt.subplots(ncols=n_params, figsize=(6*n_params,6))
        if n_params == 1:
            ax = [ax]
        
        data = np.array(self.conversion_points[structure_type][params[0]])
        fit_params = self.conversion_fits[structure_type][params[0]]
        x = np.arange(np.amin(data[0])-5, np.amax(data[0])+5, 1)
        sns.scatterplot(x=data[0], y=data[1], palette='colorblind', ax=ax[0])
        sns.lineplot(x=x, y=x*fit_params[0]+fit_params[1], color='k', linewidth=2, ax=ax[0])
        ax[0].annotate(f'y = {fit_params[0]:.1e} * x + {fit_params[1]:.2f}', xy=(0.05,0.95), xycoords='axes fraction', fontsize=10)
        ax[0].set_xlabel('Sum of atomic radii', fontsize=12, fontweight='bold')
        ax[0].set_ylabel(f'Cell length [Å]', fontsize=12, fontweight='bold')
        ax[0].set_title(f'Cellparameter {params[0]}', fontsize=14, fontweight='bold')
        ax[0].yaxis.tick_left()
        ax[0].xaxis.tick_bottom()
        
        if n_params == 2:
            data = np.array(self.conversion_points[structure_type][params[1]])
            fit_params = self.conversion_fits[structure_type][params[1]]
            x = np.arange(np.amin(data[0])-5, np.amax(data[0])+5, 1)
            sns.scatterplot(x=data[0], y=data[1], palette='colorblind', ax=ax[1])
            sns.lineplot(x=x, y=x*fit_params[0]+fit_params[1], color='k', linewidth=2, ax=ax[1])
            ax[1].set_xlabel('Sum of atomic radii [pm]', fontsize=12, fontweight='bold')
            ax[1].set_ylabel(f'Cell length [Å]', fontsize=12, fontweight='bold')
            ax[1].set_title(f'Cellparameter {params[1]}', fontsize=14, fontweight='bold')
            ax[1].annotate(f'y = {fit_params[0]:.1e} * x + {fit_params[1]:.2f}', xy=(0.05,0.95), xycoords='axes fraction', fontsize=10)
            ax[1].yaxis.tick_right()
            ax[1].xaxis.tick_bottom()
            ax[1].yaxis.set_label_position('right')
        
        plt.suptitle(f'{structure_type}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return None
    
    def get_radius(self, atom):
        try:
            radius = element(atom).metallic_radius
            if radius == None:
                radius = element(atom).atomic_radius
        except:
            print(atom)
            raise Exception
        return radius
    
    def estimate_cell_parameters(self, crystal_type, compound, hea_mean_radius=None):
        atom_list = [re.sub('[1-9]*|H[1-9]+', '', atom) for atom in re.findall('[A-Z][^A-Z]*', compound)]
        if crystal_type in ['NaCl', 'ZincBlende', 'CsCl']:
            atom_occurence = [1, 1]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params = self.conversion_fits[crystal_type]['a']
            cellparams = dict(
                a=radii_sum * fit_params[0] + fit_params[1],
            )
        elif crystal_type == 'Fluorite':
            atom_occurence = [1, 2]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params = self.conversion_fits[crystal_type]['a']
            cellparams = dict(
                a=radii_sum * fit_params[0] + fit_params[1],
            )
        elif crystal_type == 'AntiFluorite':
            atom_occurence = [2, 1]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params = self.conversion_fits[crystal_type]['a']
            cellparams = dict(
                a=radii_sum * fit_params[0] + fit_params[1],
            )
        elif crystal_type == 'Wurtzite':
            atom_occurence = [1, 1]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params_a = self.conversion_fits[crystal_type]['a']
            fit_params_c = self.conversion_fits[crystal_type]['c']
            cellparams = dict(
                a=radii_sum * fit_params_a[0] + fit_params_a[1],
                c=radii_sum * fit_params_c[0] + fit_params_c[1],
                w=0.375,
            )
        elif crystal_type == 'NiAs':
            atom_occurence = [1, 1]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params_a = self.conversion_fits[crystal_type]['a']
            fit_params_c = self.conversion_fits[crystal_type]['c']
            cellparams = dict(
                a=radii_sum * fit_params_a[0] + fit_params_a[1],
                c=radii_sum * fit_params_c[0] + fit_params_c[1],
            )
        elif crystal_type == 'Rutile':
            atom_occurence = [1, 2]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params_a = self.conversion_fits[crystal_type]['a']
            fit_params_c = self.conversion_fits[crystal_type]['c']
            cellparams = dict(
                a=radii_sum * fit_params_a[0] + fit_params_a[1],
                c=radii_sum * fit_params_c[0] + fit_params_c[1],
                x=0.3,
            )
        elif crystal_type == 'CdI2':
            atom_occurence = [1, 2]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params_a = self.conversion_fits[crystal_type]['a']
            fit_params_c = self.conversion_fits[crystal_type]['c']
            cellparams = dict(
                a=radii_sum * fit_params_a[0] + fit_params_a[1],
                c=radii_sum * fit_params_c[0] + fit_params_c[1],
            )
        elif crystal_type == 'CdCl2':
            atom_occurence = [1, 2]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params_a = self.conversion_fits[crystal_type]['a']
            fit_params_c = self.conversion_fits[crystal_type]['c']
            cellparams = dict(
                a=radii_sum * fit_params_a[0] + fit_params_a[1],
                c=radii_sum * fit_params_c[0] + fit_params_c[1],
                anti=False,
            )
        elif crystal_type == 'Perovskite':
            atom_occurence = [1, 1, 3]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params = self.conversion_fits[crystal_type]['a']
            cellparams = dict(
                a=radii_sum * fit_params[0] + fit_params[1],
            )
        elif crystal_type == 'ReO3':
            atom_occurence = [1, 3]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params = self.conversion_fits[crystal_type]['a']
            cellparams = dict(
                a=radii_sum * fit_params[0] + fit_params[1],
            )
        elif crystal_type == 'Spinel':
            atom_occurence = [1, 2, 4]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params = self.conversion_fits[crystal_type]['a']
            cellparams = dict(
                a=radii_sum * fit_params[0] + fit_params[1],
                structure='Normal',
                charges=(2,3),
            )
        elif crystal_type == 'K2NiF4':
            atom_occurence = [2, 1, 4]
            if not hea_mean_radius:
                atom_radii = [self.get_radius(atom) for atom in atom_list]
            else:
                atom_radii = [hea_mean_radius for i in atom_occurence]
            radii_sum = np.sum(np.multiply(atom_radii, atom_occurence))
            compound = ''.join([atom + str(occurence) if occurence != 1 else atom for atom, occurence in zip(atom_list, atom_occurence)])
            fit_params_a = self.conversion_fits[crystal_type]['a']
            fit_params_c = self.conversion_fits[crystal_type]['c']
            cellparams = dict(
                a=radii_sum * fit_params_a[0] + fit_params_a[1],
                c=radii_sum * fit_params_c[0] + fit_params_c[1],
                z_M_ion=0.355, 
                z_anion=0.155,
            )
        else:
            raise NameError(
                f'''"{crystal_type}" is not a valid crystal type!
            Valid crystal types are:
            [SC, FCC, BCC, SH, HCP, DIA, NaCl, 
            ZincBlende, Fluorite, AntiFluorite, 
            Wurtzite, NiAs, CsCl, Rutile, CdI2, 
            CdCl2, Perovskite, ReO3, Spinel, K2NiF4]
                '''
                )
        return compound, cellparams
    
    def infer_cell_parameters(self, crystal_type:str, atoms:Union[list, str], hea_mean_radius:bool=None, verbose:bool=False) -> Tuple[str, dict]:
        if isinstance(atoms, list):
            compound = ''.join(atoms)
        elif isinstance(atoms, str):
            compound = atoms
        if crystal_type == 'SC':
            if hea_mean_radius:
                a = 2. * hea_mean_radius / 100.
            else:
                atom_info = element(compound)
                if atom_info.lattice_structure == 'SC':
                    a=atom_info.lattice_constant
                else:
                    r = atom_info.metallic_radius_c12 / 100.
                    a = 2. * r
            cellparams = dict(a=a)
        elif crystal_type == 'FCC':
            if hea_mean_radius:
                a = 2. * np.sqrt(2.) * hea_mean_radius / 100.
            else:
                atom_info = element(compound)
                if atom_info.lattice_structure == 'FCC':
                    a=atom_info.lattice_constant
                else:
                    r = atom_info.metallic_radius_c12 / 100.
                    a = 2. * np.sqrt(2.) * r
            cellparams = dict(a=a)
        elif crystal_type == 'BCC':
            if hea_mean_radius:
                a = (4. / np.sqrt(3.)) * hea_mean_radius / 100.
            else:
                atom_info = element(compound)
                if atom_info.lattice_structure == 'BCC':
                    a = atom_info.lattice_constant
                else:
                    r = atom_info.metallic_radius_c12 / 100.
                    a = (4. / np.sqrt(3.)) * r
            cellparams = dict(a=a)
        elif crystal_type == 'SH':
            if hea_mean_radius:
                a = 2. * hea_mean_radius / 100.
                c = 1. * a
            else:
                atom_info = element(compound)
                if atom_info.lattice_structure == 'HEX':
                    a = atom_info.lattice_constant
                    c = 1. * a 
                else:
                    r = atom_info.metallic_radius_c12 / 100.
                    a = 2. * r
                    c = 1. * a
            cellparams = dict(a=a, c=c)
        elif crystal_type == 'HCP':
            if hea_mean_radius:
                a = 2. * hea_mean_radius / 100.
                c = np.sqrt(8. / 3.) * a
            else:
                atom_info = element(compound)
                if atom_info.lattice_structure == 'HEX':
                    a = atom_info.lattice_constant
                    c = np.sqrt(8. / 3.) * a 
                else:
                    r = atom_info.metallic_radius_c12 / 100.
                    a = 2. * r
                    c = np.sqrt(8. / 3.) * a
            cellparams = dict(a=a, c=c)
        elif crystal_type == 'DIA':
            if hea_mean_radius:
                a = (8. * hea_mean_radius / 100.) / np.sqrt(3.)
            else:
                atom_info = element(compound)
                if atom_info.lattice_structure == 'DIA':
                    a = atom_info.lattice_constant
                else:
                    r = atom_info.metallic_radius_c12 / 100.
                    a = (8. * r) / np.sqrt(3.)
            cellparams = dict(a=a)
        else:
            try:
                table_values = self.parameter_lookup[crystal_type]
            except KeyError as err:
                raise KeyError(f"'{crystal_type}' does not match the keys in the lookup table. Available keys are: {list(self.parameter_lookup.keys())}") from err
            try:
                cellparams = table_values[compound]
            except KeyError as err:
                dict_keys = [key for key in table_values if compound in key]
                if len(dict_keys) == 1:
                    compound_key = dict_keys[0]
                    cellparams = table_values[compound_key]
                    compound = compound_key
                elif len(dict_keys) > 1:
                    print(f"Multiple compounds matching '{compound}'.\n\t{dict_keys}")
                    key_index = int(input('Input the index (0 indexing) of the compound to use:'))
                    compound_key = dict_keys[key_index]
                    cellparams = table_values[compound_key]
                    compound = compound_key
                else:
                    if verbose:
                        print(f'No match found for {atoms}.\nEstimating cell parameters from table values.')
                    compound, cellparams = self.estimate_cell_parameters(crystal_type, compound, hea_mean_radius=hea_mean_radius)
        cellparams.update(alpha=90., gamma=120.)
        return compound, cellparams
    
    def create_cell(
        self, 
        crystal_type:str, 
        atoms:Union[list, str], 
        cellparams:Union[dict, str]='infer', 
        size:Union[list, np.ndarray]=[1,1,1], 
        hea_mean_radius:bool=None,
        return_name:bool=False, 
        verbose:bool=False
    ) -> Atoms:
        if cellparams == 'infer':
            compound, cellparams = self.infer_cell_parameters(crystal_type, atoms, hea_mean_radius=hea_mean_radius, verbose=verbose)
        compound_name = compound
        compound = re.sub('H[1-9]+|H[)]+|[()]|[1-9]*', '', compound.split('_')[-1])
        if crystal_type == 'SC':
            cell = crystal(
                compound, 
                basis=[
                    (0,0,0),
                ], 
                spacegroup=221, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'FCC':
            cell = crystal(
                compound, 
                basis=[
                    (0,0,0),
                ], 
                spacegroup=225, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'BCC':
            cell = crystal(
                compound, 
                basis=[
                    (0,0,0),
                ], 
                spacegroup=229, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'SH':
            cell = crystal(
                compound, 
                basis=[
                    (0,0,0),
                ], 
                spacegroup=191, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['gamma'],
                ], 
                size=size
            )
        elif crystal_type == 'HCP':
            cell = crystal(
                compound, 
                basis=[
                    (1./3., 2./3., 3./4.),
                ], 
                spacegroup=194, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['gamma'],
                ], 
                size=size
            )
        elif crystal_type == 'DIA':
            cell = crystal(
                compound, 
                basis=[
                    (0,0,0),
                ], 
                spacegroup=227, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'NaCl':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (0.5, 0.5, 0.5),
                ], 
                spacegroup=225, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'ZincBlende':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (0.25, 0.25, 0.25),
                ], 
                spacegroup=216,
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'Fluorite':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (0.25, 0.25, 0.25), 
                    # (-0.25, -0.25, -0.25),
                ], 
                spacegroup=225,
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'AntiFluorite':
            cell = crystal(
                compound, 
                basis=[
                    (0.25, 0.25, 0.25), 
                    # (-0.25, -0.25, -0.25), 
                    (0, 0, 0),
                ],
                spacegroup=225,
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'Wurtzite':
            cell = crystal(
                compound, 
                basis=[
                    (1/3, 2/3, 0), 
                    (1/3, 2/3, cellparams['w']), 
                ], 
                spacegroup=186, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['gamma'],
                ], 
                size=size
            )
        elif crystal_type == 'NiAs':
            cell = crystal(
                compound, 
                basis=[
                    (0.0000006667, 0.0000003333, 0.), 
                    (0.3333346667, 0.6666673333, 0.25), 
                ], 
                spacegroup=194, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['gamma'],
                ], 
                size=size
            )
        elif crystal_type == 'CsCl':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (0.5, 0.5, 0.5),
                ], 
                spacegroup=221,
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'Rutile':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (cellparams['x'], cellparams['x'], 0),
                ], 
                spacegroup=136, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'CdI2':
            cell = crystal(
                compound, 
                basis=[
                    (0., 0., 0.), 
                    (1/3, 2/3, 0.24),
                ], 
                spacegroup=164,
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['gamma'],
                ], 
                size=size
            )
        elif crystal_type == 'CdCl2':
            if cellparams['anti']:
                basis=[
                    # (0., 0., 0.26),
                    (0., 0., 0.26),
                    (0., 0., 0.),  
                ]
            else:
                basis=[
                    (0., 0., 0.), 
                    (0., 0., 0.26),
                ]
            cell = crystal(
                compound, 
                basis=basis, 
                spacegroup=166, 
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['gamma'],
                ], 
                size=size
            )
        elif crystal_type == 'Perovskite':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (0.5, 0.5, 0.5), 
                    (0.5, 0., 0.5),
                ], 
                spacegroup=221,  
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'ReO3':
            cell = crystal(
                compound, 
                basis=[
                    (0, 0, 0), 
                    (0., 0.5, 0.),
                ], 
                spacegroup=221,  
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        elif crystal_type == 'Spinel':
            if cellparams['structure'] == 'Normal':
                cell = crystal(
                    compound, 
                    basis=[
                        (0.5, 0.5, 0.5), 
                        (0.125, 0.125, 0.125), 
                        (0.3626, 0.3626, 0.3626),
                    ], # Aflow
                    # basis=[
                    #     (0., 0., 0.), 
                    #     (0.625, 0.625, 0.625), 
                    #     (0.3873, 0.3873, 0.3873)
                    # ], # West
                    spacegroup=227,  
                    cellpar=[
                        cellparams['a'], 
                        cellparams['a'], 
                        cellparams['a'], 
                        cellparams['alpha'], 
                        cellparams['alpha'], 
                        cellparams['alpha'],
                    ], 
                    size=size
                )
            elif cellparams['structure'] == 'Inverse':
                raise NotImplementedError
            elif cellparams['structure'] == 'Li in tet':
                raise NotImplementedError
        elif crystal_type == 'K2NiF4':
            cell = crystal(
                compound + '2', 
                basis=[
                    (0, 0, cellparams['z_M_ion']),
                    (0, 0, 0),
                    (1/2, 0, 0),
                    (0, 0, cellparams['z_anion']),
                ], 
                spacegroup=139,   
                cellpar=[
                    cellparams['a'], 
                    cellparams['a'], 
                    cellparams['c'], 
                    cellparams['alpha'], 
                    cellparams['alpha'], 
                    cellparams['alpha'],
                ], 
                size=size
            )
        else:
            raise NameError(
                f'''"{crystal_type}" is not a valid crystal type!
            Valid crystal types are:
            [SC, FCC, BCC, SH, HCP, DIA, NaCl, 
            ZincBlende, Fluorite, AntiFluorite, 
            Wurtzite, NiAs, CsCl, Rutile, CdI2, 
            CdCl2, Perovskite, ReO3, Spinel, K2NiF4]
                '''
                )
        if return_name:
            return cell, compound_name
        else:
            return cell

    def create_hea(
        self, 
        crystal_type:str, 
        atoms:list, 
        stoichiometry: Union[list, np.ndarray, str]='equimolar',
        cellparams:Union[dict, str]='infer', 
        size:Union[list, np.ndarray]=[1,1,1], 
        return_name:bool=False, 
        verbose:bool=False
    ) -> Atoms:
        if stoichiometry == 'equimolar':
            stoichiometry = None
        
        hea_mean_radius = np.average([self.get_radius(atom) for atom in atoms], weights=stoichiometry)
        
        cell = self.create_cell(
            crystal_type=crystal_type,
            atoms=atoms,
            cellparams=cellparams,
            size=size,
            hea_mean_radius=hea_mean_radius,
            return_name=False,
            verbose=verbose,
        )
        
        n_atoms = len(cell)
        
        hea_atoms = np.random.choice(atoms, size=(n_atoms,), p=stoichiometry, replace=True)
        cell.set_chemical_symbols(hea_atoms)
        if return_name:
            compound_name = ''.join(atoms)
            return cell, compound_name
        else:
            return cell
    
    def visualize_cell(self, cell):
        try:
            return view(cell, viewer='ngl')
        except:
            return view(cell, 'x3d')
       
    def create_cif_dataset(
        self, 
        n_species:int, 
        required_atoms:list=None, 
        optional_atoms:list=None,
        crystal_types:Union[list, str]='all', 
        unit_cell_size:Union[list, np.ndarray]=[1,1,1],
        from_table_values:bool=False, 
        strict_number_of_species:bool=False, 
        save_folder:str='./SimDataset/'
    ) -> None:
        #Create save directory if it doesn't exist
        save_directory = Path(save_folder)
        if not save_directory.exists():
            save_directory.mkdir(parents=True)
        if crystal_types == 'all':
            crystal_types = self.structure_types
        if (not from_table_values) and (not self.conversion_fits):
            self.calculate_cell_parameter_approximations()
        print('Simulating CIFs')
        for crystal_type in tqdm(crystal_types, desc='Structure types'):
            print(crystal_type)
            if (n_species == 2) and (crystal_type in ['Perovskite', 'K2NiF4']):
                print('Here')
                continue
            # elif ('O' in required_atoms) and (crystal_type in ['NiAs', 'CsCl', 'CdCl2', 'ZincBlende']):
            #     continue
            if from_table_values:
                for structure in self.parameter_lookup[crystal_type]:
                    atom_list = [re.sub('[1-9]*|H[1-9]*', '', atom) for atom in re.findall('[A-Z][^A-Z]*', structure)]
                    atom_list = np.unique(atom_list)
                    if required_atoms:
                        required_atoms_check = all(req_atom in structure for req_atom in required_atoms)
                    else:
                        required_atoms_check = True
                    if strict_number_of_species:
                        number_of_species_check = (len(atom_list) == n_species)
                    else:
                        number_of_species_check = (len(atom_list) <= n_species)
                    if required_atoms_check and number_of_species_check:
                        cif = self.create_cell(crystal_type, structure, size=unit_cell_size)
                        write(f"{save_folder}{crystal_type}_{structure.split('_')[-1]}.cif", format='cif', images=cif)
            else:
                structure_list = [''.join(atom_tuple) for atom_tuple in zip(optional_atoms, repeat(''.join(required_atoms)))]
                for structure in tqdm(structure_list, desc='Structures', leave=False):
                    if (crystal_type == 'Spinel') and (n_species == 2):
                        atom_list = re.findall('[A-Z][^A-Z]*', structure)
                        structure = ''.join([atom_list[0], atom_list[0], atom_list[1]])
                    cif, structure = self.create_cell(crystal_type, structure, size=unit_cell_size, return_name=True)
                    write(f"{save_folder}{crystal_type}_{structure.split('_')[-1]}.cif", format='cif', images=cif)
        return None