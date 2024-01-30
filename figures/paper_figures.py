##! Imports
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from mendeleev import element
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.dataset_class import InOrgMatDatasets

# Mute warnings
warnings.simplefilter(action='ignore')

##! Setup
root = '../Dataset/'

print('Loading datasets...\n\n')
# Load datasets
chili_sim = InOrgMatDatasets(root=root, dataset='CHILI-SIM')
chili_cod = InOrgMatDatasets(root=root, dataset='CHILI-COD')

# Read data splits
try:
    chili_sim.load_data_split()
    chili_cod.load_data_split()
except FileNotFoundError:
    # Create data splits
    chili_sim.create_data_split()
    chili_cod.create_data_split()

    chili_sim.load_data_split()
    chili_cod.load_data_split()

# Get statistics
stats_sim = chili_sim.get_statistics(return_dataframe=True)
stats_cod = chili_cod.get_statistics(return_dataframe=True)

stats_sim['dataset'] = 'CHILI-SIM'
stats_cod['dataset'] = 'CHILI-COD'

stats_combined = pd.concat([stats_sim, stats_cod], ignore_index=True)

####! Plotting
print('Plotting...\n\n')
##! Statistics

# Set palette
palette = sns.color_palette('tab10')
color_dict_set = {'Train': palette[0], 'Validation': palette[1], 'Test': palette[2]}
hue_order_set = ['Train', 'Validation', 'Test']
color_dict_data = {'CHILI-SIM': palette[1], 'CHILI-COD': palette[0]}
hue_order_data = ['CHILI-COD', 'CHILI-SIM']


print('Crystal system comparison...')
# Histogram comparing the crystal systems of the two datasets
# Plot
plt.figure(figsize=(6,5))
ax = sns.histplot(data=stats_combined, x='Crystal system (Number)', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=color_dict_data, hue_order=hue_order_data, common_norm=False, shrink=0.9)
# Legend
new_title = 'Dataset'
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
# Axes
ax.set_xlim(-0.5, 6.5)
ax.set_xticks(ticks=[0,1,2,3,4,5,6], labels=['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal', 'Hexagonal', 'Cubic'], rotation=45)
ax.set_xlabel('')
ax.set_ylabel('Percentage of dataset')
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
ax.set_ylim(0, 62)
# Inset plot of the Hexagonal crystal system
ip = ax.inset_axes([0.5,0.5,0.3,0.4])
sns.histplot(data=stats_combined, x='Crystal system (Number)', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=color_dict_data, hue_order=hue_order_data, common_norm=False, ax=ip, shrink=0.9)
ip.set_xlim(4.5, 5.5)
ip.set_ylim(0, 1)
ip.set_xticks(ticks=[5], labels=['Hexagonal'])
ip.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0.00%', '0.25%', '0.50%', '0.75%', '1.00%'])
ip.set_xlabel('')
ip.set_ylabel('')
ip.set_title('')
ip.set_facecolor('white')
ip.legend_.remove()
ax.indicate_inset_zoom(ip)
# Save
plt.tight_layout()
plt.savefig('./statistics_crystalSystem_comparison.pdf', format='pdf', dpi=300)
print('✓\n\n')

print('Number of elements comparison...')
# Histogram comparing the number of elements of the two datasets
# Plot
plt.figure(figsize=(6,5))
ax = sns.histplot(data=stats_combined, x='# of elements', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=color_dict_data, hue_order=hue_order_data, common_norm=False, shrink=0.9)
# Legend
new_title = 'Dataset'
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
# Axes
ax.set_xlim(0.5, 7.5)
ax.set_xlabel('# of elements')
ax.set_ylabel('Percentage of dataset')
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
ax.set_ylim(0, 102)
# Inset plot of 6 and 7 elements
ip = ax.inset_axes([0.6,0.4,0.3,0.5])
sns.histplot(data=stats_combined, x='# of elements', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=color_dict_data, hue_order=hue_order_data, common_norm=False, ax=ip, shrink=0.9)
ip.set_xlim(5.5, 7.5)
ip.set_ylim(0, 0.6)
ip.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ['0.00%', '0.10%', '0.20%', '0.30%', '0.40%', '0.50%', '0.60%'])
ip.set_xlabel('')
ip.set_ylabel('')
ip.set_title('')
ip.set_facecolor('white')
ip.legend_.remove()
ax.indicate_inset_zoom(ip)
# Save
plt.tight_layout()
plt.savefig('./statistics_nElements_comparison.pdf', format='pdf', dpi=300)
print('✓\n\n')

print('Crystal type in CHILI-SIM...')
# Histogram showing the distribution of crystal types in CHILI-SIM
# Plot
plt.figure(figsize=(6,5))
ax = sns.histplot(data=stats_sim, x='Crystal type', discrete=True, stat='percent', color=palette[1], shrink=0.9)
# Axes
ax.set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11], labels=stats_sim['Crystal type'].unique(),rotation=90)
ax.set_xlabel('')
ax.set_ylabel('Percentage of dataset')
ax.set_yticks([0, 1, 2, 3 ,4, 5, 6, 7, 8, 9], ['0%', '1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%'])
ax.set_ylim(0, 9.2)
# Save
plt.tight_layout()
plt.savefig('./statistics_crystalType_sim.pdf', format='pdf', dpi=300)
print('✓\n\n')

print('NP size comparison...')
# Histogram comparing the distribution of NP sizes in the two datasets
# Plot
plt.figure(figsize=(6,5))
ax = sns.histplot(data=stats_combined, x='NP size (Å)', hue='dataset', multiple='layer', discrete=False, stat='density', palette=color_dict_data, hue_order=hue_order_data, common_norm=False, shrink=1, binwidth=0.1, binrange=(0,60), element='step')
# Legend
new_title = 'Dataset'
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
# Axes
ax.set_xlim(0, 60)
ax.set_xlabel('Nanoparticle size (Å)')
ax.set_ylabel('Density')
# Save
plt.tight_layout()
plt.savefig('./statistics_NPsize_comparison.pdf', format='pdf', dpi=300)
print('✓\n\n')

##! Periodic table figure
print('Periodic table figure...')

# Elements in CHILI-SIM
elements_sim = []
for i in range(len(stats_sim)):
    elements_sim.append(stats_sim['Elements'].to_numpy()[i][0])
    elements_sim.append(stats_sim['Elements'].to_numpy()[i][1])
elements_sim = np.unique(elements_sim)

# Elements in CHILI-COD
elements_cod = []
for i in range(len(stats_cod)):
    for elm in stats_cod['Elements'].to_numpy()[i]:
        elements_cod.append(elm)
elements_cod = np.unique(elements_cod)

# Non-metals
non_metals = [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53]

# List of all element symbols in the periodic table without lanthanides and actinides
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
            'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
            'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
            'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

# List of all atom numbers in the periodic table without lanthanides and actinides
atom_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 72, 73, 74, 75, 76, 77, 78,
                79, 80, 81, 82, 83, 84, 85, 86,
                87, 88, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118]

# List of all lanthanides
lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
               'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
# List of all atom numbers in the lanthanides
lanthanide_numbers = [57, 58, 59, 60, 61, 62, 63, 64, 65,
                      66, 67, 68, 69, 70, 71]

# List of all actinides
actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
             'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
# List of all atom numbers in the actinides
actinide_numbers = [89, 90, 91, 92, 93, 94, 95, 96, 97,
                    98, 99, 100, 101, 102, 103]

# Plot
# 10 x 18 subplots with no whitespace and no axes
fig, axs = plt.subplots(10, 18, figsize=(18, 10), sharex=True, sharey=True, gridspec_kw=dict(wspace=0, hspace=0), subplot_kw=dict(xticks=[], yticks=[]))

# Fill in all elements in the periodic table without lanthanides and actinides
atom_index = 0
for i in range(10):
    if i == 7:
        break
    for j in range(18):
        if j == 0 and i < 7:
            # Label the periods
            axs[i,j].set_ylabel(f'{i+1}', rotation=0, labelpad=10, fontsize=14, fontweight='bold')
        elif i == 0 and 1 <= j <= 16:
            # Remove the top row of the periodic table
            axs[i,j].axis('off')
            continue
        elif i in [1,2] and 2 <= j <= 11:
            axs[i,j].axis('off')
            continue
        elif i in [5,6] and j == 2:
            # Skip the lanthanides and actinides
            if i == 5:
                axs[i,j].annotate('57-71', (0.95, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14)
            elif i == 6:
                axs[i,j].annotate('89-103', (0.95, 0.5), xycoords='axes fraction', va='center', ha='right', fontsize=14)
            
            for s in axs[i,j].spines:
                axs[i,j].spines[s].set_visible(False)
            # axs[i,j].axis('off')
            continue
        # Write atomic number in upper left corner of subplot
        axs[i,j].annotate(f'{atom_numbers[atom_index]}', (0.05, 0.93), xycoords='axes fraction', va='top', ha='left', fontsize=11)
        # Write atomic symbol in center of subplot
        axs[i,j].annotate(f'{elements[atom_index]}', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=18, fontweight='bold')
        
        # Set the opacity of the background color based on if it is a ligans or not
        if atom_numbers[atom_index] in non_metals:
            color_1 = plt.cm.tab20(1)
            color_2 = plt.cm.tab20(3)
        else:
            color_1 = 'tab:blue'
            color_2 = 'tab:orange'
        
        # Color the background of the subplot
        if atom_numbers[atom_index] in elements_sim and atom_numbers[atom_index] in elements_cod:
            axs[i,j].set_facecolor(color_2)
            # Add a blue box covering half of the subplot
            axs[i,j].add_patch(plt.Rectangle((0, 0), 0.5, 1, color=color_1))
        elif atom_numbers[atom_index] in elements_sim:
            axs[i,j].set_facecolor(color_2)
        elif atom_numbers[atom_index] in elements_cod:
            axs[i,j].set_facecolor(color_1)
        atom_index += 1
# Fill in all lanthanides
for i, (elm, num) in enumerate(zip(lanthanides, lanthanide_numbers)):
    if i == 0:
        # Label the periods
        axs[8,3+i].set_ylabel('6', rotation=0, labelpad=10, fontsize=14, fontweight='bold')
    axs[8,3+i].annotate(f'{num}', (0.05, 0.93), xycoords='axes fraction', va='top', ha='left', fontsize=11)
    axs[8,3+i].annotate(f'{elm}', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=18, fontweight='bold')
    # Color the background of the subplot
    if num in elements_sim and num in elements_cod:
        axs[8,3+i].set_facecolor('tab:orange')
        # Add a blue box covering half of the subplot
        axs[8,3+i].add_patch(plt.Rectangle((0, 0), 0.5, 1, color='tab:blue'))
    elif num in elements_sim:
        axs[8,3+i].set_facecolor('tab:orange')
    elif num in elements_cod:
        axs[8,3+i].set_facecolor('tab:blue')
# Fill in all actinides
for i, (elm, num) in enumerate(zip(actinides, actinide_numbers)):
    if i == 0:
        # Label the periods
        axs[9,3+i].set_ylabel('7', rotation=0, labelpad=10, fontsize=14, fontweight='bold')
    axs[9,3+i].annotate(f'{num}', (0.05, 0.93), xycoords='axes fraction', va='top', ha='left', fontsize=11)
    axs[9,3+i].annotate(f'{elm}', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=18, fontweight='bold')
    # Color the background of the subplot
    if num in elements_sim and num in elements_cod:
        axs[9,3+i].set_facecolor('tab:orange')
        # Add a blue box covering half of the subplot
        axs[9,3+i].add_patch(plt.Rectangle((0, 0), 0.5, 1, color='tab:blue'))
    elif num in elements_sim:
        axs[9,3+i].set_facecolor('tab:orange')
    elif num in elements_cod:
        axs[9,3+i].set_facecolor('tab:blue')
    
# Remove axes from all blank subplots
for i in range(18):
    if i == 2:
        for s in axs[7,i].spines:
            axs[7,i].spines[s].set_visible(False)
    else:
        axs[7,i].axis('off')
for i in range(3):
    if i == 2:
        for s in axs[8,i].spines:
            axs[8,i].spines[s].set_visible(False)
        for s in axs[9,i].spines:
            axs[9,i].spines[s].set_visible(False)
    else:
        axs[8,0+i].axis('off')
        axs[9,0+i].axis('off')
    
# Color connection to lanthanides and actinides blue
for i in range(5):
    axs[5+i,2].set_facecolor(plt.cm.tab20(15))

# Add annotations which indicate the groups on the periodic table
axs[0,0].annotate('1', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[1,1].annotate('2', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,2].annotate('3', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,3].annotate('4', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,4].annotate('5', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,5].annotate('6', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,6].annotate('7', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,7].annotate('8', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,8].annotate('9', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,9].annotate('10', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,10].annotate('11', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[3,11].annotate('12', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=14, fontweight='bold')
axs[1,12].annotate('13', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,13].annotate('14', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,14].annotate('15', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,15].annotate('16', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,16].annotate('17', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[0,17].annotate('18', (0.5, 1.1), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')

# Add annotations to the top middle of the plot that show which dataset each color corresponds to
# Show the color for CHILI-COD
axs[1,3].axis('on')
axs[1,3].set_facecolor('tab:blue')
axs[1,4].axis('on')
axs[1,4].set_facecolor(plt.cm.tab20(1))
axs[1,3].annotate('CHILI-COD', (1, 1.25), xycoords='axes fraction', va='center', ha='center', fontsize=16, fontweight='bold')
axs[1,3].annotate('Metal', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,4].annotate('Non-\nmetal', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')

# Show the color for both datasets
axs[1,6].axis('on')
axs[1,6].set_facecolor('tab:orange')
axs[1,6].add_patch(plt.Rectangle((0, 0), 0.5, 1, color='tab:blue'))
axs[1,7].axis('on')
axs[1,7].set_facecolor(plt.cm.tab20(3))
axs[1,7].add_patch(plt.Rectangle((0, 0), 0.5, 1, color=plt.cm.tab20(1)))
axs[1,6].annotate('Both', (1, 1.25), xycoords='axes fraction', va='center', ha='center', fontsize=16, fontweight='bold')
axs[1,6].annotate('Metal', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,7].annotate('Non-\nmetal', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')

# Show the color for CHILI-SIM
axs[1,9].axis('on')
axs[1,9].set_facecolor('tab:orange')
axs[1,10].axis('on')
axs[1,10].set_facecolor(plt.cm.tab20(3))
axs[1,9].annotate('CHILI-SIM', (1, 1.25), xycoords='axes fraction', va='center', ha='center', fontsize=16, fontweight='bold')
axs[1,9].annotate('Metal', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')
axs[1,10].annotate('Non-\nmetal', (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=12, fontweight='bold')

# Save
fig.tight_layout()
fig.savefig('./periodicTable.pdf', format='pdf', dpi=300)
print('✓\n\n')