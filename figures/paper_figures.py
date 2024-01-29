##! Imports

from benchmark.dataset_class import InOrgMatDatasets
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from mendeleev import element
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
warnings.simplefilter(action='ignore')

##! Setup
root = '../Dataset/'


# Load datasets
chili_sim = InOrgMatDatasets('CHILI-SIM', root=root)
chili_cod = InOrgMatDatasets('CHILI-COD', root=root)

# Read data splits
chili_sim.load_data_split()
chili_cod.load_data_split()

# Get statistics
stats_sim = chili_sim.get_statistics(return_dataframe=True)
stats_cod = chili_cod.get_statistics(return_dataframe=True)

stats_sim['dataset'] = 'CHILI-SIM'
stats_cod['dataset'] = 'CHILI-COD'

stats_combined = pd.concat([stats_sim, stats_cod], ignore_index=True)

##! Plotting

# Histogram comparing the crystal systems of the two datasets
# Plot
plt.figure(figsize=(6,5))
ax = sns.histplot(data=stats_combined, x='Crystal system (Number)', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=palette, common_norm=False, shrink=0.9)
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
sns.histplot(data=stats_combined, x='Crystal system (Number)', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=palette, common_norm=False, ax=ip, shrink=0.9)
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

# Histogram comparing the number of elements of the two datasets
# Plot
plt.figure(figsize=(6,5))
ax = sns.histplot(data=stats_combined, x='# of elements', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=palette, common_norm=False, shrink=0.9)
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
sns.histplot(data=stats_combined, x='# of elements', hue='dataset', multiple='dodge', discrete=True, stat='percent', palette=palette, common_norm=False, ax=ip, shrink=0.9)
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
plt.savefig('./test_stats_nElements_comparison.pdf', format='pdf', dpi=300)
