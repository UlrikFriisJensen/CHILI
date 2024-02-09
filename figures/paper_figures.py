# %% Imports
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.dataset_class import CHILI

# Mute warnings
warnings.simplefilter(action="ignore")

# %% Setup
root = "../Dataset/"

print("Loading datasets...\n\n")
# Load datasets
chili_3k = CHILI(root=root, dataset="CHILI-3K")
chili_100k = CHILI(root=root, dataset="CHILI-100K")

# Read data splits
try:
    chili_3k.load_data_split()
    chili_100k.load_data_split()
except FileNotFoundError:
    # Create data splits
    chili_3k.create_data_split()
    chili_100k.create_data_split()

    chili_3k.load_data_split()
    chili_100k.load_data_split()

# Get statistics
stats_3k = chili_3k.get_statistics(return_dataframe=True)
stats_100k = chili_100k.get_statistics(return_dataframe=True)

stats_3k["dataset"] = "CHILI-3K"
stats_100k["dataset"] = "CHILI-100K"

stats_combined = pd.concat([stats_3k, stats_100k], ignore_index=True)

# %% Plotting
print("Plotting:\n")

# Set font size
plt.rcParams.update({"font.size": 13})

# %% Statistics

# Set palette
palette = sns.color_palette("tab10")
color_dict_set = {"Train": palette[0], "Validation": palette[1], "Test": palette[2]}
hue_order_set = ["Train", "Validation", "Test"]
color_dict_data = {"CHILI-3K": palette[0], "CHILI-100K": palette[1]}
hue_order_data = ["CHILI-3K", "CHILI-100K"]

# %% Histogram comparing the crystal systems of the two datasets
print("Crystal system comparison...")
# Plot
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=stats_combined,
    x="Crystal system (Number)",
    hue="dataset",
    multiple="dodge",
    discrete=True,
    stat="percent",
    palette=color_dict_data,
    hue_order=hue_order_data,
    common_norm=False,
    shrink=0.9,
)
# Legend
new_title = "Dataset"
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
# Axes
ax.set_xlim(-0.5, 6.5)
ax.set_xticks(
    ticks=[0, 2, 4, 6], labels=["Triclinic", "Orthorhombic", "Trigonal", "Cubic"]
)  # , rotation=45)
ax.set_xticks(
    ticks=[1, 3, 5], labels=["Monoclinic", "Tetragonal", "Hexagonal"], minor=True
)
ax.tick_params(axis="x", which="minor", length=20, width=1)
ax.set_xlabel("")
ax.set_ylabel("Percentage of dataset")
ax.set_yticks(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"],
)
ax.set_ylim(0, 62)
# Place a) in the top left corner
ax.annotate(
    "a)",
    (0.04, 0.96),
    xycoords="axes fraction",
    va="top",
    ha="left",
    fontsize=18,
    fontweight="bold",
)
# Save
plt.tight_layout()
plt.savefig(
    "./statistics_crystalSystem_comparison.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
)
print("✓\n")

# %% Histogram comparing the number of elements of the two datasets
print("Number of elements comparison...")
# Plot
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=stats_combined,
    x="# of elements",
    hue="dataset",
    multiple="dodge",
    discrete=True,
    stat="percent",
    palette=color_dict_data,
    hue_order=hue_order_data,
    common_norm=False,
    shrink=0.9,
)
# Legend
new_title = "Dataset"
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
# Axes
ax.set_xlim(0.5, 7.5)
ax.set_xlabel("# of elements")
ax.set_ylabel("Percentage of dataset")
ax.set_yticks(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
)
ax.set_ylim(0, 102)
# Inset plot of 6 and 7 elements
ip = ax.inset_axes([0.6, 0.4, 0.3, 0.5])
sns.histplot(
    data=stats_combined,
    x="# of elements",
    hue="dataset",
    multiple="dodge",
    discrete=True,
    stat="percent",
    palette=color_dict_data,
    hue_order=hue_order_data,
    common_norm=False,
    ax=ip,
    shrink=0.9,
)
ip.set_xlim(5.5, 7.5)
ip.set_ylim(0, 0.6)
ip.set_yticks(
    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ["0.00%", "0.10%", "0.20%", "0.30%", "0.40%", "0.50%", "0.60%"],
)
ip.set_xlabel("")
ip.set_ylabel("")
ip.set_title("")
ip.set_facecolor("white")
ip.legend_.remove()
ax.indicate_inset_zoom(ip)
# Place b) in the top left corner
ax.annotate(
    "b)",
    (0.04, 0.96),
    xycoords="axes fraction",
    va="top",
    ha="left",
    fontsize=18,
    fontweight="bold",
)
# Save
plt.tight_layout()
plt.savefig(
    "./statistics_nElements_comparison.pdf", format="pdf", dpi=300, bbox_inches="tight"
)
print("✓\n")

print("Crystal type in CHILI-3K...")
# Histogram showing the distribution of crystal types in CHILI-3K
# Plot
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=stats_3k,
    x="Crystal type",
    discrete=True,
    stat="percent",
    color=palette[0],
    shrink=0.9,
)
# Axes
ax.set_xticks(
    ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    labels=stats_3k["Crystal type"].unique(),
    rotation=90,
)
ax.set_xlabel("")
ax.set_ylabel("Percentage of dataset")
ax.set_yticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ["0%", "1%", "2%", "3%", "4%", "5%", "6%", "7%", "8%", "9%"],
)
ax.set_ylim(0, 9.2)
# Save
plt.tight_layout()
plt.savefig(
    "./statistics_crystalType_sim.pdf", format="pdf", dpi=300, bbox_inches="tight"
)
print("✓\n")

# %% Histogram comparing the distribution of NP sizes in the two datasets
print("NP size comparison...")
# Plot
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=stats_combined,
    x="NP size (Å)",
    hue="dataset",
    multiple="layer",
    discrete=False,
    stat="density",
    palette=color_dict_data,
    hue_order=hue_order_data,
    common_norm=False,
    shrink=1,
    binwidth=0.1,
    binrange=(0, 60),
    element="step",
)
# Legend
new_title = "Dataset"
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
# Axes
ax.set_xlim(0, 60)
ax.set_xlabel("Nanoparticle size (Å)")
ax.set_ylabel("Density")
# Place c) in the top left corner
ax.annotate(
    "c)",
    (0.04, 0.96),
    xycoords="axes fraction",
    va="top",
    ha="left",
    fontsize=18,
    fontweight="bold",
)
# Save
plt.tight_layout()
plt.savefig(
    "./statistics_NPsize_comparison.pdf", format="pdf", dpi=300, bbox_inches="tight"
)
print("✓\n")

# %% Periodic table figure
print("Periodic table figure...")

# Elements in CHILI-3K
elements_3k = []
for i in range(len(stats_3k)):
    elements_3k.append(stats_3k["Elements"].to_numpy()[i][0])
    elements_3k.append(stats_3k["Elements"].to_numpy()[i][1])
elements_3k = np.unique(elements_3k)

# Elements in CHILI-100K
elements_100k = []
for i in range(len(stats_100k)):
    for elm in stats_100k["Elements"].to_numpy()[i]:
        elements_100k.append(elm)
elements_100k = np.unique(elements_100k)

# Non-metals
non_metals = [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53]

# List of all element symbols in the periodic table without lanthanides and actinides
elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

# List of all atom numbers in the periodic table without lanthanides and actinides
atom_numbers = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
]

# List of all lanthanides
lanthanides = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]
# List of all atom numbers in the lanthanides
lanthanide_numbers = [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]

# List of all actinides
actinides = [
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
]
# List of all atom numbers in the actinides
actinide_numbers = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]

# Plot
# Text sizes
element_size = 26
number_size = 16
period_size = 18


# 10 x 18 subplots with no whitespace and no axes
fig, axs = plt.subplots(
    10,
    18,
    figsize=(18, 10),
    sharex=True,
    sharey=True,
    gridspec_kw=dict(wspace=0, hspace=0),
    subplot_kw=dict(xticks=[], yticks=[]),
)

# Fill in all elements in the periodic table without lanthanides and actinides
atom_index = 0
for i in range(10):
    if i == 7:
        break
    for j in range(18):
        if j == 0 and i < 7:
            # Label the periods
            axs[i, j].annotate(
                f"{i+1}",
                (-0.1, 0.5),
                xycoords="axes fraction",
                va="center",
                ha="right",
                fontsize=period_size,
                fontweight="bold",
            )
        elif i == 0 and 1 <= j <= 16:
            # Remove the top row of the periodic table
            axs[i, j].axis("off")
            continue
        elif i in [1, 2] and 2 <= j <= 11:
            axs[i, j].axis("off")
            continue
        elif i in [5, 6] and j == 2:
            # Skip the lanthanides and actinides
            if i == 5:
                axs[i, j].annotate(
                    "57-71",
                    (0.95, 0.5),
                    xycoords="axes fraction",
                    va="center",
                    ha="right",
                    fontsize=14,
                )
            elif i == 6:
                axs[i, j].annotate(
                    "89-103",
                    (0.95, 0.5),
                    xycoords="axes fraction",
                    va="center",
                    ha="right",
                    fontsize=14,
                )

            for s in axs[i, j].spines:
                axs[i, j].spines[s].set_visible(False)
            # axs[i,j].axis('off')
            continue
        # Write atomic number in upper left corner of subplot
        axs[i, j].annotate(
            f"{atom_numbers[atom_index]}",
            (0.05, 0.93),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=number_size,
        )
        # Write atomic symbol in center of subplot
        axs[i, j].annotate(
            f"{elements[atom_index]}",
            (0.5, 0.35),
            xycoords="axes fraction",
            va="center",
            ha="center",
            fontsize=element_size,
        )

        # Set the opacity of the background color based on if it is a ligans or not
        if atom_numbers[atom_index] in non_metals:
            color_1 = plt.cm.tab20(1)
            color_2 = plt.cm.tab20(3)
        else:
            color_1 = "tab:blue"
            color_2 = "tab:orange"

        # Color the background of the subplot
        if (
            atom_numbers[atom_index] in elements_100k
            and atom_numbers[atom_index] in elements_3k
        ):
            axs[i, j].set_facecolor(color_2)
            # Add a blue box covering half of the subplot
            axs[i, j].add_patch(plt.Rectangle((0, 0), 0.5, 1, color=color_1))
        elif atom_numbers[atom_index] in elements_100k:
            axs[i, j].set_facecolor(color_2)
        elif atom_numbers[atom_index] in elements_3k:
            axs[i, j].set_facecolor(color_1)
        else:
            axs[i, j].set_facecolor(plt.cm.tab20c(19))
        atom_index += 1
# Fill in all lanthanides
for i, (elm, num) in enumerate(zip(lanthanides, lanthanide_numbers)):
    if i == 0:
        # Label the periods
        axs[8, 3 + i].annotate(
            "6",
            (-0.1, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="right",
            fontsize=period_size,
            fontweight="bold",
        )
    axs[8, 3 + i].annotate(
        f"{num}",
        (0.05, 0.93),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=number_size,
    )
    axs[8, 3 + i].annotate(
        f"{elm}",
        (0.5, 0.35),
        xycoords="axes fraction",
        va="center",
        ha="center",
        fontsize=element_size,
    )
    # Color the background of the subplot
    if num in elements_100k and num in elements_3k:
        axs[8, 3 + i].set_facecolor("tab:orange")
        # Add a blue box covering half of the subplot
        axs[8, 3 + i].add_patch(plt.Rectangle((0, 0), 0.5, 1, color="tab:blue"))
    elif num in elements_100k:
        axs[8, 3 + i].set_facecolor("tab:orange")
    elif num in elements_3k:
        axs[8, 3 + i].set_facecolor("tab:blue")
    else:
        axs[8, 3 + i].set_facecolor(plt.cm.tab20c(19))
# Fill in all actinides
for i, (elm, num) in enumerate(zip(actinides, actinide_numbers)):
    if i == 0:
        # Label the periods
        axs[9, 3 + i].annotate(
            "7",
            (-0.1, 0.5),
            xycoords="axes fraction",
            va="center",
            ha="right",
            fontsize=period_size,
            fontweight="bold",
        )
    axs[9, 3 + i].annotate(
        f"{num}",
        (0.05, 0.93),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=number_size,
    )
    axs[9, 3 + i].annotate(
        f"{elm}",
        (0.5, 0.35),
        xycoords="axes fraction",
        va="center",
        ha="center",
        fontsize=element_size,
    )
    # Color the background of the subplot
    if num in elements_100k and num in elements_3k:
        axs[9, 3 + i].set_facecolor("tab:orange")
        # Add a blue box covering half of the subplot
        axs[9, 3 + i].add_patch(plt.Rectangle((0, 0), 0.5, 1, color="tab:blue"))
    elif num in elements_100k:
        axs[9, 3 + i].set_facecolor("tab:orange")
    elif num in elements_3k:
        axs[9, 3 + i].set_facecolor("tab:blue")
    else:
        axs[9, 3 + i].set_facecolor(plt.cm.tab20c(19))

# Remove axes from all blank subplots
for i in range(18):
    if i == 2:
        for s in axs[7, i].spines:
            axs[7, i].spines[s].set_visible(False)
    else:
        axs[7, i].axis("off")
for i in range(3):
    if i == 2:
        for s in axs[8, i].spines:
            axs[8, i].spines[s].set_visible(False)
        for s in axs[9, i].spines:
            axs[9, i].spines[s].set_visible(False)
    else:
        axs[8, 0 + i].axis("off")
        axs[9, 0 + i].axis("off")

# Color connection to lanthanides and actinides blue
for i in range(5):
    axs[5 + i, 2].set_facecolor("darkgrey")  # plt.cm.tab20(15)

# Add annotations which indicate the groups on the periodic table
axs[0, 0].annotate(
    "1",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[1, 1].annotate(
    "2",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 2].annotate(
    "3",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 3].annotate(
    "4",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 4].annotate(
    "5",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 5].annotate(
    "6",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 6].annotate(
    "7",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 7].annotate(
    "8",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 8].annotate(
    "9",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 9].annotate(
    "10",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 10].annotate(
    "11",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[3, 11].annotate(
    "12",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[1, 12].annotate(
    "13",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[1, 13].annotate(
    "14",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[1, 14].annotate(
    "15",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[1, 15].annotate(
    "16",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[1, 16].annotate(
    "17",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)
axs[0, 17].annotate(
    "18",
    (0.5, 1.15),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=period_size,
    fontweight="bold",
)

# Add annotations to the top middle of the plot that show which dataset each color corresponds to
# Show the color for CHILI-3K
axs[1, 4].axis("on")
axs[1, 4].set_facecolor("tab:blue")
axs[1, 5].axis("on")
axs[1, 5].set_facecolor(plt.cm.tab20(1))
axs[1, 4].annotate(
    "CHILI-3K",
    (1, 1.25),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=24,
    fontweight="bold",
)
axs[1, 4].annotate(
    "Metal",
    (0.5, 0.5),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)
axs[1, 5].annotate(
    "Non-\nmetal",
    (0.5, 0.5),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

# Show the color for CHILI-100K
axs[1, 8].axis("on")
axs[1, 8].set_facecolor("tab:orange")
axs[1, 9].axis("on")
axs[1, 9].set_facecolor(plt.cm.tab20(3))
axs[1, 8].annotate(
    "CHILI-100K",
    (1, 1.25),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=24,
    fontweight="bold",
)
axs[1, 8].annotate(
    "Metal",
    (0.5, 0.5),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)
axs[1, 9].annotate(
    "Non-\nmetal",
    (0.5, 0.5),
    xycoords="axes fraction",
    va="center",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

# Save
fig.tight_layout()
fig.savefig("./periodicTable.pdf", format="pdf", dpi=300, bbox_inches="tight")
print("✓\n")

# %% Plots for CHILI-100K subset

# Read data split
try:
    chili_100k.load_data_split(
        split_strategy="stratified",
        stratify_on="Crystal system (Number)",
        stratify_distribution="equal",
    )
except FileNotFoundError:
    # Create data splits
    chili_100k.create_data_split(
        split_strategy="stratified",
        stratify_on="Crystal system (Number)",
        stratify_distribution="equal",
        n_sample_per_class=425,
    )

    chili_100k.load_data_split(
        split_strategy="stratified",
        stratify_on="Crystal system (Number)",
        stratify_distribution="equal",
    )

stats_100k_subset = chili_100k.get_statistics(return_dataframe=True)

# %% Crystal system distribution in CHILI-100K subset
print("Crystal system distribution in CHILI-100K subset...")
# Plot
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=stats_100k_subset,
    x="Crystal system (Number)",
    discrete=True,
    stat="percent",
    common_norm=True,
    shrink=0.9,
    hue="Stratified data split (Crystal system (Number), Equal classes)",
    palette=color_dict_set,
    hue_order=hue_order_set,
    multiple="stack",
)  # , color=palette[1])
# Legend
new_title = "Data split"
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
# Axes
ax.set_xlim(-0.5, 6.5)
ax.set_xticks(
    ticks=[0, 2, 4, 6], labels=["Triclinic", "Orthorhombic", "Trigonal", "Cubic"]
)  # , rotation=45)
ax.set_xticks(
    ticks=[1, 3, 5], labels=["Monoclinic", "Tetragonal", "Hexagonal"], minor=True
)
ax.tick_params(axis="x", which="minor", length=20, width=1)
ax.set_xlabel("")
ax.set_ylabel("Percentage of subset")
ax.set_yticks([0, 5, 10, 15, 20], ["0%", "5%", "10%", "15%", "20%"])
ax.set_ylim(0, 16)
# Save
plt.tight_layout()
plt.savefig(
    "./statistics_crystalSystem_100kSubset.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
)
print("✓\n")

# %% Number of elements in CHILI-100K subset
print("Number of elements in CHILI-100K subset...")
# Plot
plt.figure(figsize=(6, 5))
ax = sns.histplot(
    data=stats_100k_subset,
    x="# of elements",
    hue="Stratified data split (Crystal system (Number), Equal classes)",
    multiple="dodge",
    discrete=True,
    stat="percent",
    palette=color_dict_set,
    hue_order=hue_order_set,
    common_norm=False,
    shrink=0.9,
)
# Legend
new_title = "Dataset"
ax.legend_.set_title(new_title)
sns.move_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
# Axes
ax.set_xlim(0.5, 7.5)
ax.set_xlabel("# of elements")
ax.set_ylabel("Percentage of data split")
ax.set_yticks(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
)
ax.set_ylim(0, 62)
# Inset plot of 6 and 7 elements
ip = ax.inset_axes([0.67, 0.45, 0.3, 0.5])
sns.histplot(
    data=stats_100k_subset,
    x="# of elements",
    hue="Stratified data split (Crystal system (Number), Equal classes)",
    multiple="dodge",
    discrete=True,
    stat="percent",
    palette=color_dict_set,
    hue_order=hue_order_set,
    common_norm=False,
    ax=ip,
    shrink=0.9,
)
ip.set_xlim(5.5, 7.5)
ip.set_ylim(0, 1.0)
ip.set_yticks(
    [0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0%", "0.2%", "0.4%", "0.6%", "0.8%", "1.0%"]
)
ip.set_xlabel("")
ip.set_ylabel("")
ip.set_title("")
ip.set_facecolor("white")
ip.legend_.remove()
ax.indicate_inset_zoom(ip)
# Save
plt.tight_layout()
plt.savefig(
    "./statistics_nElements_100kSubset.pdf", format="pdf", dpi=300, bbox_inches="tight"
)
print("✓\n")
