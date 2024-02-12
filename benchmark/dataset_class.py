import os
import h5py
import numpy as np
import pandas as pd
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset, download_url, extract_zip
from tqdm.auto import tqdm

class CHILI(Dataset):
    def __init__(
        self, root, dataset, transform=None, pre_transform=None, pre_filter=None
    ):
        self.dataset = dataset
        self.root = os.path.join(root, self.dataset)
        # Create root directory if not exits
        if not os.path.exists(self.root):
            os.mkdir(self.root)

        # Train Val Test sets as Subsets
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        # Something is wrong with super, setup manually:
        self.transform = lambda data: data
        self.pre_transform = lambda data: data
        self.pre_filter = lambda data: data

        # Download if data if there are no raw files
        if len(self.raw_file_names) == 0:
            # Make raw folder
            if not os.path.exists(os.path.join(self.root, "raw")):
                os.mkdir(os.path.join(self.root, "raw"))
            # Download
            self.download()
        # Process if processed folder is empty
        if len(self.processed_file_names) == 0:
            # Make processed folder
            if not os.path.exists(os.path.join(self.root, "processed")):
                os.mkdir(os.path.join(self.root, "processed"))
            self.process()

        self._indices = range(self.len())

    @property
    def raw_file_names(self):
        paths = glob(os.path.join(self.raw_dir, "**/*.h5"))
        return paths

    @property
    def processed_file_names(self):
        paths = glob(os.path.join(self.processed_dir, "[!pre]*.pt"))
        return paths

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(
            f"https://sid.erda.dk/share_redirect/h6ktCBGzPF/{self.dataset}.zip",
            self.raw_dir,
        )
        # Extract zip and delete zip
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def crystal_system_to_number(self, crystal_system):
        if crystal_system == "Triclinic":
            return 1
        elif crystal_system == "Monoclinic":
            return 2
        elif crystal_system == "Orthorhombic":
            return 3
        elif crystal_system == "Tetragonal":
            return 4
        elif crystal_system == "Trigonal":
            return 5
        elif crystal_system == "Hexagonal":
            return 6
        elif crystal_system == "Cubic":
            return 7
        else:
            raise ValueError(
                'Crystal system not recognized. Please use either "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal" or "Cubic"'
            )

    def write_to_log(log_file, s):
        try:
            with open(log_file, 'a') as f:
                f.write(s + '\n')
        except Exception as e:
            print(f'Error while writing {s} to file: {log_file}')

    def process(self):

        idx = 0
        process_pbar = tqdm(desc="Processing data...", total=len(self.raw_file_names), leave=False)
        for raw_path in self.raw_file_names:

            # Read data from `raw_path`
            try:
                with h5py.File(raw_path, "r") as h5f:

                    # Unit cell
                    unit_cell_node_feat = torch.tensor(h5f["UnitCellGraph"]["NodeFeatures"][:], dtype=torch.float32)
                    unit_cell_edge_index = torch.tensor(h5f["UnitCellGraph"]["EdgeDirections"][:], dtype=torch.long)
                    unit_cell_edge_attr = torch.tensor(h5f["UnitCellGraph"]["EdgeFeatures"][:], dtype=torch.float32)
                    unit_cell_pos_abs = torch.tensor(h5f["UnitCellGraph"]["AbsoluteCoordinates"][:], dtype=torch.float32)
                    unit_cell_pos_frac = torch.tensor(h5f["UnitCellGraph"]["FractionalCoordinates"][:], dtype=torch.float32)

                    # Cell parameters
                    cell_params = torch.tensor(h5f["GlobalLabels"]["CellParameters"][:], dtype=torch.float32)

                    # Atomic species
                    atomic_species = torch.tensor(h5f["GlobalLabels"]["ElementsPresent"][:], dtype=torch.float32)

                    # Crystal type
                    crystal_type = h5f["GlobalLabels"]["CrystalType"][()].decode()

                    # Space group
                    space_group_symbol = h5f["GlobalLabels"]["SpaceGroupSymbol"][()].decode()
                    space_group_number = h5f["GlobalLabels"]["SpaceGroupNumber"][()]

                    # Crystal system
                    crystal_system = h5f["GlobalLabels"]["CrystalSystem"][()].decode()
                    crystal_system_number = self.crystal_system_to_number(crystal_system)

                    # Loop through all particle sizes
                    for key in h5f["DiscreteParticleGraphs"].keys():
                        node_feat = torch.tensor(h5f["DiscreteParticleGraphs"][key]["NodeFeatures"][:], dtype=torch.float32)
                        edge_index = torch.tensor(h5f["DiscreteParticleGraphs"][key]["EdgeDirections"][:],dtype=torch.long)
                        edge_attr = torch.tensor(h5f["DiscreteParticleGraphs"][key]["EdgeFeatures"][:], dtype=torch.float32)

                        # Create graph data object
                        data = Data(
                            data_id = raw_path.split(".")[0].split("/")[-1],
                            x = node_feat,
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos_abs = torch.tensor(h5f["DiscreteParticleGraphs"][key]["AbsoluteCoordinates"][:], dtype=torch.float32),
                            pos_frac=torch.tensor(h5f["DiscreteParticleGraphs"][key]["FractionalCoordinates"][:], dtype=torch.float32),

                            y=dict(
                                crystal_type=crystal_type,
                                space_group_symbol=space_group_symbol,
                                space_group_number=space_group_number,
                                crystal_system=crystal_system,
                                crystal_system_number=crystal_system_number,
                                atomic_species=atomic_species,
                                n_atomic_species=len(atomic_species),
                                np_size=h5f["DiscreteParticleGraphs"][key]["NP size (Å)"][()],
                                n_atoms=node_feat.shape[0],
                                n_bonds=edge_index.shape[1],

                                cell_params=cell_params,
                                unit_cell_x=unit_cell_node_feat,
                                unit_cell_edge_index=unit_cell_edge_index,
                                unit_cell_edge_attr=unit_cell_edge_attr,
                                unit_cell_pos_abs=unit_cell_pos_abs,
                                unit_cell_pos_frac=unit_cell_pos_frac,
                                unit_cell_n_atoms=unit_cell_node_feat.shape[0],
                                unit_cell_n_bonds=unit_cell_edge_index.shape[1],

                                # Scattering data
                                nd=torch.tensor(h5f["ScatteringData"][key]["ND"][:], dtype=torch.float32),
                                xrd=torch.tensor(h5f["ScatteringData"][key]["XRD"][:], dtype=torch.float32),
                                nPDF=torch.tensor(h5f["ScatteringData"][key]["nPDF"][:], dtype=torch.float32),
                                xPDF=torch.tensor(h5f["ScatteringData"][key]["xPDF"][:], dtype=torch.float32),
                                sans=torch.tensor(h5f["ScatteringData"][key]["SANS"][:], dtype=torch.float32),
                                saxs=torch.tensor(h5f["ScatteringData"][key]["SAXS"][:], dtype=torch.float32),
                            ),
                        )

                        # Apply filters
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue

                        # Apply transforms
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        # Save to `self.processed_dir`.
                        torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))

                        # Update index
                        idx += 1

                # Update process pbar
                process_pbar.update(1)

            except Exception as e:
                write_to_log('processing_error_log.out', raw_path + '\n' + e + '\n')

        process_pbar.close()

    def len(self, split=None):
        if split is None:
            length = len(self.processed_file_names)
        elif split.lower() == "train":
            length = len(self.train_set)
        elif split.lower() in ["validation", "val"]:
            length = len(self.validation_set)
        elif split.lower() == "test":
            length = len(self.test_set)
        else:
            raise ValueError(
                'Split not recognized. Please use either "train", "validation" or "test"'
            )
        return length

    def get(self, idx, split=None):
        if split is None:
            data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        elif split.lower() == "train":
            data = self.train_set[idx]
        elif split.lower() in ["validation", "val"]:
            data = self.validation_set[idx]
        elif split.lower() == "test":
            data = self.test_set[idx]
        else:
            raise ValueError(
                'Split not recognized. Please use either "train", "validation" or "test"'
            )
        return data

    def create_data_split(
        self,
        test_size=0.1,
        validation_size=None,
        split_strategy="random",
        stratify_on="Space group (Number)",
        stratify_distribution="match",
        n_sample_per_class="max",
        random_state=42,
        return_idx=False,
    ):
        """
        Split the dataset into train, validation and test sets. The indices of the split are saved to csv files in the processed directory.
        """

        if validation_size is None:
            validation_size = test_size

        df_stats = self.get_statistics(return_dataframe=True)

        if split_strategy == "random":

            # Split data into train, validation and test sets
            train_idx, test_idx = train_test_split(
                np.arange(self.len()),
                test_size = test_size,
                random_state = random_state
            )
            train_idx, validation_idx = train_test_split(
                train_idx,
                test_size = validation_size / (1 - test_size),
                random_state = random_state
            )

            # Save indices to csv files
            np.savetxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_train.csv"),
                train_idx,
                delimiter=",",
                fmt="%i",
            )
            np.savetxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_validation.csv"),
                validation_idx,
                delimiter=",",
                fmt="%i",
            )
            np.savetxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_test.csv"),
                test_idx,
                delimiter=",",
                fmt="%i",
            )

            # Update statistics dataframe
            df_stats[f"{split_strategy.capitalize()} data split"] = ""
            df_stats[f"{split_strategy.capitalize()} data split"].loc[train_idx] = "Train"
            df_stats[f"{split_strategy.capitalize()} data split"].loc[validation_idx] = "Validation"
            df_stats[f"{split_strategy.capitalize()} data split"].loc[test_idx] = "Test"

        elif split_strategy == "stratified":
            if stratify_distribution == "match":
                # Split data into train, validation and test sets
                train_idx, test_idx = train_test_split(
                    np.arange(self.len()),
                    test_size=test_size,
                    random_state=random_state,
                    stratify=df_stats[stratify_on],
                )
                train_idx, validation_idx = train_test_split(
                    train_idx,
                    test_size=validation_size / (1 - test_size),
                    random_state=random_state,
                    stratify=df_stats.loc[train_idx][stratify_on],
                )

                # Save indices to csv files
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ", "")}_train.csv'),
                    train_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ", "")}_validation.csv'),
                    validation_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ", "")}_test.csv'),
                    test_idx,
                    delimiter=",",
                    fmt="%i",
                )

                # Update statistics dataframe
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on})"] = ""
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on})"].loc[train_idx] = "Train"
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on})"].loc[validation_idx] = "Validation"
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on})"].loc[test_idx] = "Test"

            elif stratify_distribution == "equal":

                if n_sample_per_class == "max":
                    # Find the class with the least number of samples
                    min_samples = df_stats[stratify_on].value_counts().min()
                elif isinstance(n_sample_per_class, int):
                    min_samples = n_sample_per_class
                else:
                    raise ValueError(
                        'n_sample_per_class not recognized. Please use either "max" or an integer'
                    )
                # Randomly sample the same number of samples from each class
                subset_idx = []
                for group in df_stats[stratify_on].unique():
                    subset_idx += list(
                        df_stats[df_stats[stratify_on] == group]
                        .sample(min_samples, random_state=random_state)
                        .index
                    )

                # Find the total number of samples
                n_samples = len(subset_idx)

                # Find the number of samples to use for train, validation and test sets
                n_test = int(n_samples * test_size)
                n_validation = int((n_samples - n_test) * validation_size / (1 - test_size))
                n_train = n_samples - n_test - n_validation

                # Split data into train, validation and test sets
                train_idx, test_idx = train_test_split(
                    subset_idx,
                    train_size = n_train + n_validation,
                    test_size = n_test,
                    random_state = random_state,
                    stratify = df_stats.loc[subset_idx][stratify_on],
                )
                train_idx, validation_idx = train_test_split(
                    train_idx,
                    train_size = n_train,
                    test_size = n_validation,
                    random_state = random_state,
                    stratify = df_stats.loc[train_idx][stratify_on],
                )

                # Save indices to csv files
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ", "")}_{stratify_distribution}_train.csv'),
                    train_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ", "")}_{stratify_distribution}_validation.csv'),
                    validation_idx,
                    delimiter=",",
                    fmt="%i",
                )
                np.savetxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ", "")}_{stratify_distribution}_test.csv'),
                    test_idx,
                    delimiter=",",
                    fmt="%i",
                )

                # Update statistics dataframe
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"] = ""
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"].loc[train_idx] = "Train"
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"].loc[validation_idx] = "Validation"
                df_stats[f"{split_strategy.capitalize()} data split ({stratify_on}, Equal classes)"].loc[test_idx] = "Test"
            else:
                raise ValueError(
                    'Stratify distribution not recognized. Please use either "match" or "equal"'
                )
        else:
            # Raise error if split strategy is not recognized
            raise ValueError(
                'Split strategy not recognized. Please use either "random" or "stratified"'
            )

        # Update statistics file
        df_stats.to_pickle(os.path.join(self.root, "dataset_statistics.pkl"))

        if return_idx:
            return train_idx, validation_idx, test_idx
        else:
            return None

    def load_data_split(
        self,
        split_strategy="random",
        stratify_on="Space group (Number)",
        stratify_distribution="match",
    ) -> None:
        """
        Load the indices of the train, validation and test sets from csv files in the processed directory.
        """
        if split_strategy == "random":

            # Load indices from csv files
            train_idx = np.loadtxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_train.csv"),
                delimiter=",",
                dtype=int,
            )
            validation_idx = np.loadtxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_validation.csv"),
                delimiter=",",
                dtype=int,
            )
            test_idx = np.loadtxt(
                os.path.join(self.root, f"datasplit_{split_strategy}_test.csv"),
                delimiter=",",
                dtype=int,
            )

        elif split_strategy == "stratified":

            if stratify_distribution == "match":

                # Load indices from csv files
                train_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_train.csv'),
                    delimiter=",",
                    dtype=int,
                )
                validation_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_validation.csv'),
                    delimiter=",",
                    dtype=int,
                )
                test_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_test.csv'),
                    delimiter=",",
                    dtype=int,
                )

            elif stratify_distribution == "equal":

                # Load indices from csv files
                train_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_train.csv'),
                    delimiter=",",
                    dtype=int,
                )
                validation_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_validation.csv'),
                    delimiter=",",
                    dtype=int,
                )
                test_idx = np.loadtxt(
                    os.path.join(self.root, f'datasplit_{split_strategy}_{stratify_on.replace(" ","")}_{stratify_distribution}_test.csv'),
                    delimiter=",",
                    dtype=int,
                )

        # Split the dataset into train, validation and test sets
        self.train_set = Subset(self, train_idx)
        self.validation_set = Subset(self, validation_idx)
        self.test_set = Subset(self, test_idx)

    def get_statistics(self, return_dataframe=False):

        # Get statistics path
        stat_path = os.path.join(self.root, "dataset_statistics.pkl")

        # Read pkl or generate
        if os.path.exists(stat_path):
            df_stats = pd.read_pickle(stat_path)
        else:
            df_stats = pd.DataFrame(
                columns=[
                    "idx",
                    "# of nodes",
                    "# of edges",
                    "edge/node ratio",
                    "# of elements",
                    "Space group (Symbol)",
                    "Space group (Number)",
                    "Crystal type",
                    "Crystal system",
                    "Crystal system (Number)",
                    "NP size (Å)",
                    "Elements",
                ]
            )

            stat_pbar = tqdm(desc='Generating statistics...', total=self.len(), leave=False)
            for idx in tqdm(range(self.len())):
                graph = self.get(
                    idx=idx,
                )
                df_stats.loc[df_stats.shape[0]] = [
                    idx,
                    float(graph.num_nodes),
                    float(graph.num_edges),
                    float(graph.num_edges) / float(graph.num_nodes),
                    float(graph.y["n_atomic_species"]),
                    graph.y["space_group_symbol"],
                    float(graph.y["space_group_number"]),
                    graph.y["crystal_type"],
                    graph.y["crystal_system"],
                    graph.y["crystal_system_number"],
                    graph.y["np_size"],
                    graph.y["atomic_species"],
                ]
                stat_pbar.update(1)
            stat_pbar.close()

        df_stats.to_pickle(stat_path)

        if return_dataframe:
            if self.train_set is not None:
                return df_stats.loc[
                    list(self.train_set.indices)
                    + list(self.validation_set.indices)
                    + list(self.test_set.indices)
                ].reset_index(drop=True)
            else:
                return df_stats
        else:
            return None
