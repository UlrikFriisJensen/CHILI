# %% Imports
import argparse
import os
import time
import warnings
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dataset_class import InOrgMatDatasets
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.models import GAT, GCN, GIN, PMLP, EdgeCNN, GraphSAGE, GraphUNet
from torch_geometric.seed import seed_everything
from torch_geometric.utils import unbatch
from torcheval.metrics import MulticlassF1Score
from tqdm.auto import tqdm

# Ignore warnings
warnings.simplefilter(action="ignore")
warnings.filterwarnings(
    "ignore",
    message="Some classes do not exist in the target. F1 scores for these classes will be cast to zeros.",
    category=UserWarning,
)

# %% Functions and classes


# Simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, num_layers):
        super(SimpleMLP, self).__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, batch):
        if len(x.shape) < 2:
            batch_size = torch.max(batch) + 1
            x = x.reshape(batch_size, x.shape[-1] // batch_size)

        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


# Regression secondary module
class Secondary(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Secondary, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x, batch):
        # Pool the graph
        x = torch.cat(
            (
                global_mean_pool(x, batch),
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ),
            dim=1,
        )
        # Linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Define position mean absolute error function
def position_MAE(pred_xyz, true_xyz):
    """
    Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
    """
    return torch.mean(
        torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction="none"), dim=1)),
        dim=0,
    )


# %% Main function

# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmarking script")
parser.add_argument(
    "--config_folder", type=str, help="Path to folder containing configuration files"
)
parser.add_argument("--config_index", type=str, help="Index for cluster array")
args = parser.parse_args()

# Read configuration file
config_files = sorted(glob(os.path.join(args.config_folder, "*.yaml")))
config_path = config_files[int(args.config_index)]

with open(config_path, "r") as file:
    config_dict = yaml.safe_load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your dataset
dataset = InOrgMatDatasets(root=config_dict["root"], dataset=config_dict["dataset"])

# Load / Create data split
if config_dict["Data_config"]["split_strategy"] == "random":
    test_size = 0.1
try:
    dataset.load_data_split(
        split_strategy=config_dict["Data_config"]["split_strategy"],
        stratify_on=config_dict["Data_config"]["stratify_on"],
        stratify_distribution=config_dict["Data_config"]["stratify_distribution"],
    )
except FileNotFoundError:
    dataset.create_data_split(
        split_strategy=config_dict["Data_config"]["split_strategy"],
        stratify_on=config_dict["Data_config"]["stratify_on"],
        stratify_distribution=config_dict["Data_config"]["stratify_distribution"],
        test_size=0.1,
    )
    dataset.load_data_split(
        split_strategy=config_dict["Data_config"]["split_strategy"],
        stratify_on=config_dict["Data_config"]["stratify_on"],
        stratify_distribution=config_dict["Data_config"]["stratify_distribution"],
    )

# Filter on number of atoms if the task is PositionRegression from Signal
if config_dict["task"] in [
    "AbsPositionRegressionxPDF",
    "AbsPositionRegressionXRD",
    "AbsPositionRegressionSAXS",
]:
    # Train set
    filtered_idx = []
    for idx in dataset.train_set.indices:
        data = dataset[idx]
        if len(data.pos_abs) <= config_dict["Model_config"]["out_channels"] // 3:
            filtered_idx.append(idx)

    dataset.train_set = Subset(dataset, filtered_idx)

    # Validation set
    filtered_idx = []
    for idx in dataset.validation_set.indices:
        data = dataset[idx]
        if len(data.pos_abs) <= config_dict["Model_config"]["out_channels"] // 3:
            filtered_idx.append(idx)

    dataset.validation_set = Subset(dataset, filtered_idx)

    # Test set
    filtered_idx = []
    for idx in dataset.test_set.indices:
        data = dataset[idx]
        if len(data.pos_abs) <= config_dict["Model_config"]["out_channels"] // 3:
            filtered_idx.append(idx)

    dataset.test_set = Subset(dataset, filtered_idx)

if config_dict["task"] in [
    "UnitCellPositionRegressionxPDF",
    "UnitCellPositionRegressionXRD",
    "UnitCellPositionRegressionSAXS",
]:
    # Train set
    filtered_idx = []
    for idx in dataset.train_set.indices:
        data = dataset[idx]
        if (
            len(data.y["unit_cell_pos_frac"])
            <= config_dict["Model_config"]["out_channels"] // 3
        ):
            filtered_idx.append(idx)

    dataset.train_set = Subset(dataset, filtered_idx)

    # Validation set
    filtered_idx = []
    for idx in dataset.validation_set.indices:
        data = dataset[idx]
        if (
            len(data.y["unit_cell_pos_frac"])
            <= config_dict["Model_config"]["out_channels"] // 3
        ):
            filtered_idx.append(idx)

    dataset.validation_set = Subset(dataset, filtered_idx)

    # Test set
    filtered_idx = []
    for idx in dataset.test_set.indices:
        data = dataset[idx]
        if (
            len(data.y["unit_cell_pos_frac"])
            <= config_dict["Model_config"]["out_channels"] // 3
        ):
            filtered_idx.append(idx)

    dataset.test_set = Subset(dataset, filtered_idx)

# Create dataframe for saving results
results_df = pd.DataFrame(
    columns=[
        "Model",
        "Dataset",
        "Task",
        "Seed",
        "Train samples",
        "Val samples",
        "Test samples",
        "Train time",
        "Trainable parameters",
        "Train loss",
        "Val F1-score",
        "Test F1-score",
        "Val posMAE/MSE",
        "Test posMAE/MSE",
    ]
)

print(
    f'\nModel: {config_dict["model"]}\nDataset: {config_dict["dataset"]}\nTask: {config_dict["task"]}',
    flush=True,
)
print("\n", flush=True)
print(f"Number of training samples: {len(dataset.train_set)}", flush=True)
print(f"Number of validation samples: {len(dataset.validation_set)}", flush=True)
print(f"Number of test samples: {len(dataset.test_set)}", flush=True)
print("\n", flush=True)
print(f"Device: {device}", flush=True)

# Train model for each seed
for i, seed in enumerate(config_dict["Train_config"]["seeds"]):
    # Set seed
    seed_everything(seed)

    print(f"\nSeed: {seed}\n", flush=True)

    # Define your model
    if config_dict["model"] == "GCN":
        model = GCN(**config_dict["Model_config"]).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "GraphSAGE":
        model = GraphSAGE(**config_dict["Model_config"]).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "GIN":
        model = GIN(**config_dict["Model_config"]).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "GAT":
        model = GAT(**config_dict["Model_config"], v2=False).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "EdgeCNN":
        model = EdgeCNN(**config_dict["Model_config"]).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "GraphUNet":
        model = GraphUNet(**config_dict["Model_config"]).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "PMLP":
        model = PMLP(**config_dict["Model_config"]).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    elif config_dict["model"] == "MLP":
        model = SimpleMLP(**config_dict["Model_config"], device=device).to(device)
        secondary = Secondary(**config_dict["Secondary_config"]).to(device)
    else:
        raise ValueError("Model not supported")

    # Define forward pass
    if config_dict["task"] == "AtomClassification":
        if config_dict["model"] == "GraphUNet":

            def forward_pass(data):
                return model.forward(
                    x=data.pos_abs, edge_index=data.edge_index, batch=data.batch
                )
        elif config_dict["model"] == "PMLP":

            def forward_pass(data):
                return model.forward(x=data.pos_abs, edge_index=data.edge_index)
        else:

            def forward_pass(data):
                return model.forward(
                    x=data.pos_abs,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    edge_weight=data.edge_attr,
                    batch=data.batch,
                )
    elif config_dict["task"] in [
        "SpacegroupClassification",
        "CrystalSystemClassification",
        "SAXSRegression",
        "XRDRegression",
        "xPDFRegression",
    ]:
        if config_dict["model"] in ["GraphUNet"]:

            def forward_pass(data):
                out = model.forward(
                    x=torch.cat((data.x, data.pos_abs), dim=1),
                    edge_index=data.edge_index,
                    batch=data.batch,
                )
                out = secondary.forward(out, data.batch)
                return out
        elif config_dict["model"] == "PMLP":

            def forward_pass(data):
                out = model.forward(
                    x=torch.cat((data.x, data.pos_abs), dim=1),
                    edge_index=data.edge_index,
                )
                out = secondary.forward(out, data.batch)
                return out
        else:

            def forward_pass(data):
                out = model.forward(
                    x=torch.cat((data.x, data.pos_abs), dim=1),
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    edge_weight=data.edge_attr,
                    batch=data.batch,
                )
                out = secondary.forward(out, data.batch)
                return out
    elif config_dict["task"] == "PositionRegression":
        if config_dict["model"] == "GraphUNet":

            def forward_pass(data):
                return model.forward(
                    x=data.x, edge_index=data.edge_index, batch=data.batch
                )
        elif config_dict["model"] == "PMLP":

            def forward_pass(data):
                return model.forward(x=data.x, edge_index=data.edge_index)
        else:

            def forward_pass(data):
                return model.forward(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    edge_weight=data.edge_attr,
                    batch=data.batch,
                )
    elif config_dict["task"] == "DistanceRegression":
        if config_dict["model"] == "PMLP":

            def forward_pass(data):
                return model.forward(
                    x=torch.cat((data.x, data.pos_abs), dim=1),
                    edge_index=data.edge_index,
                )
        else:

            def forward_pass(data):
                return model.forward(
                    x=torch.cat((data.x, data.pos_abs), dim=1),
                    edge_index=data.edge_index,
                    batch=data.batch,
                )
    elif config_dict["task"] in [
        "AbsPositionRegressionSAXS",
        "UnitCellPositionRegressionSAXS",
        "CellParamsRegressionSAXS",
    ]:

        def forward_pass(data):
            signal = data.y["saxs"][1::2, :]
            signal = (signal - torch.min(signal, dim=-1, keepdim=True)[0]) / (
                torch.max(signal, dim=-1, keepdim=True)[0]
                - torch.min(signal, dim=-1, keepdim=True)[0]
            )
            return model.forward(x=signal, batch=data.batch)
    elif config_dict["task"] in [
        "CrystalSystemClassificationSAXS",
        "SpacegroupClassificationSAXS",
    ]:

        def forward_pass(data):
            signal = data.y["saxs"][1::2, :]
            signal = (signal - torch.min(signal, dim=-1, keepdim=True)[0]) / (
                torch.max(signal, dim=-1, keepdim=True)[0]
                - torch.min(signal, dim=-1, keepdim=True)[0]
            )
            return model.forward(x=signal, batch=data.batch)
    elif config_dict["task"] in [
        "AbsPositionRegressionXRD",
        "UnitCellPositionRegressionXRD",
        "CellParamsRegressionXRD",
    ]:

        def forward_pass(data):
            signal = data.y["xrd"][1::2, :]
            signal = (signal - torch.min(signal, dim=-1, keepdim=True)[0]) / (
                torch.max(signal, dim=-1, keepdim=True)[0]
                - torch.min(signal, dim=-1, keepdim=True)[0]
            )
            return model.forward(x=signal, batch=data.batch)
    elif config_dict["task"] in [
        "CrystalSystemClassificationXRD",
        "SpacegroupClassificationXRD",
    ]:

        def forward_pass(data):
            signal = data.y["xrd"][1::2, :]
            signal = (signal - torch.min(signal, dim=-1, keepdim=True)[0]) / (
                torch.max(signal, dim=-1, keepdim=True)[0]
                - torch.min(signal, dim=-1, keepdim=True)[0]
            )
            return model.forward(x=signal, batch=data.batch)
    elif config_dict["task"] in [
        "AbsPositionRegressionxPDF",
        "UnitCellPositionRegressionxPDF",
        "CellParamsRegressionxPDF",
    ]:

        def forward_pass(data):
            signal = data.y["xPDF"][1::2, :]
            signal = (signal - torch.min(signal, dim=-1, keepdim=True)[0]) / (
                torch.max(signal, dim=-1, keepdim=True)[0]
                - torch.min(signal, dim=-1, keepdim=True)[0]
            )
            return model.forward(x=signal, batch=data.batch)
    elif config_dict["task"] in [
        "CrystalSystemClassificationxPDF",
        "SpacegroupClassificationxPDF",
    ]:

        def forward_pass(data):
            signal = data.y["xPDF"][1::2, :]
            signal = (signal - torch.min(signal, dim=-1, keepdim=True)[0]) / (
                torch.max(signal, dim=-1, keepdim=True)[0]
                - torch.min(signal, dim=-1, keepdim=True)[0]
            )
            return model.forward(x=signal, batch=data.batch)
    else:
        raise NotImplementedError

    # Define dataloader
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=config_dict["Train_config"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset.validation_set,
        batch_size=config_dict["Train_config"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset.test_set,
        batch_size=config_dict["Train_config"]["batch_size"],
        shuffle=False,
    )

    # Rwp criterion
    def rwp(
        pred,
        truth,
    ):
        return torch.mean(
            torch.sqrt(
                torch.sum(torch.square(pred - truth), dim=-1)
                / torch.sum(torch.square(truth), dim=-1)
            )
        )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(secondary.parameters()),
        lr=config_dict["Train_config"]["learning_rate"],
    )
    if config_dict["task"] in [
        "AtomClassification",
        "SpacegroupClassification",
        "CrystalSystemClassification",
    ]:
        criterion = torch.nn.CrossEntropyLoss()
        if config_dict["task"] == "AtomClassification":
            n_classes = 118
        elif config_dict["task"] == "SpacegroupClassification":
            n_classes = 230
        elif config_dict["task"] == "CrystalSystemClassification":
            n_classes = 7
    elif config_dict["task"] in ["PositionRegression", "DistanceRegression"]:
        n_classes = 1
        criterion = torch.nn.SmoothL1Loss()
        metric = torch.nn.MSELoss()
    elif config_dict["task"] in ["SAXSRegression", "XRDRegression", "xPDFRegression"]:
        n_classes = 1
        criterion = nn.SmoothL1Loss()
        metric = nn.MSELoss()
    elif config_dict["task"] in [
        "AbsPositionRegressionSAXS",
        "AbsPositionRegressionXRD",
        "AbsPositionRegressionxPDF",
    ]:
        n_classes = 1
        criterion = torch.nn.SmoothL1Loss(reduction="none")
        metric = torch.nn.L1Loss(reduction="none")  # MAE
    elif config_dict["task"] in [
        "UnitCellPositionRegressionSAXS",
        "UnitCellPositionRegressionXRD",
        "UnitCellPositionRegressionxPDF",
    ]:
        n_classes = 1
        criterion = torch.nn.SmoothL1Loss(reduction="none")
        metric = torch.nn.L1Loss(reduction="none")  # MAE
    elif config_dict["task"] in [
        "CellParamsRegressionSAXS",
        "CellParamsRegressionXRD",
        "CellParamsRegressionxPDF",
    ]:
        n_classes = 1
        criterion = torch.nn.SmoothL1Loss()
        metric = torch.nn.MSELoss()
    elif config_dict["task"] in [
        "CrystalSystemClassificationSAXS",
        "CrystalSystemClassificationXRD",
        "CrystalSystemClassificationxPDF",
    ]:
        n_classes = 7
        criterion = torch.nn.CrossEntropyLoss()
    elif config_dict["task"] in [
        "SpacegroupClassificationSAXS",
        "SpacegroupClassificationXRD",
        "SpacegroupClassificationxPDF",
    ]:
        n_classes = 230
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # Define TensorBoard writer
    save_dir = f"{config_dict['log_dir']}/{config_dict['dataset']}/{config_dict['task']}/{config_dict['model']}/seed{seed}"
    writer = SummaryWriter(save_dir)

    # Set training time (in seconds)
    max_training_time = config_dict["Train_config"]["train_time"]
    start_time = time.time()
    epoch = 0

    # Patience
    max_patience = config_dict["Train_config"]["max_patience"]
    patience = 0

    # Training loop
    for epoch in range(config_dict["Train_config"]["epochs"]):
        # Stop training if max training time is exceeded
        if time.time() - start_time > max_training_time:
            break
        if patience >= max_patience:
            print("Max Patience reached, quitting...", flush=True)
            break
        model.train()
        total_loss = 0

        # batches pbar
        batches_pbar = tqdm(
            desc="Training epoch...", total=len(train_loader), disable=True
        )
        for data in train_loader:
            # Send to device
            data = data.to(device)

            if config_dict["task"] == "AtomClassification":
                out = forward_pass(data)
                ground_truth = data.x[:, 0].long()
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "SpacegroupClassification":
                out = forward_pass(data)
                ground_truth = torch.tensor(data.y["space_group_number"], device=device)
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "CrystalSystemClassification":
                out = forward_pass(data)
                ground_truth = torch.tensor(
                    data.y["crystal_system_number"], device=device
                )
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "PositionRegression":
                out = forward_pass(data)
                ground_truth = data.pos_abs
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "SAXSRegression":
                out = forward_pass(data)
                ground_truth = data.y["saxs"][1::2, :]
                # Min-max normalize saxs data
                ground_truth = (
                    ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                ) / (
                    torch.max(ground_truth, dim=-1, keepdim=True)[0]
                    - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                )
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "XRDRegression":
                out = forward_pass(data)
                ground_truth = data.y["xrd"][1::2, :]
                # Min-max normalize xrd data
                ground_truth = (
                    ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                ) / (
                    torch.max(ground_truth, dim=-1, keepdim=True)[0]
                    - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                )
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "xPDFRegression":
                out = forward_pass(data)
                ground_truth = data.y["xPDF"][1::2, :]
                # Min-max normalize xpdf data
                ground_truth = (
                    ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                ) / (
                    torch.max(ground_truth, dim=-1, keepdim=True)[0]
                    - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                )
                loss = criterion(out, ground_truth)
            elif config_dict["task"] == "DistanceRegression":
                out = forward_pass(data)
                out = torch.sum(
                    out[data.edge_index[0, :]] * out[data.edge_index[1, :]], dim=-1
                )
                ground_truth = data.edge_attr
                loss = criterion(out, ground_truth)
            elif config_dict["task"] in [
                "CellParamsRegressionSAXS",
                "CellParamsRegressionXRD",
                "CellParamsRegressionxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)
                # Get ground truth
                ground_truth = data.y["cell_params"].reshape(
                    torch.max(data.batch) + 1, 6
                )
                # Calculate loss
                loss = criterion(out, ground_truth)
            elif config_dict["task"] in [
                "CrystalSystemClassificationSAXS",
                "CrystalSystemClassificationXRD",
                "CrystalSystemClassificationxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)
                # Get ground truth
                ground_truth = torch.tensor(
                    data.y["crystal_system_number"], device=device
                )
                # Calculate loss
                loss = criterion(out, ground_truth)
            elif config_dict["task"] in [
                "SpacegroupClassificationSAXS",
                "SpacegroupClassificationXRD",
                "SpacegroupClassificationxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)
                # Get ground truth
                ground_truth = torch.tensor(data.y["space_group_number"], device=device)
                # Calculate loss
                loss = criterion(out, ground_truth)
            elif config_dict["task"] in [
                "AbsPositionRegressionSAXS",
                "AbsPositionRegressionXRD",
                "AbsPositionRegressionxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)  # (B, X+Y+Z)
                # Get ground truth by sorting and padding
                batch_size = torch.max(data.batch) + 1
                ground_truth = torch.zeros(
                    batch_size, config_dict["Model_config"]["out_channels"]
                ).to(device=device)
                for i, x in enumerate(unbatch(data.pos_abs, data.batch)):
                    # Sort according to norm
                    norms = torch.norm(x, p=2, dim=-1)
                    _, indices = torch.sort(norms, descending=False, dim=0)
                    x = x[indices]

                    # Pad
                    padding_size = config_dict["Model_config"][
                        "out_channels"
                    ] // 3 - x.size(0)
                    if padding_size > 0:
                        padding = torch.full(
                            (padding_size, x.size(1)), 100, dtype=x.dtype
                        ).to(device=device)
                        x = torch.cat([x, padding], dim=0)

                    # Append
                    ground_truth[i] = x.T.flatten()

                # mask out the padding
                mask = ground_truth < 100
                loss = criterion(out, ground_truth)[mask].mean()
            elif config_dict["task"] in [
                "UnitCellPositionRegressionSAXS",
                "UnitCellPositionRegressionXRD",
                "UnitCellPositionRegressionxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)  # (B, X+Y+Z)
                # Get ground truth by sorting and padding
                batch_size = torch.max(data.batch) + 1
                ground_truth = torch.zeros(
                    batch_size, config_dict["Model_config"]["out_channels"]
                ).to(device=device)
                # We cannot unbatch, since data.batch refers to the nanoparticle positions. Instead
                lidx = 0
                for i, size in enumerate(data.y["unit_cell_n_atoms"]):
                    # Extract the unit cell
                    x = data.y["unit_cell_pos_frac"][lidx:size]
                    lidx += size

                    # Sort according to norm
                    norms = torch.norm(x, p=2, dim=-1)
                    _, indices = torch.sort(norms, descending=False, dim=0)
                    x = x[indices]

                    # Pad
                    padding_size = config_dict["Model_config"][
                        "out_channels"
                    ] // 3 - x.size(0)
                    if padding_size > 0:
                        padding = torch.full(
                            (padding_size, x.size(1)), 100, dtype=x.dtype
                        ).to(device=device)
                        x = torch.cat([x, padding], dim=0)

                    # Append
                    ground_truth[i] = x.T.flatten()

                    # mask out the padding
                mask = ground_truth < 100
                loss = criterion(out, ground_truth)[mask].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batches_pbar.update(1)

        batches_pbar.close()

        train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        correct = 0
        error = 0
        total = 0

        with torch.no_grad():
            MC_F1 = MulticlassF1Score(num_classes=n_classes, average="weighted")
            for data in val_loader:
                data = data.to(device)
                if config_dict["task"] == "AtomClassification":
                    out = forward_pass(data)
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = data.x[:, 0].long()
                    MC_F1.update(predicted, ground_truth)
                elif config_dict["task"] == "SpacegroupClassification":
                    out = forward_pass(data)
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = torch.tensor(
                        data.y["space_group_number"], device=device
                    )
                    MC_F1.update(predicted, ground_truth)
                elif config_dict["task"] == "CrystalSystemClassification":
                    out = forward_pass(data)
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = torch.tensor(
                        data.y["crystal_system_number"], device=device
                    )
                    MC_F1.update(predicted, ground_truth)
                elif config_dict["task"] == "PositionRegression":
                    out = forward_pass(data)
                    error += position_MAE(out, data.pos_abs)
                elif config_dict["task"] == "SAXSRegression":
                    out = forward_pass(data)
                    ground_truth = data.saxs[1::2, :]
                    # Min-max normalize saxs data
                    ground_truth = (
                        ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                    ) / (
                        torch.max(ground_truth, dim=-1, keepdim=True)[0]
                        - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                    )
                    error += metric(out, ground_truth)
                elif config_dict["task"] == "XRDRegression":
                    out = forward_pass(data)
                    ground_truth = data.y["xrd"][1::2, :]
                    # Min-max normalize xrd data
                    ground_truth = (
                        ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                    ) / (
                        torch.max(ground_truth, dim=-1, keepdim=True)[0]
                        - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                    )
                    error += metric(out, ground_truth)
                elif config_dict["task"] == "xPDFRegression":
                    out = forward_pass(data)
                    ground_truth = data.y["xPDF"][1::2, :]
                    # Min-max normalize xpdf data
                    ground_truth = (
                        ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                    ) / (
                        torch.max(ground_truth, dim=-1, keepdim=True)[0]
                        - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                    )
                    error += metric(out, ground_truth)
                elif config_dict["task"] == "DistanceRegression":
                    out = forward_pass(data)
                    out = torch.sum(
                        out[data.edge_index[0, :]] * out[data.edge_index[1, :]], dim=-1
                    )
                    ground_truth = data.edge_attr
                    error += metric(out, ground_truth)
                elif config_dict["task"] in [
                    "CellParamsRegressionSAXS",
                    "CellParamsRegressionXRD",
                    "CellParamsRegressionxPDF",
                ]:
                    # Get prediction
                    out = forward_pass(data)
                    # Get ground truth
                    ground_truth = data.y["cell_params"].reshape(
                        torch.max(data.batch) + 1, 6
                    )
                    # Calculate loss
                    error += metric(out, ground_truth)
                elif config_dict["task"] in [
                    "CrystalSystemClassificationSAXS",
                    "CrystalSystemClassificationXRD",
                    "CrystalSystemClassificationxPDF",
                ]:
                    # Get prediction
                    out = forward_pass(data)
                    # Get ground truth
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = torch.tensor(
                        data.y["crystal_system_number"], device=device
                    )
                    # Calculate loss
                    MC_F1.update(predicted, ground_truth)
                elif config_dict["task"] in [
                    "SpacegroupClassificationSAXS",
                    "SpacegroupClassificationXRD",
                    "SpacegroupClassificationxPDF",
                ]:
                    # Get prediction
                    out = forward_pass(data)
                    # Get ground truth
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = torch.tensor(
                        data.y["space_group_number"], device=device
                    )
                    # Calculate loss
                    MC_F1.update(predicted, ground_truth)
                elif config_dict["task"] in [
                    "AbsPositionRegressionSAXS",
                    "AbsPositionRegressionXRD",
                    "AbsPositionRegressionxPDF",
                ]:
                    # Get prediction
                    out = forward_pass(data)  # (B, X+Y+Z)
                    # Get ground truth by sorting and padding
                    batch_size = torch.max(data.batch) + 1
                    ground_truth = torch.zeros(
                        batch_size, config_dict["Model_config"]["out_channels"]
                    ).to(device=device)
                    for i, x in enumerate(unbatch(data.pos_abs, data.batch)):
                        # Sort according to norm
                        norms = torch.norm(x, p=2, dim=-1)
                        _, indices = torch.sort(norms, descending=False, dim=0)
                        x = x[indices]

                        # Pad
                        padding_size = config_dict["Model_config"][
                            "out_channels"
                        ] // 3 - x.size(0)
                        if padding_size > 0:
                            padding = torch.full(
                                (padding_size, x.size(1)), 100, dtype=x.dtype
                            ).to(device=device)
                            x = torch.cat([x, padding], dim=0)

                        # Append
                        ground_truth[i] = x.T.flatten()

                    # mask out the padding
                    mask = ground_truth < 100
                    error += metric(out, ground_truth)[mask].mean()

                elif config_dict["task"] in [
                    "UnitCellPositionRegressionSAXS",
                    "UnitCellPositionRegressionXRD",
                    "UnitCellPositionRegressionxPDF",
                ]:
                    # Get prediction
                    out = forward_pass(data)  # (B, X+Y+Z)
                    # Get ground truth by sorting and padding
                    batch_size = torch.max(data.batch) + 1
                    ground_truth = torch.zeros(
                        batch_size, config_dict["Model_config"]["out_channels"]
                    ).to(device=device)

                    # We cannot unbatch, since data.batch refers to the nanoparticle positions. Instead
                    lidx = 0
                    for i, size in enumerate(data.y["unit_cell_n_atoms"]):
                        # Extract the unit cell
                        x = data.y["unit_cell_pos_frac"][lidx:size]
                        lidx += size

                        # Sort according to norm
                        norms = torch.norm(x, p=2, dim=-1)
                        _, indices = torch.sort(norms, descending=False, dim=0)
                        x = x[indices]

                        # Pad
                        padding_size = config_dict["Model_config"][
                            "out_channels"
                        ] // 3 - x.size(0)
                        if padding_size > 0:
                            padding = torch.full(
                                (padding_size, x.size(1)), 100, dtype=x.dtype
                            ).to(device=device)
                            x = torch.cat([x, padding], dim=0)

                        # Append
                        ground_truth[i] = x.T.flatten()

                    # mask out the padding
                    mask = ground_truth < 100
                    error += metric(out, ground_truth)[mask].mean()

        # Log training progress
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Log validation progress
        if "Classification" in config_dict["task"]:
            val_error = torch.tensor(0)
            val_f1 = MC_F1.compute()
            writer.add_scalar("F1-score/val", val_f1, epoch)

            # Save model if validation accuracy is improved
            if epoch == 0:
                best_val_f1 = val_f1
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "secondary_state_dict": secondary.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config_dict": config_dict,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/best.pt",
                )
            elif val_f1 > best_val_f1:
                patience = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "secondary_state_dict": secondary.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config_dict": config_dict,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/best.pt",
                )
                best_val_f1 = val_f1
            else:
                patience += 1

            print(
                f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val F1-score: {val_f1:.4f}',
                flush=True,
            )
        elif "Regression" in config_dict["task"]:
            val_f1 = 0
            val_error = error / len(val_loader)

            # Save model if validation error is improved
            if epoch == 0:
                best_val_error = val_error
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "secondary_state_dict": secondary.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config_dict": config_dict,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/best.pt",
                )
            elif val_error < best_val_error:
                patience = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "secondary_state_dict": secondary.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config_dict": config_dict,
                        "train_subset_indices": dataset.train_set.indices,
                        "validation_subset_indices": dataset.validation_set.indices,
                        "test_subset_indices": dataset.test_set.indices,
                    },
                    f"{save_dir}/best.pt",
                )
                best_val_error = val_error
            else:
                patience += 1

            if "PositionRegression" in config_dict["task"]:
                writer.add_scalar("posMAE/val", val_error, epoch)
                print(
                    f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val position MAE: {val_error:.4f}',
                    flush=True,
                )
            else:
                writer.add_scalar("MSE/val", val_error, epoch)
                print(
                    f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val MSE: {val_error:.4f}',
                    flush=True,
                )

        # Save latest model
        if config_dict["save_latest_model"]:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "secondary_state_dict": secondary.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config_dict": config_dict,
                    "train_subset_indices": dataset.train_set.indices,
                    "validation_subset_indices": dataset.validation_set.indices,
                    "test_subset_indices": dataset.test_set.indices,
                },
                f"{save_dir}/latest.pt",
            )
    print("Finished logging validation", flush=True)

    # Record stop time
    stop_time = time.time()

    # Load best model
    checkpoint = torch.load(f"{save_dir}/best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    # Evaluate the model on the test set using the best epoch
    model.eval()
    correct = 0
    error = 0
    total = 0

    with torch.no_grad():
        MC_F1 = MulticlassF1Score(num_classes=n_classes, average="weighted")
        for data in test_loader:
            data = data.to(device)
            if config_dict["task"] == "AtomClassification":
                out = forward_pass(data)
                _, predicted = torch.max(out.data, 1)
                ground_truth = data.x[:, 0].long()
                MC_F1.update(predicted, ground_truth)
            elif config_dict["task"] == "SpacegroupClassification":
                out = forward_pass(data)
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(data.y["space_group_number"], device=device)
                MC_F1.update(predicted, ground_truth)
            elif config_dict["task"] == "CrystalSystemClassification":
                out = forward_pass(data)
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(
                    data.y["crystal_system_number"], device=device
                )
                MC_F1.update(predicted, ground_truth)
            elif config_dict["task"] == "PositionRegression":
                out = forward_pass(data)
                error += position_MAE(out, data.pos_abs)
            elif config_dict["task"] == "SAXSRegression":
                out = forward_pass(data)
                ground_truth = data.saxs[1::2, :]
                # Min-max normalize saxs data
                ground_truth = (
                    ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                ) / (
                    torch.max(ground_truth, dim=-1, keepdim=True)[0]
                    - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                )
                error += metric(out, ground_truth)
            elif config_dict["task"] == "XRDRegression":
                out = forward_pass(data)
                ground_truth = data.y["xrd"][1::2, :]
                # Min-max normalize xrd data
                ground_truth = (
                    ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                ) / (
                    torch.max(ground_truth, dim=-1, keepdim=True)[0]
                    - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                )
                error += metric(out, ground_truth)
            elif config_dict["task"] == "xPDFRegression":
                out = forward_pass(data)
                ground_truth = data.y["xPDF"][1::2, :]
                # Min-max normalize xpdf data
                ground_truth = (
                    ground_truth - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                ) / (
                    torch.max(ground_truth, dim=-1, keepdim=True)[0]
                    - torch.min(ground_truth, dim=-1, keepdim=True)[0]
                )
                error += metric(out, ground_truth)
            elif config_dict["task"] == "DistanceRegression":
                out = forward_pass(data)
                out = torch.sum(
                    out[data.edge_index[0, :]] * out[data.edge_index[1, :]], dim=-1
                )
                ground_truth = data.edge_attr
                error += metric(out, ground_truth)
            elif config_dict["task"] in [
                "CellParamsRegressionSAXS",
                "CellParamsRegressionXRD",
                "CellParamsRegressionxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)
                # Get ground truth
                ground_truth = data.y["cell_params"].reshape(
                    torch.max(data.batch) + 1, 6
                )
                # Calculate loss
                error += metric(out, ground_truth)
            elif config_dict["task"] in [
                "CrystalSystemClassificationSAXS",
                "CrystalSystemClassificationXRD",
                "CrystalSystemClassificationxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)
                # Get ground truth
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(
                    data.y["crystal_system_number"], device=device
                )
                # Calculate loss
                MC_F1.update(predicted, ground_truth)
            elif config_dict["task"] in [
                "SpacegroupClassificationSAXS",
                "SpacegroupClassificationXRD",
                "SpacegroupClassificationxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)
                # Get ground truth
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(data.y["space_group_number"], device=device)
                # Calculate loss
                MC_F1.update(predicted, ground_truth)
            elif config_dict["task"] in [
                "AbsPositionRegressionSAXS",
                "AbsPositionRegressionXRD",
                "AbsPositionRegressionxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)  # (B, X+Y+Z)
                # Get ground truth by sorting and padding
                batch_size = torch.max(data.batch) + 1
                ground_truth = torch.zeros(
                    batch_size, config_dict["Model_config"]["out_channels"]
                ).to(device=device)
                for i, x in enumerate(unbatch(data.pos_abs, data.batch)):
                    # Sort according to norm
                    norms = torch.norm(x, p=2, dim=-1)
                    _, indices = torch.sort(norms, descending=False, dim=0)
                    x = x[indices]

                    # Pad
                    padding_size = config_dict["Model_config"][
                        "out_channels"
                    ] // 3 - x.size(0)
                    if padding_size > 0:
                        padding = torch.full(
                            (padding_size, x.size(1)), 100, dtype=x.dtype
                        ).to(device=device)
                        x = torch.cat([x, padding], dim=0)

                    # Append
                    ground_truth[i] = x.T.flatten()

                # mask out the padding
                mask = ground_truth < 100
                error += metric(out, ground_truth)[mask].mean()

            elif config_dict["task"] in [
                "UnitCellPositionRegressionSAXS",
                "UnitCellPositionRegressionXRD",
                "UnitCellPositionRegressionxPDF",
            ]:
                # Get prediction
                out = forward_pass(data)  # (B, X+Y+Z)
                # Get ground truth by sorting and padding
                batch_size = torch.max(data.batch) + 1
                ground_truth = torch.zeros(
                    batch_size, config_dict["Model_config"]["out_channels"]
                ).to(device=device)

                # We cannot unbatch, since data.batch refers to the nanoparticle positions. Instead
                lidx = 0
                for i, size in enumerate(data.y["unit_cell_n_atoms"]):
                    # Extract the unit cell
                    x = data.y["unit_cell_pos_frac"][lidx:size]
                    lidx += size

                    # Sort according to norm
                    norms = torch.norm(x, p=2, dim=-1)
                    _, indices = torch.sort(norms, descending=False, dim=0)
                    x = x[indices]

                    # Pad
                    padding_size = config_dict["Model_config"][
                        "out_channels"
                    ] // 3 - x.size(0)
                    if padding_size > 0:
                        padding = torch.full(
                            (padding_size, x.size(1)), 100, dtype=x.dtype
                        ).to(device=device)
                        x = torch.cat([x, padding], dim=0)

                    # Append
                    ground_truth[i] = x.T.flatten()

                # mask out the padding
                mask = ground_truth < 100
                error += metric(out, ground_truth)[mask].mean()

    if "Classification" in config_dict["task"]:
        test_error = torch.tensor(0)
        test_f1 = MC_F1.compute()

        writer.add_scalar("F1-score/test", test_f1, epoch)

        print(f"Test F1-score: {test_f1:.4f}", flush=True)
    elif "Regression" in config_dict["task"]:
        test_f1 = 0
        test_error = error / len(test_loader)
        if "PositionRegression" in config_dict["task"]:
            writer.add_scalar("posMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        elif "AbsPositionRegressionSAXS" in config_dict["task"]:
            writer.add_scalar("absposMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        elif "AbsPositionRegressionXRD" in config_dict["task"]:
            writer.add_scalar("absposMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        elif "AbsPositionRegressionxPDF" in config_dict["task"]:
            writer.add_scalar("absposMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        elif "UnitCellPositionRegressionSAXS" in config_dict["task"]:
            writer.add_scalar("unitcellposMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        elif "UnitCellPositionRegressionXRD" in config_dict["task"]:
            writer.add_scalar("unitcellposMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        elif "UnitCellPositionRegressionxPDF" in config_dict["task"]:
            writer.add_scalar("unitcellposMAE/test", test_error, epoch)

            print(f"Test position MAE: {test_error:.4f}", flush=True)
        else:
            writer.add_scalar("MSE/test", test_error, epoch)

            print(f"Test MSE: {test_error:.4f}", flush=True)

    # Close TensorBoard writer
    writer.close()

    # Add results to dataframe
    results_df.loc[i] = [
        config_dict["model"],
        config_dict["dataset"],
        config_dict["task"],
        seed,
        len(dataset.train_set),
        len(dataset.validation_set),
        len(dataset.test_set),
        stop_time - start_time,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_loss,
        val_f1,
        test_f1,
        val_error.detach().cpu().numpy(),
        test_error.detach().cpu().numpy(),
    ]

# Save results to csv file
results_df.to_csv(f"{save_dir}/../results.csv")
