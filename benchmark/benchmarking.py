import os
import yaml
import warnings
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.utils import unbatch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT, GCN, GIN, PMLP, EdgeCNN, GraphSAGE, GraphUNet
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.seed import seed_everything

from torcheval.metrics import MulticlassF1Score, MulticlassAccuracy

import pandas as pd
from glob import glob
from tqdm.auto import tqdm
import numpy as np

from dataset_class import CHILI

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
    
def position_MAE(
    self,
    pred_xyz,
    true_xyz
):
    """
    Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
    """
    return torch.mean(
        torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction="none"), dim=1)),
        dim=0,
    )

def run_benchmarking(args):

    # Read configuration file
    config_files = sorted(glob(os.path.join(args.config_folder, "*.yaml")))
    config_path = config_files[int(args.config_index)]
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    dataset = CHILI(root=config_dict["root"], dataset=config_dict["dataset"])
    
    # Load / Create data split
    if config_dict["Data_config"]["split_strategy"] == "random":
        test_size = 0.1
    try:
        dataset.load_data_split(
            split_strategy = config_dict["Data_config"]["split_strategy"],
            stratify_on = config_dict["Data_config"]["stratify_on"],
            stratify_distribution = config_dict["Data_config"]["stratify_distribution"],
        )
    except FileNotFoundError:
        dataset.create_data_split(
            split_strategy = config_dict["Data_config"]["split_strategy"],
            stratify_on = config_dict["Data_config"]["stratify_on"],
            stratify_distribution = config_dict["Data_config"]["stratify_distribution"],
            test_size=0.1,
        )
        dataset.load_data_split(
            split_strategy = config_dict["Data_config"]["split_strategy"],
            stratify_on = config_dict["Data_config"]["stratify_on"],
            stratify_distribution = config_dict["Data_config"]["stratify_distribution"],
        )
        
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
            "Metric name",
            "Val metric",
            "Test metric",
        ]
    )
    
    print(f'\nModel: {config_dict["model"]}\nDataset: {config_dict["dataset"]}\nTask: {config_dict["task"]}\n', flush=True)
    print(f"Number of training samples: {len(dataset.train_set)}", flush=True)
    print(f"Number of validation samples: {len(dataset.validation_set)}", flush=True)
    print(f"Number of test samples: {len(dataset.test_set)}\n", flush=True)
    print(f"Device: {device}", flush=True)

    # Define a dictionary to map model names to their classes and specific kwargs
    model_configurations = {
        "GCN": {
            "class": GCN,
            "kwargs": {"x": "data.pos_abs", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"}
        },
        "GraphSAGE": {
            "class": GraphSAGE,
            "kwargs": {"x": "data.pos_abs", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"}
        },
        "GIN": {
            "class": GIN,
            "kwargs": {"x": "data.pos_abs", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"}
        },
        "GAT": {
            "class": lambda **kwargs: GAT(v2=False, **kwargs),
            "kwargs": {"x": "data.pos_abs", "edge_index": "data.edge_index", "v2": False, "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"}
        },
        "EdgeCNN": {
            "class": EdgeCNN,
            "kwargs": {"x": "data.pos_abs", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"}
        },
        "GraphUNet": {
            "class": GraphUNet,
            "kwargs": {"x": "torch.cat((data.x, data.pos_abs), dim=1)", "edge_index": "data.edge_index", "batch": "data.batch"}
        },
        "PMLP": {
            "class": PMLP,
            "kwargs": {"x": "data.x", "edge_index": "data.edge_index"}
        },
        "MLP": {
            "class": lambda **kwargs: SimpleMLP(device=device, **kwargs),
            "kwargs": {"x": "torch.cat((data.x, data.pos_abs), dim=1)", "edge_index": "data.edge_index", "device": "device"}
        },
    }

    # Define a dictionary for tasks
    task_functions = {
        "AtomClassification": {
            "task_function": atom_classification,
            "loss_function": nn.CrossEntropyLoss(),
            "metric_function": MulticlassF1Score(num_classes=118, average='weighted').compute,
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "SpacegroupClassification": {
            "task_function": spacegroup_classification,
            "loss_function": nn.CrossEntropyLoss(),
            "metric_function": MulticlassF1Score(num_classes=230, average='weighted').compute,
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "CrystalSystemClassification": {
            "task_function": crystal_system_classification,
            "loss_function": nn.CrossEntropyLoss(),
            "metric_function": MulticlassF1Score(num_classes=7, average='weighted').compute,
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        # Add other tasks
    }

    # Define atom classification task
    def atom_classification(data, model, secondary, model_kwargs, device):
        pred = secondary(model.forward(**model_kwargs))
        truth = data.x[:,0]
        return pred, truth
    
    # Define space group classification task
    def space_group_classification(data, model, secondary, model_kwargs, device):
        pred = secondary(model.forward(**model_kwargs))
        truth = torch.tensor(data.y['space_group_number'], device=device)
        return pred, truth

    # Define crystal system classification task
    def crystal_system_classification(data, model, secondary, model_kwargs, device):
        pred = secondary(model.forward(**model_kwargs))
        truth = torch.tensor(data.y['crystal_system_number'], device=device)
        return pred, truth

    def pos_abs_regression():
        pass

    def edge_attr_regression():
        pass

    def saxs_regression():
        pass

    def xrd_regression():
        pass

    def xPDF_regression():
        pass

    def pos_abs_from_saxs():
        pass

    def pos_abs_from_xrd():
        pass
    
    def pos_abs_from_xPDF():
        pass

    def unit_cell_pos_frac_from_saxs():
        pass

    def unit_cell_pos_frac_from_xrd():
        pass

    def unit_cell_pos_frac_from_xPDF():
        pass

    # Seed loop
    for seed_idx, seed in enumerate(config['Train_config']['seeds']):

        # Seed
        seed_everything(seed)
        print(f'\nSeed: {seed}\n', flush=True)

        # Model config
        model_configuration = model_configurations.get(config_dict['model'])
        if model_configuration is None:
            raise ValueError("Model not supported")

        # model class and kwargs
        model_class = model_configuration['class']
        model_kwargs = model_configuration['kwargs']

        # define models
        model = model_class(**config_dict['Model_config']).to(device=device)
        secondary = Secondary(**config_dict['Secondary_config']).to(device=device)

        # Task configuration
        task_configuration = task_function.get(config_dict['task'])
        if task_configuration is None:
            raise NotImplementedError("Task not implemented")

        # Get task function, loss and metric and improved
        task_function = task_configuration['task_function']
        loss_function = task_configuration['loss_function']
        metric_function = task_configuration['metric_function']
        metric_name = task_configuration['metric_name']
        improved_function = task_configuration['improved_function']

        # Create optimizer
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(secondary.parameters()),
            lr=config_dict["Train_config"]["learning_rate"],
        )

        # Define TensorBoard writer
        save_dir = f"{config_dict['log_dir']}/{config_dict['dataset']}/{config_dict['task']}/{config_dict['model']}/seed{seed}"
        writer = SummaryWriter(save_dir)
        
        # Set training time (in seconds)
        max_training_time = config_dict["Train_config"]["train_time"]
        start_time = time.time()
        
        # Patience
        max_patience = config_dict["Train_config"]["max_patience"]
        patience = 0

        # Error
        best_error = None
        
        # Epoch loop
        for epoch in range(config_dict['Train_config']['epochs']):

            # Stop training if max training time is exceeded
            if time.time() - start_time > max_training_time:
                break
            
            # Patience
            if patience >= max_patience:
                print("Max Patience reached, quitting...", flush=True)
                break

            # Training loop
            model.train()
            train_loss = 0
            for data in train_loader:

                # Send to device
                data = data.to(device)

                # Perform forward pass
                pred, truth = task_function(data, model, secondary, model_kwargs, device)
                loss = loss_function(pred, truth)

                # Back prop. loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Train loss
            train_loss = train_loss / len(train_loader)

            # Validation loop
            model.eval()
            val_error = 0
            for data in val_loader:
                
                # Send to device
                data = data.to(device)

                # Perform forward pass
                with torch.no_grad():
                    pred, truth = task_function(data, model, secondary, model_kwargs, device)
                    metric = metric_function(pred, truth)

                # Aggregate errors
                val_error += metric.item()

            val_error = val_error / len(val_loader)

            # Save model if improved
            if improved_function(best_error, val_error):
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
                best_error = val_error
            else:
                patience += 1

            # Save latest model?
            if config_dict['save_latest']:
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
                    f"{save_dir}/lastest.pt",
                )
    
            # Save metrics
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar(f"{metric_name}/val", val_error, epoch)

            # Print checkpoint
            print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val {metric_name}: {val_f1:.4f}', flush=True)

        # Stop time
        stop_time = time.time()
    
        # Load best model
        checkpoint = torch.load(f"{save_dir}/best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]

        # Test loop
        model.eval()
        test_error = 0
        for data in test_loader:

            # Send to device
            data = data.to(device)

            # Perform forward pass
            with torch.no_grad():
                pred, truth = task_function(data, model, secondary, model_kwargs, device)
                metric = metric_function(pred, truth)

            # Aggregate errors
            test_error += metric.item()
        
        # Final test error
        test_error = test_error / len(test_loader)

        # Save metrics
        writer.add_scalar(f"{metric_name}/test", test_error, epoch)
            
        # Print checkpoint
        print(f"Test {metric_name}: {test_error:.4f}", flush=True)
        
        # Close TensorBoard writer
        writer.close()
    
        # Add results to dataframe
        results_df.loc[seed_idx] = [
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
            metric_name,
            val_error.detach().cpu().numpy(),
            test_error.detach().cpu().numpy(),
        ]
    
    # Save results to csv file
    results_df.to_csv(f"{save_dir}/../results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("--config_folder", type=str, help="Path to folder containing configuration files")
    parser.add_argument("--config_index", type=str, help="Index for cluster array")
    args = parser.parse_args()
    run_benchmarking(args)
    
