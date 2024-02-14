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
from torch_geometric.seed import seed_everything

from torcheval.metrics.functional import multiclass_f1_score
from torch.nn.functional import cross_entropy

import pandas as pd
from glob import glob

from dataset_class import CHILI
from modules import MLP, Secondary

def run_benchmarking(args):

    # Read configuration file
    config_files = sorted(glob(os.path.join(args.config_folder, "*.yaml")))
    config_path = config_files[int(args.config_index)]
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    
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
            if (len(data.y["unit_cell_pos_frac"]) <= config_dict["Model_config"]["out_channels"] // 3):
                filtered_idx.append(idx)
    
        dataset.train_set = Subset(dataset, filtered_idx)
    
        # Validation set
        filtered_idx = []
        for idx in dataset.validation_set.indices:
            data = dataset[idx]
            if (len(data.y["unit_cell_pos_frac"]) <= config_dict["Model_config"]["out_channels"] // 3):
                filtered_idx.append(idx)
    
        dataset.validation_set = Subset(dataset, filtered_idx)
    
        # Test set
        filtered_idx = []
        for idx in dataset.test_set.indices:
            data = dataset[idx]
            if (len(data.y["unit_cell_pos_frac"]) <= config_dict["Model_config"]["out_channels"] // 3):
                filtered_idx.append(idx)
    
        dataset.test_set = Subset(dataset, filtered_idx)
        
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

    # Default model configs
    default_model_configurations = {
        "GCN": {
            "class": GCN,
            "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
            "skip_training": False,
        },
        "GraphSAGE": {
            "class": GraphSAGE,
            "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
            "skip_training": False,
        },
        "GIN": {
            "class": GIN,
            "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
            "skip_training": False,
        },
        "GAT": {
            "class": GAT,
            "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
            "skip_training": False,
        },
        "EdgeCNN": {
            "class": EdgeCNN,
            "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
            "skip_training": False,
        },
        "GraphUNet": {
            "class": GraphUNet,
            "kwargs": {"x": "None", "edge_index": "data.edge_index", "batch": "data.batch"},
            "skip_training": False,
        },
        "PMLP": {
            "class": PMLP,
            "kwargs": {"x": "None", "edge_index": "data.edge_index"},
            "skip_training": False,
        },
        "MLP": {
            "class": MLP,
            "kwargs": {"x": "None", "batch": "data.batch"},
            "skip_training": False,
        },
    }

    def position_MAE(
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
    
    def atom_classification(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = data.pos_abs
        pred = model.forward(**evaluated_kwargs)
        truth = data.x[:,0].long()
        return pred, truth
    
    def crystal_system_classification(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
        pred = secondary(model.forward(**evaluated_kwargs), batch=data.batch)
        truth = data.y['crystal_system_number']
        return pred, truth
    
    def space_group_classification(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
        pred = secondary(model.forward(**evaluated_kwargs), batch=data.batch)
        truth = torch.tensor(data.y['space_group_number'], device=device)
        return pred, truth

    def pos_abs_regression(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = data.x
        pred = model.forward(**evaluated_kwargs)
        truth = data.pos_abs
        return pred, truth

    def edge_attr_regression(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
        evaluated_kwargs['edge_attr'] = None
        evaluated_kwargs['edge_weight'] = None
        pred = model.forward(**evaluated_kwargs)
        pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
        truth = data.edge_attr
        return pred, truth

    def saxs_regression(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
        pred = secondary(model.forward(**evaluated_kwargs), batch=data.batch)
        truth = data.y['saxs'][1::2, :]
        truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
        truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
        truth = (truth - truth_min) / (truth_max - truth_min)
        return pred, truth

    def xrd_regression(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
        pred = secondary(model.forward(**evaluated_kwargs), batch=data.batch)
        truth = data.y['xrd'][1::2, :]
        truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
        truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
        truth = (truth - truth_min) / (truth_max - truth_min)
        return pred, truth

    def xPDF_regression(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
        pred = secondary(model.forward(**evaluated_kwargs), batch=data.batch)
        truth = data.y['xPDF'][1::2, :]
        truth_min = torch.min(truth, dim=-1, keepdim=True)[0]
        truth_max = torch.max(truth, dim=-1, keepdim=True)[0]
        truth = (truth - truth_min) / (truth_max - truth_min)
        return pred, truth
    
    def pos_abs_padded(data, config_dict, device):
        batch_size = torch.max(data.batch) + 1
        truth = torch.zeros((batch_size, config_dict['Model_config']['out_channels'])).to(device=device)
        for i, x in enumerate(unbatch(data.pos_abs, data.batch)):

            # Sort according to norm
            norms = torch.norm(x, p = 2, dim = -1)
            indices = torch.sort(norms, descending=False, dim=0)[1]
            x = x[indices]

            # Padding
            padding_size = config_dict['Model_config']['out_channels'] // 3 - x.size(0)
            if padding_size > 0:
                padding = torch.full((padding_size, x.size(1)), 100, dtype = x.dtype).to(device=device)
                x = torch.cat([x, padding], dim = 0)

            # Append
            truth[i] = x.flatten()

        return truth

    def pos_abs_from_saxs(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['saxs'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)

        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = pos_abs_padded(data, config_dict, device)
        
        return pred[truth < 100], truth[truth < 100]

    def pos_abs_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xrd'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)

        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = pos_abs_padded(data, config_dict, device)
        
        return pred[truth < 100], truth[truth < 100]
    
    def pos_abs_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xPDF'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)

        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = pos_abs_padded(data, config_dict, device)
        
        return pred[truth < 100], truth[truth < 100]
    
    def unit_cell_pos_frac_padded(data, config_dict, device):
        batch_size = torch.max(data.batch) + 1
        truth = torch.zeros((batch_size, config_dict['Model_config']['out_channels'])).to(device=device)
        l_idx = 0
        for i, size in enumerate(data.y['unit_cell_n_atoms']):

            # Extract the unit cell
            x = data.y['unit_cell_pos_frac'][l_idx : l_idx + size]
            l_idx += size

            # Sort according to norm
            norms = torch.norm(x, p = 2, dim = -1)
            indices = torch.sort(norms, descending=False, dim=0)[1]
            x = x[indices]

            # Padding
            padding_size = config_dict['Model_config']['out_channels'] // 3 - x.size(0)
            if padding_size > 0:
                padding = torch.full((padding_size, x.size(1)), 100, dtype = x.dtype).to(device=device)
                x = torch.cat([x, padding], dim = 0)

            # Append
            truth[i] = x.flatten()

        return truth
    
    def unit_cell_pos_frac_from_saxs(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['saxs'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)

        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = unit_cell_pos_frac_padded(data, config_dict, device)
        
        return pred[truth < 100], truth[truth < 100]
    
    def unit_cell_pos_frac_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xrd'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)

        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = unit_cell_pos_frac_padded(data, config_dict, device)
        
        return pred[truth < 100], truth[truth < 100]
    
    def unit_cell_pos_frac_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xPDF'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)

        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = unit_cell_pos_frac_padded(data, config_dict, device)
        
        return pred[truth < 100], truth[truth < 100]

    def cell_params_from_saxs(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['saxs'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = data.y['cell_params'].reshape(torch.max(data.batch)+1, 6)
        return pred, truth

    def cell_params_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xrd'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = data.y['cell_params'].reshape(torch.max(data.batch)+1, 6)
        return pred, truth
    
    def cell_params_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xPDF'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = data.y['cell_params'].reshape(torch.max(data.batch)+1, 6)
        return pred, truth

    def crystal_system_from_saxs(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['saxs'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = torch.tensor(data.y['crystal_system_number'], device=device)
        return pred, truth
    
    def crystal_system_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xrd'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = torch.tensor(data.y['crystal_system_number'], device=device)
        return pred, truth
    
    def crystal_system_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xPDF'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = torch.tensor(data.y['crystal_system_number'], device=device)
        return pred, truth

    def space_group_from_saxs(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['saxs'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = torch.tensor(data.y['space_group_number'], device=device)
        return pred, truth
    
    def space_group_from_xrd(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xrd'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = torch.tensor(data.y['space_group_number'], device=device)
        return pred, truth
    
    def space_group_from_xPDF(data, model, secondary, model_kwargs, device, config_dict):
        evaluated_kwargs = {}
        for key, value in model_kwargs.items():
            evaluated_kwargs[key] = eval(value)
        sct = data.y['xPDF'][1::2, :]
        sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
        sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
        sct = (sct - sct_min) / (sct_max - sct_min)
        
        evaluated_kwargs['x'] = sct
        pred = model(**evaluated_kwargs)
        truth = torch.tensor(data.y['space_group_number'], device=device)
        return pred, truth

    # Define a dictionary for tasks
    task_configurations = {
        "AtomClassification": {
            "task_function": atom_classification,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=118, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "CrystalSystemClassification": {
            "task_function": crystal_system_classification,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=7, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "SpacegroupClassification": {
            "task_function": space_group_classification,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=230, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "PositionRegression": {
            "task_function": pos_abs_regression,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": position_MAE,
            "metric_name": 'PositionMAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "DistanceRegression": {
            "task_function": edge_attr_regression,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "SAXSRegression": {
            "task_function": saxs_regression,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "XRDRegression": {
            "task_function": xrd_regression,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "xPDFRegression": {
            "task_function": xPDF_regression,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "AbsPositionRegressionSAXS": {
            "task_function": pos_abs_from_saxs,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "AbsPositionRegressionXRD": {
            "task_function": pos_abs_from_xrd,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "AbsPositionRegressionxPDF": {
            "task_function": pos_abs_from_xPDF,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "UnitCellPositionRegressionSAXS": {
            "task_function": unit_cell_pos_frac_from_saxs,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "UnitCellPositionRegressionXRD": {
            "task_function": unit_cell_pos_frac_from_xrd,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "UnitCellPositionRegressionxPDF": {
            "task_function": unit_cell_pos_frac_from_xPDF,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "CellParamsRegressionSAXS": {
            "task_function": cell_params_from_saxs,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "CellParamsRegressionXRD": {
            "task_function": cell_params_from_xrd,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "CellParamsRegressionxPDF": {
            "task_function": cell_params_from_xPDF,
            "loss_function": nn.SmoothL1Loss(),
            "metric_function": nn.L1Loss(),
            "metric_name": 'MAE',
            "improved_function": lambda best, new: new < best if best is not None else True,
        },
        "CrystalSystemClassificationSAXS": {
            "task_function": crystal_system_from_saxs,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=7, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "CrystalSystemClassificationXRD": {
            "task_function": crystal_system_from_xrd,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=7, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "CrystalSystemClassificationxPDF": {
            "task_function": crystal_system_from_xPDF,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=7, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "SpacegroupClassificationSAXS": {
            "task_function": space_group_from_saxs,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=230, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "SpacegroupClassificationXRD": {
            "task_function": space_group_from_xrd,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=230, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        "SpacegroupClassificationxPDF": {
            "task_function": space_group_from_xPDF,
            "loss_function": lambda x,y: cross_entropy(x, y.long() - 1),
            "metric_function": lambda x,y: multiclass_f1_score(x, y.long() - 1, num_classes=230, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "improved_function": lambda best, new: new > best if best is not None else True,
        },
        # Add more tasks here...
    }


    # Seed loop
    for seed_idx, seed in enumerate(config_dict['Train_config']['seeds']):

        # Seed
        seed_everything(seed)
        print(f'\nSeed: {seed}\n', flush=True)

        # Model config
        model_configuration = default_model_configurations.get(config_dict['model'])
        if model_configuration is None:
            raise ValueError("Model not supported")

        # model class and kwargs
        model_class = model_configuration['class']
        model_kwargs = model_configuration['kwargs']

        # define models
        model = model_class(**config_dict['Model_config']).to(device=device)
        secondary = Secondary(**config_dict['Secondary_config']).to(device=device)

        # Task configuration
        task_configuration = task_configurations.get(config_dict['task'])
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
            
            # Skip training and validation if baseline
            if model_configuration['skip_training']:
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
                train_loss = 0
                val_error = 0
                break

            # Stop training if max training time is exceeded
            if time.time() - start_time > max_training_time:
                break
            
            # Patience
            if patience >= max_patience:
                print("Max Patience reached, quitting...", flush=True)
                break

            # Train loop
            model.train()
            train_loss = 0
            for data in train_loader:

                # Send to device
                data = data.to(device)

                # Perform forward pass
                pred, truth = task_function(data, model, secondary, model_kwargs, device, config_dict)
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
                    pred, truth = task_function(data, model, secondary, model_kwargs, device, config_dict)
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
                patience = 0
            else:
                patience += 1
            

            # Save latest model?
            if config_dict['save_latest_model']:
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
            print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val {metric_name}: {val_error:.4f}', flush=True)

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
                pred, truth = task_function(data, model, secondary, model_kwargs, device, config_dict)
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
            val_error,
            test_error,
        ]
    
    # Save results to csv file
    results_df.to_csv(f"{save_dir}/../results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("-f", "--config_folder", type=str, help="Path to folder containing configuration files")
    parser.add_argument("-i", "--config_index", type=str, help="Index for cluster array")
    args = parser.parse_args()
    run_benchmarking(args)
