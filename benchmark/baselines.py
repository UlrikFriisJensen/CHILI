from typing import Optional
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
from torcheval.metrics.functional import multiclass_f1_score
import numpy as np
from dataset_class import CHILI

def run_baselines(root: str,
                  dataset: str,
                  split_strategy: str,
                  stratify_on: Optional[str],
                  stratify_distribution: Optional[str]) -> None:
    """
    Run baseline benchmarks for classification and regression tasks.

    Args:
        root (str): Root directory for the dataset.
        dataset (str): Name of the dataset.
        split_strategy (str): Split strategy for train/test split.
        stratify_on (Optional[str]): Attribute to stratify on.
        stratify_distribution (Optional[str]): Distribution to stratify on.
    """

    # Set device
    device = 'cpu'

    # Create dataset
    dataset = CHILI(root=root, dataset=dataset)

    # Load / Create data split
    if split_strategy == "random":
        test_size = 0.1
    try:
        dataset.load_data_split(
            split_strategy=split_strategy,
            stratify_on=stratify_on,
            stratify_distribution=stratify_distribution,
        )
    except FileNotFoundError:
        dataset.create_data_split(
            split_strategy=split_strategy,
            stratify_on=stratify_on,
            stratify_distribution=stratify_distribution,
            test_size=0.1,
        )
        dataset.load_data_split(
            split_strategy=split_strategy,
            stratify_on=stratify_on,
            stratify_distribution=stratify_distribution,
        )

    test_loader = DataLoader(
        dataset.test_set,
        batch_size=16,
        shuffle=False,
    )

    def position_MAE(pred_xyz, true_xyz):
        """
        Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
        """
        return torch.mean(
            torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction="none"), dim=1)),
            dim=0,
        )

    def random_class(x, num_classes):
        pred = torch.zeros(len(x), num_classes)
        pred_ints = torch.randint(num_classes, size=(len(x),))
        for i, p in enumerate(pred_ints):
            pred[i][p] = 1.0
        return pred

    def most_frequent_class(x, num_classes, mfc):
        pred = torch.zeros(len(x), num_classes)
        pred_ints = torch.full(fill_value=mfc, size=(len(x),), dtype=torch.long)
        for i, p in enumerate(pred_ints):
            pred[i][p-1] = 1.0
        return pred

    def mean_graph(x, numbers):
        pred = torch.zeros_like(x)
        l_idx = 0
        for n in numbers:
            pred[l_idx:l_idx+n] = x[l_idx:l_idx+n].mean(dim=0)
            l_idx += n
        return pred, x

    def mean_scattering(x, numbers):
        pred = torch.zeros_like(x)
        truth = torch.zeros_like(x)
        for i, sct in enumerate(x):
            sct_min = torch.min(sct, dim=-1, keepdim=True)[0]
            sct_max = torch.max(sct, dim=-1, keepdim=True)[0]
            sct = (sct - sct_min) / (sct_max - sct_min)
            pred[i] = torch.full(fill_value=torch.mean(sct), size=(pred.shape[-1],))
            truth[i] = sct
        return pred, truth

    # Define a dictionary for tasks
    classification_task_configurations = {
        "AtomClassification": {
            "metric_function": lambda x, y: multiclass_f1_score(x, y.long() - 1, num_classes=118, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "x": "data.x[:,0]",
            "num_classes": 118,
            "most_frequent_class": 8,
        },
        "CrystalSystemClassification": {
            "metric_function": lambda x, y: multiclass_f1_score(x, y.long() - 1, num_classes=7, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "x": "data.y['crystal_system_number']",
            "num_classes": 7,
            "most_frequent_class": 7,
        },
        "SpacegroupClassification": {
            "metric_function": lambda x, y: multiclass_f1_score(x, y.long() - 1, num_classes=230, average='weighted'),
            "metric_name": 'WeightedF1Score',
            "x": "torch.tensor(data.y['space_group_number'])",
            "num_classes": 230,
            "most_frequent_class": 225,
        },
    }
    regression_task_configurations = {
        "PositionRegression": {
            "mean_function": mean_graph,
            "metric_function": position_MAE,
            "metric_name": 'PositionMAE',
            "x": "data.pos_abs",
            "numbers": "data.y['n_atoms']",
        },
        "DistanceRegression": {
            "mean_function": mean_graph,
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "x": "data.edge_attr",
            "numbers": "data.y['n_bonds']",
        },
        "SAXSRegression": {
            "mean_function": mean_scattering,
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "x": "data.y['saxs'][1::2,:]",
            "numbers": "None",
        },
        "XRDRegression": {
            "mean_function": mean_scattering,
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "x": "data.y['xrd'][1::2,:]",
            "numbers": "None",
        },
        "xPDFRegression": {
            "mean_function": mean_scattering,
            "metric_function": nn.MSELoss(),
            "metric_name": 'MSE',
            "x": "data.y['xPDF'][1::2,:]",
            "numbers": "None",
        },
    }

    def evaluate_classification_task(task: str, fields: dict) -> None:
        """
        Evaluate classification task.

        Args:
            task (str): Task name.
            fields (dict): Dictionary containing task configurations.
        """
        metric_function = fields['metric_function']
        task_error_rc = []
        task_error_mfc = []

        for i, seed in enumerate([42, 43, 44]):
            seed_everything(seed)
            error_rc = 0
            error_mfc = 0

            for data in test_loader:
                truth = eval(fields['x'])
                num_classes = fields['num_classes']
                mfc = fields['most_frequent_class']

                pred_rc = random_class(truth, num_classes)
                pred_mfc = most_frequent_class(truth, num_classes, mfc)

                metric_rc = metric_function(pred_rc, truth)
                metric_mfc = metric_function(pred_mfc, truth)

                error_rc += metric_rc.item()
                error_mfc += metric_mfc.item()

            task_error_rc.append(error_rc / len(test_loader))
            task_error_mfc.append(error_mfc / len(test_loader))

        task_error_rc_mean = np.mean(task_error_rc)
        task_error_rc_std = np.std(task_error_rc)
        task_error_mfc_mean = np.mean(task_error_mfc)
        task_error_mfc_std = np.std(task_error_mfc)

        print(task + ':')
        print(f'RC Error: {task_error_rc_mean:1.3f} +- {task_error_rc_std:1.3f}')
        print(f'MFC Error: {task_error_mfc_mean:1.3f} +- {task_error_mfc_std:1.3f}')
        print()

    def evaluate_regression_task(task: str, fields: dict) -> None:
        """
        Evaluate regression task.

        Args:
            task (str): Task name.
            fields (dict): Dictionary containing task configurations.
        """
        metric_function = fields['metric_function']
        mean_function = fields['mean_function']
        task_error = []

        for i, seed in enumerate([42, 43, 44]):
            seed_everything(seed)
            error = 0

            for data in test_loader:
                truth = eval(fields['x'])
                numbers = eval(fields['numbers'])

                pred, truth = mean_function(truth, numbers)

                metric = metric_function(pred, truth)
                error += metric.item()

            task_error.append(error / len(test_loader))

        task_error_mean = np.mean(task_error)
        task_error_std = np.std(task_error)

        print(task + ':')
        print(f'Error: {task_error_mean:1.3f} +- {task_error_std:1.3f}')
        print()

    for task, fields in classification_task_configurations.items():
        evaluate_classification_task(task, fields)

    for task, fields in regression_task_configurations.items():
        evaluate_regression_task(task, fields)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline benchmarking script")
    parser.add_argument("-r", "--dataset_root", type=str, required=True)
    parser.add_argument("-n", "--dataset_name", type=str, required=True)
    parser.add_argument("-ss", "--split_strategy", type=str, default='random')
    parser.add_argument("-so", "--stratify_on", type=Optional[str], default=None)
    parser.add_argument("-sd", "--stratify_distribution", type=Optional[str], default=None)
    args = parser.parse_args()
    run_baselines(args.dataset_root, args.dataset_name, args.split_strategy, args.stratify_on, args.stratify_distribution)
