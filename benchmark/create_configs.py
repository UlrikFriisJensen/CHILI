# %% Imports
import os
import yaml

# %% Top level configuration

# Models to test
models = [
    "GCN",
    "GAT",
    "GIN",
    "GraphSAGE",
    "EdgeCNN",
    "GraphUNet",
    "PMLP",
    "MLP",
]

# Path to datasets
dataset_dir = "dataset"
dataset_names = ["CHILI-3K", "CHILI-100K"]

# Data distribution
split_strategy = ["stratified", "random"]
stratify_on = ["Crystal system (Number)", None]
stratify_distribution = ["equal", None]

# Directory to save results in
save_dir = "results"

# tasks to test
classification_tasks = [
    "AtomClassification",
    "CrystalSystemClassification",
    "SpacegroupClassification",
]

regression_tasks = [
    "PositionRegression",
    "DistanceRegression",
    "SAXSRegression",
    "XRDRegression",
    "xPDFRegression",
]

generative_position_tasks = [
    "AbsPositionRegressionSAXS",
    "AbsPositionRegressionXRD",
    "AbsPositionRegressionxPDF",
    "UnitCellPositionRegressionSAXS",
    "UnitCellPositionRegressionXRD",
    "UnitCellPositionRegressionxPDF",
]

p2p_classification_tasks = [
    "CrystalSystemClassificationSAXS",
    "CrystalSystemClassificationXRD",
    "CrystalSystemClassificationxPDF",
    "SpacegroupClassificationSAXS",
    "SpacegroupClassificationXRD",
    "SpacegroupClassificationxPDF",
]

p2p_regression_tasks = [
    "CellParamsRegressionSAXS",
    "CellParamsRegressionXRD",
    "CellParamsRegressionxPDF",
]

prediction_tasks = classification_tasks + regression_tasks
generative_tasks = generative_position_tasks + p2p_classification_tasks + p2p_regression_tasks
tasks = prediction_tasks + generative_tasks

# %% Training configuration

learning_rate = 0.001
batch_size = 16
max_epochs = 1
training_time_seconds = 3600
seeds = [42]#, 43, 44]
max_patience = 50  # Epochs
save_latest_model = False

# %% Config dir
config_dir = "configs"
if not os.path.exists(config_dir):
    os.mkdir(config_dir)

# %% Create config files
for dataset_name, strategy, on, distribution in zip(
    dataset_names, split_strategy, stratify_on, stratify_distribution
):
    config_dataset_dir = os.path.join(config_dir, dataset_name)
    if not os.path.exists(config_dataset_dir):
        os.mkdir(config_dataset_dir)

    for model in models:
        # Default
        num_layers = 2
        num_layers_name = "num_layers"
        hidden_channels = 32

        # GAT
        if model == "GAT":
            hidden_channels = 64

        elif model == "EdgeCNN":
            num_layers = 4
            hidden_channels = 64

        elif model == "GraphUNet":
            num_layers_name = "depth"

        elif model == "MLP":
            num_layers = 4
            hidden_channels = 128

        for task in tasks:
            # MLP only for generative tasks
            if (model == "MLP") and (task not in generative_tasks):
                continue

            # Generative tasks only for MLP
            if (model != "MLP") and (task in generative_tasks):
                continue

            if task == "AtomClassification":
                input_channels = 3
                output_channels = 118
                num_classes = 118
                most_frequent_class = 8 # Oxygen
            elif task == "PositionRegression":
                input_channels = 4
                output_channels = 3
            elif task == "AbsPositionRegressionSAXS":
                input_channels = 300
                output_channels = 600  # 200 atoms
            elif task == "AbsPositionRegressionXRD":
                input_channels = 580
                output_channels = 600  # 200 atoms
            elif task == "AbsPositionRegressionxPDF":
                input_channels = 6000
                output_channels = 600  # 200 atoms
            elif task == "UnitCellPositionRegressionSAXS":
                input_channels = 300
                output_channels = 60  # 20 atoms
            elif task == "UnitCellPositionRegressionXRD":
                input_channels = 580
                output_channels = 60  # 20 atoms
            elif task == "UnitCellPositionRegressionxPDF":
                input_channels = 6000
                output_channels = 60  # 20 atoms
            elif task == "CrystalSystemClassificationSAXS":
                input_channels = 300
                output_channels = 7
            elif task == "CrystalSystemClassificationXRD":
                input_channels = 580
                output_channels = 7
            elif task == "CrystalSystemClassificationxPDF":
                input_channels = 6000
                output_channels = 7
            elif task == "SpacegroupClassificationSAXS":
                input_channels = 300
                output_channels = 230
            elif task == "SpacegroupClassificationXRD":
                input_channels = 580
                output_channels = 230
            elif task == "SpacegroupClassificationxPDF":
                input_channels = 6000
                output_channels = 230
            elif task == "CellParamsRegressionSAXS":
                input_channels = 300
                output_channels = 6
            elif task == "CellParamsRegressionXRD":
                input_channels = 580
                output_channels = 6
            elif task == "CellParamsRegressionxPDF":
                input_channels = 6000
                output_channels = 6
            else:
                input_channels = 7
                output_channels = 64
                num_classes = None
                most_frequent_class = None

            if task == "SpacegroupClassification":
                sec_output_channels = 230
                num_classes = 230
                most_frequent_class = 225
            elif task == "CrystalSystemClassification":
                sec_output_channels = 7
                num_classes = 7
                most_frequent_class = 7
            elif task == "SAXSRegression":
                sec_output_channels = 300
            elif task == "XRDRegression":
                sec_output_channels = 580
            elif task == "xPDFRegression":
                sec_output_channels = 6000
            else:
                sec_output_channels = 1

            sec_hidden_channels = (output_channels + sec_output_channels) // 2
            # Create config
            config = {
                "dataset": dataset_name,
                "root": dataset_dir,
                "log_dir": save_dir,
                "model": model,
                "task": task,
                "save_latest_model": save_latest_model,
                "Model_config": {
                    num_layers_name: num_layers,
                    "in_channels": input_channels,
                    "hidden_channels": hidden_channels,
                    "out_channels": output_channels,
                },
                "Secondary_config": {
                    "in_channels": output_channels * 3,
                    "hidden_channels": sec_hidden_channels,
                    "out_channels": sec_output_channels,
                },
                "Train_config": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": max_epochs,
                    "train_time": training_time_seconds,
                    "seeds": seeds,
                    "max_patience": max_patience,
                },
                "Data_config": {
                    "split_strategy": strategy,
                    "stratify_on": on,
                    "stratify_distribution": distribution,
                    "most_frequent_class": most_frequent_class,
                    "num_classes": num_classes,
                },
            }

            # Save config file
            with open(
                os.path.join(config_dataset_dir, f"{task}_{model}.yaml"), "w"
            ) as file:
                documents = yaml.dump(config, file)
