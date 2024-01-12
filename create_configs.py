import yaml
from pathlib import Path

# Top level configuration
# Models to test
models = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'EdgeCNN', 'GraphUNet', 'PMLP']
# Path to datasets
datasetRoot = './Dataset/'
datasetNames = ['Simulated_rmax60_v4', 'COD_subset_v4']
# Directory to save results in
saveDir = './Results/'
# tasks to test
tasks = ['AtomClassification', 'PositionRegression', 'DistanceRegression', 'CrystalSystemClassification', 'SpacegroupClassification', 'SAXSRegression', 'XRDRegression', 'xPDFRegression']

# Model configuration
# Number of GNN layers to use
num_layers = 2
# Number of hidden features to use
hidden_features = 32


# Training configuration
# Learning rate to use
learning_rate = 0.01
# Batch size to use
batch_size = 10
# Number of epochs to train for
epochs = 2
# Max training time in seconds
train_time = 3600 # 3600 seconds = 1 hour
# Seeds to use
seeds = [42, 43, 44]

# Create config files
for datasetName in datasetNames:
    for model in models:
        for task in tasks:
            # Number of input features to use
            if task == 'AtomClassification':
                input_features = 3
            elif task == 'PositionRegression':
                input_features = 4
            else:
                input_features = 7

            # Number of output features to use
            if task == 'AtomClassification':
                output_features = 118
            elif task == 'PositionRegression':
                output_features = 3
            else:
                output_features = 64
            # Create config
            config = {
                'dataset': datasetName,
                'root': datasetRoot,
                'log_dir': saveDir,
                'model': model,
                'task': task,
                'Model_config': {
                    'num_layers': num_layers,
                    'in_channels': input_features,
                    'hidden_channels': hidden_features,
                    'out_channels': output_features,
                },
                'Train_config': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'train_time': train_time,
                    'seeds': seeds
                }
            }
            # Create config file path
            configPath = Path(f'./benchmark_configs/config_{task}_{model}.yaml')
            # Create config file
            with open(configPath, 'w') as file:
                documents = yaml.dump(config, file)