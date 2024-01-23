import yaml
from pathlib import Path

# Top level configuration
# Models to test
models = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'EdgeCNN', 'GraphUNet', 'PMLP']#, 'MLP']
models = ['EdgeCNN']
#models = ['GCN']
#models = ['MLP']
# Path to datasets
datasetRoot = './Dataset/'
#datasetNames = ['Simulated_rmax60_v4', 'COD_subset_v4']
datasetNames = ['Simulated_rmax60_v4']
datasetNames = ['COD_subset_v4']
# Directory to save results in
saveDir = './Results_COD_TEST/'
# tasks to test
tasks = ['AtomClassification', 'PositionRegression', 'DistanceRegression', 'CrystalSystemClassification', 'SpacegroupClassification', 'SAXSRegression', 'XRDRegression', 'xPDFRegression', 'PositionRegressionSAXS', 'PositionRegressionXRD', 'PositionRegressionxPDF']
tasks = ['SpacegroupClassification', 'XRDRegression', 'xPDFRegression', 'SAXSRegression']
tasks = ['AtomClassification']

# Training configuration
# Learning rate to use
learning_rate = 0.01
# Batch size to use
batch_size = 16
# Number of epochs to train for
epochs = 1000
# Max training time in seconds
train_time = 3600 # 3600 seconds = 1 hour
# Seeds to use
seeds = [42, 43, 44]
# Patience
max_patience = 10
# save latest model
save_latest_model = False

config_folder = 'benchmark_config_COD'

# Create save directory if it doesn't exist
if not Path(config_folder).exists():
    Path(config_folder).mkdir()

# Create config files
for datasetName in datasetNames:
    for model in models:
        # Model configuration
        # Number of GNN layers to use
        num_layers = 2
        num_layers_name = 'num_layers'
        # Number of hidden features to use
        hidden_features = 32

        if model == 'GAT':
            hidden_features = 64
        elif model == 'EdgeCNN':
            num_layers = 4
            hidden_features = 64
        elif model == 'GraphUNet':
            num_layers = 2
            num_layers_name = 'depth'
        elif model == 'MLP':
            num_layers = 4
            hidden_features = 512

        for task in tasks:
            # Number of input features to use
            if task == 'AtomClassification':
                input_features = 3
            elif task == 'PositionRegression':
                input_features = 4
            elif task == 'PositionRegressionSAXS':
                input_features = 300
            elif task == 'PositionRegressionXRD':
                input_features = 580
            elif task == 'PositionRegressionxPDF':
                input_features = 6000
            else:
                input_features = 7

            # Number of output features to use
            if task == 'AtomClassification':
                output_features = 118
            elif task == 'PositionRegression':
                output_features = 3
            elif task in ['PositionRegressionSAXS', 'PositionRegressionXRD', 'PositionRegressionxPDF']:
                output_features = 3 * 200 # MAX SIZE
            else:
                output_features = 64
            
            # Secondary FF models
            if task == 'SpacegroupClassification':
                sec_output_features = 230
            elif task == 'CrystalSystemClassification':
                sec_output_features = 7
            elif task == 'SAXSRegression':
                sec_output_features = 300
            elif task == 'XRDRegression':
                sec_output_features = 580
            elif task == 'xPDFRegression':
                sec_output_features = 6000
            else:
                sec_output_features = 1

            sec_hidden_features = (output_features + sec_output_features) // 2
            # Create config
            config = {
                'dataset': datasetName,
                'root': datasetRoot,
                'log_dir': saveDir,
                'model': model,
                'task': task,
                'save_latest_model': save_latest_model,
                'Model_config': {
                    num_layers_name: num_layers,
                    'in_channels': input_features,
                    'hidden_channels': hidden_features,
                    'out_channels': output_features,
                },
                'Secondary_config': {
                    'in_channels': output_features * 3,
                    'hidden_channels': sec_hidden_features,
                    'out_channels': sec_output_features,
                },
                'Train_config': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'train_time': train_time,
                    'seeds': seeds,
                    'max_patience': max_patience,
                }
            }

            # Create config file path
            configPath = Path(f'./{config_folder}/config_{datasetName}_{task}_{model}.yaml')
            # Create config file
            with open(configPath, 'w') as file:
                documents = yaml.dump(config, file)
