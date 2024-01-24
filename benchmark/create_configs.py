import yaml
import os

## Top level configuration

# Models to test
models = ['GCN', 'GAT', 'GIN', 'GraphSAGE', 'EdgeCNN', 'GraphUNet', 'PMLP', 'MLP']

# Path to datasets
dataset_dir = 'dataset'
dataset_names = ['COD','SIM']

# Data distribution
split_strategy = ['stratified', 'random']
stratify_on = ['Crystal system (Number)', None]
stratify_distribution = ['equal', None]

# Directory to save results in
save_dir = 'results'

# tasks to test
tasks = ['AtomClassification', 'PositionRegression', 'DistanceRegression', 'CrystalSystemClassification', 'SpacegroupClassification',
         'SAXSRegression', 'XRDRegression', 'xPDFRegression', 
         'PositionRegressionSAXS', 'PositionRegressionXRD', 'PositionRegressionxPDF']

## Training configuration

learning_rate = 0.01
batch_size = 16
max_epochs = 1000
training_time_seconds = 3600
seeds = [42, 43, 44]
max_patience = 10 # Epochs
save_latest_model = False

## Config dir
config_dir = 'configs'
if not os.path.exists(config_dir):
    os.mkdir(config_dir)

## Create config files
for dataset_name, strategy, on, distribution in zip(dataset_names, split_strategy, stratify_on, stratify_distribution):

    config_dataset_dir = os.path.join(config_dir, f'{dataset_name}_{strategy}_{on}_{distribution}')
    if not os.path.exists(config_dataset_dir):
        os.mkdir(config_dataset_dir)

    for model in models:
        
        # Default
        num_layers = 2
        num_layers_name = 'num_layers'
        hidden_channels = 32

        # GAT
        if model == 'GAT':
            hidden_channels = 64

        elif model == 'EdgeCNN':
            num_layers = 4
            hidden_channels = 64

        elif model == 'GraphUNet':
            num_layers_name = 'depth'

        elif model == 'MLP':
            num_layers = 4
            hidden_channels = 128

        for task in tasks:

            if task == 'AtomClassification':
                input_channels = 3
                output_channels = 118
            elif task == 'PositionRegression':
                input_channels = 4
                output_channels = 3
            elif task == 'PositionRegressionSAXS':
                input_channels = 300
                output_channels = 600 # 200 atoms
            elif task == 'PositionRegressionXRD':
                input_channels = 580
                output_channels = 600 # 200 atoms
            elif task == 'PositionRegressionxPDF':
                input_channels = 6000
                output_channels = 600 # 200 atoms
            else:
                input_channels = 7
                output_channels = 64

            if task == 'SpacegroupClassification':
                sec_output_channels = 230
            elif task == 'CrystalSystemClassification':
                sec_output_channels = 7
            elif task == 'SAXSRegression':
                sec_output_channels = 300
            elif task == 'XRDRegression':
                sec_output_channels = 580
            elif task == 'xPDFRegression':
                sec_output_channels = 6000
            else:
                sec_output_channels = 1

            sec_hidden_channels = (output_channels + sec_output_channels) // 2
            # Create config
            config = {
                'dataset': dataset_name,
                'root': dataset_dir,
                'log_dir': os.path.join(save_dir, dataset_name),
                'model': model,
                'task': task,
                'save_latest_model': save_latest_model,
                'Model_config': {
                    num_layers_name: num_layers,
                    'in_channels': input_channels,
                    'hidden_channels': hidden_channels,
                    'out_channels': output_channels,
                },
                'Secondary_config': {
                    'in_channels': output_channels * 3,
                    'hidden_channels': sec_hidden_channels,
                    'out_channels': sec_output_channels,
                },
                'Train_config': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': max_epochs,
                    'train_time': training_time_seconds,
                    'seeds': seeds,
                    'max_patience': max_patience,
                },
                'Data_config': {
                    'split_strategy': strategy,
                    'stratify_on': on,
                    'stratify_distribution': distribution,
                },
            }

            # Save config file
            with open(os.path.join(config_dataset_dir, f'{task}_{model}.yaml'), 'w') as file:
                documents = yaml.dump(config, file)
