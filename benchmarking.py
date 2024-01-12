# Import libraries
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from Code.datasetClass import InOrgMatDatasets
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, EdgeCNN, GraphUNet, PMLP
from torch_geometric.nn import global_mean_pool, Linear, global_add_pool, global_max_pool
from torch.utils.tensorboard import SummaryWriter
import yaml
import warnings
import time
import pandas as pd
from torch_geometric.seed import seed_everything
from torcheval.metrics import MulticlassF1Score, MeanSquaredError
from glob import glob
import os

warnings.simplefilter(action='ignore')

# Define position mean absolute error function
def position_MAE(pred_xyz, true_xyz):
    '''
    Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
    '''
    return torch.mean(torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction='none'), dim=1)), dim=0)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmarking script')
parser.add_argument('--config_folder', type=str, help='Path to folder containing configuration files')
parser.add_argument('--config_index', type=str, help='Index for cluster array')
args = parser.parse_args()

# Read configuration file
config_files = glob(os.path.join(args.config_folder, '*.yaml'))
config_path = config_files[int(args.config_index)]

with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your dataset
dataset = InOrgMatDatasets(dataset=config_dict['dataset'], root=config_dict['root'])
try:
    dataset.load_data_split(split_strategy='random')
except FileNotFoundError:
    dataset.create_data_split(split_strategy='random', test_size=0.1)
    dataset.load_data_split(split_strategy='random')

# Min-max normalize saxs, xrd and xpdf data
# if config_dict['task'] in ['SAXSRegression', 'XRDRegression', 'xPDFRegression']:
    
    
# Create dataframe for saving results
results_df = pd.DataFrame(columns=['Model', 'Dataset', 'Task', 'Seed', 'Train samples', 'Val samples', 'Test samples', 'Train time', 'Trainable parameters', 'Train loss', 'Val F1-score', 'Test F1-score', 'Val MAE', 'Test MAE'])

print(f'\nModel: {config_dict["model"]}\nDataset: {config_dict["dataset"]}\nTask: {config_dict["task"]}', flush=True)
print('\n', flush=True)
print(f'Number of training samples: {len(dataset.train_set)}', flush=True)
print(f'Number of validation samples: {len(dataset.validation_set)}', flush=True)
print(f'Number of test samples: {len(dataset.test_set)}', flush=True)
print('\n', flush=True)
print(f'Device: {device}', flush=True)

# Train model for each seed
for i, seed in enumerate(config_dict['Train_config']['seeds']):
    # Set seed
    seed_everything(seed)
    
    print(f'\nSeed: {seed}\n', flush=True)

    # Define your model
    if config_dict['model'] == 'GCN':
        model = GCN(**config_dict['Model_config']).to(device)
    elif config_dict['model'] == 'GraphSAGE':
        model = GraphSAGE(**config_dict['Model_config']).to(device)
    elif config_dict['model'] == 'GIN':
        model = GIN(**config_dict['Model_config']).to(device)
    elif config_dict['model'] == 'GAT':
        model = GAT(**config_dict['Model_config']).to(device)
    elif config_dict['model'] == 'EdgeCNN':
        model = EdgeCNN(**config_dict['Model_config']).to(device)
    elif config_dict['model'] == 'GraphUNet':
        model = GraphUNet(**config_dict['Model_config']).to(device)
    elif config_dict['model'] == 'PMLP':
        model = PMLP(**config_dict['Model_config']).to(device)
    else:
        raise ValueError('Model not supported')

    # Define forward pass
    if config_dict['task'] == 'AtomClassification':
        if config_dict['model'] == 'GraphUNet':
            def forward_pass(data):
                return model.forward(x=data.pos_abs, edge_index=data.edge_index, batch=data.batch)
        elif config_dict['model'] == 'PMLP':
            def forward_pass(data):
                return model.forward(x=data.pos_abs, edge_index=data.edge_index)
        else:
            def forward_pass(data):
                return model.forward(x=data.pos_abs, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    elif config_dict['task'] in ['SpacegroupClassification', 'CrystalSystemClassification', 'SAXSRegression', 'XRDRegression', 'xPDFRegression']:
        if config_dict['model'] in ['GraphUNet']:
            def forward_pass(data, out_dim):
                out = model.forward(x=torch.cat((data.x, data.pos_abs),dim=1), edge_index=data.edge_index, batch=data.batch)
                out = torch.cat((global_mean_pool(out, data.batch), global_add_pool(out, data.batch), global_max_pool(out, data.batch)), dim=1)
                middle_dim = (out.size(-1) + out_dim) // 2
                out = F.relu(Linear(out.size(-1), middle_dim).to(device)(out))
                out = Linear(middle_dim, out_dim).to(device)(out)
                return out
        elif config_dict['model'] == 'PMLP':
            def forward_pass(data, out_dim):
                out = model.forward(x=torch.cat((data.x, data.pos_abs),dim=1), edge_index=data.edge_index)
                out = torch.cat((global_mean_pool(out, data.batch), global_add_pool(out, data.batch), global_max_pool(out, data.batch)), dim=1)
                middle_dim = (out.size(-1) + out_dim) // 2
                out = F.relu(Linear(out.size(-1), middle_dim).to(device)(out))
                out = Linear(middle_dim, out_dim).to(device)(out)
                return out
        else:
            def forward_pass(data, out_dim):
                out = model.forward(x=torch.cat((data.x, data.pos_abs),dim=1), edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
                out = torch.cat((global_mean_pool(out, data.batch), global_add_pool(out, data.batch), global_max_pool(out, data.batch)), dim=1)
                middle_dim = (out.size(-1) + out_dim) // 2
                out = F.relu(Linear(out.size(-1), middle_dim).to(device)(out))
                out = Linear(middle_dim, out_dim).to(device)(out)
                return out
    elif config_dict['task'] == 'PositionRegression':
        if config_dict['model'] == 'GraphUNet':
            def forward_pass(data):
                return model.forward(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif config_dict['model'] == 'PMLP':
            def forward_pass(data):
                return model.forward(x=data.x, edge_index=data.edge_index)
        else:
            def forward_pass(data):
                return model.forward(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    elif config_dict['task'] == 'DistanceRegression':
        if config_dict['model'] == 'PMLP':
            def forward_pass(data):
                return model.forward(x=torch.cat((data.x, data.pos_abs),dim=1), edge_index=data.edge_index)
        else:
            def forward_pass(data):
                return model.forward(x=torch.cat((data.x, data.pos_abs),dim=1), edge_index=data.edge_index, batch=data.batch)
    else:
        raise NotImplementedError

    # Define dataloader
    train_loader = DataLoader(dataset.train_set, batch_size=config_dict['Train_config']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset.validation_set, batch_size=config_dict['Train_config']['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=config_dict['Train_config']['batch_size'], shuffle=False)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['Train_config']['learning_rate'])
    if config_dict['task'] in ['AtomClassification', 'SpacegroupClassification', 'CrystalSystemClassification']:
        criterion = torch.nn.CrossEntropyLoss()
        if config_dict['task'] == 'AtomClassification':
            n_classes = 118
        elif config_dict['task'] == 'SpacegroupClassification':
            n_classes = 230
        elif config_dict['task'] == 'CrystalSystemClassification':
            n_classes = 7
    elif config_dict['task'] in ['PositionRegression', 'SAXSRegression', 'XRDRegression', 'xPDFRegression', 'DistanceRegression']:
        n_classes = 1
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise NotImplementedError
    
    # Define TensorBoard writer
    save_dir = f"{config_dict['log_dir']}{config_dict['dataset']}/{config_dict['task']}/{config_dict['model']}/seed{seed}"
    writer = SummaryWriter(save_dir)

    # Set training time (in seconds)
    max_training_time = config_dict['Train_config']['train_time']
    start_time = time.time()
    epoch = 0
    

    # Training loop
    for epoch in range(config_dict['Train_config']['epochs']):
        # Stop training if max training time is exceeded
        if time.time() - start_time > max_training_time:
            break
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if config_dict['task'] == 'AtomClassification':
                out = forward_pass(data)
                ground_truth = data.x[:,0].long()
            elif config_dict['task'] == 'SpacegroupClassification':
                out = forward_pass(data, 230)
                ground_truth = torch.tensor(data.y['space_group_number'], device=device)
            elif config_dict['task'] == 'CrystalSystemClassification':
                out = forward_pass(data, 7)
                ground_truth = torch.tensor(data.y['crystal_system_number'], device=device)
            elif config_dict['task'] == 'PositionRegression':
                out = forward_pass(data)
                ground_truth = data.pos_abs
            elif config_dict['task'] == 'SAXSRegression':
                out = forward_pass(data, 300)
                ground_truth = data.y['saxs'][1,:]
                # Min-max normalize saxs data
                ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
            elif config_dict['task'] == 'XRDRegression':
                out = forward_pass(data, 580)
                ground_truth = data.y['xrd'][1,:]
                # Min-max normalize xrd data
                ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
            elif config_dict['task'] == 'xPDFRegression':
                out = forward_pass(data, 6000)
                ground_truth = data.y['xPDF'][1,:]
                # Min-max normalize xpdf data
                ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
            elif config_dict['task'] == 'DistanceRegression':
                out = forward_pass(data)
                out = torch.sum(out[data.edge_index[0,:]]*out[data.edge_index[1,:]], dim=-1)
                ground_truth = data.edge_attr
            loss = criterion(out, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        correct = 0
        error = 0
        total = 0

        with torch.no_grad():
            MC_F1 = MulticlassF1Score(num_classes=n_classes, average='weighted')
            for data in val_loader:
                data = data.to(device)
                if config_dict['task'] == 'AtomClassification':
                    out = forward_pass(data)
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = data.x[:,0].long()
                    MC_F1.update(predicted, ground_truth)
                    # total += data.x[:,0].size(0)
                    # correct += (predicted == data.x[:,0].long()).sum().item()
                elif config_dict['task'] == 'SpacegroupClassification':
                    out = forward_pass(data, 230)
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = torch.tensor(data.y['space_group_number'], device=device)
                    MC_F1.update(predicted, ground_truth)
                    # total += ground_truth.size(0)
                    # correct += (predicted == ground_truth).sum().item()
                elif config_dict['task'] == 'CrystalSystemClassification':
                    out = forward_pass(data, 7)
                    _, predicted = torch.max(out.data, 1)
                    ground_truth = torch.tensor(data.y['crystal_system_number'], device=device)
                    MC_F1.update(predicted, ground_truth)
                    # total += ground_truth.size(0)
                    # correct += (predicted == ground_truth).sum().item()
                elif config_dict['task'] == 'PositionRegression':
                    out = forward_pass(data)
                    error += position_MAE(out, data.pos_abs)
                elif config_dict['task'] == 'SAXSRegression':
                    out = forward_pass(data, 300)
                    ground_truth = data.y['saxs'][1,:]
                    # Min-max normalize saxs data
                    ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
                    error += criterion(out, ground_truth)
                elif config_dict['task'] == 'XRDRegression':
                    out = forward_pass(data, 580)
                    ground_truth = data.y['xrd'][1,:]
                    # Min-max normalize xrd data
                    ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
                    error += criterion(out, ground_truth)
                elif config_dict['task'] == 'xPDFRegression':
                    out = forward_pass(data, 6000)
                    ground_truth = data.y['xPDF'][1,:]
                    # Min-max normalize xpdf data
                    ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
                    error += criterion(out, ground_truth)
                elif config_dict['task'] == 'DistanceRegression':
                    out = forward_pass(data)
                    out = torch.sum(out[data.edge_index[0,:]]*out[data.edge_index[1,:]], dim=-1)
                    ground_truth = data.edge_attr   
                    error += criterion(out, ground_truth) 
        
        # Log training progress
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Log validation progress
        if 'Classification' in config_dict['task']:
            val_error = torch.tensor(0)
            val_f1 = MC_F1.compute()
            writer.add_scalar('F1-score/val', val_f1, epoch)
            
            
            # Save model if validation accuracy is improved
            if epoch == 0:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f"{save_dir}/best.pt"
                    )
            elif val_f1 > best_val_f1:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f"{save_dir}/best.pt"
                    )
                best_val_f1 = val_f1
            
            print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val F1-score: {val_f1:.4f}', flush=True)
        elif 'Regression' in config_dict['task']:
            val_f1 = 0
            val_error = error / len(val_loader)
            
            # Save model if validation error is improved
            if epoch == 0:
                best_val_error = val_error
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f"{save_dir}/best.pt"
                    )
            elif val_error < best_val_error:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    },
                    f"{save_dir}/best.pt"
                    )
                best_val_error = val_error
            if 'PositionRegression' in config_dict['task']:
                writer.add_scalar('posMAE/val', val_error, epoch)
                print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val position MAE: {val_error:.4f}', flush=True)
            else:
                writer.add_scalar('MAE/val', val_error, epoch)
                print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val MAE: {val_error:.4f}', flush=True)
        
        # Save latest model
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            },
            f"{save_dir}/latest.pt"
            )
    # Record stop time
    stop_time = time.time()
    
    # Load best model
    checkpoint = torch.load(f"{save_dir}/best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    # Evaluate the model on the test set using the best epoch
    model.eval()
    correct = 0
    error = 0
    total = 0

    with torch.no_grad():
        MC_F1 = MulticlassF1Score(num_classes=n_classes, average='weighted')
        for data in test_loader:
            data = data.to(device)
            if config_dict['task'] == 'AtomClassification':
                out = forward_pass(data)
                _, predicted = torch.max(out.data, 1)
                ground_truth = data.x[:,0].long()
                MC_F1.update(predicted, ground_truth)
            elif config_dict['task'] == 'SpacegroupClassification':
                out = forward_pass(data, 230)
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(data.y['space_group_number'], device=device)
                MC_F1.update(predicted, ground_truth)
            elif config_dict['task'] == 'CrystalSystemClassification':
                out = forward_pass(data, 7)
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(data.y['crystal_system_number'], device=device)
                MC_F1.update(predicted, ground_truth)
            elif config_dict['task'] == 'PositionRegression':
                out = forward_pass(data)
                error += position_MAE(out, data.pos_abs)
            elif config_dict['task'] == 'SAXSRegression':
                out = forward_pass(data, 300)
                ground_truth = data.y['saxs'][1,:]
                # Min-max normalize saxs data
                ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
                error += criterion(out, ground_truth)
            elif config_dict['task'] == 'XRDRegression':
                out = forward_pass(data, 580)
                ground_truth = data.y['xrd'][1,:]
                # Min-max normalize xrd data
                ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
                error += criterion(out, ground_truth)
            elif config_dict['task'] == 'xPDFRegression':
                out = forward_pass(data, 6000)
                ground_truth = data.y['xPDF'][1,:]
                # Min-max normalize xpdf data
                ground_truth = (ground_truth - torch.min(ground_truth, dim=-1)[0]) / (torch.max(ground_truth, dim=-1)[0] - torch.min(ground_truth, dim=-1)[0])
                error += criterion(out, ground_truth)
            elif config_dict['task'] == 'DistanceRegression':
                out = forward_pass(data)
                out = torch.sum(out[data.edge_index[0,:]]*out[data.edge_index[1,:]], dim=-1)
                ground_truth = data.edge_attr   
                error += criterion(out, ground_truth)   

    if 'Classification' in config_dict['task']:
        test_error = torch.tensor(0)
        test_f1 = MC_F1.compute()

        writer.add_scalar('F1-score/test', test_f1, epoch)

        print(f'Test F1-score: {test_f1:.4f}', flush=True)
    elif 'Regression' in config_dict['task']:
        test_f1 = 0
        test_error = error / len(test_loader)
        if 'PositionRegression' in config_dict['task']:
            writer.add_scalar('posMAE/test', test_error, epoch)

            print(f'Test position MAE: {test_error:.4f}', flush=True)
        else:
            writer.add_scalar('MAE/test', test_error, epoch)

            print(f'Test MAE: {test_error:.4f}', flush=True)

    # Close TensorBoard writer
    writer.close()

    # Add results to dataframe
    results_df.loc[i] = [config_dict['model'], config_dict['dataset'], config_dict['task'], seed, len(dataset.train_set), len(dataset.validation_set), len(dataset.test_set), stop_time - start_time, sum(p.numel() for p in model.parameters() if p.requires_grad), train_loss, val_f1, test_f1, val_error.detach().cpu().numpy(), test_error.detach().cpu().numpy()]
    
# Save results to csv file
results_df.to_csv(f"{save_dir}/../results.csv")
