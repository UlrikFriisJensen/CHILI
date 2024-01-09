# Import libraries
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from Code.datasetClass import InOrgMatDatasets
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, EdgeCNN, DimeNetPlusPlus, SchNet, AttentiveFP
from torch_geometric.nn import global_mean_pool, Linear
from torch.utils.tensorboard import SummaryWriter
import yaml

# Define position mean absolute error function
def position_MAE(pred_xyz, true_xyz):
    '''
    Calculates the mean absolute error between the predicted and true positions of the atoms in units of Ångstrøm.
    '''
    return torch.mean(torch.sqrt(torch.sum(F.mse_loss(pred_xyz, true_xyz, reduction='none'), dim=1)), dim=0)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmarking script')
parser.add_argument('--config', type=str, help='Path to configuration file')
args = parser.parse_args()

# Read configuration file
config_path = args.config

with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)

# Set device
device = 'cpu'# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed
torch.manual_seed(config_dict['Train_config']['seed'])

# Define your dataset
dataset = InOrgMatDatasets(dataset=config_dict['dataset'], root=config_dict['root'])
try:
    dataset.load_data_split(split_strategy='random')
except FileNotFoundError:
    dataset.create_data_split(split_strategy='random', test_size=0.1)
    dataset.load_data_split(split_strategy='random')

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
elif config_dict['model'] == 'AttentiveFP':
    model = AttentiveFP(**config_dict['Model_config']).to(device)
elif config_dict['model'] == 'DimeNetPlusPlus':
    model = DimeNetPlusPlus().to(device)
elif config_dict['model'] == 'SchNet':
    model = SchNet().to(device)
else:
    raise ValueError('Model not supported')

# Define forward pass
if config_dict['task'] == 'AtomClassification':
    if config_dict['model'] in ['DimeNetPlusPlus', 'SchNet']:
        raise NotImplementedError
    else:
        def forward_pass(data):
            return model.forward(x=data.pos_abs, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
elif config_dict['task'] in ['SpacegroupClassification', 'CrystalSystemClassification']:
    if config_dict['model'] in ['DimeNetPlusPlus', 'SchNet']:
        def forward_pass(data):
            return model.forward(data.x[:,0], data.pos_abs)
    else:
        def forward_pass(data):
            return model.forward(x=torch.cat((data.x, data.pos_abs),dim=1), edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
elif config_dict['task'] == 'PositionRegression':
    if config_dict['model'] in ['DimeNetPlusPlus', 'SchNet']:
        raise NotImplementedError
    else:
        def forward_pass(data):
            return model.forward(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
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
elif config_dict['task'] in ['PositionRegression']:
    criterion = torch.nn.MSELoss()
elif config_dict['task'] in ['SAXSRegression', 'XRDRegression', 'xPDFRegression']:
    pass

print(f'\nModel: {config_dict["model"]}\nDataset: {config_dict["dataset"]}\nTask: {config_dict["task"]}')
print('\n')
print(f'Number of training samples: {len(dataset.train_set)}')
print(f'Number of validation samples: {len(dataset.validation_set)}')
print(f'Number of test samples: {len(dataset.test_set)}')
print('\n')
print(f'Device: {device}')
print('\n')
print('Model architecture:')
print(model)
print('\n')
# Define TensorBoard writer
writer = SummaryWriter(f"{config_dict['log_dir']}{config_dict['dataset']}/{config_dict['model']}_{config_dict['task']}")

# Training loop
for epoch in range(config_dict['Train_config']['epochs']):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        if config_dict['task'] == 'AtomClassification':
            out = forward_pass(data)
            ground_truth = data.x[:,0].long()
        elif config_dict['task'] == 'SpacegroupClassification':
            out = forward_pass(data)
            out = global_mean_pool(out, data.batch)
            out = Linear(out.size(-1), 230).to(device)(out)
            ground_truth = torch.tensor(data.y['space_group_number'], device=device)
        elif config_dict['task'] == 'CrystalSystemClassification':
            # print(data.x.size())
            # print(data.pos_abs.size())
            # print(data.edge_attr.size())
            # print(data.edge_index.size())
            # print(data.edge_index.max())
            out = forward_pass(data)
            out = global_mean_pool(out, data.batch)
            out = Linear(out.size(-1), 7).to(device)(out)
            ground_truth = torch.tensor(data.y['crystal_system_number'], device=device)
        elif config_dict['task'] == 'PositionRegression':
            out = forward_pass(data)
            ground_truth = data.pos_abs
        elif config_dict['task'] == 'SAXSRegression':
            pass
        elif config_dict['task'] == 'XRDRegression':
            pass
        elif config_dict['task'] == 'xPDFRegression':
            pass
        elif config_dict['task'] == 'GraphReconstruction':
            pass
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
        for data in val_loader:
            data = data.to(device)
            if config_dict['task'] == 'AtomClassification':
                out = forward_pass(data)
                _, predicted = torch.max(out.data, 1)
                total += data.x[:,0].size(0)
                correct += (predicted == data.x[:,0].long()).sum().item()
            elif config_dict['task'] == 'SpacegroupClassification':
                out = forward_pass(data)
                out = global_mean_pool(out, data.batch)
                out = Linear(out.size(-1), 230).to(device)(out)
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(data.y['space_group_number'], device=device)
                total += ground_truth.size(0)
                correct += (predicted == ground_truth).sum().item()
            elif config_dict['task'] == 'CrystalSystemClassification':
                out = forward_pass(data)
                out = global_mean_pool(out, data.batch)
                out = Linear(out.size(-1), 7).to(device)(out)
                _, predicted = torch.max(out.data, 1)
                ground_truth = torch.tensor(data.y['crystal_system_number'], device=device)
                total += ground_truth.size(0)
                correct += (predicted == ground_truth).sum().item()
            elif config_dict['task'] == 'PositionRegression':
                out = forward_pass(data)
                error += position_MAE(out, data.pos_abs)
            elif config_dict['task'] == 'SAXSRegression':
                pass
            elif config_dict['task'] == 'XRDRegression':
                pass
            elif config_dict['task'] == 'xPDFRegression':
                pass
            elif config_dict['task'] == 'GraphReconstruction':
                pass     
    
    # Log training progress
    writer.add_scalar('Loss/train', train_loss, epoch)
    
    # Log validation progress
    if 'Classification' in config_dict['task']:
        val_accuracy = correct / total
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Save model if validation accuracy is improved
        if epoch == 0:
            best_val_accuracy = val_accuracy
        elif val_accuracy > best_val_accuracy:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                },
                f"{config_dict['log_dir']}{config_dict['dataset']}/{config_dict['model']}_{config_dict['task']}/best.pt"
                )
            best_val_accuracy = val_accuracy
        
        print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    elif 'Regression' in config_dict['task']:
        val_error = error / len(val_loader)
        writer.add_scalar('posMAE/val', val_error, epoch)
        
        # Save model if validation error is improved
        if epoch == 0:
            best_val_error = val_error
        elif val_error < best_val_error:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                },
                f"{config_dict['log_dir']}{config_dict['dataset']}/{config_dict['model']}_{config_dict['task']}/best.pt"
                )
            best_val_error = val_error
        
        print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val position MAE: {val_error:.4f}')
    
    # Save latest model
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(),
        },
        f"{config_dict['log_dir']}{config_dict['dataset']}/{config_dict['model']}_{config_dict['task']}/latest.pt"
        )

# Evaluate the model on the test set using the best epoch
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        if config_dict['task'] == 'AtomClassification':
            out = forward_pass(data)
            _, predicted = torch.max(out.data, 1)
            total += data.x[:,0].size(0)
            correct += (predicted == data.x[:,0].long()).sum().item()
        elif config_dict['task'] == 'SpacegroupClassification':
            out = forward_pass(data)
            out = global_mean_pool(out, data.batch)
            out = Linear(out.size(-1), 230).to(device)(out)
            _, predicted = torch.max(out.data, 1)
            ground_truth = torch.tensor(data.y['space_group_number'], device=device)
            total += ground_truth.size(0)
            correct += (predicted == ground_truth).sum().item()
        elif config_dict['task'] == 'CrystalSystemClassification':
            out = forward_pass(data)
            out = global_mean_pool(out, data.batch)
            out = Linear(out.size(-1), 7).to(device)(out)
            _, predicted = torch.max(out.data, 1)
            ground_truth = torch.tensor(data.y['crystal_system_number'], device=device)
            total += ground_truth.size(0)
            correct += (predicted == ground_truth).sum().item()
        elif config_dict['task'] == 'PositionRegression':
            out = forward_pass(data)
            error += position_MAE(out, data.pos_abs)
        elif config_dict['task'] == 'SAXSRegression':
            pass
        elif config_dict['task'] == 'XRDRegression':
            pass
        elif config_dict['task'] == 'xPDFRegression':
            pass
        elif config_dict['task'] == 'GraphReconstruction':
            pass  

if 'Classification' in config_dict['task']:
    test_accuracy = correct / total

    writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    print(f'Test Accuracy: {test_accuracy:.4f}')
elif 'Regression' in config_dict['task']:
    test_error = error / len(test_loader)

    writer.add_scalar('posMAE/test', test_error, epoch)

    print(f'Test position MAE: {test_error:.4f}')

# Close TensorBoard writer
writer.close()