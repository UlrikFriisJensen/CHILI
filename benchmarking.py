# Import libraries
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from Code.datasetClass import InOrgMatDatasets
from torch_geometric.nn.models import GCN
from torch.utils.tensorboard import SummaryWriter
import yaml

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmarking script')
parser.add_argument('--config', type=str, help='Path to configuration file')
args = parser.parse_args()

# Read configuration file
config_path = args.config

with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model = GCN(**config_dict['Model_config']).to(device)

# Define dataloader
train_loader = DataLoader(dataset.train_set, batch_size=config_dict['Train_config']['batch_size'], shuffle=True)
val_loader = DataLoader(dataset.validation_set, batch_size=config_dict['Train_config']['batch_size'], shuffle=False)
test_loader = DataLoader(dataset.test_set, batch_size=config_dict['Train_config']['batch_size'], shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['Train_config']['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

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
        out = model.forward(data.pos_frac, data.edge_index) # TODO: Fix bug with size of edge_attr
        loss = criterion(out, data.x[:,0].long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # Validation loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model.forward(data.pos_frac, data.edge_index) # TODO: Fix bug with size of edge_attr
            _, predicted = torch.max(out.data, 1)
            total += data.x[:,0].size(0)
            correct += (predicted == data.x[:,0].long()).sum().item()

    val_accuracy = correct / total

    # Log training and validation progress
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print(f'Epoch: {epoch+1}/{config_dict["Train_config"]["epochs"]}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Evaluate the model on the test set using the best epoch
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model.forward(data.pos_frac, data.edge_index) # TODO: Fix bug with size of edge_attr
        _, predicted = torch.max(out.data, 1)
        total += data.x[:,0].size(0)
        correct += (predicted == data.x[:,0].long()).sum().item()

test_accuracy = correct / total

writer.add_scalar('Accuracy/test', test_accuracy, epoch)

print(f'Test Accuracy: {test_accuracy:.4f}')

# Close TensorBoard writer
writer.close()