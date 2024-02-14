#!/usr/bin/env python
# coding: utf-8

import warnings
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN
from benchmark.dataset_class import CHILI

# Hyperparamters
learning_rate = 0.001
batch_size = 16
max_epochs = 50
seeds = 42
max_patience = 50  # Epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
root = 'benchmark/dataset/'
dataset='CHILI-3K'

# Create dataset using the CHILI dataset from this repo
dataset = CHILI(root, dataset)

# Create random split and load that into the dataset class
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    dataset.create_data_split(split_strategy = 'random', test_size=0.1)
    dataset.load_data_split(split_strategy = 'random')

# Define dataloader
train_loader = DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset.validation_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset.test_set, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(dataset.train_set)}", flush=True)
print(f"Number of validation samples: {len(dataset.validation_set)}", flush=True)
print(f"Number of test samples: {len(dataset.test_set)}", flush=True)
print()

# Intialise model and optimizer
model = GCN(
    in_channels = 7,
    hidden_channels = 32,
    out_channels = 1,
    num_layers = 4,
).to(device=device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = learning_rate,
)

# Initialise loss function and metric function
loss_function = nn.SmoothL1Loss()
metric_function = nn.MSELoss()
improved_function = lambda best, new: new < best if best is not None else True

# Training & Validation
patience = 0
best_error = None
for epoch in range(max_epochs):
    
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
        pred = model.forward(
            x = torch.cat((data.x, data.pos_abs), dim=1),
            edge_index = data.edge_index,
            edge_attr = None,
            edge_weight = None,
            batch = data.batch
        )
        pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
        truth = data.edge_attr
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
            pred = model.forward(
                x = torch.cat((data.x, data.pos_abs), dim=1),
                edge_index = data.edge_index,
                edge_attr = None,
                edge_weight = None,
                batch = data.batch
            )
            pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
            truth = data.edge_attr
            metric = metric_function(pred, truth)

        # Aggregate errors
        val_error += metric.item()

    val_error = val_error / len(val_loader)

    if improved_function(best_error, val_error):
        best_error = val_error
        patience = 0
    else:
        patience += 1

    # Print checkpoint
    print(f'Epoch: {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val WeightedF1Score: {val_error:.4f}')

# Testing
model.eval()
test_error = 0
for data in test_loader:

    # Send to device
    data = data.to(device)

    # Perform forward pass
    with torch.no_grad():
        pred = model.forward( 
            x = torch.cat((data.x, data.pos_abs), dim=1),
            edge_index = data.edge_index,
            edge_attr = None,
            edge_weight = None,
            batch = data.batch
        )
        pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
        truth = data.edge_attr
        metric = metric_function(pred, truth)

    # Aggregate errors
    test_error += metric.item()

# Final test error
test_error = test_error / len(test_loader)
print(f"Test WeightedF1Score: {test_error:.4f}")