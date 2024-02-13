import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

# Simple MLP model
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, batch):
        if len(x.shape) < 2:
            batch_size = torch.max(batch) + 1
            x = x.reshape(batch_size, x.shape[-1] // batch_size)

        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Regression secondary module
class Secondary(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Secondary, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x, batch):
        # Pool the graph
        x = torch.cat(
            (
                global_mean_pool(x, batch),
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ),
            dim=1,
        )
        # Linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class RandomClass(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(RandomClass, self).__init__()

    def forward(self, x, device, num_classes):
        pred = torch.zeros(len(x), num_classes, requires_grad=True).to(device=device)
        pred_ints = torch.randint(num_classes, size=(len(x),))
        for i, p in enumerate(pred_ints):
            pred[i][p] = 1.0
        return pred

class MostFrequentClass(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MostFrequentClass, self).__init__()

    def forward(self, x, device, num_classes, mfc):
        pred = torch.zeros(len(x), num_classes, requires_grad=True).to(device=device)
        pred_ints = torch.full(fill_value=mfc, size=(len(x),), dtype=torch.long)
        for i, p in enumerate(pred_ints):
            pred[i][p-1] = 1.0
        return pred
