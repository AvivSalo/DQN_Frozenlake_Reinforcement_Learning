import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DeepQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.layers(x)
