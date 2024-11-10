import torch
import torch.nn as nn
class MLP(nn.Module):
    # TODO ask jiale to update
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.output = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
breakpoint()
model = torch.load("hybrid.pth")