import torch
from torch import nn


class AgentNN(nn.Module):
    def __init__(self, network, freeze=False):
        super().__init__()
        self.network = network

        if freeze:
            self._freeze()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def _freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False