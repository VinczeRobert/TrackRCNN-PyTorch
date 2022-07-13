from torch import nn
import torch.nn.functional as F


class AssociationHead(nn.Module):
    def __init__(self, input_channels, representation_size):
        super(AssociationHead, self).__init__()
        self.fc = nn.Linear(in_features=input_channels, out_features=representation_size, bias=False)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))

        return x
