from torch import nn


class AssociationHead(nn.Module):
    def __init__(self, input_channels, representation_size):
        super(AssociationHead, self).__init__()
        self.fc = nn.Linear(input_channels, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
