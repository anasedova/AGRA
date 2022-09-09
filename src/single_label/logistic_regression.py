from torch import nn


class MaxEntNetwork(nn.Module):
    def __init__(self, num_features, num_outputs):
        super(MaxEntNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(num_features, num_outputs),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits