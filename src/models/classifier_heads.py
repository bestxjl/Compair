"""Classification heads for the paper.

The Linear head lives here. The Small MLP and Matched MLP heads are in
cnn_bilstm.py, and the KAN head is in kan_layers.py. All heads share the same
interface: __init__(in_dim, num_classes), forward(x) returns the logits.
"""
import torch.nn as nn


class LinearClassifier(nn.Module):
    """A single linear layer from 128 features to the class logits.

    If this head already reaches high accuracy, the backbone features are
    nearly linearly separable, and a larger head adds little.
    """
    def __init__(self, in_dim=128, num_classes=6):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
