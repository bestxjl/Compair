"""CNN-BiLSTM backbone and MLP classifiers used in the paper."""
import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    """Multi-scale 1D CNN with three parallel branches (k=1, 3, 5)."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 42, kernel_size=1),
            nn.BatchNorm1d(42),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, 43, kernel_size=3, padding=1),
            nn.BatchNorm1d(43),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, 43, kernel_size=5, padding=2),
            nn.BatchNorm1d(43),
            nn.ReLU(inplace=True),
        )
        # Bottleneck: 128 -> 64, stride-2 max pooling.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # x: (B, C, T)
        h1 = self.conv1(x)  # (B, 42, T)
        h3 = self.conv3(x)  # (B, 43, T)
        h5 = self.conv5(x)  # (B, 43, T)
        h = torch.cat([h1, h3, h5], dim=1)  # (B, 128, T)
        h = self.bottleneck(h)  # (B, 64, T/2)
        return h


class BiLSTMModule(nn.Module):
    """Three-layer bidirectional LSTM with global average pooling."""
    def __init__(self, input_size=64, hidden_size=64, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(hidden_size * 2)  # 128

    def forward(self, x):
        # x: (B, 64, T/2), needs (B, T/2, 64) for the LSTM.
        x = x.permute(0, 2, 1)  # (B, T/2, 64)
        x, _ = self.lstm(x)      # (B, T/2, 128)
        x = x.permute(0, 2, 1)   # (B, 128, T/2)
        x = self.bn(x)           # (B, 128, T/2)
        x = x.mean(dim=2)        # (B, 128)
        return x


class CNNBiLSTMBackbone(nn.Module):
    """Feature extraction backbone: CNN + BiLSTM, returns a 128-d vector."""
    def __init__(self, in_channels):
        super().__init__()
        self.cnn = MultiScaleCNN(in_channels)
        self.bilstm = BiLSTMModule(input_size=64, hidden_size=64, num_layers=3)

    def forward(self, x):
        # x: (B, C, T)
        h = self.cnn(x)       # (B, 64, T/2)
        z = self.bilstm(h)    # (B, 128)
        return z

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class SmallMLPClassifier(nn.Module):
    """Compact MLP head: 128 -> 64 -> 6 with ReLU and Dropout(0.2)."""
    def __init__(self, in_dim=128, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class EqualMLPClassifier(nn.Module):
    """MLP head matched to the KAN's parameter budget.

    128 -> 440 -> 256 -> 6 with BN, ReLU, and Dropout(0.3). The total is
    172,590 parameters, close to the KAN's 167,116.
    """
    def __init__(self, in_dim=128, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 440),
            nn.BatchNorm1d(440),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(440, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class CNNBiLSTMClassifier(nn.Module):
    """Full model: backbone plus a classification head."""
    def __init__(self, backbone, classifier, return_features=False):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.return_features = return_features

    def forward(self, x):
        z = self.backbone(x)  # (B, 128)
        out = self.classifier(z)
        if self.return_features:
            return out, z
        return out

    def get_param_count(self):
        return self.backbone.get_param_count() + sum(
            p.numel() for p in self.classifier.parameters()
        )
