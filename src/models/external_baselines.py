"""External baseline models: LSTM-FCN and DeepConvLSTM."""
import torch
import torch.nn as nn


class LSTMFCN(nn.Module):
    """LSTM-FCN for time series classification.

    Karim et al., "Multivariate LSTM-FCNs for Time Series Classification", Neural Networks, 2019.

    FCN branch: 3 Conv1D blocks (128->256->128, k=8,5,3) + GAP
    LSTM branch: 1 LSTM(8) with Dropout(0.8)
    Concatenated output -> Linear classifier
    """
    def __init__(self, in_channels, seq_len, num_classes=6):
        super().__init__()

        # FCN Branch
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

        # LSTM Branch (Karim et al. "dimension shuffle"): transpose (C, T) -> (T, C)
        # and feed ALL C channels to the LSTM, so input_size = in_channels.
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=8,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(0.8)

        # Final classifier
        self.classifier = nn.Linear(128 + 8, num_classes)

    def forward(self, x):
        # x: (B, C, T)

        # FCN path
        fcn = self.conv1(x)
        fcn = self.conv2(fcn)
        fcn = self.conv3(fcn)
        fcn = self.gap(fcn).squeeze(-1)  # (B, 128)

        # LSTM path: dimension shuffle (Karim et al.), transpose (C, T) -> (T, C)
        # and feed ALL channels (input_size = C) to the LSTM, not a channel mean.
        lstm_in = x.permute(0, 2, 1)      # (B, T, C)
        lstm_out, _ = self.lstm(lstm_in)   # (B, T, 8)
        lstm_out = lstm_out[:, -1, :]       # (B, 8), last timestep
        lstm_out = self.lstm_dropout(lstm_out)

        # Concat and classify
        combined = torch.cat([fcn, lstm_out], dim=1)
        return self.classifier(combined)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class DeepConvLSTM(nn.Module):
    """DeepConvLSTM for wearable activity recognition.

    Ordóñez & Roggen, "Deep Convolutional and LSTM Recurrent Neural Networks
    for Multimodal Wearable Activity Recognition", Sensors, 2016.

    Architecture: 4×Conv1D(64, k=5) -> 2×LSTM(128) -> Dense
    No pooling between conv layers (valid convolution shrinks sequence).
    """
    def __init__(self, in_channels, num_classes=6, dropout=0.5):
        super().__init__()

        # 4 Convolutional layers (valid padding, no pooling)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
        )

        # 2 LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=64, hidden_size=128,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=128, hidden_size=128,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)  # (B, 64, T')

        # LSTM: (B, T', 64)
        h = h.permute(0, 2, 1)
        h, _ = self.lstm1(h)
        h = self.dropout1(h)
        h, _ = self.lstm2(h)     # (B, T', 128)
        h = self.dropout2(h)
        h = h[:, -1, :]           # (B, 128), last timestep

        return self.classifier(h)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
