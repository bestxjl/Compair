"""Training utilities: loop, metrics, feature extraction."""
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, silhouette_score,
)
from sklearn.decomposition import PCA


def train_epoch(model, dataloader, optimizer, criterion, device,
                grad_accum=1, grad_clip=1.0, use_amp=False, frozen_modules=None):
    """Single training epoch with optional AMP (mixed precision).

    ``frozen_modules`` (optional list of nn.Module) is switched to eval() after
    ``model.train()`` so their BatchNorm running stats and Dropout stay frozen
    while the rest of the model trains (used by the frozen-backbone experiment).
    """
    model.train()
    if frozen_modules:
        for m in frozen_modules:
            m.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, y) / grad_accum
        else:
            output = model(x)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, y) / grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping epoch")
            return float('nan'), 0.0

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        pred = output.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    if len(dataloader) == 0 or total == 0:
        return 0.0, 0.0
    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, extract_features=False):
    """Evaluate model, optionally extracting features."""
    model.eval()
    all_preds, all_labels = [], []
    all_features = [] if extract_features else None
    n_classes = None

    # Try to extract features from backbone
    has_return_features = hasattr(model, 'return_features') and model.return_features

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        output = model(x)
        # Handle models that return (logits, features)
        if isinstance(output, tuple):
            output, z = output
            if all_features is not None:
                all_features.append(z.cpu().numpy())
        elif extract_features and hasattr(model, 'backbone'):
            z = model.backbone(x)
            if all_features is not None:
                all_features.append(z.cpu().numpy())

        if n_classes is None:
            n_classes = output.shape[1]
        pred = output.argmax(dim=1)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds)
    # Restrict F1 to classes actually present in this fold's test set.
    # sklearn's default (labels=None) unions y_true and y_pred, so a class the
    # model wrongly predicts but that has no samples in y_true (e.g. a LOSO
    # test subject who never performed that activity) gets F1=0 and drags
    # macro-F1 down. Using only y_true's classes gives the correct macro-F1.
    present_labels = np.unique(labels)
    macro_f1 = f1_score(labels, preds, labels=present_labels, average='macro')
    per_class = precision_recall_fscore_support(labels, preds, labels=present_labels,
                                                  average=None, zero_division=0)
    # Pad the per-class vectors to the full class space (0 for absent classes)
    # so per_class_precision/recall/f1 always have length n_classes. Variable-
    # length vectors break downstream np.array() aggregation when a test fold
    # (e.g. a PAMAP2 subject) never performed some activity.
    if n_classes is not None and len(present_labels) < n_classes:
        def _pad(vals):
            out = [0.0] * n_classes
            for lab, v in zip(present_labels, vals):
                out[int(lab)] = float(v)
            return out
        per_class = [_pad(per_class[0]), _pad(per_class[1]), _pad(per_class[2])]

    result = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'predictions': preds,
        'labels': labels,
        'per_class_precision': per_class[0],
        'per_class_recall': per_class[1],
        'per_class_f1': per_class[2],
    }

    if extract_features and all_features is not None and len(all_features) > 0:
        feats = np.concatenate(all_features, axis=0)
        # Filter NaN/inf
        valid = np.isfinite(feats).all(axis=1)
        result['features'] = feats[valid]
        result['labels_for_features'] = labels[valid] if valid.sum() < len(labels) else labels

    return result


def compute_silhouette(features, labels, n_samples=2000, pca_dim=50, random_state=42):
    """Compute Silhouette score on features.

    Uses PCA reduction and subsampling for efficiency.
    Fixed random_state for reproducibility.
    """
    # Filter NaN/inf values
    valid_mask = np.isfinite(features).all(axis=1)
    if valid_mask.sum() < 50:
        return 0.0
    features = features[valid_mask]
    labels = labels[valid_mask]

    rng = np.random.RandomState(random_state)
    if len(features) > n_samples:
        indices = rng.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # PCA to 50 dims (as in paper)
    if features.shape[1] > pca_dim:
        pca = PCA(n_components=min(pca_dim, features.shape[0] - 1), random_state=random_state)
        features = pca.fit_transform(features)

    n_labels = len(np.unique(labels))
    if n_labels < 2 or len(features) <= n_labels:
        return 0.0

    return silhouette_score(features, labels)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_gaussian_noise(x, snr_db):
    """Add Gaussian noise to signal at specified SNR (dB).

    Args:
        x: torch.Tensor of shape (B, C, T), clean signal.
        snr_db: float, target SNR in dB (∞ for no noise).

    Returns:
        torch.Tensor, noisy signal with same shape.
    """
    if snr_db is None:
        return x
    # Signal power per sample
    signal_power = (x ** 2).mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def measure_latency(model, input_shape, device, n_warmup=10, n_measure=100):
    """Measure inference latency."""
    model.eval()
    x = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    # Measure
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(n_measure):
            _ = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.time() - start

    return (elapsed / n_measure) * 1000  # ms per sample


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 warmup_lr=1e-4, target_lr=0.001, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.current_epoch = 0

        # Set initial LR to warmup_lr so epoch 0 starts the warmup correctly.
        # (The training loop calls scheduler.step() AFTER each epoch,
        #  so the first epoch would otherwise train at target_lr with no warmup.)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            frac = self.current_epoch / self.warmup_epochs
            return self.warmup_lr + (self.target_lr - self.warmup_lr) * frac
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            return self.min_lr + (self.target_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * progress)) / 2
