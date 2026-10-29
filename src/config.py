"""Unified configuration for all experiments."""
import os
import torch

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results_v3')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
VRAM_GB = (_props.total_memory if hasattr(_props, 'total_memory') else _props.total_mem) / 1e9 if _props is not None else 0
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_MULTI_GPU = N_GPUS > 1

# cuDNN auto-tuner, significant speedup for CNN models on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Training (from paper)
BATCH_SIZE = 1024
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
BATCH_SIZE_KAN = 1024   # unified with MLP for a clean parameter-fair comparison (FastKAN trains fine at bs=1024)
GRAD_ACCUM_KAN = 1

# Per-dataset batch-size overrides: small datasets need a smaller batch to get
# enough gradient updates per epoch. At bs=1024 these give only ~6-8 batches/epoch
# (too few for stable convergence): PAMAP2 ~7.3K windows/fold, UCI-HAR ~8.2K/fold.
DATASET_BATCH_SIZES = {
    'pamap2': {'batch_size': 256, 'batch_size_kan': 256},
    'ucihar': {'batch_size': 256, 'batch_size_kan': 256},
}
USE_AMP = True
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 1.0
EPOCHS = 100
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-4

# Data (from paper)
WINDOW_SEC = 2.56
OVERLAP = 0.5

# KAN architecture (from paper)
KAN_LAYERS = [128, 64, 32, 6]
KAN_GRID_SIZE = 5
KAN_SPLINE_ORDER = 3
# FastKAN grid adjusted higher to match ~170K parameter budget
# G=15 gives ~167K params with [128,64,32,6] architecture
FASTKAN_GRID_SIZE = 15

# G ablation (Reviewer 2 request): FastKAN grid-size sensitivity around the
# parameter-matched default G=15. G=8 is the FastKAN reference default; G=10
# brackets below G=15, and G=20/25 exceed the Matched-MLP budget (they grant
# the KAN more parameters than the MLP). Run on all seven datasets so the
# KAN-vs-MLP ordering is checked at every G on the full benchmark.
G_ABLATION_VALUES = [8, 10, 15, 20, 25]
G_ABLATION_DATASETS = ['ucihar', 'wisdm', 'pamap2', 'motionsense',
                       'uschad', 'hhar', 'realworld']

# MLP classifiers (from paper)
SMALL_MLP_DIMS = [128, 64, 6]
EQUAL_MLP_DIMS = [128, 440, 256, 6]

# Random seeds, all classifiers use identical seeds for controlled comparison
UNIFIED_SEEDS = [42, 43, 44]

# Subject-wise CV
N_FOLDS = 5
LOSO_THRESHOLD = 20  # Use LOSO for datasets with <20 subjects

# Training configuration override for KAN (memory constraints)
# BATCH_SIZE_KAN and GRAD_ACCUM_KAN defined above, this block intentionally empty

# Datasets info
DATASETS = {
    'ucihar': {
        'name': 'UCI-HAR',
        'subjects': 30, 'channels': 9, 'sampling_rate': 50,
        'window_samples': 128, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 30,
        'classes': ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying'],
    },
    'wisdm': {
        'name': 'WISDM',
        'subjects': 36, 'channels': 3, 'sampling_rate': 20,
        'window_samples': 100, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 36,
        'classes': ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
    },
    'pamap2': {
        'name': 'PAMAP2',
        'subjects': 9, 'channels': 9, 'sampling_rate': 50,
        'window_samples': 128, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 9,
        'classes': ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling'],
    },
    'motionsense': {
        'name': 'MotionSense',
        'subjects': 24, 'channels': 12, 'sampling_rate': 50,
        'window_samples': 128, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 24,
        'classes': ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
    },
    'uschad': {
        'name': 'USC-HAD',
        'subjects': 14, 'channels': 6, 'sampling_rate': 50,
        'window_samples': 128, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 14,
        'classes': ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Lying'],
    },
    'hhar': {
        'name': 'HHAR',
        'subjects': 9, 'channels': 6, 'sampling_rate': 50,
        'window_samples': 128, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 9,
        'classes': ['Bike', 'Sit', 'StairsDown', 'StairsUp', 'Stand', 'Walk'],
    },
    'realworld': {
        'name': 'RealWorld',
        'subjects': 15, 'channels': 6, 'sampling_rate': 50,
        'window_samples': 128, 'n_classes': 6,
        'cv_method': 'loso', 'n_folds': 15,
        'classes': ['Walking', 'Running', 'Sitting', 'Standing', 'StairsUp', 'StairsDown'],
    },
}
