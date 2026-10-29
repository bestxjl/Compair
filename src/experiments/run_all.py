"""Master experiment script, 4 heads + baselines + G ablation + frozen backbone.

Usage:
    python -m src.experiments.run_all --phase heads                # Main comparison
    python -m src.experiments.run_all --phase g_ablation           # KAN grid-size ablation
    python -m src.experiments.run_all --phase frozen_backbone      # Head-capacity isolation
    python -m src.experiments.run_all --phase heads --dataset hhar # Single dataset
"""
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.config import (
    DEVICE, DATASETS, RESULTS_DIR,
    KAN_LAYERS, KAN_GRID_SIZE, KAN_SPLINE_ORDER, FASTKAN_GRID_SIZE,
    G_ABLATION_VALUES, G_ABLATION_DATASETS,
    BATCH_SIZE, BATCH_SIZE_KAN, GRAD_ACCUM_KAN, DATASET_BATCH_SIZES,
    N_GPUS, USE_MULTI_GPU,
    UNIFIED_SEEDS,
    LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING, GRAD_CLIP_NORM,
    EPOCHS, WARMUP_EPOCHS, WARMUP_LR,
    NUM_WORKERS, PERSISTENT_WORKERS, USE_AMP,
)
from src.models import (
    CNNBiLSTMBackbone,
    SmallMLPClassifier, EqualMLPClassifier, CNNBiLSTMClassifier,
    KANClassifier, LSTMFCN, DeepConvLSTM,
    LinearClassifier,
)
from src.training.cv_runner import run_cv
from src.data.datasets import (
    load_dataset, get_cv_splits, subject_wise_split, HARWindowDataset,
)
from src.training.trainer import train_epoch, evaluate, CosineWarmupScheduler


# ============================================================
# Model Builders
# ============================================================

def build_cnn_bilstm_linear(in_channels, n_classes):
    backbone = CNNBiLSTMBackbone(in_channels)
    classifier = LinearClassifier(in_dim=128, num_classes=n_classes)
    return CNNBiLSTMClassifier(backbone, classifier, return_features=True)


def build_cnn_bilstm_small_mlp(in_channels, n_classes):
    backbone = CNNBiLSTMBackbone(in_channels)
    classifier = SmallMLPClassifier(in_dim=128, num_classes=n_classes)
    return CNNBiLSTMClassifier(backbone, classifier, return_features=True)


def build_cnn_bilstm_equal_mlp(in_channels, n_classes):
    backbone = CNNBiLSTMBackbone(in_channels)
    classifier = EqualMLPClassifier(in_dim=128, num_classes=n_classes)
    return CNNBiLSTMClassifier(backbone, classifier, return_features=True)


def build_cnn_bilstm_kan(in_channels, n_classes, grid_size=None,
                          spline_order=KAN_SPLINE_ORDER, kan_type='fast'):
    backbone = CNNBiLSTMBackbone(in_channels)
    layers = [128] + KAN_LAYERS[1:-1] + [n_classes]
    if grid_size is None:
        grid_size = FASTKAN_GRID_SIZE if kan_type == 'fast' else KAN_GRID_SIZE
    classifier = KANClassifier(
        layer_dims=layers, grid_size=grid_size,
        spline_order=spline_order, kan_type=kan_type, dropout=0.0,
    )
    return CNNBiLSTMClassifier(backbone, classifier, return_features=True)


def build_lstm_fcn(in_channels, n_classes, seq_len=128):
    return LSTMFCN(in_channels, seq_len, n_classes)


def build_deepconvlstm(in_channels, n_classes):
    return DeepConvLSTM(in_channels, n_classes)


# ============================================================
# Experiment Configs
# ============================================================

ALL_DATASETS = ['ucihar', 'wisdm', 'pamap2', 'motionsense', 'uschad', 'hhar', 'realworld']
SEEDS = UNIFIED_SEEDS  # [42, 43, 44]


# ============================================================
# Phase 1: Classification Head Comparison (4 heads × 7 datasets)
# ============================================================

def run_heads_phase():
    """All four classification heads on all datasets, unified 3 seeds.

    All heads share the same training configuration (identical batch size and
    gradient clipping) for a clean parameter-fair comparison.
    """
    print("=" * 60)
    print("PHASE 1: Classification Head Comparison (4 heads × 7 datasets)")
    print("=" * 60)

    all_summaries = []

    for ds in ALL_DATASETS:
        print(f"\n{'='*40}\n{ds}: {len(SEEDS)} seeds\n{'='*40}")
        bs = DATASET_BATCH_SIZES.get(ds, {}).get('batch_size', BATCH_SIZE)
        bs_kan = DATASET_BATCH_SIZES.get(ds, {}).get('batch_size_kan', BATCH_SIZE_KAN)

        # 1. Linear (0.8K params)
        print(f"  Linear...")
        _, summary = run_cv(
            build_cnn_bilstm_linear, ds,
            config={'model_name': 'head_linear', 'seeds': SEEDS, 'batch_size': bs})
        if summary:
            all_summaries.append(summary)
            print(f"    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

        # 2. Small MLP (8.6K params)
        print(f"  Small MLP...")
        _, summary = run_cv(
            build_cnn_bilstm_small_mlp, ds,
            config={'model_name': 'head_small_mlp', 'seeds': SEEDS, 'batch_size': bs})
        if summary:
            all_summaries.append(summary)
            print(f"    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

        # 3. Equal MLP (173K params), capacity-matched control
        print(f"  Equal MLP...")
        _, summary = run_cv(
            build_cnn_bilstm_equal_mlp, ds,
            config={'model_name': 'head_equal_mlp', 'seeds': SEEDS, 'batch_size': bs})
        if summary:
            all_summaries.append(summary)
            print(f"    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

        # 4. KAN (167K params), FastKAN, no grad_clip
        print(f"  KAN...")
        _, summary = run_cv(
            lambda ic, nc: build_cnn_bilstm_kan(ic, nc, kan_type='fast'), ds,
            config={'model_name': 'head_kan', 'seeds': SEEDS,
                    'batch_size': bs_kan, 'grad_accum': GRAD_ACCUM_KAN,
                    'is_kan': True})
        if summary:
            all_summaries.append(summary)
            print(f"    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

    out_path = os.path.join(RESULTS_DIR, 'heads_summaries.json')
    with open(out_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved {len(all_summaries)} summaries -> {out_path}")
    return all_summaries


# ============================================================
# Phase 2: External Baselines (LSTM-FCN, DeepConvLSTM)
# ============================================================

def run_baseline_phase():
    """Conventional HAR baselines (LSTM-FCN, DeepConvLSTM) on all datasets."""
    print("=" * 60)
    print("PHASE 2: External Baselines (LSTM-FCN, DeepConvLSTM)")
    print("=" * 60)

    all_summaries = []

    for ds in ALL_DATASETS:
        bs = DATASET_BATCH_SIZES.get(ds, {}).get('batch_size', BATCH_SIZE)
        print(f"\n{'='*40}\nLSTM-FCN, {ds} (bs={bs})\n{'='*40}")
        _, summary = run_cv(build_lstm_fcn, ds,
            config={'model_name': 'lstm_fcn', 'seeds': SEEDS, 'batch_size': bs})
        if summary:
            all_summaries.append(summary)
            print(f"  {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

    for ds in ALL_DATASETS:
        bs = DATASET_BATCH_SIZES.get(ds, {}).get('batch_size', BATCH_SIZE)
        print(f"\n{'='*40}\nDeepConvLSTM, {ds} (bs={bs})\n{'='*40}")
        _, summary = run_cv(build_deepconvlstm, ds,
            config={'model_name': 'deepconvlstm', 'seeds': SEEDS, 'batch_size': bs})
        if summary:
            all_summaries.append(summary)
            print(f"  {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

    out_path = os.path.join(RESULTS_DIR, 'baseline_summaries.json')
    with open(out_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved {len(all_summaries)} summaries -> {out_path}")
    return all_summaries


# ============================================================
# Phase 3c: FastKAN Grid-Size (G) Ablation ,  Reviewer 2 request
# ============================================================

def run_g_ablation_phase(datasets=None):
    """FastKAN grid-size (G) sensitivity around the default G=15.

    Runs the KAN head with G in G_ABLATION_VALUES on the given datasets
    (default G_ABLATION_DATASETS), 3 seeds, to show the KAN-vs-MLP gap is not
    an artifact of the fixed G=15. G=15 reproduces the main head_kan condition;
    G=8 is the FastKAN reference default, and G=20/25 exceed the MLP budget.
    """
    datasets = datasets or G_ABLATION_DATASETS
    print("=" * 60)
    print("PHASE 3c: FastKAN Grid-Size (G) Ablation")
    print("=" * 60)

    all_summaries = []
    for ds in datasets:
        for G in G_ABLATION_VALUES:
            print(f"\n  FastKAN G={G}, {ds}")
            _, summary = run_cv(
                lambda ic, nc, g=G: build_cnn_bilstm_kan(ic, nc, grid_size=g, kan_type='fast'),
                ds, config={'model_name': f'kan_G{G}', 'seeds': SEEDS,
                            'grid_size': G,
                            'batch_size': BATCH_SIZE_KAN,
                            'grad_accum': GRAD_ACCUM_KAN, 'is_kan': True})
            if summary:
                summary['grid_size'] = G
                summary['kan_type'] = 'fast'
                all_summaries.append(summary)
                print(f"    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")

    out_path = os.path.join(RESULTS_DIR, 'g_ablation_summaries.json')
    with open(out_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved {len(all_summaries)} summaries -> {out_path}")
    return all_summaries


# ============================================================
# Phase 6: Frozen-Backbone experiment (Reviewer 3 C1 / Reviewer 1.5)
# ============================================================

def _train_model(model, train_loader, test_loader, config=None, freeze_backbone=False):
    """Train a CNNBiLSTMClassifier for the fixed schedule; return final metrics.

    If ``freeze_backbone`` is True, only the classifier head is trained (the
    backbone parameters are frozen), isolating head capacity from the features.
    """
    config = config or {}
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]

    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    wd = config.get('weight_decay', WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=epochs,
        warmup_lr=WARMUP_LR, target_lr=lr,
    )
    model = model.to(DEVICE)

    frozen = [model.backbone] if freeze_backbone else None
    t0 = time.time()
    converged = True
    for epoch in range(epochs):
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            grad_accum=config.get('grad_accum', 1),
            grad_clip=config.get('grad_clip', GRAD_CLIP_NORM),
            use_amp=config.get('use_amp', USE_AMP),
            frozen_modules=frozen,
        )
        if np.isnan(train_loss):
            converged = False
            break
        scheduler.step()
    train_time = time.time() - t0

    result = evaluate(model, test_loader, DEVICE)
    return {
        'accuracy': float(result['accuracy']),
        'macro_f1': float(result['macro_f1']),
        'per_class_f1': (result['per_class_f1'].tolist()
                         if hasattr(result['per_class_f1'], 'tolist')
                         else result['per_class_f1']),
        'train_time': train_time,
        'converged': converged,
    }


_FROZEN_HEAD_SPECS = [
    ('frozen_linear',    build_cnn_bilstm_linear),
    ('frozen_small_mlp', build_cnn_bilstm_small_mlp),
    ('frozen_equal_mlp', build_cnn_bilstm_equal_mlp),
    ('frozen_kan',       lambda ic, nc: build_cnn_bilstm_kan(ic, nc, kan_type='fast')),
]


def _run_frozen_unit(in_channels, n_classes, train_loader, test_loader, seed, config):
    """One (fold, seed) unit: train the ref (matched MLP) model once, then
    retrain each head on the frozen backbone. Returns one record per head.

    Each head is re-seeded before construction so its classifier has a
    deterministic, ref-independent initialization.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    ref_model = build_cnn_bilstm_equal_mlp(in_channels, n_classes)
    ref_res = _train_model(ref_model, train_loader, test_loader, config=config)
    if not ref_res['converged']:
        # Ref model diverged, the frozen features are garbage; skip the four
        # heads so this (fold, seed) unit stays incomplete and is retried.
        return []

    records = []
    for head_name, builder in _FROZEN_HEAD_SPECS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = builder(in_channels, n_classes)
        model.backbone.load_state_dict(ref_model.backbone.state_dict())
        res = _train_model(model, train_loader, test_loader,
                           config=config, freeze_backbone=True)
        records.append({
            'model': head_name,
            'accuracy': res['accuracy'],
            'macro_f1': res['macro_f1'],
            'per_class_f1': res['per_class_f1'],
            'head_param_count': sum(p.numel() for p in model.classifier.parameters()),
            'ref_accuracy': ref_res['accuracy'],
            'ref_macro_f1': ref_res['macro_f1'],
            'converged': bool(res['converged'] and ref_res['converged']),
        })
    return records


def run_frozen_backbone_phase():
    """Frozen-backbone experiment (Reviewer 3 C1, Reviewer 1.5).

    For each fold x seed: train the full model ONCE (backbone + matched MLP
    head), freeze the backbone, then retrain each of the four heads on the
    frozen features. This isolates the head's own capacity from backbone
    quality and directly tests the linear-separability reading.

    A (fold, seed) unit is atomic and written only after all four heads finish;
    resume skips units that already have all four heads recorded. On a
    multi-GPU machine the incomplete folds are distributed across the worker
    pool (one subprocess per GPU).
    """
    print("=" * 60)
    print("PHASE 6: Frozen-Backbone (head capacity isolation)")
    print("=" * 60)

    head_names = [n for n, _ in _FROZEN_HEAD_SPECS]
    results_path = os.path.join(RESULTS_DIR, 'frozen_backbone_results.json')

    # Existing results keyed by (dataset, model, fold, seed), merges across
    # --dataset invocations and resumes after a crash.
    existing = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                for r in json.load(f):
                    existing[(r['dataset'], r['model'], r['fold'], r['seed'])] = r
        except Exception:
            pass

    def unit_done(ds, fold, seed):
        return all((ds, hn, fold, seed) in existing for hn in head_names)

    def save_all():
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(list(existing.values()), f, indent=2)

    worker_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        'src', 'training', '_win_worker.py')

    for ds in ALL_DATASETS:
        data = load_dataset(ds)
        if not data:
            print(f"  SKIP {ds}: no data")
            continue
        n_classes = len(set(l for d in data.values() for l in np.unique(d['y'])))
        in_channels = next(iter(data.values()))['x'].shape[1]
        splits = list(get_cv_splits(data, ds, n_folds=5, seed=42))
        bs = DATASET_BATCH_SIZES.get(ds, {}).get('batch_size', BATCH_SIZE)

        incomplete_folds = sorted({fi for fi in range(len(splits))
                                   for s in SEEDS if not unit_done(ds, fi, s)})
        if not incomplete_folds:
            print(f"  {ds}: all {len(splits)} folds x {len(SEEDS)} seeds done")
            continue
        print(f"\n{ds}: {len(splits)} folds, {in_channels} ch, {n_classes} classes; "
              f"{len(incomplete_folds)} incomplete folds")

        done_units = [[fi, s] for fi in range(len(splits)) for s in SEEDS
                      if unit_done(ds, fi, s)]
        train_config = {
            'batch_size': bs, 'epochs': EPOCHS, 'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP_NORM,
            'use_amp': USE_AMP,
        }

        if USE_MULTI_GPU:
            gpu_assignments = [[] for _ in range(N_GPUS)]
            for i, fi in enumerate(incomplete_folds):
                gpu_assignments[i % N_GPUS].append(fi)

            procs = []
            for gpu_id in range(N_GPUS):
                if not gpu_assignments[gpu_id]:
                    continue
                task = {
                    'dataset_key': ds,
                    'folds': gpu_assignments[gpu_id],
                    'seeds': SEEDS,
                    'frozen': True,
                    'model_name': 'frozen_backbone',
                    'config': dict(train_config),
                    'completed_runs': done_units,
                }
                task_file = os.path.join(
                    RESULTS_DIR, f'_winfrozentask_{ds}_gpu{gpu_id}.json')
                with open(task_file, 'w') as f:
                    json.dump(task, f)
                log_file = task_file.replace('.json', '_log.txt')
                log_fh = open(log_file, 'w')
                proc = subprocess.Popen(
                    [sys.executable, worker_script, str(gpu_id), task_file],
                    stdout=log_fh, stderr=subprocess.STDOUT,
                )
                log_fh.close()
                procs.append((gpu_id, proc, task_file, log_file))
                print(f"  [GPU {gpu_id}] launched -> folds {gpu_assignments[gpu_id]}",
                      flush=True)

            for gpu_id, proc, task_file, log_file in procs:
                proc.wait()
                if os.path.exists(log_file):
                    with open(log_file) as f:
                        content = f.read()
                    if content:
                        print(content, end='', flush=True)
                    os.remove(log_file)
                if proc.returncode != 0:
                    print(f"  [GPU {gpu_id}] FAILED (exit {proc.returncode})",
                          flush=True)
                out_file = task_file.replace('.json', f'_out_gpu{gpu_id}.json')
                if os.path.exists(out_file):
                    try:
                        with open(out_file) as f:
                            worker_results = json.load(f)
                    except Exception:
                        print(f"  [GPU {gpu_id}] out file corrupt, skipping",
                              flush=True)
                        worker_results = []
                    for r in worker_results:
                        existing[(r['dataset'], r['model'], r['fold'], r['seed'])] = r
                    print(f"  [GPU {gpu_id}] collected {len(worker_results)} records",
                          flush=True)
                    save_all()
                    os.remove(out_file)
                else:
                    print(f"  [GPU {gpu_id}] no output, all tasks failed",
                          flush=True)
                if os.path.exists(task_file):
                    os.remove(task_file)
        else:
            for fold_idx in incomplete_folds:
                test_subjects = splits[fold_idx]
                X_train, y_train, X_test, y_test, train_s, test_s = \
                    subject_wise_split(data, test_subjects, normalize=True)
                train_loader = DataLoader(
                    HARWindowDataset(X_train, y_train), batch_size=bs,
                    shuffle=True, drop_last=True, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=PERSISTENT_WORKERS,
                )
                test_loader = DataLoader(
                    HARWindowDataset(X_test, y_test), batch_size=bs,
                    shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=True, persistent_workers=PERSISTENT_WORKERS,
                )
                for seed in SEEDS:
                    if unit_done(ds, fold_idx, seed):
                        continue
                    records = _run_frozen_unit(in_channels, n_classes,
                                               train_loader, test_loader, seed,
                                               train_config)
                    for r in records:
                        r['dataset'] = ds
                        r['fold'] = fold_idx
                        r['seed'] = seed
                        r['test_subjects'] = test_s
                        r['train_subjects'] = train_s
                        r['n_train'] = len(X_train)
                        r['n_test'] = len(X_test)
                        existing[(ds, r['model'], fold_idx, seed)] = r
                    print(f"  fold {fold_idx + 1}/{len(splits)} seed {seed}: "
                          + ", ".join(f"{r['model']}={r['accuracy']:.4f}"
                                      for r in records), flush=True)
                    save_all()

    print(f"\nSaved {len(existing)} frozen-backbone records -> {results_path}")
    return list(existing.values())


# ============================================================
# Phase Registry
# ============================================================

PHASES = {
    'heads':             run_heads_phase,
    'baselines':         run_baseline_phase,
    'g_ablation':        run_g_ablation_phase,
    'frozen_backbone':   run_frozen_backbone_phase,
}


def main():
    parser = argparse.ArgumentParser(description='CNN-BiLSTM HAR Experiments')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'heads', 'baselines', 'g_ablation', 'frozen_backbone'])
    parser.add_argument('--dataset', type=str, default=None,
                       help='Run on specific dataset only')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(N_GPUS):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({vram:.1f} GB)")
        if USE_MULTI_GPU:
            print(f"  -> multi-GPU pool: {N_GPUS} GPUs "
                  f"(subprocess per GPU, no DataParallel overhead)")
        else:
            print(f"  -> single-GPU mode")

    if args.dataset:
        global ALL_DATASETS, G_ABLATION_DATASETS
        ALL_DATASETS = [args.dataset]
        G_ABLATION_DATASETS = [args.dataset]

    if args.phase == 'all':
        phases = ['heads', 'baselines', 'g_ablation', 'frozen_backbone']
    else:
        phases = [args.phase]

    for phase in phases:
        if phase in PHASES:
            PHASES[phase]()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
