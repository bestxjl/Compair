"""Subject-wise cross-validation training orchestration.

Single-GPU     : uses DEVICE (cuda:0 or cpu).
Multi-GPU Linux : DistributedDataParallel (DDP) with NCCL, one process per GPU.
Multi-GPU Win   : subprocess pool, one process per GPU, independent training.
"""
import os
import sys
import json
import time
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR,
    BATCH_SIZE, BATCH_SIZE_KAN, GRAD_ACCUM_KAN,
    LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING,
    GRAD_CLIP_NORM, EPOCHS, WARMUP_EPOCHS, WARMUP_LR,
    N_GPUS, USE_MULTI_GPU,
    NUM_WORKERS, PERSISTENT_WORKERS, USE_AMP,
)
from ..data.datasets import (
    load_dataset, get_cv_splits, subject_wise_split,
    HARWindowDataset, NoisyHARWindowDataset,
)
from .trainer import (
    train_epoch, evaluate, compute_silhouette,
    count_parameters, measure_latency,
    CosineWarmupScheduler,
)


# ============================================================
# Single-GPU training
# ============================================================

def run_single_fold(model, train_loader, test_loader, fold_name, config):
    """Train and evaluate on one CV fold (single-GPU / Windows worker).

    Fixed-schedule protocol: train for the full ``epochs`` schedule and report
    the FINAL epoch. There is no early stopping and no test-based checkpoint
    selection. The test set is evaluated exactly once, at the final epoch; only
    training-set metrics are logged to ``history`` (for training curves).

    Args:
        model: nn.Module
        train_loader, test_loader: DataLoader
        fold_name: str identifier for checkpoint naming
        config: dict with training overrides

    Returns:
        dict with metrics (plus per-epoch ``history``)
    """
    epochs = config.get('epochs', EPOCHS)
    lr = config.get('lr', LEARNING_RATE)
    wd = config.get('weight_decay', WEIGHT_DECAY)
    grad_accum = config.get('grad_accum', 1)
    is_kan = config.get('is_kan', False)
    batch_size = config.get('batch_size', BATCH_SIZE)
    noise_snr = config.get('noise_snr', None)  # SNR in dB for noise robustness tests
    # grad_clip: unified across all heads (explicit override still wins)
    grad_clip = config.get('grad_clip', GRAD_CLIP_NORM)

    model = model.to(DEVICE)

    # If noise SNR is specified, wrap test data with noise for evaluation
    if noise_snr is not None:
        test_dataset_clean = test_loader.dataset
        # Guard: verify the dataset has .X/.y numpy access
        if not hasattr(test_dataset_clean, 'X') or not hasattr(test_dataset_clean, 'y'):
            raise TypeError(
                f"Cannot apply noise: test_loader.dataset is "
                f"{type(test_dataset_clean).__name__}, expected HARWindowDataset"
            )
        X_test = test_dataset_clean.X.numpy()
        y_test = test_dataset_clean.y.numpy()
        noisy_test_dataset = NoisyHARWindowDataset(X_test, y_test, noise_snr)
        eval_test_loader = DataLoader(
            noisy_test_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=0,  # No multi-worker for small noise dataset
        )
    else:
        eval_test_loader = test_loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=epochs,
        warmup_lr=WARMUP_LR, target_lr=lr,
    )

    train_time_start = time.time()
    # KAN epochs are ~20 min each; first eval at epoch 5 needs ~100 min.
    # MLP epochs are ~1-2 min each on typical datasets.
    MAX_FOLD_SECONDS = 21600 if is_kan else 7200  # 6 h / 2 h
    history = []  # per-epoch metrics for training/validation curves

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            grad_accum=grad_accum, grad_clip=grad_clip,
            use_amp=config.get('use_amp', USE_AMP),
        )
        scheduler.step()

        # Detect NaN loss -> abort fold
        if np.isnan(train_loss):
            print(f"  [CONVERGENCE FAIL] NaN loss, aborting fold: training diverged, "
                  f"batch size may be too large, try a smaller bs")
            return {'accuracy': 0.0, 'macro_f1': 0.0, 'silhouette': 0.0,
                    'train_time': 0, 'latency_ms': 0, 'final_epoch_acc': 0.0,
                    'per_class_f1': [0.0] * 6, 'history': [], 'status': 'failed'}

        # Safety timeout (should never trigger on the fixed schedule)
        if time.time() - train_time_start > MAX_FOLD_SECONDS:
            print(f"  WARNING: fold timeout after {MAX_FOLD_SECONDS}s, "
                  f"stopping at epoch {epoch + 1}")
            break

        # Log per-epoch TRAINING metrics only. The test set is evaluated exactly
        # once, at the final epoch (below), never during training, so test-fold
        # information cannot influence any step of the protocol.
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
        })

    train_time = time.time() - train_time_start

    # Fixed-schedule protocol: the model is already at its final trained state
    # (last epoch of the schedule). No best-epoch reload, no test-based selection.
    # Final evaluation: for noise experiments use noisy data (the metric of
    # interest); for all others use clean data (needed for Silhouette).
    if noise_snr is not None:
        # Noise experiment: report accuracy ON the noisy test set
        final_result = evaluate(model, eval_test_loader, DEVICE, extract_features=False)
        sil_score = 0.0  # Silhouette on noisy features is not meaningful
    else:
        final_result = evaluate(model, test_loader, DEVICE, extract_features=True)
        features = final_result.get('features', None)
        sil_score = 0.0
        if features is not None and len(features) > 50:
            sil_labels = final_result.get('labels_for_features',
                                          final_result['labels'])
            sil_score = compute_silhouette(features, sil_labels)

    # Convergence guard: flag folds that barely beat chance (non-convergence,
    # e.g. batch size too large), so the user can reduce bs and re-run.
    if final_result['accuracy'] < 0.25:
        print(f"  [CONVERGENCE FAIL] fold accuracy {final_result['accuracy']:.3f} "
              f"near chance (about 0.167 for 6 classes), likely not converged, try a smaller batch size")

    # Measure latency
    sample_shape = next(iter(test_loader))[0][:1].shape
    latency = measure_latency(model, sample_shape, DEVICE)

    return {
        'accuracy': final_result['accuracy'],
        'macro_f1': final_result['macro_f1'],
        'per_class_f1': (final_result['per_class_f1'].tolist()
                         if hasattr(final_result['per_class_f1'], 'tolist')
                         else final_result['per_class_f1']),
        'silhouette': float(sil_score),
        'train_time': train_time,
        'latency_ms': latency,
        'final_epoch_acc': float(final_result['accuracy']),
        'history': history,
    }


# Main CV entry point
# ============================================================

def run_cv(model_builder, dataset_key, config=None):
    """Run full subject-wise cross-validation.

    Args:
        model_builder: fn(in_channels, n_classes) -> model
        dataset_key: str, one of DATASETS keys
        config: dict with keys:
            - n_folds: int (default from DATASETS)
            - seeds: list of int (for multi-seed runs)
            - epochs, lr, weight_decay, batch_size, grad_accum, is_kan

    Returns:
        (all_results: list[dict], summary: dict)
    """
    if config is None:
        config = {}
    seeds = config.get('seeds', [42])
    is_kan = config.get('is_kan', False)

    # ---- load data -------------------------------------------------------
    print(f"Loading {dataset_key}...")
    try:
        data = load_dataset(dataset_key)
    except (FileNotFoundError, Exception) as e:
        print(f"  SKIP: Cannot load {dataset_key} - {e}")
        return [], {}
    if not data:
        print(f"  SKIP: no data for {dataset_key}")
        return [], {}

    n_classes = len(set(l for d in data.values()
                        for l in np.unique(d['y'])))
    in_channels = next(iter(data.values()))['x'].shape[1]

    # Compute param count once (same architecture for all folds × seeds)
    _temp = model_builder(in_channels, n_classes)
    param_count = count_parameters(_temp)
    del _temp

    # ---- CV splits -------------------------------------------------------
    splits = list(get_cv_splits(data, dataset_key,
                                n_folds=config.get('n_folds', 5),
                                seed=42))
    n_folds = len(splits)

    # Protocol signature: guards resume against silently mixing results from a
    # different CV protocol (e.g. 5-fold -> LOSO) or a fixed data loader (which
    # changes the window count). Stamped into every record so stale files are
    # detected on the next run.
    protocol = f"{dataset_key}|f={n_folds}|s={len(data)}|w={sum(len(d['x']) for d in data.values())}"
    config['protocol'] = protocol

    # ---- resume from partial results -------------------------------------
    result_path = os.path.join(
        RESULTS_DIR,
        f"{dataset_key}_{config.get('model_name', 'model')}_cv.json"
    )
    completed_runs = set()
    all_results = []
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                prev = json.load(f)
            if prev and prev[0].get('protocol') != protocol:
                print(f"  Protocol changed ({protocol}), discarding stale "
                      f"results, starting fresh", flush=True)
                prev = []
            completed_runs = set((r['fold'], r['seed']) for r in prev)
            all_results = prev
            done_folds = set(r['fold'] for r in prev)
            print(f"  Resuming: {len(completed_runs)} runs done "
                  f"across {len(done_folds)} folds", flush=True)
        except Exception:
            pass

    if USE_MULTI_GPU:
        mode = f"multi-GPU pool ({N_GPUS} GPUs)"
    else:
        mode = "single-GPU"
    print(f"  {n_folds} folds, {len(data)} subjects, {in_channels} channels, "
          f"{n_classes} classes  [{mode}]", flush=True)

    # ================================================================
    # PATH A: Windows multi-GPU  ->  subprocess pool (one per GPU)
    # ================================================================
    if USE_MULTI_GPU:

        # 0. Merge any orphaned worker output files into completed_runs
        #    (left over from a previous run that was killed)
        _model_key = config.get('model_name', 'model')
        _orphan_pattern = f'_wintask_{dataset_key}_{_model_key}_gpu'
        _seen = set((r['fold'], r['seed']) for r in all_results)
        _seed_set = set(seeds)
        _orphan_files = []
        for _gid in range(N_GPUS):
            _out = os.path.join(RESULTS_DIR,
                                f'{_orphan_pattern}{_gid}_out_gpu{_gid}.json')
            if os.path.exists(_out):
                try:
                    with open(_out) as _f:
                        _prev = json.load(_f)
                    _added = _stale = 0
                    for _r in _prev:
                        _key = (_r['fold'], _r['seed'])
                        # Reject stale orphans (old protocol / 5-fold run /
                        # foreign seed) so they can't contaminate a new protocol.
                        if _r.get('protocol') != protocol:
                            _stale += 1
                            continue
                        if not (_key[0] < n_folds and _key[1] in _seed_set):
                            _stale += 1
                            continue
                        if _key not in _seen:
                            _seen.add(_key)
                            completed_runs.add(_key)
                            all_results.append(_r)
                            _added += 1
                    print(f"  [GPU {_gid}] recovered {_added} orphan results "
                          f"({_stale} stale, {len(_prev) - _added - _stale} known)",
                          flush=True)
                    _orphan_files.append(_out)
                except Exception:
                    pass
        # Persist BEFORE deleting the orphan files: the orphan is the only copy
        # until result_path is written, so a crash in between must not lose it.
        if _orphan_files:
            with open(result_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            for _out in _orphan_files:
                try:
                    os.remove(_out)
                except OSError:
                    pass

        # 1. Assign folds to GPUs round-robin
        all_fold_indices = [i for i in range(n_folds)
                            if not {s for (f, s) in completed_runs if f == i} == set(seeds)]
        gpu_assignments = [[] for _ in range(N_GPUS)]
        for i, fi in enumerate(all_fold_indices):
            gpu_assignments[i % N_GPUS].append(fi)

        # 2. Write task files + launch subprocesses (all GPUs in parallel)
        worker_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            'src', 'training', '_win_worker.py')

        procs = []
        for gpu_id in range(N_GPUS):
            if not gpu_assignments[gpu_id]:
                continue

            task = {
                'dataset_key': dataset_key,
                'folds': gpu_assignments[gpu_id],
                'seeds': seeds,
                'model_name': config.get('model_name', 'model'),
                'config': config,
                'completed_runs': [[f, s] for (f, s) in completed_runs],
            }

            task_file = os.path.join(
                RESULTS_DIR,
                f'_wintask_{dataset_key}_{config.get("model_name","model")}'
                f'_gpu{gpu_id}.json')
            with open(task_file, 'w') as f:
                json.dump(task, f)

            log_file = task_file.replace('.json', '_log.txt')
            log_fh = open(log_file, 'w')
            proc = subprocess.Popen(
                [sys.executable, worker_script, str(gpu_id), task_file],
                stdout=log_fh, stderr=subprocess.STDOUT,
            )
            log_fh.close()  # close parent's copy; child inherited its own
            procs.append((gpu_id, proc, task_file, log_file))
            print(f"  [GPU {gpu_id}] launched -> folds {gpu_assignments[gpu_id]}",
                  flush=True)

        # 3. Wait for ALL workers, then print logs + collect results
        for gpu_id, proc, task_file, log_file in procs:
            proc.wait()
            # Print worker output
            if os.path.exists(log_file):
                with open(log_file) as f:
                    content = f.read()
                if content:
                    print(content, end='', flush=True)
                os.remove(log_file)

            if proc.returncode != 0:
                print(f"  [GPU {gpu_id}] FAILED (exit {proc.returncode})",
                      flush=True)

            # Collect results
            out_file = task_file.replace('.json', f'_out_gpu{gpu_id}.json')
            if os.path.exists(out_file):
                try:
                    with open(out_file) as f:
                        worker_results = json.load(f)
                except Exception:
                    print(f"  [GPU {gpu_id}] out file corrupt/truncated, skipping",
                          flush=True)
                    worker_results = []
                all_results.extend(worker_results)
                print(f"  [GPU {gpu_id}] collected {len(worker_results)} results",
                      flush=True)
                # Save incrementally (crash-proof)
                os.makedirs(RESULTS_DIR, exist_ok=True)
                with open(result_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                os.remove(out_file)
            else:
                print(f"  [GPU {gpu_id}] no output, all tasks failed",
                      flush=True)
            # Clean up task file
            if os.path.exists(task_file):
                os.remove(task_file)

    # ================================================================
    # PATH B: DDP (Linux + NCCL)  or  PATH C: single-GPU
    # ================================================================
    else:
        for fold_idx, test_subjects in enumerate(splits):
            fold_seeds_done = {s for (f, s) in completed_runs if f == fold_idx}
            if fold_seeds_done == set(seeds):
                print(f"  Fold {fold_idx + 1}/{n_folds}, SKIP "
                      f"(all {len(seeds)} seeds done)", flush=True)
                continue
            elif fold_seeds_done:
                print(f"  Fold {fold_idx + 1}/{n_folds} "
                      f"(test subjects: {test_subjects}) "
                      f",  resuming, {len(fold_seeds_done)}/{len(seeds)} seeds done",
                      flush=True)
            else:
                print(f"  Fold {fold_idx + 1}/{n_folds} "
                      f"(test subjects: {test_subjects})", flush=True)

            # Split
            X_train, y_train, X_test, y_test, train_s, test_s = \
                subject_wise_split(data, test_subjects, normalize=True)

            # Subsampling for data efficiency experiments
            data_ratio = config.get('data_ratio', 1.0)
            if data_ratio < 1.0:
                rng = np.random.RandomState(42)  # fixed seed for reproducibility
                n_total = len(X_train)
                bsz = config.get('batch_size', BATCH_SIZE)
                n_subset = max(int(n_total * data_ratio), bsz * 2)  # at least 2 full batches
                indices = rng.choice(n_total, n_subset, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]

            # DataLoaders
            train_loader = DataLoader(
                HARWindowDataset(X_train, y_train),
                batch_size=config.get('batch_size', BATCH_SIZE),
                shuffle=True, drop_last=True,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=PERSISTENT_WORKERS,
            )
            test_loader = DataLoader(
                HARWindowDataset(X_test, y_test),
                batch_size=config.get('batch_size', BATCH_SIZE),
                shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=PERSISTENT_WORKERS,
            )

            # Multi-seed
            for seed in seeds:
                if (fold_idx, seed) in completed_runs:
                    print(f"    seed={seed}: SKIP (already done)", flush=True)
                    continue

                fold_name = f"{dataset_key}_fold{fold_idx}_seed{seed}"

                torch.manual_seed(seed)
                np.random.seed(seed)
                model = model_builder(in_channels, n_classes)
                fold_result = run_single_fold(
                    model, train_loader, test_loader, fold_name, config,
                )
                if fold_result.get('status') == 'failed':
                    print(f"    seed={seed}: FAILED (NaN), not saved, will retry", flush=True)
                    continue

                fold_result['fold'] = fold_idx
                fold_result['seed'] = seed
                fold_result['test_subjects'] = test_s
                fold_result['train_subjects'] = train_s
                fold_result['n_train'] = len(X_train)
                fold_result['n_test'] = len(X_test)
                fold_result['param_count'] = param_count
                fold_result['protocol'] = protocol

                all_results.append(fold_result)
                print(f"    seed={seed}: acc={fold_result['accuracy']:.4f}, "
                      f"F1={fold_result['macro_f1']:.4f}, "
                      f"sil={fold_result['silhouette']:.3f}, "
                      f"time={fold_result['train_time']:.0f}s", flush=True)

                # Save after EACH run, crash-proof
                os.makedirs(RESULTS_DIR, exist_ok=True)
                with open(result_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # ---- summary ---------------------------------------------------------
    if not all_results:
        print(f"  WARNING: No results collected for {dataset_key}/{config.get('model_name', '?')} "
              f",  skipping summary", flush=True)
        return [], {}

    accs = [r['accuracy'] for r in all_results]
    f1s = [r['macro_f1'] for r in all_results]
    silhs = [r['silhouette'] for r in all_results]

    summary = {
        'dataset': dataset_key,
        'model': config.get('model_name', 'unknown'),
        'n_folds': n_folds,
        'n_seeds': len(seeds),
        'n_runs': len(all_results),
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'accuracy_min': float(np.min(accs)),
        'accuracy_max': float(np.max(accs)),
        'macro_f1_mean': float(np.mean(f1s)),
        'macro_f1_std': float(np.std(f1s)),
        'silhouette_mean': float(np.mean(silhs)),
        'silhouette_std': float(np.std(silhs)),
        'param_count': all_results[0]['param_count'] if all_results else 0,
    }

    return all_results, summary
