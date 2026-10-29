"""Windows multi-GPU worker, launched as a subprocess per GPU.

Usage (internal, called by cv_runner):
    python _win_worker.py <gpu_id> <task_file.json>

Sets CUDA_VISIBLE_DEVICES BEFORE any torch import so each worker
sees exactly one GPU.  Each worker processes its assigned folds
independently; no GPU-to-GPU communication needed.
"""
import os
import sys
import json

if __name__ != '__main__':
    pass
else:
    gpu_id = sys.argv[1]
    task_file = sys.argv[2]

    # -- CRITICAL: set before any torch import ------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # Now safe to import torch + project modules
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    _root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sys.path.insert(0, _root)

    from src.config import (
        BATCH_SIZE, NUM_WORKERS, PERSISTENT_WORKERS,
    )
    from src.data.datasets import (
        load_dataset, get_cv_splits, subject_wise_split,
        HARWindowDataset,
    )
    from src.training.trainer import count_parameters
    from src.training.cv_runner import run_single_fold
    from src.models import CNNBiLSTMClassifier

    # -- Model builders (must match run_all.py) ------------------
    from src.experiments.run_all import (
        build_cnn_bilstm_small_mlp,
        build_cnn_bilstm_equal_mlp,
        build_cnn_bilstm_kan,
        build_cnn_bilstm_linear,
        build_lstm_fcn,
        build_deepconvlstm,
        _run_frozen_unit,
    )

    # -- Load tasks ---------------------------------------------
    with open(task_file, 'r') as f:
        task_spec = json.load(f)

    dataset_key = task_spec['dataset_key']
    assigned_folds = task_spec['folds']
    seeds = task_spec['seeds']
    model_name = task_spec['model_name']
    config = task_spec['config']
    completed_runs = set(tuple(r) for r in task_spec.get('completed_runs', []))

    # -- Model dispatch -----------------------------------------
    # Classification heads (prefix-matched so ablation names like
    # 'head_small_mlp_bs256_es20' still resolve to the right builder).
    _head_prefixes = (
        ('head_linear',    build_cnn_bilstm_linear),
        ('head_small_mlp', build_cnn_bilstm_small_mlp),
        ('head_equal_mlp', build_cnn_bilstm_equal_mlp),
    )

    # KAN models: the KAN head (main) and the G-ablation grid sizes.
    _kan_prefixes = ('head_kan', 'kan_G')

    # Conventional baselines
    _baseline_names = {'lstm_fcn', 'deepconvlstm'}

    frozen = task_spec.get('frozen', False)

    model_builder = None
    if not frozen:
        if any(model_name.startswith(p) for p, _ in _head_prefixes):
            model_builder = next(b for p, b in _head_prefixes
                                 if model_name.startswith(p))
        elif model_name in _baseline_names:
            model_builder = build_lstm_fcn if model_name == 'lstm_fcn' else build_deepconvlstm
        elif any(model_name.startswith(p) for p in _kan_prefixes):
            kan_type = config.get('kan_type', 'fast')
            grid_size = config.get('grid_size', None)
            model_builder = lambda ic, nc: build_cnn_bilstm_kan(ic, nc, grid_size=grid_size, kan_type=kan_type)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Expected one of {[p for p, _ in _head_prefixes]} "
                f"or starts with {list(_kan_prefixes)} "
                f"or {sorted(_baseline_names)}"
            )

    is_kan = any(model_name.startswith(p) for p in _kan_prefixes)

    # -- Load data ----------------------------------------------
    print(f"[GPU {gpu_id}] loading {dataset_key}...", flush=True)
    data = load_dataset(dataset_key)

    n_classes = len(set(l for d in data.values()
                        for l in np.unique(d['y'])))
    in_channels = next(iter(data.values()))['x'].shape[1]

    splits = list(get_cv_splits(data, dataset_key,
                                n_folds=config.get('n_folds', 5), seed=42))

    if not frozen:
        temp_model = model_builder(in_channels, n_classes)
        param_count = count_parameters(temp_model)
        del temp_model
    else:
        param_count = 0

    print(f"[GPU {gpu_id}] {n_classes} classes, {in_channels} channels, "
          f"assigned folds {assigned_folds}", flush=True)

    # -- Process assigned folds ---------------------------------
    results = []
    for fold_idx in assigned_folds:
        test_subjects = splits[fold_idx]
        X_train, y_train, X_test, y_test, train_s, test_s = \
            subject_wise_split(data, test_subjects, normalize=True)

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

        for seed in seeds:
            if (fold_idx, seed) in completed_runs:
                print(f"[GPU {gpu_id}] fold={fold_idx} seed={seed} SKIP (done)",
                      flush=True)
                continue

            if frozen:
                records = _run_frozen_unit(in_channels, n_classes,
                                           train_loader, test_loader, seed,
                                           config)
                for r in records:
                    r['dataset'] = dataset_key
                    r['fold'] = fold_idx
                    r['seed'] = seed
                    r['test_subjects'] = test_s
                    r['train_subjects'] = train_s
                    r['n_train'] = len(X_train)
                    r['n_test'] = len(X_test)
                results.extend(records)
                out_file = task_file.replace('.json', f'_out_gpu{gpu_id}.json')
                with open(out_file, 'w') as f:
                    json.dump(results, f)
                print(f"[GPU {gpu_id}] fold={fold_idx} seed={seed} "
                      + ", ".join(f"{r['model']}={r['accuracy']:.4f}"
                                  for r in records), flush=True)
                continue

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_builder(in_channels, n_classes)

            fold_name = f"{dataset_key}_fold{fold_idx}_seed{seed}"
            fold_result = run_single_fold(
                model, train_loader, test_loader, fold_name, config)
            if fold_result.get('status') == 'failed':
                print(f"[GPU {gpu_id}] fold={fold_idx} seed={seed} FAILED (NaN), skip",
                      flush=True)
                continue

            fold_result['fold'] = fold_idx
            fold_result['seed'] = seed
            fold_result['test_subjects'] = test_s
            fold_result['train_subjects'] = train_s
            fold_result['n_train'] = len(X_train)
            fold_result['n_test'] = len(X_test)
            fold_result['param_count'] = param_count
            fold_result['protocol'] = config.get('protocol')

            results.append(fold_result)
            # Save incrementally, crash-proof
            out_file = task_file.replace('.json', f'_out_gpu{gpu_id}.json')
            with open(out_file, 'w') as f:
                json.dump(results, f)
            print(f"[GPU {gpu_id}] fold={fold_idx} seed={seed} "
                  f"acc={fold_result['accuracy']:.4f} "
                  f"F1={fold_result['macro_f1']:.4f}",
                  flush=True)

    print(f"[GPU {gpu_id}] done, {len(results)} runs",
          flush=True)
