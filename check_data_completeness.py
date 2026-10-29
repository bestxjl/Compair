#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data completeness check for HHAR / RealWorld / PAMAP2.

Run this once before starting experiments to confirm the data loads and has the
expected subject counts, channels, and class distributions.

Usage: python check_data_completeness.py

Checks:
  1. Data paths exist and the loader runs.
  2. Subject count matches expectation.
  3. Channel count per subject matches expectation.
  4. Per-subject class distribution (samples per class).
  5. Missing classes (a subject with zero samples for a class, which would
     zero out that class's F1 in macro-F1).
  6. Total window count near the historical value (to catch data loss).

Exit code: 0 = all pass, 1 = anomalies found.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR
from src.data.datasets import load_hhar, load_realworld, load_pamap2


# Expected values verified against a known-good run.
EXPECTED = {
    'hhar': {
        'name': 'HHAR',
        'n_subjects': 9,
        'n_classes': 6,
        'n_channels': 6,
        'classes': ['Bike', 'Sit', 'StairsD', 'StairsU', 'Stand', 'Walk'],
        'expected_windows': (170000, 185000),   # 176,225 including samsungold
        'note': 'Four phone models (nexus4/s3/s3mini/samsungold), all kept.',
    },
    'realworld': {
        'name': 'RealWorld',
        'n_subjects': 15,
        'n_classes': 6,
        'n_channels': 6,
        'classes': ['Walking', 'Running', 'Sitting', 'Standing', 'StairsUp', 'StairsDown'],
        'expected_windows': (300000, 320000),   # 308,521 including nested stair-climb sessions
        'note': 'Stair-climb data is nested sessions; the loader reads them recursively.',
    },
    'pamap2': {
        'name': 'PAMAP2',
        'n_subjects': 8,                         # subject 109 is empty and skipped
        'n_classes': 6,
        'n_channels': 9,
        'classes': ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling'],
        'expected_windows': (8000, 9000),        # 8,340
        'note': 'Subject 103 lacks Running and Cycling, subject 104 lacks Running; this is genuine missing data.',
    },
}


def check_dataset(key, loader, cfg):
    print('=' * 78)
    print(f"[{cfg['name']}] checking ... (loader: {loader.__name__})")
    print('=' * 78)

    ok = True

    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  note: {cfg['note']}")

    try:
        data = loader()
    except Exception as e:
        print(f"  [FAIL] load error: {e}")
        return False

    n_subj = len(data)
    if n_subj == 0:
        print(f"  [FAIL] zero subjects loaded: check the data path")
        return False

    if n_subj != cfg['n_subjects']:
        print(f"  [WARN] subject count {n_subj} != expected {cfg['n_subjects']}")
        ok = False
    else:
        print(f"  [OK]   subject count = {n_subj}")

    print(f"  {'subject':>8s} {'windows':>8s} {'ch':>3s}   " + "  ".join(f"{c:>8s}" for c in cfg['classes']))
    total_windows = 0
    n_missing_class = 0
    for sid in sorted(data.keys()):
        d = data[sid]
        x = d['x']
        y = d['y']
        n = len(y)
        total_windows += n

        ch = x.shape[1] if x.ndim == 3 else x.shape[0]
        ch_flag = '' if ch == cfg['n_channels'] else f'  <-- channel mismatch (expected {cfg["n_channels"]})'
        if ch != cfg['n_channels']:
            ok = False

        counts = [int((y == c).sum()) for c in range(cfg['n_classes'])]
        missing = [cfg['classes'][c] for c in range(cfg['n_classes']) if counts[c] == 0]

        row = f"  {sid:>8d} {n:>8d} {ch:>3d}   " + "  ".join(f"{c:>8d}" for c in counts)
        print(row + ch_flag)

        if missing:
            n_missing_class += 1
            print(f"         ^^ missing classes: {missing}  (F1 will be zero for these; see the macro-F1 note)")

    lo, hi = cfg['expected_windows'] if cfg['expected_windows'] else (0, 0)
    print(f"  total windows = {total_windows}", end='')
    if lo and hi:
        if lo <= total_windows <= hi:
            print(f"  [OK] in expected range [{lo}, {hi}]")
        else:
            print(f"  [WARN] outside expected range [{lo}, {hi}]: data may be incomplete or duplicated")
            ok = False
    else:
        print(f"  (first run, no historical reference)")

    if n_missing_class == 0:
        print(f"  [OK]   no missing classes")
    else:
        print(f"  [WARN] {n_missing_class} subjects have missing classes: macro-F1 uses present classes")
        # Missing classes in PAMAP2 are a known property of the dataset, not a failure.
        if key != 'pamap2':
            ok = False

    return ok


def main():
    print('#' * 78)
    print('#  Data completeness check  HHAR / RealWorld / PAMAP2')
    print('#' * 78)
    print()

    results = {}
    results['hhar'] = check_dataset('hhar', load_hhar, EXPECTED['hhar'])
    print()
    results['realworld'] = check_dataset('realworld', load_realworld, EXPECTED['realworld'])
    print()
    results['pamap2'] = check_dataset('pamap2', load_pamap2, EXPECTED['pamap2'])
    print()

    print('=' * 78)
    all_ok = all(results.values())
    for k, v in results.items():
        print(f"  {EXPECTED[k]['name']:<12s}: {'PASS' if v else 'FAIL/CHECK'}")
    print('=' * 78)
    print(f"  summary: {'all pass, ready to run experiments' if all_ok else 'anomalies found, check the [WARN]/[FAIL] lines above'}")
    print('=' * 78)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
