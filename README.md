# Rethinking the Role of the Classification Head in CNN-BiLSTM-Based Human Activity Recognition

Code for the paper "Rethinking the Role of the Classification Head in CNN-BiLSTM-Based Human Activity Recognition" (Electronics, MDPI, 2026).

## What the study does

The paper compares four classification heads on a fixed CNN-BiLSTM backbone across seven public human activity recognition datasets. The heads are a 774-parameter linear projection, a small MLP, a parameter-matched MLP, and a Gaussian-RBF KAN. The main finding is that the head has a bounded effect on accuracy: the best-to-worst spread across heads is under 3 percentage points, and the linear head stays within 1.5 percentage points of the best head on every dataset. A frozen-backbone experiment shows the four heads collapse to within about 1 percentage point once the backbone is fixed.

## Layout

- `src/config.py`: hyperparameters and dataset metadata.
- `src/data/datasets.py`: the dataset loaders, preprocessing, windowing, and the subject-wise splits.
- `src/models/`: the CNN-BiLSTM backbone, the four classification heads, the KAN layers, and two external baselines (LSTM-FCN and DeepConvLSTM).
- `src/training/`: the training loop, the cross-validation runner, and the multi-GPU worker.
- `src/experiments/run_all.py`: the command-line entry point for all experiments.
- `check_data_completeness.py`: optional sanity check that prints per-subject class counts before training.

## Requirements

Python 3.8 or newer. Install the dependencies:

    pip install -r requirements.txt

The code was tested with PyTorch 2.13.

## Data

The study uses seven public datasets: UCI-HAR, WISDM, PAMAP2, MotionSense, USC-HAD, HHAR, and RealWorld. Download each one from its original source and put it under `data/raw/` with the folder name its loader expects. The paths are in `src/data/datasets.py`, and `DATA_DIR` is set in `src/config.py`.

## Running the experiments

All commands run from the repository root.

    python -m src.experiments.run_all --phase heads           # four heads, seven datasets
    python -m src.experiments.run_all --phase baselines       # LSTM-FCN and DeepConvLSTM
    python -m src.experiments.run_all --phase g_ablation      # FastKAN grid-size sweep
    python -m src.experiments.run_all --phase frozen_backbone # frozen-backbone experiment

To run one dataset only, add `--dataset NAME`:

    python -m src.experiments.run_all --phase heads --dataset hhar

Results are written to `results_v3/` as JSON, one file per dataset and model. The heads, baselines, and grid-size sweep resume after an interruption; completed fold and seed pairs are skipped.

## Cross-validation and folds

Every dataset uses leave-one-subject-out cross-validation. The splits are produced by `get_cv_splits` in `src/data/datasets.py`, using the dataset metadata in `src/config.py`. All four heads and both baselines share the same folds and the same three random seeds (42, 43, 44), so every pairwise comparison is paired within subjects.

## Reported statistics

The means and standard deviations in the paper are fold-level: the three seeds are averaged within each fold first, and the standard deviation is the population standard deviation over folds. The files in `results_v3/` store one record per fold and seed, so these statistics can be recomputed directly from them.

## License

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{shi2026head,
  title = {Rethinking the Role of the Classification Head in CNN-BiLSTM-Based Human Activity Recognition},
  author = {Shi, Junbing and Wang, Yingmin and Yang, Libo},
  journal = {Electronics},
  year = {2026}
}
```
