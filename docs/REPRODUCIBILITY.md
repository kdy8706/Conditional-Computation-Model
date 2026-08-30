# Reproducibility guide

This repository separates the code needed to reproduce the workflow from research datasets that require permission review.

## Workflow

1. Install the package with `python -m pip install -e ".[dev]"`.
2. Obtain authorized model-ready MATLAB data. Each input file must contain `Dinput2`, `Doutput2`, and `Dpressure2`; their expected shapes are documented in the [data card](DATA_CARD.md).
3. Copy `configs/archive_take5_epoch996_candidate.yaml` and set the two data paths to the authorized local files.
4. Train with:

   ```bash
   python scripts/train.py --config configs/archive_take5_epoch996_candidate.yaml
   ```

   The configuration makes the train/validation split, seed, number of epochs, batch size, learning rate, dropout, loss assignment, and output directory explicit. Edit those values rather than changing source code.
5. Evaluate a checkpoint on an independent authorized dataset:

   ```bash
   python scripts/evaluate.py ^
     --data path/to/independent.mat ^
     --checkpoint path/to/checkpoint.pth ^
     --normalization path/to/stats.npz ^
     --feature-layout take5_10spatial ^
     --pool-mode avg ^
     --vector-grid-index 0 0 ^
     --missing-value-policy sentinel ^
     --output outputs/metrics.json
   ```
6. Compare the depth-resolved temperature and salinity metrics in the output. Select the model only after reviewing both internal validation and independent validation results.

## Training and model selection

The original archive used 20 repeated attempts, an 80/20 shuffled split, 1,500 epochs, batch size 5,000, learning rate 0.0005, dropout 0.2, and saved checkpoints every two epochs from epoch 500 onward. The current `scripts/train.py` creates a reproducible split from the configured seed and saves the best validation-loss checkpoint as `best.pth`, plus `last.pth`.

The supplied local archive includes many intermediate checkpoints. It should not be committed as a 1 GB ZIP. The published release package should instead contain one chosen checkpoint, its matching normalization statistics, resolved configuration, checksum, and expected metrics. See [model selection](MODEL_SELECTION.md).

## Data and release assets

Raw observations, bulk MATLAB files, and model artifacts remain excluded until their data-provider terms and redistribution permissions are documented. The local archive checkpoint `model_epoch_996.pth` is recorded as a candidate for the take5 10-spatial-channel layout; it must not be confused with the separate 9-spatial epoch-996 artifact described elsewhere in this repository.
