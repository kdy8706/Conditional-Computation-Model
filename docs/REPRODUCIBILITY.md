# Reproducibility guide

This repository separates the code needed to reproduce the workflow from research datasets that require permission review.

## Workflow

1. Install the package with `python -m pip install -e ".[dev]"`.
2. Obtain authorized model-ready MATLAB data. Each input file must contain `Dinput2`, `Doutput2`, and `Dpressure2`; their expected shapes are documented in the [data card](DATA_CARD.md).
3. Copy `configs/archive_take5_epoch996_candidate.yaml` and set the two data paths to the authorized local files. The verified ZIP-archive workflow requires these **14-channel** inputs:

   - `dataset3.mat`: `Dinput2 = (8, 8, 14, 9866)`;
   - `dataset_ARGO(1).mat`: `Dinput2 = (8, 8, 14, 3469)`.

   Do **not** substitute `dataset_ARGO_kma.mat`: the inspected local file has 16 channels and fails the repository's take5 data validation. The last (14th) channel of `dataset_ARGO(1).mat` is the binary eddy-routing signal required by the model.
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

The verified `last_result(1).zip` archive used 20 repeated attempts, an 80/20 shuffled split, **1,000 epochs**, batch size 5,000, learning rate 0.0005, dropout 0.2, average pooling, and saved a checkpoint every two epochs. It contains 500 `.pth` checkpoints. The current `scripts/train.py` creates one reproducible run from the configured seed and saves the best validation-loss checkpoint as `best.pth`, plus `last.pth`; it does not reproduce the archive's 20-attempt checkpoint sweep by itself.

The supplied local archive includes many intermediate checkpoints. It should not be committed as a 1 GB ZIP. The published release package should instead contain one chosen checkpoint, its matching normalization statistics, resolved configuration, checksum, and expected metrics. See [model selection](MODEL_SELECTION.md).

## Data and release assets

Raw observations, bulk MATLAB files, and model artifacts remain excluded until their data-provider terms and redistribution permissions are documented. The verified local archive checkpoint is `13/model_epoch_996.pth`. It is an `OrderedDict` state dictionary whose first expert-convolution weights have shape `(20, 10, 3, 3)`, confirming the take5 10-spatial-channel layout. Extract its containing `13/` directory before running `convert_legacy_normalization.py`, because the matching normalization files are stored beside the checkpoint. It must not be confused with the separate 9-spatial epoch-996 artifact described elsewhere in this repository.
