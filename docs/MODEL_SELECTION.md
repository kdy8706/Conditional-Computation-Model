# Model selection policy

## Scope

The local `last_result(1).zip` archive contains 500 saved checkpoints for the take5 layout, from `model_epoch_10.pth` through `model_epoch_998.pth`, plus normalization files and `model_result.mat`. It is approximately 1.14 GB compressed and is not a suitable Git repository asset.

## Candidate designation

The archive's `model_epoch_996.pth` is the user-designated candidate checkpoint. It uses the **take5 10-spatial-channel layout**, which includes net heat flux. It is distinct from the repository's previously documented `final_checkpoint_9spatial` epoch-996 artifact.

## Required selection record before release

A final release must record:

- exact checkpoint filename and SHA-256 checksum;
- matching input, temperature, salinity, and pressure normalization files;
- feature layout, pooling mode, vector-grid index, and missing-value policy;
- training seed and resolved YAML configuration;
- internal validation and independent-validation data identifiers;
- a primary selection metric and its value; and
- depth-resolved temperature and salinity metrics.

## Recommended rule

Use the independent validation dataset as the primary decision surface. Rank checkpoints by the mean depth-resolved RMSE for temperature and salinity after applying a predeclared weighting or normalization. Report the full depth profiles of R2, RMSE, and MAE, and reject a checkpoint that improves the average only by materially degrading one variable or a critical depth range.

## Release package

Publish the selected checkpoint, matching normalization assets, resolved configuration, checksum file, and expected metrics as a GitHub Release or Zenodo asset. Keep raw profiles, bulk MATLAB datasets, and the 500-checkpoint archive outside Git unless redistribution permission is confirmed.
