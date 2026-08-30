# Model selection policy

## Scope

The published `v0.1.0` Release asset `last_result.1.zip` is the GitHub
filename for the local `last_result(1).zip`. It contains one archived take5
attempt: 500 checkpoints from `model_epoch_10.pth` through
`model_epoch_998.pth`, legacy `cnn.py`, matching normalization files,
train/test splits, and `model_result.mat`. It is a Release asset rather than a
Git-tracked file.

## Candidate designation

The user-designated candidate is `13/model_epoch_996.pth`. It is a 542-tensor
`OrderedDict`; its first normal and eddy expert convolution weights are both
`(20, 10, 3, 3)`. This confirms the **take5 10-spatial-channel layout**, which
includes net heat flux. Its SHA-256 is:

```text
79b6a661d77eca4fb8ca8c7d978a530e54234e8ea77a8b0941c09e1887766e02
```

It is distinct from the historical `final_checkpoint_9spatial` epoch-996
artifact.

## Release record and remaining selection requirements

Release v0.1.0 records the candidate checkpoint, checksum, matching
normalization assets, take5/average-pooling compatibility settings, and
held-out-split metrics in
[`results/release_test_metrics.json`](../results/release_test_metrics.json).

For every future selected model, also record:

- exact checkpoint filename and SHA-256 checksum;
- matching input, temperature, salinity, and pressure normalization files;
- feature layout, pooling mode, vector-grid index, and missing-value policy;
- training seed and resolved YAML configuration;
- internal and independent-validation data identifiers;
- a predeclared primary selection metric and its value; and
- depth-resolved temperature and salinity metrics.

## Recommended rule

Use a genuinely independent validation dataset as the primary decision
surface. Rank checkpoints by a predeclared aggregate of the depth-resolved
temperature and salinity errors, report the full R2/RMSE/MAE profiles, and
reject a model that improves an average only by materially degrading a critical
depth or variable.

## Release package

Publish a selected checkpoint, matching normalization assets, resolved
configuration, checksum file, and expected metrics as a GitHub Release or
Zenodo asset. v0.1.0 follows this approach and additionally publishes the
author-authorized historical archive; it remains a Release asset rather than a
Git blob.

