# Reproducibility guide

## First-time route

Follow the [README](../README.md) from **Install** through **Retrain the
released archived split** without substituting local data paths or changing a
model option. That is the supported first-time-user route. It has been checked
with the two v0.1.0 Release assets and uses only these paths after extraction:

```text
artifacts/release/execution/model_epoch_996.pth
artifacts/release/execution/input_normalization.mat
artifacts/release/execution/output_normalization_t.mat
artifacts/release/execution/output_normalization_s.mat
artifacts/release/execution/pressure_normalization.mat
artifacts/release/archive/13/trainset.mat
artifacts/release/archive/13/testset.mat
```

Run `python scripts/preflight_release.py` before evaluation or training. It
checks all paths, the checkpoint SHA-256, the 10-spatial-channel checkpoint
signature, and the two archived MATLAB split layouts.

The supported evaluation uses `testset.mat`, which is an archived held-out
split, not an independent dataset. Its expected depth-resolved output is
recorded in [`results/release_test_metrics.json`](../results/release_test_metrics.json).

## What the archived retraining run does

`configs/release_archive_take5.yaml` trains one deterministic configured
attempt using `trainset.mat` and validates it with `testset.mat`. It writes
`best.pth`, `last.pth`, `stats.npz`, `split_indices.npz`, and
`resolved_config.yaml` into `runs/release_archive_take5/`.

The original `last_result.1.zip` archive records one historical attempt with
500 checkpoints saved every two epochs. Its legacy `cnn.py` used 20 attempts;
the refactored trainer deliberately does not infer or hide this sweep. To
perform a new sweep, choose and record each seed and output directory, then
compare held-out and independent results under a predeclared selection rule.

## Independent evaluation

An independent MATLAB file must contain `Dinput2`, `Doutput2`, and
`Dpressure2`, with shapes `(8, 8, 14, N)`, `(N, 13, 2)`, and `(14, N)`.
Use the README evaluation command but replace `--data` only. Keep
`take5_10spatial`, average pooling, `[0, 0]`, and `sentinel` for the released
checkpoint. Record the dataset owner, preprocessing version, and whether the
file is truly independent before reporting the result.

The historical nine-spatial-channel artifacts are a different compatibility
path. They must not be mixed with the v0.1.0 take5 checkpoint or its released
normalization files.

