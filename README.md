# Conditional Computation Model for Ocean Subsurface Reconstruction

This repository reconstructs 13-level subsurface temperature and salinity
profiles in the East Sea from satellite-based surface fields and 0 m in-situ
observations. It accompanies Kim et al. (2026), *A Wobbling Ratio for
diagnosing phase evolution of the Ulleung Warm Eddy from its three-dimensional
tilt structure*.

> **Scope and paper relationship:** this public repository is a runnable,
> archive-compatible implementation for researchers to inspect, evaluate, and
> retrain the released workflow. Its `take5_10spatial` checkpoint is not
> claimed to be the exact final production artifact used for every result in
> Kim et al. (2026). In particular, the repository preserves both a released
> 10-spatial-channel take5 archive and a separate historical 9-spatial-channel
> paper-result lineage. See [evidence status](docs/EVIDENCE_STATUS.md) before
> comparing this Release directly with figures in the article.


## Why Conditional Computation?

![Conditional computation model architecture](docs/figures/codex-clipboard-32df9674-8335-4a86-819f-0491c5ecd83f.png)

### The scientific and learning problem

The objective is not merely to interpolate a subsurface profile. It is to infer
how temperature and salinity vary from 10 to 500 m under dynamically different
surface conditions, including mesoscale eddies. A profile is assigned to the
eddy regime when its upstream Okubo-Weiss (OW) criterion satisfies
`OW < -0.2 sigma`.

The available in-situ profiles have a **sampling imbalance** between profiles
collected within and outside eddies. Kim et al. (2026) and the reviewer-response
analysis report 3,141 eddy profiles out of 13,335 total profiles. The later
processed files inspected for this repository contain 3,109 eddy labels, so
these figures are data-version-specific and are not assumed to be identical to
the published archive split. Limited eddy sampling can bias a conventional
model toward more common non-eddy conditions. The broader methodological issue
is reviewed by [He and Garcia (2009)](https://doi.org/10.1109/TKDE.2008.239);
the project-specific reconciliation is recorded in the [data card](docs/DATA_CARD.md).

### Why use CCM rather than one regressor?

A single regressor trained on all profiles can be dominated by the more common
non-eddy conditions. CCM was designed to **address the potential effect of
sampling imbalance** without discarding observations: it uses the full dataset
while allowing eddy and non-eddy modules to be treated separately. Its decision
module routes a sample to the corresponding module; the two modules have the
same architecture and separate weights.

![CCM and non-CCM performance comparison](docs/figures/ccm_vs_nonccm_major1.png)

*Figure A. Author-generated CCM/non-CCM comparison prepared for the Major 1
reviewer response; it is not a figure in Kim et al. (2026). Red circles are
CCM and blue squares are non-CCM; solid and dashed lines indicate the historical
external-validation and held-out test datasets, respectively. Panels show
depth-dependent temperature and salinity RMSE and NRMSE. This historical
comparison is consistent with CCM showing more robust external-validation
behavior in that analysis, but it does not by itself prove that sampling
imbalance was fully mitigated. The non-CCM training workflow, baseline weights,
and regime-stratified metrics are not released here.*

The architecture is an ocean-reconstruction application of input-dependent
conditional computation: a gating decision selects the computation used for a
given input. [Bengio, Leonard, and Courville (2013)](https://arxiv.org/abs/1308.3432)
and [Bengio et al. (2016)](https://arxiv.org/abs/1511.06297) provide the
methodological background for conditional, input-dependent activation. They do
not by themselves establish the ocean-specific sampling imbalance; that
motivation comes from the documented composition of this project's in-situ
profiles.

### How CCM represents the ocean problem

Each expert combines two complementary feature paths:

- A CNN extracts horizontal structure from the surface patch: sea-surface
  height, wind, tide, bathymetry, net heat flux, and related gridded fields.
- An MLP encodes the point variables: 0 m in-situ SST and SSS, together with
  day of year. These surface observations retain the local thermohaline state.

The two feature representations are combined before depth-wise regression.
Thirteen depth heads predict temperature and salinity sequentially. After the
first level, each head receives the preceding predicted temperature, salinity,
and pressure through residual conditioning. In the current CCM, this residual
connection carries the previous-depth information but does not itself solve a
physical equation.

The residual structure was intentionally designed as an extension point for a
future equation-informed layer relating depth `n` to `n + 1`. This would allow
a hybrid numerical-model and deep-learning model to be developed while keeping
the present data-driven CCM as the baseline. Physics-informed machine learning
provides a broader framework for combining mathematical models and data-driven
networks [Karniadakis et al. (2021)](https://doi.org/10.1038/s42254-021-00314-5);
no such hybrid layer is implemented in this release.

### Scientific background and design references

The following references describe why these inputs and connections were chosen;
they are not all claims that CCM reproduces the cited methods exactly.

| Reference | Contribution to this repository's design |
|---|---|
| [Kim et al. (2026)](docs/references/Kim%20et%20al.%2C%202026.pdf) | East Sea eddy context and the scientific need to represent three-dimensional thermohaline structure. |
| [Ocean 3-D temperature conservation note](docs/references/ocean_3D_temperature_conservation_equation.txt) | Physical motivation from advection, mixing, surface forcing, and vertical transport. It defines the future equation-informed direction; CCM is not a numerical conservation-equation solver. |
| [Liu et al. (2024)](docs/references/liu%20et%20al.%2C%202024%20%2B%20tilting.pdf) | Satellite-observation and deep-learning reconstruction of three-dimensional eddy thermohaline structure. |
| [Yu et al. (2022)](docs/references/yu%20et%20al.%2C%202022%20%2B%20tide.pdf) | Motivation for retaining tidal information among the surface inputs for subsurface thermal inversion. |
| [Kim et al. (2023)](docs/references/README.md#referenced-but-not-stored) | CNN-based subsurface-salinity estimation, supporting the use of a spatial CNN pathway for thermohaline reconstruction. |
| [Karniadakis et al. (2021)](https://doi.org/10.1038/s42254-021-00314-5) | General framework for the planned, but not yet implemented, physics-informed hybrid extension. |
| [Complete reference notes](docs/references/README.md) | Full citation list, including the methodological conditional-computation and imbalance references. |


> **Start here:** this workflow is written for a first-time user. It uses the
> `v0.1.1` source release with the published `v0.1.0` archive assets, without
> undocumented local paths. The repository and Release assets are public.

## What you can reproduce

| Goal | Included inputs | Result |
|---|---|---|
| Evaluate the released checkpoint | execution package + archive `testset.mat` | depth-resolved temperature/salinity metrics on the archived held-out split |
| Retrain one archived split | archive `trainset.mat` + `testset.mat` | `best.pth`, `last.pth`, normalization statistics, and validation loss |
| Repeat the original 20-attempt sweep | same archive plus your own automation | not fully reproducible: original seeds and selection record are not recovered |
| Independent external validation | your own 14-channel MATLAB dataset | optional; not included in Release v0.1.0 |

The supplied `testset.mat` is a held-out split from the archived workflow. It
is **not** an independent external validation dataset.

## Release assets

Use the source from [Release v0.1.1](https://github.com/kdy8706/Conditional-Computation-Model/releases/tag/v0.1.1) or its matching commit, then download the two data/model assets from [Release v0.1.0](https://github.com/kdy8706/Conditional-Computation-Model/releases/tag/v0.1.0).

| Asset | Size | Purpose |
|---|---:|---|
| [`ccm-take5-epoch996-v0.1.1.zip`](https://github.com/kdy8706/Conditional-Computation-Model/releases/download/v0.1.1/ccm-take5-epoch996-v0.1.1.zip) | ~2 MB | User-designated checkpoint, four matching normalization files, configuration, and checksums |
| [`last_result.1.zip`](https://github.com/kdy8706/Conditional-Computation-Model/releases/download/v0.1.0/last_result.1.zip) | ~1.06 GB | Full archived attempt: source `cnn.py`, 500 checkpoints, train/test splits, normalization files, and historical results |

`last_result.1.zip` is the published filename of the locally named
`last_result(1).zip` archive. They refer to the same archive.

The release checkpoint is `13/model_epoch_996.pth` in the full archive. It is
a take5 model with 10 spatial channels and average pooling. Verify it before
use:

```text
SHA-256 (model_epoch_996.pth)
79b6a661d77eca4fb8ca8c7d978a530e54234e8ea77a8b0941c09e1887766e02
```

## 1. Install

Clone tag `v0.1.1` for a version-pinned workflow. Do not use the automatically
generated source ZIP attached to the older `v0.1.0` tag: it lacks support for
the archived `trainset.mat` and `testset.mat` layout.

```powershell
python --version
# If this does not report Python 3.10 or newer, install CPython from https://www.python.org/downloads/
git clone --branch v0.1.1 https://github.com/kdy8706/Conditional-Computation-Model.git
cd Conditional-Computation-Model

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

If PowerShell execution policy blocks activation, run the same commands with
`.\.venv\Scripts\python.exe` instead of `python`; activation is optional.

Requirements are Python 3.10 or newer plus the packages declared in
`pyproject.toml`: NumPy, SciPy, PyTorch, torch-optimizer, PyYAML, and pytest
for tests. CPU execution is supported; a CUDA GPU is optional. Leave at least
3 GB of free disk space for the Release archive, Python environment, and run
outputs; 8 GB RAM or more is recommended.

## 2. Create the documented local layout

In the repository root, create `artifacts/release/`. Download the two Release
assets through the GitHub Release page, then extract them so the layout is
exactly:

```text
Conditional-Computation-Model/
├─ artifacts/
│  └─ release/
│     ├─ execution/                         # extract ccm-take5-epoch996-v0.1.1.zip here
│     │  ├─ model_epoch_996.pth
│     │  ├─ input_normalization.mat
│     │  ├─ output_normalization_t.mat
│     │  ├─ output_normalization_s.mat
│     │  └─ pressure_normalization.mat
│     └─ archive/                           # extract last_result.1.zip here
│        ├─ cnn.py
│        └─ 13/
│           ├─ trainset.mat
│           ├─ testset.mat
│           ├─ model_result.mat
│           └─ model_epoch_2.pth ... model_epoch_1000.pth
├─ configs/release_archive_take5.yaml
└─ scripts/
```

The archive's `13/trainset.mat` provides `xtrain`, `ytrain`, and
`pressure_train`; `13/testset.mat` provides `xtest`, `ytest`, and
`pressure_test`. Both have the verified take5 shape `(8, 8, 14, N)` and can be
read directly by the current code.

Check this layout, the checkpoint checksum, the 10-channel model signature,
and both MATLAB split files before training or evaluation:

```powershell
python scripts/preflight_release.py
```

## 3. Verify the released checkpoint on the archived test split

Run these commands from the repository root. This is the shortest complete
reproduction path and uses the matching checkpoint and normalization files.

```powershell
python scripts/convert_legacy_normalization.py `
  --source artifacts/release/execution `
  --output artifacts/release/take5_epoch996.npz `
  --feature-layout take5_10spatial

python scripts/evaluate.py `
  --data artifacts/release/archive/13/testset.mat `
  --checkpoint artifacts/release/execution/model_epoch_996.pth `
  --normalization artifacts/release/take5_epoch996.npz `
  --feature-layout take5_10spatial `
  --pool-mode avg `
  --vector-grid-index 3 3 `
  --missing-value-policy sentinel `
  --output outputs/release_test_metrics.json
```

Success creates `outputs/release_test_metrics.json`, containing R2, RMSE,
relative RMSE, MAE, and normalized RMSE at each of the 13 depth levels for
temperature and salinity. Do not change `take5_10spatial`, `avg`, or
`sentinel` when evaluating this released checkpoint.

## 4. Retrain the released archived split

The supplied configuration already points to the directory layout above. It
uses the ZIP-verified settings: 10 spatial channels, average pooling, 13 depth
levels, dropout 0.2, batch size 5,000, learning rate 0.0005, and 1,000 epochs.

```powershell
python scripts/train.py --config configs/release_archive_take5.yaml
```

Outputs are written to `runs/release_archive_take5/`:

```text
best.pth                 lowest validation loss from this run
last.pth                 checkpoint from epoch 1,000
stats.npz                training-split normalization statistics
split_indices.npz        indices produced within this new configured run
resolved_config.yaml     exact configuration used
```

The historical archive used 20 attempts and retained 500 intermediate
checkpoints (every two epochs). The refactored trainer intentionally runs one
configured attempt. Original 20-attempt seeds and the full selection record are
not recovered, so this repository does not claim to reproduce that sweep.

## Data interface and compatibility

The released workflow uses `take5_10spatial`:

- `Dinput2` or the archived split input: `(8, 8, 14, N)`;
- 10 spatial fields, including net heat flux;
- SST, SSS, and day of year as three point inputs, taken at the central
  `(4, 4)` location in one-based notation (`[3, 3]` in this Python code);
- the 14th channel as the binary eddy-routing signal;
- targets: `(N, 13, 2)` for temperature and salinity; and
- pressure: `(14, N)`.

The model routes each sample to an eddy or non-eddy expert. A CNN extracts
spatial-patch features, while an MLP encodes the surface point variables.
Thirteen depth heads predict sequentially; after the first level, each head
receives the preceding temperature, salinity, and pressure through residual
conditioning.

This division expresses two parts of the physical problem: the CNN retains
horizontal surface structure, while the MLP incorporates the local 0 m
in-situ SST and SSS used as surface thermohaline conditions. The residual
depth-to-depth connection conditions a deeper prediction on the preceding water
layer. It remains in the architecture as the planned insertion point for a
future equation-informed hybrid layer; it is not direct evidence that the
current model learns physical transport.

The input choices and sequential structure were informed by the
[temperature-conservation note](docs/references/ocean_3D_temperature_conservation_equation.txt)
and the related studies retained in [docs/references](docs/references). CCM is
a data-driven reconstruction model, not a numerical solver of those equations.

The codebase also preserves a historical `final_checkpoint_9spatial` layout.
It is not the v0.1.0 release target. Never evaluate the published take5
checkpoint with that 9-channel/max-pooling configuration.

## Independent evaluation

To evaluate another 14-channel MATLAB dataset, it must contain `Dinput2`,
`Doutput2`, and `Dpressure2` with the shapes above. Reuse the commands in
step 3 but replace only `--data`; keep the released checkpoint's take5 layout,
average pooling, and sentinel missing-value policy. Clearly label those
results as independent evaluation and record the dataset provenance.

## References and project files

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Data card](docs/DATA_CARD.md)
- [Model-selection policy](docs/MODEL_SELECTION.md)
- [Evidence status and known limitations](docs/EVIDENCE_STATUS.md)
- [Tested environment](docs/TESTED_ENVIRONMENT.md)
- [Reference output for the released test split](results/release_test_metrics.json)
- [Physical motivation and references](docs/references/README.md)
- `configs/`: training profiles
- `scripts/`: training, evaluation, normalization conversion, prediction, and
  result-export utilities
- `tests/`: data-layout, preprocessing, metrics, and model tests

## Citation, access, and license

The provisional software author and maintainer is Dong-Young Kim. See
[`CITATION.cff`](CITATION.cff) for citation details; contact
[kdy8706@naver.com](mailto:kdy8706@naver.com) for questions.

Source code is distributed under the [BSD 3-Clause License](LICENSE). The
author-prepared model archive, derived train/test split, and weights are made
available under the [CC BY 4.0 terms](DATA_AND_MODEL_LICENSE.md). Third-party
reference papers and any upstream source products retain their own terms.

