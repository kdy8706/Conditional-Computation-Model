# Data card

## Dataset roles

| Role | Identified source | Samples | Repository policy |
|---|---|---:|---|
| primary training/validation | `take5/process/dataset3.mat` | 9,866 | excluded pending redistribution review |
| KMA supplement for training/validation | `validation/kma(1)/data/dataset_ARGO(1).mat` | 3,469 | excluded pending redistribution review |
| KIOST independent test | `validation/kordi(1)/data/dataset_ARGO(1).mat` | 4,100 | excluded pending redistribution review |
| final-result independent artifact | final results `dataset_ARGO(1).mat` | 8,091 | excluded; identity requires author confirmation |

The primary and KMA datasets combine to 13,335 samples, matching the article's training/validation count. The published workflow uses an 80/20 shuffled split and keeps the 4,100 KIOST profiles as the independent test set.

## MATLAB variables

The later take5 files contain:

- `Dinput2`: `(8, 8, 14, N)`
- `Doutput2`: `(N, 13, 2)`
- `Dpressure2`: `(14, N)`

The recovered epoch-996 train/test files and final 8,091-profile evaluation file instead contain `Dinput2` with 13 channels. Their other shapes are unchanged.

The first 13 input channels are standardized using statistics calculated on the training split. The binary routing channel is not standardized. Temperature and salinity outputs are standardized independently at every output level. Pressure is standardized using one global mean and standard deviation.

## Channel schema

The 14-channel order below is confirmed by `ch4_nan_processing.mlx` and the later take5 code.

| Channel | Working interpretation | Model path |
|---:|---|---|
| 1 | sea surface height | spatial |
| 2 | surface wind, component 1 | spatial |
| 3 | surface wind, component 2 | spatial |
| 4 | tidal elevation | spatial |
| 5 | tidal current, component 1 | spatial |
| 6 | tidal current, component 2 | spatial |
| 7 | net heat flux | spatial |
| 8 | longitude | spatial |
| 9 | latitude | spatial |
| 10 | bathymetry/depth | spatial |
| 11 | sea surface temperature | vector |
| 12 | sea surface salinity | vector |
| 13 | day of year | vector |
| 14 | binary eddy flag derived from OW | decision |

For the 13-channel epoch-996 layout, channel 7 (net heat flux) is absent and later channels shift left by one. Recovered normalization means identify channels 7-9 as longitude (about 130), latitude (about 37), and bathymetry (about -1,358 m), respectively. The decision channel is 13.

The article describes vector values at the central `(4, 4)` point in one-based indexing. The recovered final evaluator and take5 code use `[0, 0]`; this is the compatibility default. The `[0, 0]` and `[3, 3]` SST/SSS values differ in the final 8,091-profile artifact, so future data pipelines must choose the intended location explicitly.

## Output levels

The 14 pressure entries are approximately:

```text
0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500 m
```

The 13 outputs correspond to 10-500 m. For output levels after the first, the model uses the preceding output and preceding pressure. Therefore the legacy slice `pressure[:, :13]` is consistent with the article's “previous depth” residual conditioning.

## Filtering and missing values

The article states that samples with more than 50% missing values in the input patch were removed before training. `ch4_nan_processing.mlx` implements this as at least 32 grid cells that are finite across every input channel. The local intermediate files are consistent with upstream filtering: `dataset.mat` contains 11,714 samples, while `dataset2.mat` and `dataset3.mat` contain 9,866. The deterministic rule is ported to Python.

Remaining NaNs are converted to the sentinel `-999` after standardization. Output losses ignore targets equal to that sentinel.

## Eddy counts requiring reconciliation

The reviewer response reports 3,141 eddy profiles among 13,335 samples. The currently inspected binary channel contains 3,109 eddy profiles: 2,148 in the primary file and 961 in the KMA file. Before release, identify whether the publication used a later data version, a different OW threshold product, or an additional filtering rule.

The epoch-996 artifacts use only the 9,866-profile primary set and contain the same 2,148 eddy samples, split as 1,712 train and 436 held out.

## Redistribution requirements

Before publishing any data or normalization artifacts, document:

- data provider and product identifiers for every input variable;
- original licenses and citation requirements;
- whether in-situ profile data may be redistributed;
- preprocessing dates and product versions;
- checksums for the exact files used in the publication;
- the mapping from source observations to processed samples.

A small synthetic fixture should be released with the code even if the research datasets cannot be redistributed.
