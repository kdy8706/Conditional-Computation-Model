# Publication-to-code alignment audit

This audit separates confirmed evidence from unresolved authorship decisions. It prevents a cleaned implementation from being presented as an exact reproduction when the underlying artifacts differ.

## Confirmed final-result artifacts

- `model_condition/model_result.mat` names `model_epoch_996.pth` as the evaluated CCM checkpoint.
- The checkpoint contains 542 state-dictionary entries. Its first feature and gate convolutions both have shape `(20, 9, 3, 3)`, proving that it expects nine spatial channels.
- The associated normalized training and held-out sets contain 7,893 and 1,973 profiles, respectively: an 80/20 split of 9,866 profiles.
- Their recovered channel statistics identify the spatial order as SSH, wind u/v, tidal elevation/current u/v, longitude, latitude, and bathymetry. Net heat flux is absent.
- The final evaluation artifact contains predictions for 8,091 independent profiles and was produced on 9 October 2025.
- The non-CCM comparison result identifies `cnn_100_300_55.h5`.
- Running the cleaned epoch-996 compatibility profile against the final data reproduces the stored finite-value mask and predictions within a mean absolute difference of `6.40e-05` and maximum absolute difference of `0.00329` on CPU.

## Confirmed later take5 implementation

- Root `take5/cnn.py` expects ten spatial channels and therefore cannot load `model_epoch_996.pth` strictly.
- It adds net heat flux as spatial channel 7, shifts longitude/latitude/bathymetry to channels 8-10, then uses SST, SSS, day of year, and eddy routing.
- Its two focal-Huber branches use gamma 1.5 for non-eddy and gamma 2.0 for eddy, matching the loss description in the paper.
- It is configured for 1,500 epochs, batch size 5,000, dropout 0.2, learning rate 0.0005, and 20 attempts.

## Resolved since the initial audit

- The upstream patch, missing-data, outlier, and OW-label code was found in `ch3_patch.mlx` and `ch4_nan_processing.mlx` and its deterministic core has been ported to Python.
- The final evaluation script and both CCM/non-CCM numeric result files were found. The compact depth-wise metrics are in `results/depth_metrics.csv`.
- The final CCM checkpoint is unambiguously epoch 996 for the recovered October 2025 result set.
- The cleaned implementation exposes explicit pool and missing-value policies, plus a local checkpoint regression command, rather than silently applying the later take5 behavior to epoch 996.

## Items requiring author confirmation

### 1. Which model is the public scientific target?

The paper describes net heat flux among the inputs, and the later take5 training code includes it. The recovered epoch-996 checkpoint does not. Choose one of these release claims:

- publish epoch 996 as a recovered historical artifact with its actual 9-channel schema; or
- retrain the documented 10-channel take5 model and publish that checkpoint as a new reproducible release.

The repository supports both layouts but does not claim that they are equivalent.

### 2. Independent-test identity and size

The paper materials previously inspected describe a 4,100-profile independent KIOST set. The final October 2025 result folder instead contains an 8,091-profile `dataset_ARGO(1).mat`, with 413 indices stored separately in `argo_ind.mat`. The provenance and meaning of the 8,091/413 subsets must be confirmed before attaching publication claims to the exported metrics.

### 3. Eddy sample count

The reviewer response reports 3,141 eddy profiles among 13,335 samples. The later 14-channel primary plus KMA data contains 3,109. The final 9-channel checkpoint split contains 2,148 eddy profiles among 9,866 samples (1,712 train and 436 held out).

### 4. Exact training code for epoch 996

The final evaluator and checkpoint are present, but the exact 9-channel training script and its random split indices were not found in the supplied folders. The later 10-channel `take5/cnn.py` is not that script.

### 5. Legacy model defects retained for compatibility

- The gated-convolution validity mask can become effectively all-valid when any channel is present; standardized missing values can enter the convolution as `-999`.
- Most `expand_fc` layers and `residual_fc[0]` are created but never reached in the legacy forward pass.
- Fixing either issue changes checkpoint behavior and should be a versioned retraining change.

## Public-release policy

This repository is publicly released as a recovered, archive-compatible
research-code implementation so that other researchers can run and inspect it.
Do not label the released 10-channel take5 checkpoint as the exact final-paper
epoch-996 model, and do not attribute the 8,091-profile metrics to the paper's
4,100-profile dataset without provenance confirmation. The public scope and
evidence boundary are summarized in [EVIDENCE_STATUS.md](EVIDENCE_STATUS.md).
