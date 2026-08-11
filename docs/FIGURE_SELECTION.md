# Result figure selection

The supplied final-result directory contains 15 PNG files, including alternate exports and separate metric panels. Uploading all of them would be redundant.

## Recommended public set

| Source file | Role | Decision |
|---|---|---|
| `result_discussion_temp.png` | combined temperature R2/RMSE/NRMSE profile | keep after dataset-label confirmation |
| `result_discussion_sal.png` | combined salinity R2/RMSE/NRMSE profile | keep after dataset-label confirmation |
| `result_discussion_legend.png` | CCM/non-CCM and held-out/independent legend | keep if the two panels are kept |
| `picture/result_discussion_*_{r2,rmse,nrmse}.png` | six single-metric panels | omit from the first release; reproducible plotting should replace them |
| files suffixed `(1)` or `(2)` | alternate exports | omit unless the author identifies one as publication-final |

The repository currently includes the compact reviewer-response comparison as `docs/figures/ccm_vs_non_ccm.png` and the underlying numeric metrics as `results/depth_metrics.csv`. This keeps the first release small while preserving the scientific comparison.

Before copying the three recommended composites, confirm that “Independent” refers to the 8,091-profile final dataset rather than the 4,100-profile KIOST dataset described in the paper materials.
