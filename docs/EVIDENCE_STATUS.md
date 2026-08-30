# Evidence status and known limitations

## What this public implementation demonstrates

The `v0.1.1` source, together with the `v0.1.0` archive assets, provides a
runnable take5 workflow. It validates the released checkpoint and matching
normalization files, evaluates them on the archived held-out split, and trains
one configured model from the released `trainset.mat` and `testset.mat` files.
The reference output in `results/release_test_metrics.json` is demonstrated by
that workflow.

## Relationship to Kim et al. (2026)

This repository is an archive-compatible implementation made available so that
other researchers can run and inspect the workflow. The published take5
checkpoint has 10 spatial channels and average pooling. The repository also
preserves a separate historical 9-spatial-channel, max-pooling lineage
associated with final-paper results. Their exact identity, training lineage,
and correspondence to each article result have not been fully reconciled.

Accordingly, the take5 Release must not be described as the exact production
model for every figure or conclusion in Kim et al. (2026). It is a runnable
research artifact informed by that study.

## Sampling-imbalance evidence

Kim et al. (2026) and the Major 1 reviewer-response analysis report 3,141
eddy profiles among 13,335 profiles. Inspection of the later processed
14-channel files found 3,109 eddy labels. The released 10,668/2,667 archive
split has not yet been assigned a published regime-count table. These are
different data-version records and should not be substituted for one another.

The separate eddy/non-eddy modules were designed to address the potential
effect of sampling imbalance. The author-generated Figure A provides a
historical CCM/non-CCM comparison, but it is not a figure in the paper and is
not a fully reproducible baseline experiment in this repository. It therefore
supports a cautious interpretation of robustness rather than a conclusive
claim that imbalance was fully mitigated.

## Validation and future work

The released `testset.mat` is an archived held-out split, not an independent
external dataset. The historical reviewer comparison labels another dataset as
external validation; because its results informed selection discussion, it
should not be interpreted as an untouched final test without a separate,
locked evaluation set.

The public workflow uses central patch point `(4, 4)` in one-based notation
(`vector_grid_index: [3, 3]` in Python). The residual connection is retained
as a data-driven previous-depth conditioning structure and future insertion
point for an equation-informed hybrid layer. No numerical-model or
physics-informed layer is implemented in this release.
