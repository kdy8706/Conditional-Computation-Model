# Public release record and remaining limitations

## Completed for the public workflow

- The source is versioned as `v0.1.1`; it supports the archived
  `trainset.mat`/`testset.mat` variable layout.
- The author-designated `model_epoch_996.pth`, four normalization files, and
  the complete archived train/test split are distributed as GitHub Release
  assets. `scripts/preflight_release.py` checks SHA-256 hashes for all five
  small execution files and validates both MATLAB files.
- The public route uses the paper's central point, `(4, 4)` in one-based
  notation and `[3, 3]` in Python.
- The public route has passed the test suite and checkpoint evaluation on the
  archived held-out split in the documented Windows/CPython environment.
- Code is licensed under BSD 3-Clause; author-prepared derived data, weights,
  and figures use the repository's CC BY 4.0 terms.

## Explicit limitations (do not overstate)

- This is an archive-compatible runnable implementation for reuse, **not a
  claim that its take5 checkpoint is the exact final production artifact for
  every result in Kim et al. (2026)**.
- The original 20-attempt sweep, its seeds, and its complete model-selection
  record have not been recovered. The documented trainer executes one
  configured attempt.
- Eddy totals of 3,141 (review response) and 3,109 (inspected processed
  files) refer to different known data versions. A count for the published
  archived split has not yet been assigned.
- The final 8,091-profile independent artifact and its relationship to the
  archived held-out split require additional provenance confirmation.
- Figure A is an author-generated reviewer-response illustration, not a
  figure in Kim et al. (2026). Its non-CCM comparator is not released as a
  runnable baseline.
- Upstream product names, versions, and redistribution terms remain the
  responsibility of every downstream user. Repository terms do not supersede
  source-provider requirements.

## Recommended future additions

- release a documented non-CCM baseline and its inputs for a reproducible
  imbalance analysis;
- publish an immutable split/regime-count manifest with data-version IDs;
- record full optimizer, seed, and selection metadata for future experiments;
- add a small synthetic fixture and continuous integration for quick checks.
