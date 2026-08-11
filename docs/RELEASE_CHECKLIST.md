# Public release checklist

## Must resolve

- [x] Identify the checkpoint used by the final recovered CCM result (`model_epoch_996.pth`).
- [ ] Confirm whether epoch 996, the later 10-channel take5 model, or both should be the public target.
- [ ] Recover/confirm the exact loss configuration that generated epoch 996.
- [ ] Reconcile the 3,141 versus 3,109 eddy sample count.
- [x] Reconstruct the deterministic OW-to-binary routing rule in Python.
- [x] Recover the CCM/non-CCM evaluator outputs and export their numeric metrics.
- [ ] Confirm the provenance of the 8,091-profile final independent artifact and its 413-index subset.
- [ ] Confirm the exact input channel order, units, product names, and versions.
- [ ] Choose a software license with all code authors.
- [ ] Confirm data and trained-weight redistribution permissions.
- [x] Add provisional repository title and maintainer contact.
- [ ] Confirm software contributors and acknowledgements with the author team.

## Strongly recommended

- [ ] Export an immutable train/validation split manifest with a seed.
- [ ] Save one selected checkpoint plus optimizer/configuration metadata.
- [ ] Generate a small synthetic example that can run without restricted data.
- [ ] Add dataset and checkpoint SHA-256 checksums.
- [ ] Run training and inference tests in a clean environment with CUDA and CPU.
- [ ] Add a non-CCM baseline implementation and reproducible ablation command.
- [ ] Decide whether to preserve legacy masking or retrain with corrected masking.
- [ ] Remove unreachable layers in a versioned, checkpoint-incompatible model revision.
- [ ] Add continuous integration after the dependency versions are confirmed.

## Files that must stay out of ordinary Git history

- 87,021 legacy `.pth` files under `process`;
- raw and processed observation `.mat` files;
- the 1.28 GB `all.mat` and multi-hundred-MB matching files;
- bulk predictions and generated 3D fields;
- temporary code screenshots and repeated experiment folders.

Use an external data archive, institutional repository, or an appropriately configured Git LFS/Release workflow only after licensing and quota review. Keep the source repository small and reproducible.
