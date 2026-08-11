# Legacy source provenance

Several monolithic training variants dated February 2025 were found:

- `model2/try2/cnn.py`: five attempts; ordinary non-eddy Huber and focal eddy Huber;
- `model2/try3/cnn.py`: twenty attempts; focal non-eddy Huber and ordinary eddy Huber.
- root `take5/cnn.py`: ten spatial channels; focal-Huber gamma 1.5 for non-eddy and 2.0 for eddy; 1,500 epochs.

The supplied `last_result(1).zip` contains the `try3` behavior and the outputs of experiment folder 13.

The final result checkpoint itself is now identified as epoch 996, but it has nine spatial input channels and therefore did not come from the root 10-channel take5 script. The exact 9-channel training file remains missing. Original monolithic files are not copied into the cleaned tree; their checksums and behavior are recorded in `docs/PROVENANCE.md`.

The compatibility model in `src/ocean_ccm/model.py` preserves the legacy state-dictionary parameter names and unreachable module layout. Scientific behavior that may be erroneous is documented in `docs/PAPER_ALIGNMENT.md` rather than silently altered.
