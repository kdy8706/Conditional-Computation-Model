# Model card

## Intended use

The conditional computation model reconstructs vertical profiles of subsurface temperature and salinity from daily sea-surface observations over the East Sea. Its original scientific use was to create three-dimensional fields for analysis of the Ulleung Warm Eddy.

The model is intended for research reconstruction within the domain, variables, resolution, and data-processing assumptions described by Kim et al. (2026). It has not been validated as an operational forecasting system or for safety-critical navigation.

## Scientific motivation

In-situ sampling is imbalanced between eddy and non-eddy conditions. The reviewer response reports 3,141 eddy profiles out of 13,335 training/validation profiles. The CCM uses a hard decision module and separate parameter sets so the smaller eddy regime is not forced to share all regression parameters with the dominant non-eddy regime.

## Decision module

The article defines the eddy condition using the Okubo-Weiss parameter:

```text
OW < -0.2 sigma  -> eddy expert
otherwise        -> non-eddy expert
```

The training script receives an upstream binary eddy field and routes `signal[:, 3, 3] > 0` to the legacy `outlier` branch. The recovered `ch4_nan_processing.mlx` creates that field with `OW < -0.2 * OW_std`; the rule is now available in `ocean_ccm.preprocessing`.

## Expert architecture

The experts have identical topology and separate weights.

### Spatial encoder

```text
(9 or 10, 8, 8)
 -> gated convolution, 20 channels, 3 x 3
 -> convolution, 40 channels, 3 x 3
 -> convolution, 80 channels, 3 x 3
 -> 2 x 2 pooling
 -> 80 features
```

The first convolution computes `GELU(feature_conv(x)) * tanh(gate_conv(x))`.

### Vector encoder

```text
SST, SSS, day-of-year
 -> 3 -> 6 -> 12 -> 24
```

### Sequential depth regression

The spatial and vector encodings form 104 shared features. At the first output level they are expanded to 128 features. At later levels, the previous temperature, salinity, and pressure are encoded as 24 residual features and concatenated with the 104 shared features.

Each depth has a separate `128 -> 32 -> 8 -> 2` regression head. Predictions therefore propagate from shallower to deeper levels.

## Inputs

- spatial surface patch: `(batch, 9, 8, 8)` for epoch 996 or `(batch, 10, 8, 8)` for take5
- point variables: `(batch, 3)`
- preceding pressure levels: `(batch, 13)`
- binary eddy signal: `(batch, 8, 8)`

## Outputs

`(batch, 13, 2)`, where the last dimension is temperature and salinity and the levels are approximately 10-500 m.

## Training

The recovered epoch-996 evaluator uses max pooling, point variables at `[0, 0]`, and preserves input NaNs so invalid predictions are excluded by the original metric calculation. The later take5 code uses average pooling, maps missing values to `-999`, combines 13,335 profiles, and uses batch size 5,000, 1,500 epochs, dropout 0.2, and learning rate 0.0005. Separate RAdam optimizers update the two experts.

The later take5 loss assignment matches the paper (focal-Huber gamma 1.5 and 2.0), but its 10-channel input does not match the epoch-996 checkpoint. See [Paper alignment](PAPER_ALIGNMENT.md).

## Limitations

- hard routing cannot express uncertainty near the OW threshold;
- the smaller eddy expert receives substantially fewer samples;
- sequential regression can propagate shallow-depth errors downward;
- batch normalization may be unstable for very small routed sub-batches;
- the legacy missing-value mask is ineffective when any channel at a pixel is valid, and missing values enter the first convolution as `-999`;
- the legacy module layout contains parameters that are never used in `forward`;
- generalization outside the East Sea, 1993-2023 period, input products, and 0.25 degree processing grid is unverified.

## Ethical and scientific-use notes

Predicted profiles are model reconstructions, not direct observations. Downstream products should preserve provenance and distinguish measured from reconstructed values. Spatial maps should not be interpreted beyond the independent validation domain without additional evaluation.
