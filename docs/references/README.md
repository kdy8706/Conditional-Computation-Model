# References and design notes

This folder records the scientific and methodological sources used to explain
the Conditional Computation Model (CCM). The README links each source to a
specific design decision; the sources are not presented as claims that CCM
reimplements every cited method.

## Project background and physical motivation

| Reference | Role in CCM |
|---|---|
| [Kim et al. (2026)](Kim%20et%20al.%2C%202026.pdf), *A Wobbling Ratio for diagnosing phase evolution of the Ulleung Warm Eddy from its three-dimensional tilt structure* | East Sea eddy context and the scientific need to resolve three-dimensional thermohaline structure. |
| [ocean_3D_temperature_conservation_equation.txt](ocean_3D_temperature_conservation_equation.txt) | Design note on the three-dimensional temperature-conservation equation, including advection, mixing, surface forcing, and vertical heat transport. It motivates the sequential residual depth connection; CCM is not a numerical solver of this equation. |

## Ocean reconstruction and input-design references

| Reference | Role in CCM |
|---|---|
| [Liu et al. (2024)](liu%20et%20al.%2C%202024%20%2B%20tilting.pdf), *Reconstructing 3-D Thermohaline Structures for Mesoscale Eddies Using Satellite Observations and Deep Learning* | Supports the relevance of satellite-informed, deep-learning reconstruction for three-dimensional eddy thermohaline structure. |
| [Yu et al. (2022)](yu%20et%20al.%2C%202022%20%2B%20tide.pdf), *An offshore subsurface thermal structure inversion method by coupling ensemble learning and tide model for the South Yellow Sea* | Motivates retaining tidal information among surface inputs for subsurface thermal inversion. |
| [Kim et al. (2023)](https://doi.org/10.3389/fmars.2023.1247462), *Estimation of subsurface salinity and analysis of Changjiang diluted water volume in the East China Sea* | Supports the use of CNN-based spatial feature extraction for subsurface-salinity and broader thermohaline reconstruction. |

## Conditional computation and imbalance references

| Reference | Role in CCM |
|---|---|
| [He and Garcia (2009)](https://doi.org/10.1109/TKDE.2008.239), *Learning from Imbalanced Data* | General methodological basis for stating that underrepresented routing regimes require explicit attention. It does not identify the ocean-specific imbalance; that evidence is documented in the project data card. |
| [Bengio, Leonard, and Courville (2013)](https://arxiv.org/abs/1308.3432), *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation* | Conceptual background for conditional computation and gated, input-dependent activation. |
| [Bengio et al. (2016)](https://arxiv.org/abs/1511.06297), *Conditional Computation in Neural Networks for Faster Models* | Further methodological background for selecting computation according to the input. |

## Included and externally cited material

The repository stores the supplied copies of Kim et al. (2026), the
temperature-conservation note, Liu et al. (2024), and Yu et al. (2022). Kim et
al. (2023) is cited through its DOI because the supplied copy exceeds GitHub's
browser-upload limit. The conditional-computation and imbalance papers are
linked to their official DOI or arXiv records rather than copied into this
repository.

Keep each source's original license and publisher terms in mind before
redistributing this repository or its reference files.


