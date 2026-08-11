# Source provenance

Local absolute paths are intentionally omitted from the public-facing inventory. The hashes below identify the author-provided files without redistributing them.

## Scientific sources

- Kim, D.-Y. et al. (2026), *A Wobbling Ratio for diagnosing phase evolution of the Ulleung Warm Eddy from its three-dimensional tilt structure*, *Frontiers in Marine Science* 13:1860054, DOI `10.3389/fmars.2026.1860054`.
- Reviewer 4 response, Major Comment 1: sampling-imbalance motivation and CCM/non-CCM comparison.

## Model and preprocessing sources

| Artifact | SHA-256 | Role |
|---|---|---|
| root `take5/cnn.py` | `8A057F21F4128EC7B5807466D18F337ACB016B8B5D5B7849C48A5EDED65763B6` | later 10-spatial-channel training implementation |
| `ch3_patch.mlx` | `6DB6B238B7E4B82AF1403FEB1A74520DD7EDF37C8698BDC419471824C32DB5DF` | 8 x 8 patch construction |
| `ch4_nan_processing.mlx` | `C92557E992B912A5043483ECE517009F812BAA2E1144F99430901A529A64D91E` | channel assembly, filtering, and OW routing |
| final `model_result2_vali.py` | `D72E26AD88662CE1F5E171736580436EA8BF2D60CE0EF50D6B4229E248AFD384` | 9-channel CCM evaluation |
| `process/model2/try2/cnn.py` | `361DBC84A6FFC35B177C1D7BB2B9B066C6E72D5B9602794945FB24FDCDB6E6F0` | older loss variant |
| `process/model2/try3/cnn.py` | `EB8A2DB169ECD2B53D9D94B3216043F25C6D551D6EAF471F538341FBB6D5C9D5` | ZIP/experiment-13 variant |

## Final result artifacts

| Artifact | Size | SHA-256 |
|---|---:|---|
| `model_epoch_996.pth` | 2,420,294 bytes | `F9B88C9D30C74DCEC8D78896BC33F464CEDDBB2BDBEFB2F7CA88A17093F80350` |
| CCM `model_result.mat` | 3,752,104 bytes | `3A4B0447C4D43B49D491F1D5F5CA0F7F5EFFD31F0891D315AD09DC5F474FBE3D` |
| non-CCM `cnn_100_300_55.h5` | 2,889,192 bytes | `189118D8A042F365A3CB9BABCA89D9BBBD0C9A07F264812A1B1DEC6C4ADF2D8F` |
| non-CCM `model_result.mat` | 3,724,104 bytes | `373191D2E1349A7B526402C07F73CB52A2D0E6FD0383673CD8562C9F6F73967A` |
| final 8,091-profile `dataset_ARGO(1).mat` | 29,652,492 bytes | `7BEB2D06F5AEEE4FB1C9CD63C5B1F44B7B9C2778455E46CA78A7152A54E16895` |

## Processed datasets inspected earlier

| Role | Samples | SHA-256 |
|---|---:|---|
| later take5 primary `dataset3.mat` | 9,866 | `54F746E2EF90885BFE38AD7A4707D8D39292833C08199C4A821E35DD97200BCE` |
| KMA supplement | 3,469 | `57CEB5FD22650134EA54B52757207C08362AB13CC0547B2EF57E8DC51E58620C` |
| KIOST independent set | 4,100 | `24061F89D4907CAB0B04A1DF6ED8E7F7D39F58E152AB9E317281AD573C621BCA` |

## Exclusion policy

The legacy experiment tree contains tens of thousands of repeated checkpoints and large processed data files. They are evidence for internal audit, not suitable Git content. Only code, documentation, compact metrics, checksums, and legally redistributable examples should enter the source repository.
