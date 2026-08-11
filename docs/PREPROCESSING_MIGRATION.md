# MATLAB-to-Python preprocessing migration

## Scope recovered from take5

The root `take5` folder contains 20 MATLAB Live Scripts and 10 Python files. Most Live Scripts are data-source preparation or exploratory analysis; they should not all be copied into a public source repository.

| MATLAB stage | Purpose | Public Python status |
|---|---|---|
| `ch1.1_data_collection.mlx`, `ch1,1_data_collection(field).mlx` | ocean and in-situ source ingestion | needs provider-specific readers and license metadata |
| `ch1.2_ecmwf.mlx`, `ch1.3_ecmwf2.mlx` | ERA5/ECMWF preparation | needs xarray/netCDF implementation |
| `ch2.1_data_matching.mlx` | UTC conversion and nearest grid/time/profile matching | algorithm documented; product adapters still needed |
| `ch2.2_tide_data.mlx`, `ch2.3_wind_data.mlx`, `ch2.4_nhf_data.mlx` | daily forcing preparation | product adapters still needed |
| `ch3_patch.mlx` | centered 8 x 8 patch extraction | implemented in `ocean_ccm.preprocessing` |
| `ch4_nan_processing.mlx` | channel stacking, missing-data filtering, outlier removal, OW routing | deterministic core implemented in Python |
| `data_tend.mlx`, `기타/*` | exploratory plots and diagnostics | omit from the first software release unless needed for a paper figure |

## Recovered deterministic rules

The Python functions preserve these MATLAB behaviors:

1. An 8 x 8 patch uses the center grid index minus 3 through plus 4.
2. A sample survives the input missing-data screen when at least 32 of 64 cells are finite in every input channel.
3. SST, SSS, and profile-output outliers use a two-sided `2.56 * standard deviation` rule in the recovered script.
4. Eddy routing is set when `OW < -0.2 * OW_std` at the central cell.
5. The later take5 tensor contains 14 channels, including net heat flux.
6. The recovered epoch-996 checkpoint uses a 13-channel predecessor layout with net heat flux removed.

## What is needed to complete the Python-only raw-data pipeline

- exact product names, versions, variable names, units, temporal resolution, and download URLs for SSH/SLA, wind, tides, net heat flux, SST, SSS, and bathymetry;
- the authoritative in-situ profile file format and its redistribution terms;
- the grid definition stored in `LonLat.mat`;
- confirmation of whether the public target is the 9-spatial-channel epoch-996 model or the later 10-spatial-channel take5 model;
- one small, legally redistributable example for an end-to-end test.

Once those inputs are confirmed, the remaining Live Script stages can be replaced with xarray, netCDF4, pandas, NumPy, and TEOS-10/GSW Python equivalents without requiring MATLAB.
