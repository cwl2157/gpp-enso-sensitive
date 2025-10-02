
"""
Compute annual tropical (40S–40N) GPP totals from monthly CanESM5 files.
Steps:
1) Read all target files from D:\cangku\data
2) Regrid to 1x1° using lat/lon from gpp_abrupt-4xCO2_CanESM5_mask_1x1.nc
3) For each file, sum months (Jan–Dec) to annual totals (time-integrated using days_in_month if units are per second)
4) Integrate over 40S–40N using spherical cell areas
5) Plot 1850–1949 annual tropical GPP time series for each experiment
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# -------------------- User paths --------------------
DATA_DIR = Path(r"D:\cangku\data")
MASK_FILE = DATA_DIR / "gpp_abrupt-4xCO2_CanESM5_mask_1x1.nc"

# Specific files (will only process those that exist)
TARGET_FILES = [
    "gpp_Lmon_CanESM5_abrupt-4xCO2_185001-200012.nc",
    "gpp_Lmon_CanESM5_G1_r1i1p2f1_gn_185001-194912.nc",
    "gpp_Lmon_CanESM5_piControl_r1i1p2f1_gn_185001-194912.nc",
]

# -------------------- Helpers --------------------
def fix_lon_to_180(ds, lon_name="lon"):
    """Convert longitudes to [-180, 180) and sort, if needed."""
    if lon_name in ds.coords:
        lon = ds[lon_name]
    elif lon_name in ds.dims:
        lon = ds[lon_name]
    else:
        return ds  # no lon found
    lon_new = ((lon + 180) % 360) - 180
    ds = ds.assign_coords({lon_name: lon_new})
    ds = ds.sortby(lon_name)
    return ds

def find_gpp_var(ds):
    """Return the name of the GPP variable (contains 'gpp' case-insensitive)."""
    for v in ds.data_vars:
        if "gpp" in v.lower():
            return v
    raise ValueError("No variable containing 'gpp' found in dataset.")

def compute_cell_area_1x1(lat_1d, lon_1d):
    """Compute spherical cell areas (m^2) for a regular 1x1° grid given 1D lat/lon arrays."""
    R = 6371000.0  # Earth radius [m]
    dlat = np.deg2rad(np.abs(np.diff(lat_1d).mean()) if lat_1d.size > 1 else 1.0)
    dlon = np.deg2rad(np.abs(np.diff(lon_1d).mean()) if lon_1d.size > 1 else 1.0)

    # cell bounds by +/- 0.5 grid spacing
    lat_rad = np.deg2rad(lat_1d.values)
    # use half-step based on dlat
    lat_bnds_upper = lat_rad + dlat/2.0
    lat_bnds_lower = lat_rad - dlat/2.0
    strip_area = R**2 * dlon * (np.sin(lat_bnds_upper) - np.sin(lat_bnds_lower))  # shape (lat,)
    # broadcast to (lat, lon)
    area_2d = xr.DataArray(
        np.outer(strip_area, np.ones(lon_1d.size)),
        coords={"lat": lat_1d, "lon": lon_1d},
        dims=("lat", "lon"),
        name="cell_area",
        attrs={"units": "m2"}
    )
    return area_2d

def to_annual_integrated(da):
    """
    Convert monthly GPP to annual totals.
    If units contain 's-1' (i.e., a rate), multiply by seconds per month before summing.
    Otherwise, do a straight monthly sum.
    """
    units = (da.attrs.get("units") or "").lower()
    if "s-1" in units or "s^-1" in units or "/s" in units:
        # time-integrate monthly means
        seconds = (da["time"].dt.days_in_month * 86400).astype("float64")
        # align for broadcasting
        da_monthly_total = da * seconds
    else:
        da_monthly_total = da
    annual = da_monthly_total.groupby("time.year").sum("time")
    return annual

# -------------------- Load target 1x1° grid --------------------
if not MASK_FILE.exists():
    raise FileNotFoundError(f"Mask/grid file not found: {MASK_FILE}")
mask = xr.open_dataset(MASK_FILE)
# try common coord names
lat_name = "lat" if "lat" in mask.coords else ("latitude" if "latitude" in mask.coords else "lat")
lon_name = "lon" if "lon" in mask.coords else ("longitude" if "longitude" in mask.coords else "lon")
mask = fix_lon_to_180(mask, lon_name=lon_name)
lat_tgt = mask[lat_name]
lon_tgt = mask[lon_name]

# Precompute cell areas on the 1x1 grid (lat,lon)
area = compute_cell_area_1x1(lat_tgt, lon_tgt)

# -------------------- Process each file --------------------
results = {}  # {label: pandas.Series of annual totals}

for fname in TARGET_FILES:
    fpath = DATA_DIR / fname
    if not fpath.exists():
        # skip missing files silently but inform at the end
        print(f"[Skip] File not found: {fpath}")
        continue

    print(f"[Open] {fpath.name}")
    ds = xr.open_dataset(fpath, decode_times=True)

    # Harmonize lon to [-180, 180)
    ds = fix_lon_to_180(ds, lon_name=lon_name if lon_name in ds.coords else ("lon" if "lon" in ds.coords else list(ds.coords)[0]))

    # Identify GPP variable
    gpp_name = find_gpp_var(ds)
    gpp = ds[gpp_name]

    # Regrid to target 1x1 using linear interpolation
    # Ensure target lons match range with source
    # (both have been forced to [-180, 180))
    gpp_1x1 = gpp.interp({lat_name if lat_name in gpp.dims else "lat": lat_tgt,
                          lon_name if lon_name in gpp.dims else "lon": lon_tgt},
                         method="linear")

    # Select tropics
    gpp_tropics = gpp_1x1.sel({lat_name if lat_name in gpp_1x1.dims else "lat": slice(-40, 40)})

    # Annual integration
    gpp_annual = to_annual_integrated(gpp_tropics)

    # Area-integrate over 40S–40N
    # Align area with data dims (lat, lon)
    # Use xarray's weighted sum with explicit weights (area)
    # Mask out any NaNs to avoid propagating in sum
    area_tropics = area.sel(lat=slice(-40, 40))
    # ensure same coords names as data
    area_tropics = area_tropics.rename({ "lat": list(gpp_annual.dims)[-2] if "lat" in area_tropics.dims and "lat" not in gpp_annual.dims else "lat",
                                         "lon": list(gpp_annual.dims)[-1] if "lon" in area_tropics.dims and "lon" not in gpp_annual.dims else "lon"})
    # broadcast & multiply
    gpp_total = (gpp_annual * area_tropics).sum(dim=[d for d in gpp_annual.dims if d != "year"], skipna=True)

    # To pandas Series
    ser = gpp_total.to_series()
    # Keep 1850–1949 only (if available)
    ser = ser[(ser.index >= 1850) & (ser.index <= 1949)]
    label = fname.replace("gpp_Lmon_CanESM5_", "").replace(".nc", "")
    results[label] = ser

# -------------------- Plot --------------------
if not results:
    raise RuntimeError("No input files were found. Please check paths in DATA_DIR.")

plt.figure(figsize=(10, 5))
for label, ser in results.items():
    if len(ser) == 0:
        continue
    ser.sort_index().plot(label=label)
plt.title("Annual Tropical (40S–40N) GPP totals (1850–1949)\nRegridded to 1×1°")
plt.xlabel("Year")
plt.ylabel("GPP total (units depend on source; integrated over area)")
plt.legend()
plt.tight_layout()

# Save alongside data
out_png = DATA_DIR / "tropical_GPP_1850_1949.png"
plt.savefig(out_png, dpi=150)
print(f"[Saved] {out_png}")

print("Done.")
