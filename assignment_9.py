import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm

# =====================================================
# PATHS
# =====================================================
BASE_126 = "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 9/SSP126"
BASE_370 = "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 9/SSP370"

TEMP_FOLDER_126 = os.path.join(BASE_126, "Temp_126")
TEMP_FOLDER_370 = os.path.join(BASE_370, "Temp_370")

PR_FOLDER_126 = os.path.join(BASE_126, "Precipitation_126")
PR_FOLDER_370 = os.path.join(BASE_370, "Precipitation_370")

HUM_FOLDER_126 = os.path.join(BASE_126, "Humidity_126")
HUM_FOLDER_370 = os.path.join(BASE_370, "Humidity_370")

WB_OUTPUT_126 = os.path.join(BASE_126, "wb_126.nc")
WB_OUTPUT_370 = os.path.join(BASE_370, "wb_370.nc")

# =====================================================
# OPEN ALL FILES
# =====================================================
def open_all(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {folder}")

    print(f"\nFiles in {folder}:")
    for f in files:
        print(os.path.basename(f))

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time"
    )

    ds = ds.sortby("time")
    return ds

# =====================================================
# SUBSET REGION
# =====================================================
def subset_region(da, lat_min=15, lat_max=33, lon_min=33, lon_max=60):
    lat_name = "lat"
    lon_name = "lon"

    lat0 = float(da[lat_name].values[0])
    lat1 = float(da[lat_name].values[-1])
    lon0 = float(da[lon_name].values[0])
    lon1 = float(da[lon_name].values[-1])

    # latitude
    if lat0 < lat1:
        da = da.sel({lat_name: slice(lat_min, lat_max)})
    else:
        da = da.sel({lat_name: slice(lat_max, lat_min)})

    # longitude
    if lon0 < lon1:
        da = da.sel({lon_name: slice(lon_min, lon_max)})
    else:
        da = da.sel({lon_name: slice(lon_max, lon_min)})

    return da

# =====================================================
# SEN'S SLOPE
# =====================================================
def sens_slope(x, years):
    x = np.array(x, dtype=float)
    years = np.array(years, dtype=float)

    mask = ~np.isnan(x)
    x = x[mask]
    years = years[mask]

    n = len(x)
    if n < 2:
        return np.nan

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if years[j] != years[i]:
                slopes.append((x[j] - x[i]) / (years[j] - years[i]))

    if len(slopes) == 0:
        return np.nan

    return np.median(slopes)

# =====================================================
# STANDARD MK COMPONENTS
# =====================================================
def mk_stats(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 2:
        return np.nan, np.nan, np.nan

    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            if x[j] > x[k]:
                s += 1
            elif x[j] < x[k]:
                s -= 1

    var_s = n * (n - 1) * (2 * n + 5) / 18

    return s, var_s, n

# =====================================================
# LAG-1 AUTOCORRELATION
# =====================================================
def lag1_autocorr(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 3:
        return 0.0

    x_mean = np.mean(x)
    num = np.sum((x[:-1] - x_mean) * (x[1:] - x_mean))
    den = np.sum((x - x_mean) ** 2)

    if den == 0:
        return 0.0

    return num / den

# =====================================================
# ADJUSTED MANN-KENDALL TEST
# Simple autocorrelation-adjusted variance
# =====================================================
def adjusted_mk_test(x, alpha=0.05):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]

    s, var_s, n = mk_stats(x)
    if np.isnan(s):
        return np.nan, np.nan, "no data"

    r1 = lag1_autocorr(x)

    # effective sample size correction
    if abs(r1) < 1:
        n_eff = n * (1 - r1) / (1 + r1)
    else:
        n_eff = n

    if n_eff < 2:
        n_eff = 2

    correction_factor = n / n_eff
    var_s_adj = var_s * correction_factor

    if s > 0:
        z = (s - 1) / np.sqrt(var_s_adj)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s_adj)
    else:
        z = 0.0

    p = 2 * (1 - norm.cdf(abs(z)))

    if p < alpha:
        trend = "Increasing" if z > 0 else "Decreasing"
    else:
        trend = "No significant trend"

    return z, p, trend

# =====================================================
# LOAD + PROCESS ONE SCENARIO
# =====================================================
def process_scenario(temp_folder, pr_folder, scenario_name):
    ds_temp = open_all(temp_folder)
    ds_pr   = open_all(pr_folder)

    print(f"\nScenario: {scenario_name}")
    print("Temperature variables:", list(ds_temp.data_vars))
    print("Precipitation variables:", list(ds_pr.data_vars))

    # ---- variable names ----
    tas = ds_temp["tas"] - 273.15   # Kelvin -> Celsius
    pr  = ds_pr["pr"]               # check this variable name in your files

    # ---- subset Saudi Arabia ----
    tas = subset_region(tas, 15, 33, 33, 60)
    pr  = subset_region(pr, 15, 33, 33, 60)

    # ---- spatial mean ----
    tas_mean = tas.mean(dim=["lat", "lon"], skipna=True)
    pr_mean  = pr.mean(dim=["lat", "lon"], skipna=True)

    # ---- annual mean precipitation and annual mean temperature ----
    tas_annual = tas_mean.resample(time="YE").mean().compute()
    pr_annual  = pr_mean.resample(time="YE").mean().compute()

    years = tas_annual["time"].dt.year.values
    tas_vals = np.array(tas_annual.values, dtype=float)
    pr_vals  = np.array(pr_annual.values, dtype=float)

    # ---- trends ----
    z_tas, p_tas, trend_tas = adjusted_mk_test(tas_vals)
    z_pr,  p_pr,  trend_pr  = adjusted_mk_test(pr_vals)

    slope_tas = sens_slope(tas_vals, years)
    slope_pr  = sens_slope(pr_vals, years)

    return {
        "years": years,
        "tas": tas_vals,
        "pr": pr_vals,
        "z_tas": z_tas,
        "p_tas": p_tas,
        "trend_tas": trend_tas,
        "z_pr": z_pr,
        "p_pr": p_pr,
        "trend_pr": trend_pr,
        "slope_tas": slope_tas,
        "slope_pr": slope_pr
    }

# =====================================================
# RUN BOTH SCENARIOS
# =====================================================
ssp126 = process_scenario(TEMP_FOLDER_126, PR_FOLDER_126, "SSP1-RCP2.6")
ssp370 = process_scenario(TEMP_FOLDER_370, PR_FOLDER_370, "SSP3-RCP7.0")

# =====================================================
# PRINT RESULTS
# =====================================================
print("\n==================== RESULTS ====================")

print("\nSSP1-RCP2.6")
print("Temperature trend:", ssp126["trend_tas"])
print("Temperature p-value:", ssp126["p_tas"])
print("Temperature Sen's slope:", ssp126["slope_tas"], "°C/year")

print("Precipitation trend:", ssp126["trend_pr"])
print("Precipitation p-value:", ssp126["p_pr"])
print("Precipitation Sen's slope:", ssp126["slope_pr"], "per year")

print("\nSSP3-RCP7.0")
print("Temperature trend:", ssp370["trend_tas"])
print("Temperature p-value:", ssp370["p_tas"])
print("Temperature Sen's slope:", ssp370["slope_tas"], "°C/year")

print("Precipitation trend:", ssp370["trend_pr"])
print("Precipitation p-value:", ssp370["p_pr"])
print("Precipitation Sen's slope:", ssp370["slope_pr"], "per year")

# =====================================================
# PLOTS: TEMPERATURE
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(ssp126["years"], ssp126["tas"], marker="o", label="SSP1-RCP2.6")
plt.plot(ssp370["years"], ssp370["tas"], marker="o", label="SSP3-RCP7.0")
plt.title("Average Annual Temperature vs Year (Saudi Arabia)")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# PLOTS: PRECIPITATION
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(ssp126["years"], ssp126["pr"], marker="o", label="SSP1-RCP2.6")
plt.plot(ssp370["years"], ssp370["pr"], marker="o", label="SSP3-RCP7.0")
plt.title("Average Annual Precipitation vs Year (Saudi Arabia)")
plt.xlabel("Year")
plt.ylabel("Precipitation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# PART 3: CLIMATE EXTREMES
# Annual maximum of daily Saudi-average temperature and precipitation
# =====================================================

def process_extremes(temp_folder, pr_folder, scenario_name):
    ds_temp = open_all(temp_folder)
    ds_pr   = open_all(pr_folder)

    tas = ds_temp["tas"] - 273.15
    pr  = ds_pr["pr"]

    tas = subset_region(tas, 15, 33, 33, 60)
    pr  = subset_region(pr, 15, 33, 33, 60)

    # daily average over all Saudi grid cells
    tas_daily_mean = tas.mean(dim=["lat", "lon"], skipna=True)
    pr_daily_mean  = pr.mean(dim=["lat", "lon"], skipna=True)

    # annual maximum of daily Saudi-average
    tas_max_annual = tas_daily_mean.resample(time="YE").max().compute()
    pr_max_annual  = pr_daily_mean.resample(time="YE").max().compute()

    years = tas_max_annual["time"].dt.year.values
    tas_max_vals = np.array(tas_max_annual.values, dtype=float)
    pr_max_vals  = np.array(pr_max_annual.values, dtype=float)

    # trend analysis
    z_tas, p_tas, trend_tas = adjusted_mk_test(tas_max_vals)
    z_pr, p_pr, trend_pr = adjusted_mk_test(pr_max_vals)

    slope_tas = sens_slope(tas_max_vals, years)
    slope_pr  = sens_slope(pr_max_vals, years)

    return {
        "years": years,
        "tas_max": tas_max_vals,
        "pr_max": pr_max_vals,
        "trend_tas": trend_tas,
        "p_tas": p_tas,
        "slope_tas": slope_tas,
        "trend_pr": trend_pr,
        "p_pr": p_pr,
        "slope_pr": slope_pr
    }

# =====================================================
# RUN EXTREMES ANALYSIS
# =====================================================
ext126 = process_extremes(TEMP_FOLDER_126, PR_FOLDER_126, "SSP1-RCP2.6")
ext370 = process_extremes(TEMP_FOLDER_370, PR_FOLDER_370, "SSP3-RCP7.0")

# =====================================================
# PRINT EXTREMES RESULTS
# =====================================================
print("\n==================== PART 3: CLIMATE EXTREMES ====================")

print("\nSSP1-RCP2.6")
print("Maximum temperature trend:", ext126["trend_tas"])
print("Maximum temperature p-value:", ext126["p_tas"])
print("Maximum temperature Sen's slope:", ext126["slope_tas"], "°C/year")

print("Maximum precipitation trend:", ext126["trend_pr"])
print("Maximum precipitation p-value:", ext126["p_pr"])
print("Maximum precipitation Sen's slope:", ext126["slope_pr"], "per year")

print("\nSSP3-RCP7.0")
print("Maximum temperature trend:", ext370["trend_tas"])
print("Maximum temperature p-value:", ext370["p_tas"])
print("Maximum temperature Sen's slope:", ext370["slope_tas"], "°C/year")

print("Maximum precipitation trend:", ext370["trend_pr"])
print("Maximum precipitation p-value:", ext370["p_pr"])
print("Maximum precipitation Sen's slope:", ext370["slope_pr"], "per year")

# =====================================================
# PLOT MAXIMUM TEMPERATURE
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(ext126["years"], ext126["tas_max"], marker="o", label="SSP1-RCP2.6")
plt.plot(ext370["years"], ext370["tas_max"], marker="o", label="SSP3-RCP7.0")
plt.title("Annual Maximum of Daily Mean Temperature in Saudi Arabia")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# PLOT MAXIMUM PRECIPITATION
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(ext126["years"], ext126["pr_max"], marker="o", label="SSP1-RCP2.6")
plt.plot(ext370["years"], ext370["pr_max"], marker="o", label="SSP3-RCP7.0")
plt.title("Annual Maximum of Daily Mean Precipitation in Saudi Arabia")
plt.xlabel("Year")
plt.ylabel("Precipitation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =====================================================
# PART 4: WET BULB TEMPERATURE
# =====================================================

def calculate_wet_bulb_temperature(temp_k, rh_percent):
    """
    Wet-bulb temperature using Stull (2011)
    Input:
        temp_k     : air temperature in Kelvin
        rh_percent : relative humidity in %
    Output:
        wet-bulb temperature in Kelvin
    """
    temp_c = temp_k - 273.15

    wbt_c = (
        temp_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659))
        + np.arctan(temp_c + rh_percent)
        - np.arctan(rh_percent - 1.676331)
        + 0.00391838 * (rh_percent ** 1.5) * np.arctan(0.023101 * rh_percent)
        - 4.686035
    )

    wbt_k = wbt_c + 273.15
    return wbt_k

def make_wet_bulb_file(temp_folder, hum_folder, output_file, scenario_name):
    ds_temp = open_all(temp_folder)
    ds_hum  = open_all(hum_folder)

    tas = ds_temp["tas"]
    hurs = ds_hum["hurs"]

    wbt = calculate_wet_bulb_temperature(tas, hurs)

    ds_output = xr.Dataset(
        {
            "wet_bulb_temp": wbt
        },
        attrs={
            "description": f"Wet bulb temperature for {scenario_name}",
            "units": "K",
            "calculation_method": "Stull (2011)"
        }
    )

    ds_output.to_netcdf(output_file)
    print(f"Wet bulb temperature saved to: {output_file}")

    return ds_output

def process_wet_bulb_from_nc(wb_file, scenario_name):
    ds_wb = xr.open_dataset(wb_file)

    wb = ds_wb["wet_bulb_temp"]

    # subset Saudi Arabia
    wb = subset_region(wb, 15, 33, 33, 60)

    # spatial mean over Saudi Arabia
    wb_mean = wb.mean(dim=["lat", "lon"], skipna=True)

    # annual mean
    wb_annual = wb_mean.resample(time="YE").mean().compute()

    years = wb_annual["time"].dt.year.values
    wb_vals_k = np.array(wb_annual.values, dtype=float)
    wb_vals_c = wb_vals_k - 273.15

    # trend analysis
    z_wb, p_wb, trend_wb = adjusted_mk_test(wb_vals_c)
    slope_wb = sens_slope(wb_vals_c, years)

    return {
        "years": years,
        "wb_c": wb_vals_c,
        "trend_wb": trend_wb,
        "p_wb": p_wb,
        "slope_wb": slope_wb
    }

# =====================================================
# CREATE WET BULB NC FILES
# =====================================================
make_wet_bulb_file(TEMP_FOLDER_126, HUM_FOLDER_126, WB_OUTPUT_126, "SSP1-RCP2.6")
make_wet_bulb_file(TEMP_FOLDER_370, HUM_FOLDER_370, WB_OUTPUT_370, "SSP3-RCP7.0")

# =====================================================
# READ WET BULB FILES AND ANALYZE
# =====================================================
wb126 = process_wet_bulb_from_nc(WB_OUTPUT_126, "SSP1-RCP2.6")
wb370 = process_wet_bulb_from_nc(WB_OUTPUT_370, "SSP3-RCP7.0")

# =====================================================
# PRINT RESULTS
# =====================================================
print("\n==================== PART 4: WET BULB TEMPERATURE ====================")

print("\nSSP1-RCP2.6")
print("Wet-bulb trend:", wb126["trend_wb"])
print("Wet-bulb p-value:", wb126["p_wb"])
print("Wet-bulb Sen's slope:", wb126["slope_wb"], "°C/year")

print("\nSSP3-RCP7.0")
print("Wet-bulb trend:", wb370["trend_wb"])
print("Wet-bulb p-value:", wb370["p_wb"])
print("Wet-bulb Sen's slope:", wb370["slope_wb"], "°C/year")

# =====================================================
# PLOT WET BULB TEMPERATURE
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(wb126["years"], wb126["wb_c"], marker="o", label="SSP1-RCP2.6")
plt.plot(wb370["years"], wb370["wb_c"], marker="o", label="SSP3-RCP7.0")
plt.title("Average Annual Wet-Bulb Temperature in Saudi Arabia")
plt.xlabel("Year")
plt.ylabel("Wet-Bulb Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# PART 5: WET BULB TEMPERATURE TREND ANALYSIS
# =====================================================

def process_wetbulb_trend(wb_file, scenario_name):

    ds = xr.open_dataset(wb_file)
    wb = ds["wet_bulb_temp"]

    # clip Saudi Arabia
    wb = subset_region(wb, 15, 33, 33, 60)

    # spatial average across Saudi Arabia
    wb_mean = wb.mean(dim=["lat","lon"], skipna=True)

    # annual mean wet bulb temperature
    wb_annual = wb_mean.resample(time="YE").mean().compute()

    years = wb_annual["time"].dt.year.values
    wb_vals = np.array(wb_annual.values, dtype=float) - 273.15

    # trend analysis
    z, p, trend = adjusted_mk_test(wb_vals)
    slope = sens_slope(wb_vals, years)

    return {
        "years": years,
        "wb": wb_vals,
        "trend": trend,
        "p": p,
        "slope": slope
    }


# =====================================================
# RUN TREND ANALYSIS
# =====================================================

wbtrend126 = process_wetbulb_trend(WB_OUTPUT_126, "SSP1-RCP2.6")
wbtrend370 = process_wetbulb_trend(WB_OUTPUT_370, "SSP3-RCP7.0")


# =====================================================
# PRINT RESULTS
# =====================================================

print("\n==================== PART 5: WET BULB TREND ====================")

print("\nSSP1-RCP2.6")
print("Trend:", wbtrend126["trend"])
print("p-value:", wbtrend126["p"])
print("Sen slope:", wbtrend126["slope"], "°C/year")

print("\nSSP3-RCP7.0")
print("Trend:", wbtrend370["trend"])
print("p-value:", wbtrend370["p"])
print("Sen slope:", wbtrend370["slope"], "°C/year")


# =====================================================
# PLOT ANNUAL WET BULB TEMPERATURE
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(wbtrend126["years"], wbtrend126["wb"],
         marker="o", label="SSP1-RCP2.6")

plt.plot(wbtrend370["years"], wbtrend370["wb"],
         marker="o", label="SSP3-RCP7.0")

plt.title("Average Annual Wet Bulb Temperature in Saudi Arabia")
plt.xlabel("Year")
plt.ylabel("Wet Bulb Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =====================================================
# PART 5: WET BULB EXTREMES
# =====================================================

def process_wetbulb_extremes(wb_file):

    ds = xr.open_dataset(wb_file)
    wb = ds["wet_bulb_temp"]

    wb = subset_region(wb, 15, 33, 33, 60)

    # daily spatial average
    wb_daily_mean = wb.mean(dim=["lat","lon"], skipna=True)

    # annual maximum
    wb_max = wb_daily_mean.resample(time="YE").max().compute()

    years = wb_max["time"].dt.year.values
    wb_vals = np.array(wb_max.values, dtype=float) - 273.15

    z, p, trend = adjusted_mk_test(wb_vals)
    slope = sens_slope(wb_vals, years)

    return {
        "years": years,
        "wb_max": wb_vals,
        "trend": trend,
        "p": p,
        "slope": slope
    }


# =====================================================
# RUN EXTREMES
# =====================================================

wbext126 = process_wetbulb_extremes(WB_OUTPUT_126)
wbext370 = process_wetbulb_extremes(WB_OUTPUT_370)


# =====================================================
# PRINT EXTREMES
# =====================================================

print("\n==================== WET BULB EXTREMES ====================")

print("\nSSP1-RCP2.6")
print("Maximum trend:", wbext126["trend"])
print("p-value:", wbext126["p"])
print("Sen slope:", wbext126["slope"])

print("\nSSP3-RCP7.0")
print("Maximum trend:", wbext370["trend"])
print("p-value:", wbext370["p"])
print("Sen slope:", wbext370["slope"])


# =====================================================
# PLOT MAXIMUM WET BULB TEMPERATURE
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(wbext126["years"], wbext126["wb_max"],
         marker="o", label="SSP1-RCP2.6")
º
plt.plot(wbext370["years"], wbext370["wb_max"],
         marker="o", label="SSP3-RCP7.0")

plt.title("Annual Maximum Wet Bulb Temperature in Saudi Arabia")
plt.xlabel("Year")
plt.ylabel("Wet Bulb Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
