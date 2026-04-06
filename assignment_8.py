import xarray as xr
import geopandas as gpd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os
import rioxarray


# PART 1 - ERA5 Data Pre-processing

# Set working directory

os.chdir("/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 8")
print(os.getcwd())

# Load watershed shapefile

shapefile_path = "WS_3/WS_3.shp"
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs("EPSG:4326")

# Function to load and clip ERA5 data

def load_and_clip(nc_file, var_name, gdf):
    ds = xr.open_dataset(nc_file)
    da = ds[var_name]
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    clipped = da.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped

# Load 2001 ERA5 datasets

precip_file = "Precipitation/era5_OLR_2001_total_precipitation.nc"
et_file = "Total_Evaporation/era5_OLR_2001_total_evaporation.nc"
runoff_file = "Runoff/ambientera5_OLR_2001_total_runoff.nc"

# Convert from meters to mm

P_grid = load_and_clip(precip_file, "tp", gdf) * 1000
ET_grid = load_and_clip(et_file, "e", gdf) * 1000
Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000

# Watershed averaged variables

P = P_grid.mean(dim=["latitude", "longitude"]).values
ET = ET_grid.mean(dim=["latitude", "longitude"]).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values

# Load 2002 ERA5 datasets and apply same steps as for year 2001

precip_file_2002 = "Precipitation/era5_OLR_2002_total_precipitation.nc"
et_file_2002 = "Total_Evaporation/era5_OLR_2002_total_evaporation.nc"
runoff_file_2002 = "Runoff/ambientera5_OLR_2002_total_runoff.nc"

P_grid_2002 = load_and_clip(precip_file_2002, "tp", gdf) * 1000
ET_grid_2002 = load_and_clip(et_file_2002, "e", gdf) * 1000
Q_grid_2002 = load_and_clip(runoff_file_2002, "ro", gdf) * 1000

P_2002 = P_grid_2002.mean(dim=["latitude", "longitude"]).values
ET_2002 = ET_grid_2002.mean(dim=["latitude", "longitude"]).values
Q_obs_2002 = Q_grid_2002.mean(dim=["latitude", "longitude"]).values

ET_2002 = np.where(ET_2002 < 0, -ET_2002, ET_2002)


# Convert evaporation to positive

ET = np.where(ET < 0, -ET, ET)

print("Part 1 preprocessing completed")
print("P shape:", P.shape)
print("ET shape:", ET.shape)
print("Q_obs shape:", Q_obs.shape)

# Plot

fig, axes = plt.subplots(2,1, figsize=(12,8), sharex=False)

# ----- 2001 -----
axes[0].plot(P, color='hotpink', linewidth=3, alpha=0.4, label='Precipitation', zorder=1)
axes[0].plot(ET, color='orange', linewidth=2, label='Evaporation', zorder=2)
axes[0].plot(Q_obs, color='green', linewidth=1.5, label='Runoff', zorder=3)

axes[0].set_title("Hydrological Variables (2001)")
axes[0].set_ylabel("mm")
axes[0].legend()

# ----- 2002 -----
axes[1].plot(P_2002, color='hotpink', linewidth=3, alpha=0.4, label='Precipitation', zorder=1)
axes[1].plot(ET_2002, color='orange', linewidth=2, label='Evaporation', zorder=2)
axes[1].plot(Q_obs_2002, color='green', linewidth=1.5, label='Runoff', zorder=3)

axes[1].set_title("Hydrological Variables (2002)")
axes[1].set_ylabel("mm")
axes[1].set_xlabel("Time (hour)")
axes[1].legend()

plt.tight_layout()
plt.show()


# Part 2 — Model Validation

# Linear reservoir rainfall-runoff model

def simulate_runoff(k, P, ET, Q0, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q0
    for t in range(1, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt / k)
        Q_sim[t] = max(0, Q_t)
    return Q_sim

# Kling Gupta Efficiency

def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    KGE = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return KGE, r, alpha, beta

# Validate model with given k

k_test = 0.15
Q_sim = simulate_runoff(k_test, P, ET, Q_obs[0])
KGE, r, alpha, beta = kge(Q_obs, Q_sim)

print("Validation Results (2001)")
print("KGE:", KGE)
print("Correlation:", r)
print("Alpha:", alpha)
print("Beta:", beta)

# Time series plot

plt.figure(figsize=(10,4))
plt.plot(Q_obs, label="Observed Runoff")
plt.plot(Q_sim, label="Simulated Runoff")
plt.legend()
plt.title("Observed vs Simulated Runoff (Validation)")
plt.xlabel("Time (hour)")
plt.ylabel("Runoff (mm)")
plt.grid(True)
plt.show()

# Scatter plot

plt.figure(figsize=(6,6))
plt.scatter(Q_obs, Q_sim, alpha=0.5)
max_val = max(np.max(Q_obs), np.max(Q_sim))
plt.plot([0, max_val], [0, max_val], 'r--')

plt.xlabel("Observed Runoff")
plt.ylabel("Simulated Runoff")
plt.title("Observed vs Simulated Runoff Scatter")
plt.grid(True)
plt.show()

# Part 3 - Model Calibration + Validation

# Objective function for optimization

def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET, Q_obs[0])
    kge_val = kge(Q_obs, Q_sim)
    return 1 - kge_val[0]

# Optimize k

res = opt.minimize_scalar(
    objective,
    bounds=(0.1, 2),
    args=(P, ET, Q_obs),
    method='bounded'
)

best_k = res.x
print("Optimized k:", best_k)

Q_sim_cal = simulate_runoff(best_k, P, ET, Q_obs[0])
KGE_cal, r_cal, alpha_cal, beta_cal = kge(Q_obs, Q_sim_cal)

print("Calibration Results (2001)")
print("KGE:", KGE_cal)
print("r:", r_cal)
print("alpha:", alpha_cal)
print("beta:", beta_cal)

# Calibration plots

plt.figure(figsize=(10,4))
plt.plot(Q_obs, label="Observed Runoff 2001")
plt.plot(Q_sim_cal, label="Simulated Runoff 2001")
plt.legend()
plt.xlabel("Time (hour)")
plt.ylabel("Runoff (mm)")
plt.title("Observed vs Simulated Runoff (Calibration 2001)")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(Q_obs, Q_sim_cal, alpha=0.5)
max_val = max(np.max(Q_obs), np.max(Q_sim_cal))
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel("Observed Runoff")
plt.ylabel("Simulated Runoff")
plt.title("Scatter Plot Calibration (2001)")
plt.grid(True)
plt.show()


# Load 2002 data for validation

precip_fileVal = "Precipitation/era5_OLR_2002_total_precipitation.nc"
et_fileVal = "Total_Evaporation/era5_OLR_2002_total_evaporation.nc"
runoff_fileVal = "Runoff/ambientera5_OLR_2002_total_runoff.nc"

P_gridVal = load_and_clip(precip_fileVal, "tp", gdf) * 1000
ET_gridVal = load_and_clip(et_fileVal, "e", gdf) * 1000
Q_gridVal = load_and_clip(runoff_fileVal, "ro", gdf) * 1000

P_v = P_gridVal.mean(dim=["latitude", "longitude"]).values
ET_v = ET_gridVal.mean(dim=["latitude", "longitude"]).values
Q_obs_v = Q_gridVal.mean(dim=["latitude", "longitude"]).values

ET_v = np.where(ET_v < 0, -ET_v, ET_v)

# Run model with calibrated k

Q_sim_v = simulate_runoff(best_k, P_v, ET_v, Q_obs_v[0])
KGE_v, r_v, alpha_v, beta_v = kge(Q_obs_v, Q_sim_v)

print("Validation Results (2002)")
print("KGE:", KGE_v)
print("r:", r_v)
print("alpha:", alpha_v)
print("beta:", beta_v)

# Time series validation plot

plt.figure(figsize=(10,4))
plt.plot(Q_obs_v, label="Observed Runoff 2002")
plt.plot(Q_sim_v, label="Simulated Runoff 2002")
plt.legend()
plt.xlabel("Time (hour)")
plt.ylabel("Runoff (mm)")
plt.title("Observed vs Simulated Runoff (Validation 2002)")
plt.grid(True)
plt.show()

# Scatter validation plot

plt.figure(figsize=(6,6))
plt.scatter(Q_obs_v, Q_sim_v, alpha=0.5)
max_val = max(np.max(Q_obs_v), np.max(Q_sim_v))
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel("Observed Runoff")
plt.ylabel("Simulated Runoff")
plt.title("Scatter Plot Validation (2002)")
plt.grid(True)
plt.show()

# Part 4 - Multi Year Search for Best k   ---------------------------------

# Function to load one year of watershed data

def load_year_data(year, gdf):
    precip_file = f"Precipitation/era5_OLR_{year}_total_precipitation.nc"
    et_file = f"Total_Evaporation/era5_OLR_{year}_total_evaporation.nc"
    runoff_file = f"Runoff/ambientera5_OLR_{year}_total_runoff.nc"

    P_grid = load_and_clip(precip_file, "tp", gdf) * 1000
    ET_grid = load_and_clip(et_file, "e", gdf) * 1000
    Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000

    P = P_grid.mean(dim=["latitude", "longitude"]).values
    ET = ET_grid.mean(dim=["latitude", "longitude"]).values
    Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values

    # Convert evaporation to positive values
    ET = np.where(ET < 0, -ET, ET)

    return P, ET, Q_obs


# Find which years are available

years = []

for year in range(2001, 2021):
    precip_file = f"Precipitation/era5_OLR_{year}_total_precipitation.nc"
    et_file = f"Total_Evaporation/era5_OLR_{year}_total_evaporation.nc"
    runoff_file = f"Runoff/ambientera5_OLR_{year}_total_runoff.nc"

    if os.path.exists(precip_file) and os.path.exists(et_file) and os.path.exists(runoff_file):
        years.append(year)

print("Available years:", years)


# Load all available years once

all_data = {}

for year in years:
    print(f"Loading data for {year}...")
    all_data[year] = load_year_data(year, gdf)

print("All years loaded successfully.")


# Function to compute mean KGE for a given k

def mean_kge_for_k(k, all_data, years):
    kge_list = []

    for year in years:
        P_y, ET_y, Q_obs_y = all_data[year]
        Q_sim_y = simulate_runoff(k, P_y, ET_y, Q_obs_y[0])
        KGE_y, _, _, _ = kge(Q_obs_y, Q_sim_y)
        kge_list.append(KGE_y)

    return np.mean(kge_list)



# Search for the best k across all years

# Wide search range for robustness
k_values = np.linspace(0.1, 1.0, 200)
mean_kges = []

for k in k_values:
    mean_kges.append(mean_kge_for_k(k, all_data, years))

mean_kges = np.array(mean_kges)

best_idx = np.argmax(mean_kges)
best_k_multi = k_values[best_idx]
best_mean_kge = mean_kges[best_idx]

print("\nMulti-year optimization results")
print(f"Best k across all years: {best_k_multi:.3f}")
print(f"Best mean KGE across all years: {best_mean_kge:.3f}")



# Compute yearly metrics using the best k

yearly_results = []

for year in years:
    P_y, ET_y, Q_obs_y = all_data[year]
    Q_sim_y = simulate_runoff(best_k_multi, P_y, ET_y, Q_obs_y[0])
    KGE_y, r_y, alpha_y, beta_y = kge(Q_obs_y, Q_sim_y)

    yearly_results.append([year, KGE_y, r_y, alpha_y, beta_y])

print("\nYearly results using the multi-year optimal k:")
for row in yearly_results:
    print(f"Year {row[0]}: KGE={row[1]:.3f}, r={row[2]:.3f}, alpha={row[3]:.3f}, beta={row[4]:.3f}")


# Plot: Sensitivity plot (mean KGE vs k)

plt.figure(figsize=(9, 5))
plt.plot(k_values, mean_kges, color="#e91e63", linewidth=2.5)
plt.axvline(best_k_multi, color="black", linestyle="--", linewidth=1.5,
            label=f"Best k = {best_k_multi:.3f}")
plt.xlabel("Storage coefficient k")
plt.ylabel("Mean KGE across all years")
plt.title("Multi-year sensitivity of model performance to k")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# Plot: Yearly KGE bar plot

years_plot = [row[0] for row in yearly_results]
kge_plot = [row[1] for row in yearly_results]

plt.figure(figsize=(10, 5))
plt.bar(years_plot, kge_plot, color="#ff69b4", edgecolor="black", alpha=0.85)
plt.axhline(np.mean(kge_plot), color="black", linestyle="--", linewidth=1.5,
            label=f"Mean KGE = {np.mean(kge_plot):.3f}")
plt.xlabel("Year")
plt.ylabel("KGE")
plt.title("Yearly model performance using the multi-year optimal k")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



# Select representative years: worst, median, and best KGE

sorted_results = sorted(yearly_results, key=lambda x: x[1])

worst_year = sorted_results[0][0]
median_year = sorted_results[len(sorted_results) // 2][0]
best_year = sorted_results[-1][0]

selected_years = [worst_year, median_year, best_year]

print("\nRepresentative years selected for runoff comparison:")
print(f"Worst year: {worst_year}")
print(f"Median year: {median_year}")
print(f"Best year: {best_year}")


# Plot runoff comparison for representative years

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

for ax, year in zip(axes, selected_years):
    P_y, ET_y, Q_obs_y = all_data[year]
    Q_sim_y = simulate_runoff(best_k_multi, P_y, ET_y, Q_obs_y[0])
    KGE_y, _, _, _ = kge(Q_obs_y, Q_sim_y)

    ax.plot(Q_obs_y, label="Observed runoff", linewidth=1.8)
    ax.plot(Q_sim_y, label="Simulated runoff", linewidth=1.4)
    ax.set_title(f"Observed vs simulated runoff for {year} (KGE = {KGE_y:.3f})")
    ax.set_ylabel("Runoff (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (hour)")
plt.tight_layout()
plt.show()

