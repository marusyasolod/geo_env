import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr


# Open the NetCDF file

dset_part2 = xr.open_dataset(
    "/Users/solodim/Desktop/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc"
)

# Pause execution to inspect the dataset
pdb.set_trace()

# Part 2 (Q4, Q6, Q10 and Q11 are answered in the Overleaf document)

# Question 2 (Print the names of the variables)
print(dset_part2.keys())

# Question 3 (Dimensions of the air temperature variable)
print("Dimensions of tas:", dset_part2["tas"].dims)

# Question 5 (Data type of the air temperature variable)
print("Data type of tas:", dset_part2["tas"].dtype)

# Question 7 (Temporal span of the file)
print("Time span:", dset_part2["time"].values[0], "to", dset_part2["time"].values[-1])

# Question 8 (Units of air temperature data)
print("Units of tas:", dset_part2["tas"].attrs.get("units"))

# Question 9 (Spatial and temporal resolution of air temp data )
lat_res = dset_part2["lat"].values[1] - dset_part2["lat"].values[0]
lon_res = dset_part2["lon"].values[1] - dset_part2["lon"].values[0]
print("Spatial resolution (lat, lon):", lat_res, lon_res)
print("Temporal resolution: monthly")


# Part 3

# Open the NetCDF file

dset_part3 = xr.open_dataset(
    "/Users/solodim/Desktop/Climate_Model_Data/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc"
)

# Question 1 (Calculate the mean air temperature map for 1850–1900)

mean_1850_1900 = np.mean(dset_part3['tas'].sel(time=slice('1850-01-01','1900-12-31')), axis=0)

mean_1850_1900 = np.array(mean_1850_1900)

print("Part 3 dtype:", mean_1850_1900.dtype)

print("Part 3 shape:", mean_1850_1900.shape)

global_mean_1850_1900_K = np.nanmean(mean_1850_1900)
global_mean_1850_1900_C = global_mean_1850_1900_K - 273.15

print("Global mean temperature 1850–1900 (K):", global_mean_1850_1900_K)
print("Global mean temperature 1850–1900 (°C):", global_mean_1850_1900_C)


# Question 2 (Calculate the mean air temperature map for 2071–2100) for each climate scenario

scenario_files = {
    "ssp245": "/Users/solodim/Desktop/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc",
    "ssp119": "/Users/solodim/Desktop/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc",
    "ssp585": "/Users/solodim/Desktop/Climate_Model_Data/tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc",
}

mean_maps_2071_2100 = {}

for scen, path in scenario_files.items():

# Open NetCDF file for the given scenario
    
    ds = xr.open_dataset(path)

# Select the time period

    tas_2071_2100 = ds["tas"].sel(time=slice("2071-01-01", "2100-12-31"))
    
    print(f"{scen} selected months:", tas_2071_2100.sizes["time"])  # should be 360

    mean_map = np.mean(tas_2071_2100, axis=0)

# Convert to NumPy array

    mean_map = np.array(mean_map)

# Store the mean temperature map for this scenario

    mean_maps_2071_2100[scen] = mean_map

    print(f"{scen} mean map dtype:", mean_map.dtype)
    print(f"{scen} mean map shape:", mean_map.shape)
    print(f"{scen} global mean (K):", np.nanmean(mean_map))
    print(f"{scen} global mean (°C):", np.nanmean(mean_map) - 273.15)
    
    print("-" * 40) #for visual effect

# Question 3 (Compute and visualize the temperature differences between 2071–2100 and 1850–1900 for each scenario)

import numpy as np
import matplotlib.pyplot as plt

# Get lat/lon values for plotting

lat = dset_part3["lat"].values
lon = dset_part3["lon"].values

# Color scale limits

vmin, vmax = -5, 15   # in K 

# Compute and plot for each scenario

for scen, mean_2071_2100 in mean_maps_2071_2100.items():
    
# Compute temperature anomaly map (K)
    
    deltaT = mean_2071_2100 - mean_1850_1900  

# Plot
    plt.figure(figsize=(10, 4.8))
    pcm = plt.pcolormesh(lon, lat, deltaT, shading="auto", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pcm)
    cbar.set_label("Temperature difference ΔT (K)")

    plt.title(f"{scen}: ΔT (2071–2100 minus 1850–1900)")
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.savefig(f"deltaT_{scen}.png", dpi=300, bbox_inches="tight")
    plt.close()






