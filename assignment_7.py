import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob
import geopandas as gpd
import rioxarray


# Load Saudi shapefile
shape = gpd.read_file("/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 7/Saudi_Shape_File/Saudi_Shape.shp")
shape = shape.to_crs("EPSG:4326")

def clip_to_saudi(dataarray):
    dataarray = dataarray.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)
    dataarray = dataarray.rio.write_crs("EPSG:4326", inplace=False)
    return dataarray.rio.clip(shape.geometry, shape.crs, drop=True)


## PLOTTING PRECIPITATION

# Open NetCDF files

files = sorted(glob.glob("/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 7/Precipitation/*.nc"))
print(files)

datasets = [xr.open_dataset(f) for f in files]

# Merge all yearly precipitation files into one dataset by extending the time axis
ds = xr.concat(datasets, dim="valid_time")

print(ds)
print("Data variables:", ds.data_vars)
print("Coordinates:", ds.coords)

# Total precipitation
pr = ds["tp"] * 1000   # convert from m to mm

print("Precip dims:", pr.dims)

# Average over Saudi Arabia domain

pr_saudi = clip_to_saudi(pr)

spatial_pr = pr_saudi.mean(dim=["latitude", "longitude"])

# Monthly and yearly sums
monthly_pr = spatial_pr.resample(valid_time="1MS").sum().compute()
yearly_pr  = spatial_pr.resample(valid_time="1YS").sum().compute()

# Overlay plot
plt.figure(figsize=(12,6))
monthly_pr.plot(label="Monthly precipitation")
yearly_pr.plot(marker="o", linewidth=2, label="Yearly precipitation")

plt.title("Monthly and Yearly Total Precipitation in Saudi Arabia (2000–2020)")
plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.grid(True)
plt.legend()
plt.show()



## PLOTTING EVAPORATION

# Open NetCDF files

files = sorted(glob.glob("/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 7/Total_Evaporation/*.nc"))
print(files)

# Open and combine all yearly files

datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim="valid_time")

print(ds)
print("Data variables:", ds.data_vars)
print("Coordinates:", ds.coords)

# Total evaporation variable

evap = ds["e"] * 1000   # convert from m to mm


# Average over Saudi Arabia domain
evap = -evap

evap_saudi = clip_to_saudi(evap)
spatial_evap = evap_saudi.mean(dim=["latitude", "longitude"])

# Monthly sum

monthly_evap = spatial_evap.resample(valid_time="1MS").sum().compute()

# Yearly sum

yearly_evap = spatial_evap.resample(valid_time="1YS").sum().compute()

# Overlay both on one plot

plt.figure(figsize=(12,6))
monthly_evap.plot(label="Monthly evaporation")
yearly_evap.plot(marker="o", linewidth=2, label="Yearly evaporation")

plt.title("Monthly and Yearly Total Evaporation in Saudi Arabia (2000–2020)")
plt.xlabel("Time")
plt.ylabel("Evaporation (mm)")
plt.grid(True)
plt.legend()
plt.show()

## PLOTTING RUNOFF

# Load runoff files
files = sorted(glob.glob("/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 7/Runoff/*.nc"))
print(files)

# Open and combine
datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim="valid_time")

print(ds)
print(ds.data_vars)

# Runoff variable 
runoff = ds["ro"] * 1000   # convert from m to mm

# Spatial average over Saudi Arabia

runoff_saudi = clip_to_saudi(runoff)

spatial_ro = runoff_saudi.mean(dim=["latitude", "longitude"])
# Monthly and yearly sums
monthly_ro = spatial_ro.resample(valid_time="1MS").sum().compute()
yearly_ro  = spatial_ro.resample(valid_time="1YS").sum().compute()

# Plot both on same figure
plt.figure(figsize=(12,6))
monthly_ro.plot(label="Monthly runoff")
yearly_ro.plot(marker="o", linewidth=2, label="Yearly runoff")

plt.title("Monthly and Yearly Total Runoff in Saudi Arabia (2000–2020)")
plt.xlabel("Time")
plt.ylabel("Runoff (mm)")
plt.grid(True)
plt.legend()
plt.show()


#COMPARISON

# Compute difference
water_balance = monthly_pr - (monthly_evap + monthly_ro)

# Plot
plt.figure(figsize=(12,6))
water_balance.plot()

plt.title("Water Balance: P - (E + R) in Saudi Arabia (2000–2020)")
plt.xlabel("Time")
plt.ylabel("mm")
plt.axhline(0, linestyle="--")  
plt.grid(True)
plt.show()
