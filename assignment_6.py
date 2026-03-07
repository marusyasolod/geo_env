import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
import tools

# -------- Part 2:  Data Pre-Processing -------------

# Open the NET-CDF file

dset = xr.open_dataset('/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 6/reanalysis-era5-single-levels-timeseries-sfcyzr2jb0l.nc')

print(dset)

# Extract the relevant variables from the dataset
                           
t2m = dset.variables['t2m']
tp = dset.variables['tp']
latitude = dset.variables['latitude']
longitude = dset.variables['longitude']
time_dt = np.array(dset.variables['valid_time'])
time_dt = pd.to_datetime(time_dt)

df_era5 = pd.DataFrame(index=time_dt)

# Convert air temperature from K to C and precipitation from m h^-1 to mm h^-1

t2m = t2m - 273.15
tp = tp * 1000

if t2m.ndim == 4:
   t2m = np.nanmean(t2m, axis=1)
   tp = np.nanmean(tp, axis=1)                        

# Create a Pandas dataframe

df_era5 = pd.DataFrame(index=time_dt)
df_era5['t2m'] = t2m
df_era5['tp'] = tp

# Plot

df_era5.plot()

plt.legend(loc='upper right')
                           
plt.show()


# Annual average precipitation

annual_precip = df_era5['tp'].resample('YE').mean()*24*365.25
mean_annual_precip = np.nanmean(annual_precip)

print(annual_precip)
print(mean_annual_precip)                           

# -------- Part 3:  Calculation of Potential Evaporation (PE) -------------

# Derive all inputs for the function from the hourly ERA5 data
# using equation for the Hargreaves and Samani (1985) method

tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values
lat = 21.25
doy = df_era5['t2m'].resample('D').mean().index.dayofyear


# Compute PE
pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)

# Create daily time index for plotting
ts_index = df_era5['t2m'].resample('D').mean().index

# Plot PE time series
plt.figure()
plt.plot(ts_index, pe, label='Potential Evaporation')
plt.xlabel('Time')
plt.ylabel('Potential evaporation (mm d$^{-1}$)')
plt.legend()
plt.show()

# Mean annual PE in mm/year
annual_pe = pd.Series(pe, index=ts_index).resample('YE').sum()
mean_annual_pe = annual_pe.mean()

print("Annual PE (mm/year):")
print(annual_pe)

print("Mean annual PE (mm/year):", mean_annual_pe)
                           
