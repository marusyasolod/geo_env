import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
import tools

# -------- Part 1: Downloading and Importing Jeddah Weather Data-------------

# Open the CSV file

df_isd = tools.read_isd_csv("/Users/solodim/Desktop/41024099999.csv")


# Plot the data

plot = df_isd.plot(title="ISD data for Jeddah")
plt.savefig("ISD data for Jeddah")
plt.close()

# ---------Part 2: Heat Index (HI) Calculation (Q5, Q7, Q8, Q9 are answered in the Overleaf document)----------

# Q1 - Conversion of dewpoint temperature to relative humidity

df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW'].values,df_isd['TMP'].values)

# Q2 - Calculation of HI from air temperature and relative humidity data

df_isd['HI'] = tools.gen_heat_index(df_isd['TMP'].values, df_isd['RH'].values)

# Q3 - Highest HI level in the year

hi_max = df_isd['HI'].max()
print("Highest HI level in the year:",hi_max)

# Q4 - Day and time with the highest HI observed

t_max = df_isd['HI'].idxmax()
print("Day and time with the highest HI:",t_max)

# Q6 - Air temperature and relative humidity observed at highest HI

df_isd.loc[['2024-08-10 11:00:00']]
print(df_isd.loc[['2024-08-10 11:00:00']])

# Q10 - Plot

plt.figure(figsize=(10, 4.8))

plt.plot(df_isd.index, df_isd["HI"])

plt.title("Heat Index (HI) Time Series – Jeddah (2024)")
plt.xlabel("Time (UTC)")
plt.ylabel("Heat Index (°C)")

plt.savefig("HI_time_series_2024.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------Part 3: Potential Impact of Climate Change--------------

# Q2 - Increase in the highest HI value when additional warming is considered (2.5C)

deltaT = 2.503  # projected warming from Assignment 2 (°C)

# Recalculate HI with warming applied

HI_warm = tools.gen_heat_index(df_isd["TMP"].values + deltaT, df_isd["RH"].values)

# Compute maxima

hi_max_original = df_isd["HI"].max()
hi_max_warm = np.nanmax(HI_warm)

print("Projected warming applied (°C):", deltaT)
print("Original max HI (°C):", hi_max_original)
print("New max HI after warming (°C):", hi_max_warm)
print("Increase in max HI (°C):", hi_max_warm - hi_max_original)



