import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
import tools

# -------- Part 1/2: Downloading and Processing Data-------------

# Open the CSV file

files = [
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.00.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.03.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.06.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.09.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.12.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.15.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.18.v02r01.nc",
    "/Users/solodim/Desktop/PhD/SEMESTER 1/Geo-Env Modeling/Ex 5/Data/GRIDSAT-B1.2009.11.25.21.v02r01.nc",
]

for fn in files:

    dset = xr.open_dataset(fn)


# 2) Load irwin_cdr data

IR = np.array(dset.variables["irwin_cdr"]).squeeze()

# 3) Flip data vertically

IR = np.flipud(IR)

# 4) Apply scale of 0.01 and an offset of 200 (for brightness t in K)

IR = IR * 0.01 + 200

# 5) Convert from Kelvin to Celsius

IR = IR - 273.15

# 6) Plot

plt.figure()
plt.imshow(IR, extent=[-180.035, 180.035, -70.035, 70.035], aspect="auto")
cbar = plt.colorbar()
cbar.set_label("Brightness temperature (degrees Celsius)")

# 7) Mark Jeddah's location

jeddah_lat = 21.5
jeddah_lon = 39.2
plt.scatter(jeddah_lon, jeddah_lat, color="red", marker="o", label="Jeddah")
plt.legend()

plt.title(fn.split("/")[-1])  
plt.show()


# 11) Spatial resolution
dset = xr.open_dataset(files[0])
IR0 = np.array(dset.variables["irwin_cdr"]).squeeze()
ny, nx = IR0.shape

lon_min, lon_max = -180.035, 180.035
lat_min, lat_max = -70.035, 70.035

d_lon = (lon_max - lon_min) / nx
print("Approx spatial resolution (km):", d_lon * 111)

# 12) Lowest brightness temperature near Jeddah
best_val = 1e9
best_hour = None

for fn in files:
    dset = xr.open_dataset(fn)
    IR = np.array(dset.variables["irwin_cdr"]).squeeze()
    IR = np.flipud(IR)
    IR = IR * 0.01 + 200
    IR = IR - 273.15

    ny, nx = IR.shape
    ix = int((39.2 - lon_min) / (lon_max - lon_min) * (nx - 1))
    iy = int((21.5 - lat_min) / (lat_max - lat_min) * (ny - 1))
    val = IR[iy, ix]

    if val < best_val:
        best_val = val
        best_hour = fn.split(".")[4]  

print("Lowest BT near Jeddah:", best_val, "°C at", best_hour, "UTC")


# ---- Part 3: Rainfall estimation ----

# 2) Formula for rainfall rates

A = 1.1183e11        # mm h^-1
b = 3.6382e-2        # K^-1
c = 1.2

def IR_to_R(IR_celsius):
    T = IR_celsius + 273.15          # Conversion back to Kelvin 
    R = A * np.exp(-b * (T ** c))    # mm/hr
    return R

# 3) Create a map of the cumulative rainfall btw 00:00 and 12:00

files_00_12 = files[0:5]   # 00,03,06,09,12
dt = 3.0                   # hours between images

R_list = []
for fn in files_00_12:
    dset = xr.open_dataset(fn)

    IR = np.array(dset.variables["irwin_cdr"]).squeeze()
    IR = np.flipud(IR)
    IR = IR * 0.01 + 200
    IR = IR - 273.15

    R = IR_to_R(IR)     # mm/hr
    R_list.append(R)

# trapezoidal time integration: sum 0.5*(R_i + R_{i+1})*dt
cum = np.zeros_like(R_list[0])
for i in range(len(R_list) - 1):
    cum += 0.5 * (R_list[i] + R_list[i+1]) * dt   # mm

# plot cumulative rainfall map 
plt.figure()
plt.imshow(cum, extent=[-180.035, 180.035, -70.035, 70.035], aspect="auto")
cbar = plt.colorbar()
cbar.set_label("Cumulative rainfall (mm), 00–12 UTC")

plt.scatter(39.2, 21.5, color="red", marker="o", label="Jeddah")
plt.legend()
plt.title("Cumulative rainfall (00–12 UTC)")
plt.show()

# find hour of max rainfall rate at Jeddah pixel (then convert to local time)
lon_min, lon_max = -180.035, 180.035
lat_min, lat_max = -70.035, 70.035
jeddah_lon, jeddah_lat = 39.2, 21.5

ny, nx = R_list[0].shape
ix = int((jeddah_lon - lon_min) / (lon_max - lon_min) * (nx - 1))
iy = int((jeddah_lat - lat_min) / (lat_max - lat_min) * (ny - 1))

vals = [R[iy, ix] for R in R_list]                # mm/hr at each time
i_max = int(np.argmax(vals))
best_utc = int(files_00_12[i_max].split(".")[4])  # 00/03/06/09/12

best_local = (best_utc + 3) % 24                  # Jeddah = UTC+3
print("Highest rain rate at Jeddah:", vals[i_max], "mm/hr at", best_utc, "UTC =", best_local, "local time")
