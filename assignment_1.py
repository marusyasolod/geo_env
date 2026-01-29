import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import pdb

# Open the NetCDF file
dset = xr.open_dataset(
    "/Users/solodim/Desktop/SRTMGL1_NC.003_Data/N21E039.SRTMGL1_NC.nc"
)

# Pause execution to inspect the dataset
pdb.set_trace()

# Load elevation data
DEM = np.array(dset.variables["SRTMGL1_DEM"])

# Close the dataset
dset.close()

# Pause again to inspect DEM
pdb.set_trace()

# Visualize the data
plt.imshow(DEM)
cbar = plt.colorbar()
cbar.set_label("Elevation (m asl)")
plt.savefig("assignment_1.png", dpi=300)
