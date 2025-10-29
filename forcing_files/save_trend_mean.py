import xarray as xr
import numpy as np
from pathlib import Path
import gc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import xarray as xr
import glob
import pandas as pd


def pacific_mask(sst):
    LonsPts_pos = [130,120,110,110,110,260,260,270,267,266,300,300,300]
    LatsPts_pos =[-30,-10,-5,10,30,30,25,17,15,12,10,-30]
    
    x_pixel_pos = LonsPts_pos
    y_pixel_pos = LatsPts_pos
    
    temp_list = []
    for a, b in zip(x_pixel_pos, y_pixel_pos):
        temp_list.append([a, b])
    
    polygon = np.array(temp_list)
    
    xv, yv = sst.lon.values, sst.lat.values
    lon2d, lat2d = np.meshgrid(xv, yv)
    points = np.hstack((lon2d.reshape((-1,1)), lat2d.reshape((-1,1))))
    path = matplotlib.path.Path(polygon)
    mask = path.contains_points(points)
    mask.shape = lon2d.shape
    mask_bit_pos = mask*np.ones_like(lon2d)
    
    
    mask_mods = mask_bit_pos#+mask_bit_neg
    
    mask_mods[mask_mods==0] = np.nan
    return mask_mods
    
def find_trend_values(dt):

    #Input: xarray array with time, lat, lon coordinates and a variable to calculate the trend
    #Output: xarray array with the trend values for each grid point
    
    time, lat, lon = dt['time'].values, dt['lat'].values, dt['lon'].values
    sst_reshaped = dt.values.reshape(len(time), -1)
    degree = 1

    gradient_values = np.empty_like(sst_reshaped[0])
    for i in range(len(sst_reshaped[0])):
        y = sst_reshaped[:, i]
        coefficients = np.polyfit(np.arange(len(time)), y, degree)
        gradient_values[i] = coefficients[-2]  # Coefficient for x term (slope)

    gradient_values = gradient_values.reshape(lat.shape[0], lon.shape[0])
    gradient_da = xr.DataArray(gradient_values, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])
    return gradient_da 

def save_trend(ds_jra55,var):


    ds_pa = pacific_mask(ds_jra55[var])*ds_jra55[var]
    ds_jra55_trends = find_trend_values(ds_pa.sel(lat = slice(-20,20)))
    
    ds_jra55_trends.to_netcdf('/g/data/e14/ic0706/JRA55-do/trends/JRA55-do_'+var+'-trend_1982-2018_gr.nc')


    # 5) free memory before next year
    ds_jra55_trends.close()
    ds_pa.close()
    del ds_jra55_trends, ds_pa
    gc.collect()
    return

def save_mean(ds_jra55,var):
    
    ds_pa = pacific_mask(ds_jra55[var])*ds_jra55[var]
    ds_jra55_mean = ds_pa.sel(lat = slice(-20,20)).mean(dim = 'time')
    
    ds_jra55_mean.to_netcdf('/g/data/e14/ic0706/JRA55-do/means/JRA55-do_'+var+'-mean_1982-2018_gr.nc')


    # 5) free memory before next year
    ds_jra55_mean.close()
    ds_pa.close()
    del ds_jra55_mean, ds_pa
    gc.collect()
    return 
    
years = np.arange(1982, 2019, 1)


jra55_fps = [
    f"/g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hrPt/huss/gr/v20190429/huss_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_{y}01010000-{y}12312100.nc"
    for y in years
]

ds_jra55 = xr.open_mfdataset(
    jra55_fps, combine="by_coords"
)


save_trend(ds_jra55, 'huss')
save_mean(ds_jra55, 'huss')

