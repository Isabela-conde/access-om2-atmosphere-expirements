import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os, glob, re
import dask.array as da
import dask

print('starting - pray to the demons of dask', flush=True)


##### loop over each grid cell in the tropical pacific box - save at each point 
# 1. open jra55 and era5 files
# 2. find grid cells you want to loop over 
#      - grab 1 lat and lon of jra55 and 
# 3. start grid cell loop 
#    3.1 find trend
#    3.2 find difference and create time series with correct coords and name 
#    3.3 progressively save to nc file 
#    3.4 del variable 
# 4. fucking hopefully that works 
#####

years = np.arange(1980, 2019, 1)

######### 1. open datasets ###########

# 1. open jra55
jra55_filepaths = [
    f'/g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hrPt/uas/gr/v20190429/uas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_{year}01010000-{year}12312100.nc'
    for year in years
]
ds_jra55 = xr.open_mfdataset(jra55_filepaths, combine='by_coords')
print('1. jra55 files opened', flush=True)


# 1. open era5 datset
era5_path = '/g/data/ik11/inputs/ERA5/single-levels/reanalysis/10u/'
era5_filepaths = sorted(glob.glob(os.path.join(era5_path, '*', '10u_era5_oper_sfc_*.nc')))

files = [
    f for f in era5_filepaths
    if os.path.exists(f)
    and years[0] <= int(os.path.basename(f).split('_')[4][:4]) <= years[-1]
]
ds_era5 = xr.open_mfdataset(files, engine='netcdf4', combine='by_coords')
ds_era5=ds_era5.rename({'latitude': 'lat', 'longitude': 'lon', 'u10':'uas'}).sortby('lat') # dim and variables same as jra55

print('1. era5 files opened', flush=True)


######### 2. find lats to loop over ###########


#2.1 mask tropical pacific - box 

def mask_trop_pa(ds, return_coords=False): 
    # either return masked tropical pacific or return lats and lons in tropical pacific from a time slice
    lat_ok = (ds.lat >= -20) & (ds.lat <= 20)
    lon_ok = (ds.lon >= 110) & (ds.lon <= 290)

    masked_ds = ds.where(lat_ok & lon_ok, other=0)

    if return_coords:
        lats = ds.lat.where(lat_ok, drop=True)
        lons = ds.lon.where(lon_ok, drop=True)
        return lats.values, lons.values
    else:
        return masked_ds


jra55_slice = ds_jra55.isel(time = 0).uas.load()
lats, lons = mask_trop_pa(jra55_slice,return_coords=True)

print("2. find lats to loop over - done")

######### 3. start grid cell loop  ###########

bias_ts = ds_jra55.copy()*0

import dask.array as da
num = len(ds_jra55['time'].drop_vars(['height']).values)
times = da.linspace(0,num-1,num)       # shape (492,)
times = times[:, None, None] 

for j, nlat in enumerate(lats):
    for i, nlon in enumerate(lons):
        ds_jra55_cell = ds_jra55.sel(lat = nlat, lon = nlon)
        ds_jra55_cell_trend  = (
            ds_jra55_cell.uas
            .drop_vars(['height'])
            .polyfit(dim='time', deg=1)
            .polyfit_coefficients.sel(degree=1)  # slope
        )

        ds_era5_cell = ds_era5.sel(lat = nlat, lon = nlon)
        ds_era5_cell_trend  = (
            ds_era5_cell.uas
            .drop_vars(['height'])
            .polyfit(dim='time', deg=1)
            .polyfit_coefficients.sel(degree=1)  # slope
        )
        
        bias = ds_era5_trend_gr - ds_source_trend
        print(bias)

        bias_ts['uas'].loc[dict(lat=nlat, lon=nlon)] = times*float(bias)
        if [i,j]==[1,1]:
            print(bias_ts)


        
    







print('control trend found', flush=True)









# ERA5 trend (same operation), then align lat order for safe slicing
ds_era5_trend = (
    ds_era5.u10
    .polyfit(dim='time', deg=1)
    .polyfit_coefficients.sel(degree=1)
)
ds_era5_trend = ds_era5_trend.rename({'latitude': 'lat', 'longitude': 'lon'}).sortby('lat')
print('era5 trend found', flush=True)

# Interpolate ERA5 trend to JRA55 grid (your ops)
new_lats = ds_source_trend.lat.values
new_lons = ds_source_trend.lon.values
ds_era5_trend_gr = ds_era5_trend.sel(lat=slice(new_lats[0], new_lats[-1])).interp(lat=new_lats, lon=new_lons)



# Build time-dependent field (unchanged math)
num = len(ds_source['time'].drop_vars(['height']).values)

# Smaller time chunks to keep memory low during write
times = da.linspace(0, num-1, num, chunks=(128,))
times = times[:, None, None]
bias_ts = times * da.broadcast_to(bias_masked, (num, 320, 640))

# Rechunk to smaller (128,320,640) for safer per-chunk size
bias_ts = bias_ts.rechunk((128, 320, 640))
print("[chk] planned chunks (time,lat,lon):", bias_ts.chunksize, flush=True)

# Wrap into Dataset (unchanged)
bias_ts_da = xr.Dataset(
    {'uas': xr.DataArray(
        bias_ts,
        dims=("time", "lat", "lon"),
        coords={
            "time": ds_source['time'].drop_vars(['height']).values,
            "lat": new_lats,
            "lon": new_lons
        }
    )}
)
ds_u10m = bias_ts_da
template_ds = ds_source

# Assign coords/bounds (unchanged)
ds_u10m = ds_u10m.assign_coords(height=template_ds.height)
ds_u10m['lat_bnds']  = template_ds['lat_bnds']
ds_u10m['lon_bnds']  = template_ds['lon_bnds']
ds_u10m['time_bnds'] = template_ds['time_bnds']
ds_u10m = ds_u10m[['uas','lat_bnds','lon_bnds','time_bnds']]

print('starting to save file', flush=True)

# Free upstream refs before write (no math change)
try:
    del ds, ds_source_trend, ds_era5_trend, ds_era5_trend_gr, bias, bias_masked
except NameError:
    pass
import gc; gc.collect()

# Match on-disk chunks to our rechunk; keep everything else as-is
encoding = {'uas': {'chunksizes': (128, 320, 640)}}

out_path = "/g/data/e14/ic0706/access-om2/forcing_perturbations/uas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_Tropical_Pacific_ERA5_Trend_Bias_Perturbation.nc"

# Sequential write to keep peak memory low (no change to results)
# with dask.config.set(scheduler='single-threaded'):
ds_u10m.to_netcdf(out_path)

print('done', flush=True)

