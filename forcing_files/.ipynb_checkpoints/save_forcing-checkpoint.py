import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os, glob, re
import dask.array as da
import dask
from pathlib import Path
import time
import gc

print('starting', flush=True)
t_all_start = time.time()

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

print('[load] opening JRA55 trend', flush=True)
jra55_trend = xr.open_dataarray('/g/data/e14/ic0706/JRA55-do/trends/jra55-do_uas-trend_1982-2018.nc')

print('[load] opening ERA5 monthly u10 files', flush=True)
base = Path('/g/data/rt52/era5/single-levels/monthly-averaged/10u')
files = [str(p) for p in base.glob('*/10u_era5_moda_sfc_*.nc')
         if 1982 <= int(p.name.split('_')[4][:4]) <= 2018]
print(f'[load] found {len(files)} ERA5 files in 1982–2018', flush=True)
ds_era5 = xr.open_mfdataset(sorted(files), combine='by_coords')
# rename and label
ds_era5 = ds_era5.rename({'latitude':'lat','longitude':'lon','u10':'uas'}).sortby('lat')

ds_era5['lon']=(ds_era5['lon']) % 360
ds_era5 = ds_era5.sortby('lon')
ds_era5=ds_era5.sortby('lat')

print('[trend] computing ERA5 Pacific mask & trend (±20°)', flush=True)
ds_era5_pa = pacific_mask(ds_era5.uas)*ds_era5.uas
uas_era5_trends = find_trend_values(ds_era5_pa.sel(lat = slice(-20,20)))

print('[regrid] interpolating ERA5 trend onto JRA55 grid', flush=True)
new_lats = jra55_trend.lat.values
new_lons = jra55_trend.lon.values
ds_era5_trend_gr = uas_era5_trends.sel(lat = slice(new_lats[0],new_lats[-1])).interp(lat=new_lats, lon=new_lons)

bias_trop = jra55_trend-ds_era5_trend_gr

########
print('[load] opening JRA55 3-hourly uas (1982–2018)', flush=True)
years = np.arange(1982, 2019, 1)
filepaths = [
    f'/g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hrPt/uas/gr/v20190429/uas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_{year}01010000-{year}12312100.nc'
    for year in years
]

ds_jra55 = xr.open_mfdataset(filepaths, combine='by_coords')
print('[bias] building global bias template', flush=True)

tpl = ds_jra55.uas.isel(time=0).drop_vars('height', errors='ignore')
bias2 = bias_trop.rename({'latitude':'lat', 'longitude':'lon'}) if 'latitude' in bias_trop.dims else bias_trop
bias_global = bias2.reindex_like(tpl).fillna(0)

####################################
##### Save indivual years ##########
####################################

print('[save-years] start saving per-year files', flush=True)
t_years_start = time.time()

out_dir = Path("/g/data/e14/ic0706/jra55_bias_per_year_simple")
out_dir.mkdir(parents=True, exist_ok=True)

offset = 0  

for y in range(1982, 2020):
    print(f'[save-years] processing {y}', flush=True)
    # 1) slice one year
    dsy = ds_jra55.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))
    n = dsy.sizes.get("time", 0)
    print(f'[save-years] {y}: n={n}, offset={offset}', flush=True)
    if n == 0:
        print(f'[save-years] {y}: skipped (no time steps)', flush=True)
        continue

    t_index = xr.DataArray(
        np.arange(offset, offset + n, dtype="int32"),
        dims=["time"],
        coords={"time": dsy.time},
        name="t_index",
    )

    new_uas = (t_index * bias_global).rename("uas")
    if "uas" in dsy:
        new_uas.attrs = dsy["uas"].attrs

    dsy_out = dsy.drop_vars("uas", errors="ignore").assign(uas=new_uas)
    out_file = out_dir / f"uas_bias_times_time_{y}.nc"
    t_save_start = time.time()
    dsy_out.to_netcdf(out_file)
    t_save_end = time.time()
    print(f'[save-years] wrote {out_file} in {t_save_end - t_save_start:.2f}s', flush=True)

    offset += n
    dsy_out.close()
    dsy.close()
    del dsy_out, dsy, t_index, new_uas
    gc.collect()

t_years_end = time.time()
print(f'[save-years] all yearly files done in {(t_years_end - t_years_start)/60:.2f} min', flush=True)

print("Done.", flush=True)

###############################
#### combine to one file ######
###############################
print('[final] building final concatenated Dataset', flush=True)
t_final_start = time.time()

years = np.arange(1958, 1982, 1)

int_jra55_filepaths = [
    f'/g/data/qv56/replicas/input4MIPs/CMIP6/OMIP/MRI/MRI-JRA55-do-1-4-0/atmos/3hrPt/uas/gr/v20190429/uas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_{y}01010000-{y}12312100.nc'
    for y in years
]
int_ds_jra55 = xr.open_mfdataset(
    int_jra55_filepaths, engine='netcdf4', combine='by_coords'
)

int_ds_jra55['uas'] = int_ds_jra55['uas']*0

files = sorted(glob.glob(os.path.join(out_dir, 'uas_bias_times_time_*.nc')))
print(f'[final] found {len(files)} yearly files to combine', flush=True)
ds_uas = xr.open_mfdataset(files, engine='netcdf4', combine='by_coords')

ds_save = xr.concat([int_ds_jra55,ds_uas],dim = 'time')

t_write_start = time.time()
ds_save.to_netcdf("/g/data/e14/ic0706/access-om2/forcing_perturbations/uas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_Tropical_Pacific_ERA5_Trend_Bias_Perturbation.nc")
t_write_end = time.time()
print(f'[final] wrote final file in {t_write_end - t_write_start:.2f}s', flush=True)

print('control data set opened', flush=True)

t_all_end = time.time()
print(f'[total] script finished in {(t_all_end - t_all_start)/60:.2f} min', flush=True)
