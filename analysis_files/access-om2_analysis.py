import os
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import glob
import pandas as pd
import matplotlib
## plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

#### funcs

def relabel_sort(SST):
    sorted_sst=SST.rename({'xt_ocean':'lon', 'yt_ocean': 'lat'})
    sorted_sst['lon'] = (sorted_sst['lon'] + 360) % 360
    sorted_sst = sorted_sst.sortby('lon')
    return sorted_sst



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
        gradient_values[i] = coefficients[-2]

    gradient_values = gradient_values.reshape(lat.shape[0], lon.shape[0])
    gradient_da = xr.DataArray(gradient_values, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])
    return gradient_da 



########## Percentiles ############

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
    
    
    mask_mods = mask_bit_pos
    
    mask_mods[mask_mods==0] = np.nan
    return mask_mods

# percentile function 
from xhistogram.xarray import histogram

def quantiles_xhist(data):
    hist_T = data
    weights = np.cos(np.deg2rad(hist_T.lat))
    times = hist_T['time']
    const_area=np.r_[0:1:0.005]
    temps = np.r_[8:34:0.5]
    
    temps_interp = np.empty([len(times),len(const_area)])
    
    for i in np.r_[0:len(times):1]:  
        h=histogram(hist_T[i],bins=temps,weights=weights).cumsum()
        hh = (h - h[0])/(h[-1]-h[0])
        temps_interp[i]=np.interp(const_area,hh,temps[1:])
    
    
    sst_area= xr.DataArray(temps_interp,coords={'time':times,'area':const_area})

    return(sst_area)


def find_trend_percentile(data):
    data = data.sel(time=slice('1983','2017'))
    x_val = np.arange(0, len(data.time))
    slope_array=[np.polyfit(x_val, data.isel(area=j), 1)[0] for j in range(200)]
    da = xr.DataArray(slope_array, coords ={'area':data.area.values},name = 'trend')
    return da






def basic_om2_analysis(perturb_base):
    out_path = '/g/data/e14/ic0706/access-om2_expirements/access-om2_analysis_output/out/'
    os.makedirs(out_path, exist_ok=True)

    control_base = '/g/data/e14/ic0706/access-om2/1deg_jra55_iaf/'
    SST_control = (xr.open_mfdataset(glob.glob(control_base + 'output*/ocean/ocean-2d-surface_pot_temp-1-monthly-*.nc')).surface_pot_temp  - 273.15).sel(time = slice('1983','2017'))
    SST_perturb = xr.open_mfdataset(glob.glob(perturb_base + 'output*/ocean/ocean-2d-surface_pot_temp-1-monthly-*.nc')).surface_pot_temp  - 273.15

    print('1. Datatsets opened')

    
    SST_control = relabel_sort(SST_control)
    SST_perturb = relabel_sort(SST_perturb)

    print('2. Datatsets relabelled')


    

    ###### plot snap shots
    SST_control['time'] = [pd.Timestamp(t).replace(day=16) for t in SST_control['time'].values]
    
    fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(12,8))
    time1 = np.datetime64('1983-01-16T12:00:00')
    time2 = np.datetime64('2000-01-16T12:00:00')
    time3 = np.datetime64('2017-01-16T12:00:00')
    
    for i,time in enumerate([time1,time2,time3]):
        SST_control.sel(time=time).plot(ax=axes[0][i],vmin=12.,vmax=30)
        axes[0][i].set_title('Control SST ' + str(np.datetime64(time,'M')))
        SST_perturb.sel(time=time).plot(ax=axes[1][i],vmin=12.,vmax=30)
        axes[1][i].set_title('Perturbation SST ' + str(np.datetime64(time,'M')))
        (SST_perturb-SST_control).sel(time=time).plot(ax=axes[2][i],vmin=-2.,vmax=2.,cmap='RdBu_r')
        axes[2][i].set_title('Perturbation - Control SST ' + str(np.datetime64(time,'M')))
    
    plt.tight_layout()
    plt.savefig(out_path +'comparison-snapshots.png', dpi=300)
    plt.close()

    print('3. comparison-snapshot figure saved')



    sst_perturb_trend = find_trend_values(SST_perturb)
    sst_control_trend = find_trend_values(SST_control)
    
    sst_control_trend.to_netcdf(out_path+'control_gridcell_trend.nc')
    sst_perturb_trend.to_netcdf(out_path+'perturb_gridcell_trend.nc')
    
    print('3. trend found and saved')


    pert = sst_perturb_trend * 10 * 12
    ctrl = sst_control_trend * 10 * 12
    
    proj = ccrs.PlateCarree(central_longitude=180)
    pc0  = ccrs.PlateCarree()
    extent = [110, 290, -20, 20]
    
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.05, hspace=0.10)
    
    axes = [fig.add_subplot(gs[0, 0], projection=proj),
            fig.add_subplot(gs[0, 1], projection=proj)]
    
    ims = []
    for ax, da, title in zip(axes[:2], [pert, ctrl], ['Perturbed', 'Control']):
        im = ax.pcolormesh(da.lon, da.lat, da, cmap='RdBu_r', vmin=-0.3, vmax=0.3, transform=pc0)
        ax.set_extent(extent, crs=pc0)
        ax.coastlines()
        ax.set_title(title, fontsize=12)
        ims.append(im)

    for ax in axes:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

    cax = fig.add_axes([0.92, 0.42, 0.01, 0.15])
    fig.colorbar(ims[0], cax=cax, label=r'$^{\circ}$C$ / \, 10$ yrs')
    
    plt.tight_layout()
    
    plt.savefig(out_path+'trend.png', dpi=300)
    plt.close()

    print('4. trend plots saved')

    SST_control_pa = (pacific_mask(SST_control)*SST_control).sel(lon = slice(110,290), lat =slice(-20,20))
    SST_perturb_pa = (pacific_mask(SST_perturb)*SST_perturb).sel(lon = slice(110,290), lat =slice(-20,20))

    SST_control_percentiles = quantiles_xhist(SST_control_pa)
    SST_perturb_percentiles = quantiles_xhist(SST_perturb_pa)

    print('5. percentiles found')

    SST_control_percentiles_trend =  find_trend_percentile(SST_control_percentiles)
    SST_perturb_percentiles_trend = find_trend_percentile(SST_perturb_percentiles)
    
    SST_control_percentiles_trend.to_netcdf(out_path+'control_percentile_trend.nc')
    SST_perturb_percentiles_trend.to_netcdf(out_path+'perturb_percentile_trend.nc')
    print('7. percentiles trend saved')

    
    plt.figure(figsize=(5, 5))

    SST_perturb_percentiles_trend.plot(label = 'perturb')
    SST_control_percentiles_trend.plot(label = 'control')
    plt.legend()

    plt.savefig(out_path +'percentile_trend.png', dpi=300)
    plt.close()
    print('7. percentiles trend plot saved')

    return 



    
basic_om2_analysis('/g/data/e14/ic0706/access-om2/1deg_jra55_iaf_trend_perturbed/')
