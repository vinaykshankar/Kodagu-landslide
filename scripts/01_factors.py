"""Derive terrain predictors on the 30 m UTM 43N grid from the SRTM 1-arc-second DEM."""
import numpy as np, rasterio, sys
if not hasattr(np, "in1d"): np.in1d = np.isin   # pysheds 0.5 + numpy 2 compatibility
from pysheds.grid import Grid
D = sys.argv[1]
with rasterio.open(f'{D}/dem_utm.tif') as s:
    dem = s.read(1).astype('float64'); prof = s.profile; tr = s.transform
cs = 30.0
z = np.pad(dem, 1, mode='edge')
a,b,c = z[:-2,:-2], z[:-2,1:-1], z[:-2,2:]
d,e,f = z[1:-1,:-2], z[1:-1,1:-1], z[1:-1,2:]
g,h,i = z[2:,:-2], z[2:,1:-1], z[2:,2:]
# Horn (1981) gradients; rows increase southward
dzdx = ((c+2*f+i)-(a+2*d+g))/(8*cs)
dzdy_s = ((g+2*h+i)-(a+2*b+c))/(8*cs)          # dz per metre southward
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy_s)))
# aspect: downslope direction, degrees clockwise from north
dzdn = -dzdy_s                                   # dz per metre northward
aspect = (np.degrees(np.arctan2(-dzdx, -dzdn)) + 360) % 360
flat = slope < 1.0
north = np.where(flat, 0.0, np.cos(np.radians(aspect)))
east  = np.where(flat, 0.0, np.sin(np.radians(aspect)))
# Zevenbergen & Thorne (1987) total curvature (x100; positive = convex)
Dz = ((d+f)/2 - e)/cs**2
Ez = ((b+h)/2 - e)/cs**2
curv = -2*(Dz+Ez)*100
# TWI from D8 flow accumulation (pysheds)
grid = Grid.from_raster(f'{D}/dem_utm.tif')
r = grid.read_raster(f'{D}/dem_utm.tif')
r = grid.fill_pits(r); r = grid.fill_depressions(r); r = grid.resolve_flats(r)
fdir = grid.flowdir(r); acc = np.asarray(grid.accumulation(fdir), dtype='float64')
sca = (acc + 1) * cs                              # specific catchment area (m)
twi = np.log(sca / np.tan(np.radians(np.maximum(slope, 0.1))))
nan = np.isnan(dem)
prof.update(dtype='float32', nodata=np.nan, compress='lzw')
for name, arr in [('slope',slope),('northness',north),('eastness',east),('curvature',curv),('twi',twi),('elevation',dem)]:
    arr = arr.astype('float32'); arr[nan] = np.nan
    with rasterio.open(f'{D}/f_{name}.tif','w',**prof) as o: o.write(arr,1)
    print(name, np.nanpercentile(arr,[1,50,99]).round(2))
