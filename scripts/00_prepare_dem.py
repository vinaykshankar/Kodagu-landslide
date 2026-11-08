"""Clip the merged SRTM 1-arc-second DEM to the district (+0.02 deg margin) and reproject (bilinear) to a 30 m UTM 43N grid."""
import sys, numpy as np, geopandas as gpd, rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
D = sys.argv[1]
b = gpd.read_file(f'{D}/boundary/Kodagu.shp').to_crs(4326)
with rasterio.open(f'{D}/Kodagu_SRTM_merged.tif') as src:
    arr, tr = mask(src, b.geometry.buffer(0.02), crop=True, nodata=np.nan, filled=True); crs = src.crs
arr = arr[0].astype('float64'); arr[(arr == 32767) | (arr < -100)] = np.nan
h, w = arr.shape
t2, w2, h2 = calculate_default_transform(crs, 'EPSG:32643', w, h, *rasterio.transform.array_bounds(h, w, tr), resolution=30)
dem = np.full((h2, w2), np.nan)
reproject(arr, dem, src_transform=tr, src_crs=crs, dst_transform=t2, dst_crs='EPSG:32643', resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
with rasterio.open(f'{D}/dem_utm.tif', 'w', driver='GTiff', height=h2, width=w2, count=1, dtype='float32', crs='EPSG:32643', transform=t2, nodata=np.nan, compress='lzw') as o:
    o.write(dem.astype('float32'), 1)
print('dem_utm.tif written', dem.shape)
