"""Screen GEE candidates into the inventory, draw pseudo-absences, extract predictors."""
import numpy as np, pandas as pd, geopandas as gpd, rasterio, sys
from rasterio.mask import mask
from shapely.geometry import Point
D, O = sys.argv[1], sys.argv[2]
SEED = 42; BUF = 500.0; SLOPE_INV = 15.0; SLOPE_ABS = 5.0; AMIN, AMAX = 900.0, 20000.0
FACT = ['slope','curvature','elevation','northness','eastness','twi']
bnd = gpd.read_file(f'{D}/boundary/Kodagu.shp').to_crs(32643); poly = bnd.union_all()
c = pd.read_csv(f'{D}/Kodagu_landslide_candidates_v2.csv')
g = gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.longitude, c.latitude), crs=4326).to_crs(32643)
def sample(gdf):
    xy = list(zip(gdf.geometry.x, gdf.geometry.y)); out = {}
    for f in FACT:
        with rasterio.open(f'{D}/f_{f}.tif') as s: out[f] = np.array([v[0] for v in s.sample(xy)])
    return pd.DataFrame(out, index=gdf.index)
g = g.join(sample(g))
log = [('GEE candidates', len(g))]
g = g[g.within(poly)]; log.append(('inside district boundary', len(g)))
g = g[(g.area_m2 >= AMIN) & (g.area_m2 <= AMAX)]; log.append(('900 m2 <= patch area <= 2 ha', len(g)))
g = g[g.slope >= SLOPE_INV]; log.append(('centroid slope >= 15 deg', len(g)))
with rasterio.open(f'{D}/f_slope.tif') as s:
    rc = [s.index(x, y) for x, y in zip(g.geometry.x, g.geometry.y)]
g['cell'] = [f'{r}_{cc}' for r, cc in rc]
g = g.drop_duplicates('cell'); log.append(('one point per 30 m cell', len(g)))
pres = g.copy(); pres['presence'] = 1
n = len(pres)
# pseudo-absences: random in district, slope >= 5 deg, > 500 m from every presence, 1:1
rng = np.random.default_rng(SEED); minx, miny, maxx, maxy = poly.bounds
buf = pres.geometry.buffer(BUF).union_all(); pts = []
with rasterio.open(f'{D}/f_slope.tif') as s:
    while len(pts) < n:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy); p = Point(x, y)
        if not poly.contains(p) or buf.contains(p): continue
        v = next(s.sample([(x, y)]))[0]
        if np.isnan(v) or v < SLOPE_ABS: continue
        pts.append((x, y))
ab = gpd.GeoDataFrame({'presence': np.zeros(len(pts), int)}, geometry=[Point(p) for p in pts], crs=32643)
ab = ab.join(sample(ab))
df = pd.concat([pres[['presence','geometry']+FACT+['area_m2']], ab[['presence','geometry']+FACT]], ignore_index=True)
df['x'] = df.geometry.x; df['y'] = df.geometry.y
w = gpd.GeoSeries(df.geometry, crs=32643).to_crs(4326); df['lon'] = w.x; df['lat'] = w.y
df = df.drop(columns='geometry').dropna(subset=FACT)
df.to_csv(f'{O}/model_dataset.csv', index=False)
pres_out = df[df.presence==1][['lon','lat','x','y','area_m2']]; pres_out.to_csv(f'{O}/inventory_points.csv', index=False)
df[df.presence==0][['lon','lat','x','y']].to_csv(f'{O}/pseudo_absences.csv', index=False)
pd.DataFrame(log, columns=['step','n']).to_csv(f'{O}/inventory_screening_log.csv', index=False)
for s_, k in log: print(f'{s_:35s} {k}')
print('final presences', int(df.presence.sum()), 'absences', int((df.presence==0).sum()))
with rasterio.open(f'{D}/f_slope.tif') as s:
    m, _ = mask(s, bnd.geometry, crop=False, nodata=np.nan); cells = np.isfinite(m[0]).sum()
print('district area from 30 m grid (km2)', round(cells*900/1e6, 1))
