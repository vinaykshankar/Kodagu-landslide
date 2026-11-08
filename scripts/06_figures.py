"""Publication figures from the rebuilt pipeline outputs."""
import numpy as np, pandas as pd, geopandas as gpd, rasterio, json, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from rasterio.mask import mask
D, O, F = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
CAT = ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; INK = '#0b0b0b'; INK2 = '#52514e'
bnd = gpd.read_file(f'{D}/boundary/Kodagu.shp').to_crs(32643)
with rasterio.open(f'{D}/f_elevation.tif') as s:
    m, tr = mask(s, bnd.geometry, crop=True, nodata=np.nan); dem = m[0]
ext = [tr.c, tr.c + tr.a*dem.shape[1], tr.f + tr.e*dem.shape[0], tr.f]
hs = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)), vert_exag=2, dx=30, dy=30)
hs = np.where(np.isnan(dem), np.nan, hs)
df = pd.read_csv(f'{O}/model_dataset.csv'); fo = pd.read_csv(f'{O}/spatial_folds_4km.csv')
def frame(ax):
    bnd.boundary.plot(ax=ax, color=INK, lw=0.7); ax.set_aspect('equal')
    ax.set_xlim(ext[0]-1500, ext[1]+1500); ax.set_ylim(ext[2]-1500, ext[3]+1500)
    xt = ax.get_xticks(); yt = ax.get_yticks(); ax.set_xticks(xt[1:-1]); ax.set_yticks(yt[1:-1])
    ax.set_xticklabels([f'{v/1000:.0f}' for v in xt[1:-1]]); ax.set_yticklabels([f'{v/1000:.0f}' for v in yt[1:-1]])
    ax.set_xlabel('Easting (km, UTM 43N)'); ax.set_ylabel('Northing (km, UTM 43N)')
    x0, y0 = ext[0]+2000, ext[2]+3000; ax.plot([x0, x0+10000], [y0, y0], color=INK, lw=2); ax.text(x0+5000, y0+1200, '10 km', ha='center', fontsize=8, color=INK)
    ax.annotate('N', xy=(ext[1]-4000, ext[2]+12000), xytext=(ext[1]-4000, ext[2]+5000), ha='center', fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2))
# ---------- Figure 1: (a) study area + inventory, (b) sampling and spatial folds ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 6.6), constrained_layout=True)
ax = axs[0]
ax.imshow(hs, extent=ext, cmap='gray', vmin=0, vmax=1, alpha=0.55)
im = ax.imshow(dem, extent=ext, cmap='Greens_r', alpha=0.45, vmin=100, vmax=1700)
p = df[df.presence==1]; ax.scatter(p.x, p.y, s=10, c=CAT[1], edgecolor='white', linewidth=0.4, zorder=3, label=f'2018 landslide scars (n = {len(p)})')
frame(ax); ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.01); cb.set_label('Elevation (m)')
ax.set_title('(a) Study area and 2018 landslide inventory', loc='left', fontsize=10, color=INK)
ax = axs[1]
ax.imshow(hs, extent=ext, cmap='gray', vmin=0, vmax=1, alpha=0.35)
B = 4000; x0g, y0g = df.x.min()-1, df.y.min()-1
for gx in np.arange(x0g, ext[1]+B, B): ax.axvline(gx, color='#c3c2b7', lw=0.35, zorder=1)
for gy in np.arange(y0g, ext[3]+B, B): ax.axhline(gy, color='#c3c2b7', lw=0.35, zorder=1)
for k in range(5):
    q = fo[fo.fold==k]
    ax.scatter(q[q.presence==1].x, q[q.presence==1].y, s=11, marker='^', c=CAT[k], edgecolor='white', linewidth=0.3, zorder=3)
    ax.scatter(q[q.presence==0].x, q[q.presence==0].y, s=10, marker='o', facecolor='none', edgecolor=CAT[k], linewidth=0.8, zorder=3)
frame(ax)
h = [Patch(color=CAT[k], label=f'Fold {k+1}') for k in range(5)] + [Line2D([],[],marker='^',ls='',color=INK2,label='Scar (presence)'), Line2D([],[],marker='o',ls='',mfc='none',color=INK2,label='Pseudo-absence')]
ax.legend(handles=h, loc='upper right', fontsize=7.5, framealpha=0.9)
ax.set_title('(b) Sampling design and 4 km spatial-block folds', loc='left', fontsize=10, color=INK)
fig.savefig(f'{F}/Figure_1.png', dpi=300); plt.close()
# ---------- Figure 2: workflow ----------
steps = [('Data', 'SRTM 1″ DEM (30 m) · Sentinel-2 Level-1C with Cloud Score+ (Google Earth Engine) · district boundary'),
         ('Inventory', 'Sentinel-2 NDVI change in GEE: 5,247 patches → in district 1,834 → ≤ 2 ha 1,604 → slope ≥ 15°: 325 scars'),
         ('Sampling', '325 pseudo-absences: random, slope ≥ 5°, > 500 m from any scar (seed 42)'),
         ('Predictors', 'Slope · curvature · elevation · northness · eastness · TWI (all VIF < 2)'),
         ('Models', 'IV · Logistic Regression · Random Forest, each with and without slope'),
         ('Validation', 'Random 70/30 hold-out · random 5-fold CV · 4 km spatial-block CV (+1 km buffer; 10 km check)'),
         ('Outputs', 'Relative susceptibility map (5 equal-interval classes) · concentration ratio · SHAP')]
fig, ax = plt.subplots(figsize=(10, 7.2)); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, len(steps))
for i, (t, d) in enumerate(steps):
    yb = len(steps)-1-i
    ax.add_patch(FancyBboxPatch((0.03, yb+0.12), 0.94, 0.72, boxstyle='round,pad=0.0,rounding_size=0.03', fc='#eef3fb', ec='#86b6ef', lw=1))
    ax.text(0.06, yb+0.6, t, fontsize=11, fontweight='bold', color='#104281', va='center')
    ax.text(0.06, yb+0.32, d, fontsize=8.6, color=INK, va='center')
    if i < len(steps)-1: ax.annotate('', xy=(0.5, yb+0.02), xytext=(0.5, yb+0.13), arrowprops=dict(arrowstyle='-|>', color='#52514e', lw=1.2))
fig.savefig(f'{F}/Figure_2.png', dpi=300, bbox_inches='tight'); plt.close()
# ---------- Figure 3: susceptibility map ----------
st = pd.read_csv(f'{O}/susceptibility_class_statistics.csv'); st = st[(st.model=='LR with slope') & (st.fit=='all samples')]
with rasterio.open(f'{O}/lsi_lr_classes.tif') as s:
    m, tr2 = mask(s, bnd.geometry, crop=True, nodata=0); c = m[0].astype(float); c[c==0] = np.nan
ramp = ['#fde6cf','#fbb77f','#f4823e','#d4541a','#8f3009']
fig, ax = plt.subplots(figsize=(7.2, 7.4), constrained_layout=True)
ax.imshow(hs, extent=ext, cmap='gray', vmin=0, vmax=1, alpha=0.9)
ax.imshow(c, extent=ext, cmap=ListedColormap(ramp), vmin=0.5, vmax=5.5, alpha=0.78, interpolation='nearest')
frame(ax)
h = [Patch(color=ramp[i], label=f"{r.cls}  ({r.p_from:.1f}–{r.p_to:.1f}; {r.area_pct:.1f}%)") for i, r in enumerate(st.itertuples())]
ax.legend(handles=h, title='Class (probability range; % of district)', loc='upper right', fontsize=7.5, title_fontsize=8, framealpha=0.92)
fig.savefig(f'{F}/Figure_3.png', dpi=300); plt.close()
print('figures written')
