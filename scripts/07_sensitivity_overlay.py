"""Pseudo-absence sensitivity: exclusion buffer 250/500/750 m x 5 independent draws."""
import numpy as np, pandas as pd, geopandas as gpd, rasterio, sys, warnings
from shapely.geometry import Point
from rasterio.mask import mask
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
D, O = sys.argv[1], sys.argv[2]
F = ['slope','curvature','elevation','northness','eastness','twi']; NS = F[1:]
bnd = gpd.read_file(f'{D}/boundary/Kodagu.shp').to_crs(32643); poly = bnd.union_all()
df = pd.read_csv(f'{O}/model_dataset.csv'); pres = df[df.presence==1].copy()
src = {f: rasterio.open(f'{D}/f_{f}.tif') for f in F}
def draw(buf_m, seed):
    rng = np.random.default_rng(seed); minx, miny, maxx, maxy = poly.bounds
    buf = gpd.GeoSeries(gpd.points_from_xy(pres.x, pres.y), crs=32643).buffer(buf_m).union_all(); pts = []
    while len(pts) < len(pres):
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy); p = Point(x, y)
        if not poly.contains(p) or buf.contains(p): continue
        v = next(src['slope'].sample([(x, y)]))[0]
        if np.isnan(v) or v < 5: continue
        pts.append((x, y))
    a = pd.DataFrame(pts, columns=['x','y'])
    for f in F: a[f] = [v[0] for v in src[f].sample(pts)]
    a['presence'] = 0; return pd.concat([pres[['x','y','presence']+F], a], ignore_index=True).dropna()
rows = []
for b in (250, 500, 750):
    for seed in range(5):
        d = draw(b, 100+seed); y = d.presence.values
        for name, cols, mk in [('LR with slope', F, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1e6))),
                               ('RF with slope', F, lambda: RandomForestClassifier(500, max_features='sqrt', min_samples_leaf=5, random_state=42, n_jobs=-1)),
                               ('RF without slope', NS, lambda: RandomForestClassifier(500, max_features='sqrt', min_samples_leaf=5, random_state=42, n_jobs=-1))]:
            au = [roc_auc_score(y[t], mk().fit(d.iloc[s][cols], y[s]).predict_proba(d.iloc[t][cols])[:,1]) for s, t in StratifiedKFold(5, shuffle=True, random_state=42).split(d, y)]
            rows.append(dict(buffer_m=b, seed=100+seed, model=name, AUC=np.mean(au)))
sens = pd.DataFrame(rows); sens.to_csv(f'{O}/pseudoabsence_sensitivity.csv', index=False)
print(sens.groupby(['model','buffer_m']).AUC.agg(['mean','min','max']).round(3))
