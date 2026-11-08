"""LR coefficients, LSI maps with equal-interval classes, concentration statistics, SHAP."""
import numpy as np, pandas as pd, geopandas as gpd, rasterio, sys, json, warnings
from rasterio.mask import mask
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import shap
warnings.filterwarnings('ignore')
D, O = sys.argv[1], sys.argv[2]; SEED = 42
F = ['slope','curvature','elevation','northness','eastness','twi']; NS = [c for c in F if c != 'slope']
df = pd.read_csv(f'{O}/model_dataset.csv')
# ---- LR coefficients (all 978 samples, standardised predictors) ----
sc = StandardScaler().fit(df[F]); X = sm.add_constant(pd.DataFrame(sc.transform(df[F]), columns=F))
lr = sm.Logit(df.presence, X).fit(disp=0); ci = lr.conf_int()
coef = pd.DataFrame(dict(beta=lr.params, SE=lr.bse, OR=np.exp(lr.params), OR_lo=np.exp(ci[0]), OR_hi=np.exp(ci[1]), p=lr.pvalues))
coef.to_csv(f'{O}/lr_coefficients.csv'); print(coef.round(3).to_string())
# ---- raster stack inside the district ----
bnd = gpd.read_file(f'{D}/boundary/Kodagu.shp').to_crs(32643)
st = []
for f in F:
    with rasterio.open(f'{D}/f_{f}.tif') as s:
        m, _ = mask(s, bnd.geometry, crop=False, nodata=np.nan); st.append(m[0].ravel()); prof = s.profile; shp_ = m[0].shape
st = np.column_stack(st); valid = np.isfinite(st).all(1); Xg = pd.DataFrame(st[valid], columns=F)
ncell = valid.sum(); print('valid district cells', ncell, 'area km2', round(ncell*900/1e6,1))
BREAKS = [0, .2, .4, .6, .8, 1.0001]; LAB = ['Very Low','Low','Moderate','High','Very High']
tr, te = train_test_split(df, test_size=0.3, stratify=df.presence, random_state=SEED)
def lrfit(d, cols): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1e6)).fit(d[cols], d.presence)
def rffit(d, cols): return RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5, random_state=SEED, n_jobs=-1).fit(d[cols], d.presence)
out = {}; rows = []
for name, fitter, cols in [('LR with slope', lrfit, F), ('RF without slope', rffit, NS)]:
    for basis, data, pts in [('all samples', df, df[df.presence==1]), ('training subset (held-out scars)', tr, te[te.presence==1])]:
        m = fitter(data, cols); p = m.predict_proba(Xg[cols])[:,1]
        cls = np.digitize(p, BREAKS) - 1; area = np.bincount(cls, minlength=5)*900/1e6
        pc = np.digitize(m.predict_proba(pts[cols])[:,1], BREAKS) - 1; sc_ = np.bincount(pc, minlength=5)
        hv_area = area[3:].sum()/area.sum(); hv_scar = sc_[3:].sum()/sc_.sum()
        for i in range(5): rows.append(dict(model=name, fit=basis, cls=LAB[i], p_from=BREAKS[i], p_to=min(BREAKS[i+1],1), area_km2=area[i], area_pct=100*area[i]/area.sum(), scars=int(sc_[i]), scars_pct=100*sc_[i]/sc_.sum()))
        out[f'{name} | {basis}'] = dict(HVH_area_km2=float(area[3:].sum()), HVH_area_pct=100*hv_area, scars_in_HVH=int(sc_[3:].sum()), scars_total=int(sc_.sum()), scar_pct=100*hv_scar, concentration_ratio=hv_scar/hv_area)
        if name == 'LR with slope' and basis == 'all samples':
            grid = np.full(valid.shape, np.nan, 'float32'); grid[valid] = p
            pr = prof.copy(); pr.update(dtype='float32', nodata=np.nan, compress='lzw')
            with rasterio.open(f'{O}/lsi_lr_probability.tif','w',**pr) as o: o.write(grid.reshape(shp_),1)
            cg = np.full(valid.shape, 0, 'uint8'); cg[valid] = cls+1; pr.update(dtype='uint8', nodata=0)
            with rasterio.open(f'{O}/lsi_lr_classes.tif','w',**pr) as o: o.write(cg.reshape(shp_),1)
pd.DataFrame(rows).to_csv(f'{O}/susceptibility_class_statistics.csv', index=False)
json.dump(out, open(f'{O}/concentration_statistics.json','w'), indent=1)
for k, v in out.items(): print(k, {a: round(b,3) if isinstance(b,float) else b for a,b in v.items()})
# ---- SHAP on held-out test set (probability units, class = landslide) ----
sh = []
for name, cols in [('RF with slope', F), ('RF without slope', NS)]:
    m = rffit(tr, cols); sv = shap.TreeExplainer(m).shap_values(te[cols])
    sv = sv[1] if isinstance(sv, list) else (sv[:,:,1] if sv.ndim == 3 else sv)
    pd.DataFrame(sv, columns=cols).to_csv(f'{O}/shap_values_{name.replace(" ","_")}.csv', index=False)
    for c, v in zip(cols, np.abs(sv).mean(0)): sh.append(dict(model=name, predictor=c, mean_abs_shap=v))
shd = pd.DataFrame(sh).sort_values(['model','mean_abs_shap'], ascending=[True, False]); shd.to_csv(f'{O}/shap_importance.csv', index=False)
print(shd.round(4).to_string(index=False))
