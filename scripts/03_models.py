"""IV, LR and RF (with / without slope) under random hold-out, random 5-fold and spatial-block CV."""
import numpy as np, pandas as pd, sys, json, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, cohen_kappa_score, brier_score_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')
O = sys.argv[1]; SEED = 42
df = pd.read_csv(f'{O}/model_dataset.csv'); y = df.presence.values
CAND = ['slope','curvature','elevation','northness','eastness','twi']
# ---- VIF screening ----
def vif(cols):
    X = StandardScaler().fit_transform(df[cols]); X = np.c_[np.ones(len(X)), X]
    return {c: variance_inflation_factor(X, i+1) for i, c in enumerate(cols)}
vif_all = vif(CAND)
FINAL = ['slope','curvature','elevation','northness','eastness','twi']   # all VIF < 2: none excluded
vif_final = vif(FINAL)
SETS = {'with slope': FINAL, 'without slope': [c for c in FINAL if c != 'slope']}
# ---- IV model ----
def aspect_class(d):
    a = (np.degrees(np.arctan2(d.eastness, d.northness)) + 360) % 360
    k = ((a + 22.5) // 45).astype(int) % 8
    return np.where((d.northness == 0) & (d.eastness == 0), 8, k)
class IV:
    def __init__(s, cont): s.cont = cont
    def classes(s, d):
        out = {f: np.digitize(d[f], s.edges[f]) for f in s.cont}
        out['aspect'] = aspect_class(d); return out
    def fit(s, d, yy):
        s.edges = {f: np.quantile(d[f], [0.2,0.4,0.6,0.8]) for f in s.cont}
        cl = s.classes(d); Ns, Nn = yy.sum(), (1-yy).sum(); s.table = {}; s.rows = []
        for f, k in cl.items():
            s.table[f] = {}
            for c in np.unique(k):
                ns = yy[k==c].sum(); nn = (1-yy)[k==c].sum()
                corr = ns == 0 or nn == 0
                ns_, nn_ = max(ns, 0.5), max(nn, 0.5)
                v = np.log((ns_/Ns)/(nn_/Nn)); s.table[f][c] = v
                s.rows.append(dict(factor=f, cls=int(c), n_landslide=int(ns), n_nonlandslide=int(nn), IV=v, continuity_correction=corr))
        return s
    def score(s, d):
        cl = s.classes(d); return sum(np.array([s.table[f].get(c, 0.0) for c in k]) for f, k in cl.items())
IV_CONT = ['slope','curvature','elevation','twi']
# ---- learners ----
def make(model, cols):
    if model == 'LR': return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1e6))
    return RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=5, oob_score=True, random_state=SEED, n_jobs=-1)
MODELS = [('IV','with slope'),('IV','without slope'),('LR','with slope'),('LR','without slope'),('RF','with slope'),('RF','without slope')]
def fit_predict(model, setname, tr, te):
    cols = SETS[setname]
    if model == 'IV':
        cont = [c for c in IV_CONT if c in cols]; m = IV(cont).fit(tr, tr.presence.values)
        return m.score(te), m.score(tr), m
    m = make(model, cols).fit(tr[cols], tr.presence)
    ptr = m.oob_decision_function_[:,1] if model == 'RF' else m.predict_proba(tr[cols])[:,1]
    return m.predict_proba(te[cols])[:,1], ptr, m
def youden(yy, p):
    f, t, th = roc_curve(yy, p); return th[np.argmax(t - f)]
def thr_metrics(yy, p, th):
    pred = (p >= th).astype(int); tn, fp, fn, tp = confusion_matrix(yy, pred).ravel()
    sens = tp/(tp+fn); spec = tn/(tn+fp); prec = tp/(tp+fp)
    return dict(threshold=th, TP=tp, FN=fn, FP=fp, TN=tn, sensitivity=sens, specificity=spec, precision=prec,
                F1=2*prec*sens/(prec+sens), accuracy=(tp+tn)/len(yy), kappa=cohen_kappa_score(yy, pred))
res = {}
# ---- (1) random 70/30 hold-out ----
tr, te = train_test_split(df, test_size=0.3, stratify=df.presence, random_state=SEED)
hold = []
for m, s in MODELS:
    pte, ptr, fitted = fit_predict(m, s, tr, te); th = youden(tr.presence.values, ptr)
    r = dict(model=m, predictors=s, n_train=len(tr), n_test=len(te), AUC=roc_auc_score(te.presence, pte))
    r.update(thr_metrics(te.presence.values, pte, th))
    if m != 'IV': r['Brier'] = brier_score_loss(te.presence, pte)
    hold.append(r)
    if (m, s) == ('IV','with slope'): pd.DataFrame(fitted.rows).to_csv(f'{O}/iv_class_table_training.csv', index=False)
pd.DataFrame(hold).to_csv(f'{O}/holdout_metrics.csv', index=False)
tr[['lon','lat','presence']].assign(split='train').pipe(lambda a: pd.concat([a, te[['lon','lat','presence']].assign(split='test')])).to_csv(f'{O}/holdout_split.csv', index=False)
# ---- (2) random stratified 5-fold ----
skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rows = []
for k, (a, b) in enumerate(skf.split(df, y)):
    for m, s in MODELS:
        p, _, _ = fit_predict(m, s, df.iloc[a], df.iloc[b]); rows.append(dict(design='random 5-fold', fold=k, model=m, predictors=s, AUC=roc_auc_score(y[b], p)))
cv = pd.DataFrame(rows)
# ---- (3) spatial block CV ----
from scipy.spatial import cKDTree
x0, y0 = df.x.min() - 1, df.y.min() - 1   # grid origin: SW corner of sample extent
def block_folds(B):
    bid = (((df.x - x0)//B).astype(int)).astype(str) + '_' + (((df.y - y0)//B).astype(int)).astype(str)
    blocks = bid.unique(); rng = np.random.default_rng(SEED); rng.shuffle(blocks)
    cnt = bid.value_counts(); load = np.zeros(5); fmap = {}
    for bl in sorted(blocks, key=lambda q: -cnt[q]):   # largest first, to the lightest fold
        f = int(np.argmin(load)); fmap[bl] = f; load[f] += cnt[bl]
    return bid, bid.map(fmap).values
design_rows, fold_rows = [], []
for B, gap in [(4000, 0), (4000, 1000), (10000, 0)]:
    name = f'spatial {B//1000} km' + (f' + {gap//1000} km buffer' if gap else '')
    bid, fold = block_folds(B)
    for k in range(5):
        te_i = np.where(fold == k)[0]; tr_i = np.where(fold != k)[0]
        if gap:
            d, _ = cKDTree(df.iloc[te_i][['x','y']].values).query(df.iloc[tr_i][['x','y']].values)
            tr_i = tr_i[d > gap]
        dmin, _ = cKDTree(df.iloc[tr_i][['x','y']].values).query(df.iloc[te_i][['x','y']].values)
        fold_rows.append(dict(design=name, fold=k, n_blocks=len(set(bid.iloc[te_i])), train_presence=int(y[tr_i].sum()), train_absence=int((1-y[tr_i]).sum()),
                              test_presence=int(y[te_i].sum()), test_absence=int((1-y[te_i]).sum()), median_dist_test_to_train_m=float(np.median(dmin))))
        for m, s in MODELS:
            p, _, _ = fit_predict(m, s, df.iloc[tr_i], df.iloc[te_i]); rows.append(dict(design=name, fold=k, model=m, predictors=s, AUC=roc_auc_score(y[te_i], p)))
    if (B, gap) == (4000, 0):
        df.assign(block=bid.values, fold=fold)[['lon','lat','x','y','presence','block','fold']].to_csv(f'{O}/spatial_folds_4km.csv', index=False)
    design_rows.append(dict(design=name, n_blocks_total=int(bid.nunique())))
allcv = pd.DataFrame(rows); allcv.to_csv(f'{O}/cv_fold_auc.csv', index=False)
pd.DataFrame(fold_rows).to_csv(f'{O}/spatial_fold_composition.csv', index=False)
summ = allcv.groupby(['design','model','predictors']).AUC.agg(['mean', lambda a: a.std(ddof=1)]).rename(columns={'<lambda_0>':'sd'}).reset_index()
summ.to_csv(f'{O}/cv_summary.csv', index=False)
json.dump(dict(vif_candidates=vif_all, vif_final=vif_final, designs=design_rows), open(f'{O}/vif_and_designs.json','w'), indent=1)
pd.set_option('display.width', 200)
print('VIF candidates', {k: round(v,2) for k,v in vif_all.items()}); print('VIF final', {k: round(v,2) for k,v in vif_final.items()})
print(pd.DataFrame(hold)[['model','predictors','AUC','threshold','sensitivity','specificity','accuracy','kappa']].round(3).to_string(index=False))
print(summ.pivot_table(index=['model','predictors'], columns='design', values='mean').round(3).to_string())
print(summ.pivot_table(index=['model','predictors'], columns='design', values='sd').round(3).to_string())
print(pd.DataFrame(fold_rows).to_string(index=False)); print(design_rows)
