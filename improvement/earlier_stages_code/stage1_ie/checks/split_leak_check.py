"""Quantify train/val window-overlap leakage on real UCI HAR source persons:
random stratified 80/20 split (as in a window-level split) vs our contiguous+purged split.
Needs data/ucihar/ucihar_domain_<k>_wd.data (built by run_experiment.py). Writes nothing."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..')); os.chdir(ROOT); sys.path.insert(0, ROOT)
from sklearn.model_selection import train_test_split
import data_preprocess_ucihar as prep
H = 64
def succ_pred(X):
    first = {}
    for j in range(len(X)): first.setdefault(X[j, :H].tobytes(), []).append(j)
    s = np.full(len(X), -1)
    for i in range(len(X)):
        js = [j for j in first.get(X[i, H:].tobytes(), []) if j != i]
        if js: s[i] = js[0]
    p = np.full(len(X), -1)
    for i, j in enumerate(s):
        if j >= 0: p[j] = i
    return s, p
def leaky_val_fraction(val, train, s, p):
    tr = set(train.tolist())
    return np.mean([(s[v] in tr) or (p[v] in tr) for v in val])
print(f'{"person":>6} | {"random split: val windows with an overlapping train window":>58} | contiguous+purged')
for pid in range(5):
    X, y, _ = prep.load_domain_data(str(pid)); X = np.asarray(X); y = np.asarray(y).astype(int)
    s, p = succ_pred(X); idx = np.arange(len(y))
    fr = [leaky_val_fraction(*reversed(train_test_split(idx, test_size=0.2, stratify=y, random_state=r)), s, p) for r in range(5)]
    val = np.concatenate([np.where(y == c)[0][-max(1, round(0.2 * (y == c).sum())):] for c in np.unique(y)])
    inval = np.zeros(len(y), bool); inval[val] = True
    nb = {int(n) for v in val for n in (s[v], p[v]) if n >= 0 and not inval[n]}
    train = np.array([i for i in idx if not inval[i] and i not in nb])
    print(f'{pid:>6} | {100*np.mean(fr):56.1f}% | {100*leaky_val_fraction(val, train, s, p):.1f}%')
