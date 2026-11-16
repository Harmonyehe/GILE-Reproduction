"""Target-clustered paired analysis of the full experiment (read-only).
Seeds of the same target are not independent, so the honest unit is the target: average each paired
difference over seeds within a target, then a t-interval over the 5 targets (df = 4).
Usage: python checks/analyse_full.py [root]   (reads <root>/ucihar/all_runs.csv written by aggregate.py)"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
root = sys.argv[1] if len(sys.argv) > 1 else 'runs_full'
df = pd.read_csv(os.path.join(ROOT, root, 'ucihar', 'all_runs.csv'), dtype={'target': str})
metrics = ['acc', 'macro_f1', 'weighted_f1', 'SEC_last10_weighted_f1', 'last_weighted_f1',
           'probe_person_from_z_lin', 'probe_person_from_z_mlp', 'probe_activity_from_zd_lin', 'probe_activity_from_zd_mlp']
rows = []
for a, b in (('ie_on', 'ie_off'), ('ie_on', 'released'), ('ie_off', 'released')):
    A = df[df.variant == a].set_index(['target', 'seed']); B = df[df.variant == b].set_index(['target', 'seed'])
    j = A.index.intersection(B.index)
    for m in metrics:
        d = (A.loc[j, m] - B.loc[j, m]).astype(float)
        per_t = d.groupby(level='target').mean()
        mu, se = per_t.mean(), per_t.std(ddof=1) / np.sqrt(len(per_t))
        h = stats.t.ppf(0.975, len(per_t) - 1) * se
        rows.append({'comparison': f'{a} - {b}', 'metric': m, 'mean_diff': round(mu, 2),
                     'CI95_low': round(mu - h, 2), 'CI95_high': round(mu + h, 2),
                     'per_target': ' '.join(f'{t}:{v:+.1f}' for t, v in per_t.items()),
                     'targets_positive': int((per_t > 0).sum())})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, root, 'ucihar', 'paired_target_level_CI.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(out.to_string(index=False))
print('\nNoise calibration: released and ie_off are mathematically identical procedures (two Adam steps vs one at the '
      'summed lr); runs are seed-reproducible on T4, so their per-run paired differences (sd) measure how far float-'
      'rounding differences are amplified by training (a floor for "no real effect"):')
for m in ('acc', 'weighted_f1', 'probe_person_from_z_mlp'):
    A = df[df.variant == 'ie_off'].set_index(['target', 'seed'])[m]; B = df[df.variant == 'released'].set_index(['target', 'seed'])[m]
    print(f'   {m}: sd of (ie_off - released) per run = {(A - B).std(ddof=1):.2f}')
print('\nMeans per variant (all 25 runs):')
print(df.groupby('variant')[['acc', 'macro_f1', 'weighted_f1', 'last_weighted_f1', 'SEC_last10_weighted_f1',
                             'DIAG_leaky_weighted_f1', 'probe_person_majority', 'sel_epoch', 'train_seconds']].mean().round(2))
