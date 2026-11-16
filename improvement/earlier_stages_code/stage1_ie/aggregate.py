"""Collect runs/<dataset>/<variant>/target*/seed*/metrics.json (+ history.csv) into tables.
Usage: python aggregate.py [--root runs] [--dataset ucihar] [--expect_targets 0,1,2,3,4] [--expect_seeds 0,1,2,3,4]

PRIMARY metric  : target performance at the epoch selected on SOURCE-person validation (metrics.json).
SECONDARY (descriptive stability only, approved 15 Nov): mean target metrics over the final 10 epochs,
                  read from history.csv. Never used for model selection, tuning, or choosing a variant.
DIAGNOSTIC only : best-target-epoch ("leaky") numbers, for comparison with the paper's protocol.
Writes into <root>/<dataset>/: all_runs.csv, summary.csv, per_target.csv, paired_differences.csv, completeness.txt"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='runs'); ap.add_argument('--dataset', default='ucihar')
ap.add_argument('--expect_targets', default='0,1,2,3,4'); ap.add_argument('--expect_seeds', default='0,1,2,3,4')
ap.add_argument('--variants', default='released,ie_off,ie_on')
a = ap.parse_args()
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.root, a.dataset)
VARS = a.variants.split(',')
rows = []
for f in sorted(glob.glob(os.path.join(base, '*', 'target*', 'seed*', 'metrics.json'))):
    m = json.load(open(f)); s, l, k = m['target_at_selected_epoch'], m['target_at_last_epoch'], m['DIAGNOSTIC_ONLY_leaky_best_target_epoch']
    h = pd.read_csv(os.path.join(os.path.dirname(f), 'history.csv'))
    pr = m.get('probes_selected_epoch') or {}
    pz, pzd = pr.get('person_from_activity_code_z', {}), pr.get('activity_from_person_code_zd', {})
    rows.append({'variant': m['variant'], 'target': str(m['target']), 'seed': int(m['seed']),
                 'sel_epoch': m['selection']['selected_epoch'], 'src_val_acc': m['selection']['source_val_acc'],
                 'acc': s['accuracy'], 'macro_f1': s['macro_f1'], 'weighted_f1': s['weighted_f1'],
                 'SEC_last10_acc': h['target_acc_DIAG'].tail(10).mean(),
                 'SEC_last10_macro_f1': h['target_macro_f1_DIAG'].tail(10).mean(),
                 'SEC_last10_weighted_f1': h['target_weighted_f1_DIAG'].tail(10).mean(),
                 'last_acc': l['accuracy'], 'last_weighted_f1': l['weighted_f1'],
                 'DIAG_leaky_acc': k['accuracy'], 'DIAG_leaky_weighted_f1': k['weighted_f1'],
                 'probe_person_from_z_lin': pz.get('linear'), 'probe_person_from_z_mlp': pz.get('mlp'),
                 'probe_person_majority': pz.get('majority_baseline'),
                 'probe_activity_from_zd_lin': pzd.get('linear'), 'probe_activity_from_zd_mlp': pzd.get('mlp'),
                 'probe_leak_shared': (pr.get('leak_check') or {}).get('identical_windows_train_test'),
                 'nonfinite_losses': int((~np.isfinite(h['train_loss_per_sample'])).sum()),
                 'train_seconds': m['train_seconds']})
if not rows: raise SystemExit(f'no metrics.json under {base}')
df = pd.DataFrame(rows).sort_values(['variant', 'target', 'seed']); df.to_csv(os.path.join(base, 'all_runs.csv'), index=False)

# completeness
exp = [(v, t, int(s)) for v in VARS for t in a.expect_targets.split(',') for s in a.expect_seeds.split(',')]
have = set(zip(df.variant, df.target, df.seed)); missing = [e for e in exp if e not in have]
dup = df.duplicated(['variant', 'target', 'seed']).sum()
msg = (f'expected {len(exp)} runs, found {len(df)}, missing {len(missing)}, duplicates {dup}, '
       f'non-finite-loss runs {(df.nonfinite_losses > 0).sum()}\nmissing: {missing}')
open(os.path.join(base, 'completeness.txt'), 'w').write(msg + '\n'); print(msg)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
num = [c for c in df.columns if c not in ('variant', 'target', 'seed')]
# per variant: mean over targets within each seed, then mean +- std over seeds
summ = df.groupby(['variant', 'seed'])[num].mean().groupby('variant').agg(['mean', 'std']).round(2)
summ.to_csv(os.path.join(base, 'summary.csv'))
show = ('acc', 'macro_f1', 'weighted_f1', 'SEC_last10_macro_f1', 'SEC_last10_weighted_f1', 'DIAG_leaky_weighted_f1',
        'probe_person_from_z_lin', 'probe_person_from_z_mlp', 'probe_activity_from_zd_lin', 'sel_epoch')
print('\n=== PRIMARY (selected epoch) + SECONDARY + DIAG: mean over 5 targets, then mean/std over seeds ===')
print(summ[[c for c in summ.columns if c[0] in show]])
pt = df.groupby(['target', 'variant'])[['acc', 'macro_f1', 'weighted_f1', 'probe_person_from_z_lin']].agg(['mean', 'std']).round(2)
pt.to_csv(os.path.join(base, 'per_target.csv'))
print('\n=== per target (mean/std over seeds) ===\n', pt)

pairs = []
for va, vb in (('ie_on', 'ie_off'), ('ie_on', 'released'), ('ie_off', 'released')):
    A = df[df.variant == va].set_index(['target', 'seed']); B = df[df.variant == vb].set_index(['target', 'seed'])
    j = A.index.intersection(B.index)
    if len(j) == 0: continue
    for metric in ('acc', 'macro_f1', 'weighted_f1', 'SEC_last10_weighted_f1', 'probe_person_from_z_lin',
                   'probe_person_from_z_mlp', 'probe_activity_from_zd_lin'):
        dlt = (A.loc[j, metric] - B.loc[j, metric]).astype(float).values
        try: p = wilcoxon(dlt).pvalue if len(dlt) >= 6 and np.any(dlt != 0) else np.nan
        except ValueError: p = np.nan
        pairs.append({'comparison': f'{va} - {vb}', 'metric': metric, 'n_pairs': len(j),
                      'mean_diff': round(dlt.mean(), 2), 'std_diff': round(dlt.std(ddof=1), 2) if len(dlt) > 1 else np.nan,
                      'n_positive': int((dlt > 0).sum()), 'wilcoxon_p_descriptive': round(p, 4) if p == p else np.nan})
if pairs:
    pdf = pd.DataFrame(pairs); pdf.to_csv(os.path.join(base, 'paired_differences.csv'), index=False)
    print('\n=== paired differences over (target, seed); Wilcoxon p is descriptive (pairs share targets) ===\n',
          pdf.to_string(index=False))
