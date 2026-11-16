"""Check that the numbers in the Person-3 reports match the analysis outputs (analysis_p3/). Every expected string is
formatted from decision.json / arm_summary.csv / per_target.csv and must appear verbatim in the report.
Usage: python tools/check_report_numbers.py <report.md> [<report.md> ...]"""
import csv, json, os, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); A = os.path.join(PKG, 'analysis_p3')
d = json.load(open(os.path.join(A, 'decision.json')))
arms = {r['arm']: r for r in csv.DictReader(open(os.path.join(A, 'arm_summary.csv')))}
per = list(csv.DictReader(open(os.path.join(A, 'per_target.csv'))))
f2 = lambda x: f'{float(x):.2f}'; s2 = lambda x: f'{float(x):+.2f}'
exp = {}
for arm in ('released', 'ie_off', 'ie_on', 'ie_off_bal', 'xpc_bal', 'fix', 'aug', 'fix_aug'):
    r = arms[arm]
    for k in ('acc', 'macro_f1', 'weighted_f1', 'acc_last', 'probe_lin', 'probe_mlp', 'acc_leaky_DIAG'):
        exp[f'{arm} {k}'] = f2(r[k])
for arm in ('fix', 'aug', 'fix_aug'):
    D = d['decisions'][arm]
    exp[f'{arm} verdict'] = D['verdict']
    for k in ('acc', 'macro_f1', 'weighted_f1', 'probe_lin', 'probe_mlp'):
        exp[f'{arm} d{k} CI'] = f"{s2(D[k]['mean'])} [{s2(D[k]['ci_low'])}, {s2(D[k]['ci_high'])}]"
    exp[f'{arm} per-target'] = ', '.join(s2(v) for v in D['acc']['per_target'])
for k, v in d['factorial']['acc'].items():
    exp[f'factorial acc {k}'] = f"{s2(v['mean'])} [{s2(v['ci_low'])}, {s2(v['ci_high'])}]"
import glob, numpy as np
for r in per:   # recomputed from the run files (per_target.csv is rounded to 3 decimals, which can flip the 2nd)
    root = os.path.join(PKG, 'baseline_ie_off_metrics') if r['arm'] == 'ie_off' else os.path.join(PKG, 'code', 'stage3', 'runs_p3', 'ucihar', r['arm'])
    v = [json.load(open(os.path.join(root, f'target{r["target"]}', f'seed{s}', 'metrics.json')))['target_at_selected_epoch']['accuracy']
         for s in d['primary_seeds']]
    exp[f"per-target {r['arm']} t{r['target']}"] = f2(np.mean(v))
bad = 0
for path in sys.argv[1:]:
    text = open(path, encoding='utf-8').read().replace('−', '-')
    miss = [f'{k}: {v}' for k, v in exp.items() if v.replace('−', '-') not in text]
    print(f'{os.path.basename(path)}: {len(exp) - len(miss)}/{len(exp)} expected values found')
    for m in miss: print('   not found:', m)
    bad += len(miss)
