"""Stage-3 analysis exactly as predeclared in PREDECLARED_PLAN.md section 4.
Usage: python tools/analyse_p3.py <runs_root (…/ucihar)> --out analysis_p3
Control = baseline_ie_off_metrics/ (stage-1 ie_off, read-only copies). Primary seeds = seeds complete for ALL arms x targets.
Also tabulates every earlier arm (released / ie_on from runs_full, ie_off_bal / xpc_bal from the stage-2 T4 runs) when present."""
import argparse, csv, glob, json, math, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); IMP = os.path.dirname(PKG)
p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--out', default=os.path.join(PKG, 'analysis_p3'))
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
T = ['0', '1', '2', '3', '4']; NEW = ['fix', 'aug', 'fix_aug']


def load(root, arm):
    out = {}
    for f in glob.glob(os.path.join(root, arm, 'target*', 'seed*', 'metrics.json')):
        m = json.load(open(f)); out[(str(m['target']), int(m['seed']))] = m
    return out


R = {'ie_off': {}}
for f in glob.glob(os.path.join(PKG, 'baseline_ie_off_metrics', 'target*', 'seed*', 'metrics.json')):
    m = json.load(open(f)); R['ie_off'][(str(m['target']), int(m['seed']))] = m
for arm in NEW: R[arm] = load(a.root, arm)
REF = {'released': os.path.join(IMP, 'runs_full', 'ucihar'), 'ie_on': os.path.join(IMP, 'runs_full', 'ucihar'),
       'ie_off_bal': os.path.join(IMP, 'XPCB_TRANSFER_PACKAGE', 'code', 'stage2_xpc', 'runs_xpc_t4', 'ucihar'),
       'xpc_bal': os.path.join(IMP, 'XPCB_TRANSFER_PACKAGE', 'code', 'stage2_xpc', 'runs_xpc_t4', 'ucihar')}
for arm, root in REF.items():                  # fallback: metrics-only copies shipped inside the package (GitHub copy)
    if not os.path.isdir(os.path.join(root, arm)): root = os.path.join(PKG, 'reference_metrics_earlier_stages')
    R[arm] = load(root, arm)

seeds_all = sorted({s for (t, s) in R['ie_off']})
primary_seeds = [s for s in seeds_all if all((t, s) in R[arm] for arm in ['ie_off'] + NEW for t in T)]
partial = {arm: sorted(k for k in R[arm] if k[1] not in primary_seeds) for arm in NEW}
lines = [f'PRIMARY SEEDS (complete for all arms and targets): {primary_seeds}',
         f'runs found: ' + ', '.join(f'{k}={len(v)}' for k, v in R.items()),
         f'runs outside the primary seeds (reported, not in the primary test): {partial}']

M = {'acc': lambda m: m['target_at_selected_epoch']['accuracy'], 'macro_f1': lambda m: m['target_at_selected_epoch']['macro_f1'],
     'weighted_f1': lambda m: m['target_at_selected_epoch']['weighted_f1'],
     'acc_last': lambda m: m['target_at_last_epoch']['accuracy'],
     'acc_leaky_DIAG': lambda m: m['DIAGNOSTIC_ONLY_leaky_best_target_epoch']['accuracy'],
     'probe_lin': lambda m: m['probes_selected_epoch']['person_from_activity_code_z']['linear'],
     'probe_mlp': lambda m: m['probes_selected_epoch']['person_from_activity_code_z']['mlp'],
     'probe_act_from_zd_mlp': lambda m: m['probes_selected_epoch']['activity_from_person_code_zd']['mlp'],
     'sel_epoch': lambda m: m['selection']['selected_epoch'], 'train_s': lambda m: m['train_seconds'],
     'epoch_s': lambda m: m['mean_epoch_seconds']}


def tmean(arm, t, key, seeds):
    v = [M[key](R[arm][(t, s)]) for s in seeds if (t, s) in R[arm]]
    return float(np.mean(v)) if v else float('nan')


def paired(x_arm, y_arm, key, seeds):
    d = np.array([np.mean([M[key](R[x_arm][(t, s)]) - M[key](R[y_arm][(t, s)]) for s in seeds]) for t in T])
    m, sd = float(d.mean()), float(d.std(ddof=1)); se = sd / math.sqrt(len(d)); h = stats.t.ppf(0.975, len(d) - 1) * se
    pval = float(2 * stats.t.sf(abs(m / se), len(d) - 1)) if se > 0 else float('nan')
    return {'mean': m, 'ci_low': m - h, 'ci_high': m + h, 'p': pval, 'n_up': int((d > 0).sum()), 'per_target': d.tolist()}


# ---------------------------------------------------------------- 1. arm means (primary seeds)
S = primary_seeds
rows = []
for arm in ['released', 'ie_off', 'ie_on', 'ie_off_bal', 'xpc_bal'] + NEW:
    if not all((t, s) in R[arm] for t in T for s in S):
        rows.append({'arm': arm, 'note': 'not complete for primary seeds'}); continue
    r = {'arm': arm, 'n_runs': len(T) * len(S)}
    for k in M:
        try:
            per_run = [M[k](R[arm][(t, s)]) for t in T for s in S]
            r[k] = float(np.mean(per_run))
        except (KeyError, TypeError):
            r[k] = float('nan')
    r['acc_sd_over_seeds_mean_of_targets'] = float(np.mean([np.std([M['acc'](R[arm][(t, s)]) for s in S], ddof=1) for t in T])) if len(S) > 1 else float('nan')
    rows.append(r)
with open(os.path.join(a.out, 'arm_summary.csv'), 'w', newline='') as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'arm', k)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
lines.append('\n== ARM MEANS (primary seeds; target accuracy at source-val-selected epoch is PRIMARY) ==')
for r in rows:
    if 'note' in r: lines.append(f"{r['arm']:10s} {r['note']}"); continue
    lines.append(f"{r['arm']:10s} acc {r['acc']:6.2f}  mF1 {r['macro_f1']:6.2f}  wF1 {r['weighted_f1']:6.2f}  last-ep acc {r['acc_last']:6.2f}  "
                 f"[leaky DIAG {r['acc_leaky_DIAG']:6.2f}]  probe p<-z lin {r['probe_lin']:6.2f} mlp {r['probe_mlp']:6.2f}  "
                 f"sel.ep {r['sel_epoch']:5.1f}  sd(seeds) {r['acc_sd_over_seeds_mean_of_targets']:.2f}  {r['train_s']:.0f}s/run")

# ---------------------------------------------------------------- 2. per-target table
with open(os.path.join(a.out, 'per_target.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['target', 'arm', 'acc', 'macro_f1', 'weighted_f1', 'probe_lin', 'probe_mlp', 'acc_per_seed'])
    for t in T:
        for arm in ['ie_off'] + NEW:
            w.writerow([t, arm] + [round(tmean(arm, t, k, S), 3) for k in ('acc', 'macro_f1', 'weighted_f1', 'probe_lin', 'probe_mlp')]
                       + [' '.join(f'{M["acc"](R[arm][(t, s)]):.2f}' for s in S)])
lines.append('\n== PER TARGET accuracy (mean over primary seeds) ==')
lines.append('target ' + ' '.join(f'{x:>8s}' for x in ['ie_off'] + NEW))
for t in T: lines.append(f'  {t}    ' + ' '.join(f'{tmean(x, t, "acc", S):8.2f}' for x in ['ie_off'] + NEW))

# ---------------------------------------------------------------- 3. predeclared contrasts
dec = {}
pc = {}
for arm in NEW:
    pc[arm] = {k: paired(arm, 'ie_off', k, S) for k in ('acc', 'macro_f1', 'weighted_f1', 'probe_lin', 'probe_mlp')}
# Holm over the 3 primary accuracy p-values
order = sorted(NEW, key=lambda x: pc[x]['acc']['p'])
holm, running = {}, 0.0
for i, arm in enumerate(order):
    running = max(running, min(1.0, (len(NEW) - i) * pc[arm]['acc']['p'])); holm[arm] = running
lines.append('\n== PREDECLARED CONTRASTS vs ie_off (target-level paired, n=5, df=4) ==')
for arm in NEW:
    A = pc[arm]['acc']; F1 = pc[arm]['macro_f1']
    supported = A['mean'] > 0 and A['n_up'] >= 4 and F1['mean'] > 0
    strong = supported and A['ci_low'] > 0 and holm[arm] < 0.05
    harmful = A['mean'] < 0 and (5 - A['n_up']) >= 4
    verdict = 'STRONG IMPROVEMENT' if strong else 'IMPROVEMENT SUPPORTED' if supported else 'HARMFUL' if harmful else 'INCONCLUSIVE'
    dec[arm] = {'verdict': verdict, 'holm_p_acc': holm[arm], **{k: v for k, v in pc[arm].items()}}
    lines.append(f"{arm:8s} {verdict:22s} dAcc {A['mean']:+6.2f} [{A['ci_low']:+.2f}, {A['ci_high']:+.2f}] up {A['n_up']}/5 "
                 f"p {A['p']:.3f} Holm {holm[arm]:.3f} | dMacroF1 {F1['mean']:+.2f} [{F1['ci_low']:+.2f}, {F1['ci_high']:+.2f}] | "
                 f"dWF1 {pc[arm]['weighted_f1']['mean']:+.2f} | dProbe lin {pc[arm]['probe_lin']['mean']:+.2f} mlp {pc[arm]['probe_mlp']['mean']:+.2f}")
    lines.append(f"         per-target dAcc: {' '.join(f'{v:+.2f}' for v in A['per_target'])}")

# ---------------------------------------------------------------- 4. factorial effects (descriptive)
def fx(key):
    out = {}
    td = lambda x, y: np.array([np.mean([M[key](R[x][(t, s)]) - M[key](R[y][(t, s)]) for s in S]) for t in T])
    for name, d in (('layout_main', (td('fix', 'ie_off') + td('fix_aug', 'aug')) / 2),
                    ('aug_main', (td('aug', 'ie_off') + td('fix_aug', 'fix')) / 2),
                    ('interaction', td('fix_aug', 'aug') - td('fix', 'ie_off'))):
        m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(5)); h = stats.t.ppf(0.975, 4) * se
        out[name] = {'mean': m, 'ci_low': m - h, 'ci_high': m + h, 'n_pos': int((d > 0).sum()), 'per_target': d.tolist()}
    return out
FX = {k: fx(k) for k in ('acc', 'macro_f1', 'probe_mlp')}
lines.append('\n== FACTORIAL EFFECTS (descriptive) ==')
for k, v in FX.items():
    for name, e in v.items():
        lines.append(f"{k:9s} {name:12s} {e['mean']:+6.2f} [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}] positive for {e['n_pos']}/5 targets")

# ---------------------------------------------------------------- 5. extra context (not part of the rule)
extra = {}
for arm in ('fix', 'fix_aug'):
    for ref in ('released', 'xpc_bal', 'ie_on'):
        if all((t, s) in R[ref] for t in T for s in S):
            extra[f'{arm}-{ref}'] = paired(arm, ref, 'acc', S)
lines.append('\n== CONTEXT: accuracy vs earlier arms (same seeds; not part of the predeclared rule) ==')
for k, v in extra.items():
    lines.append(f"{k:18s} {v['mean']:+6.2f} [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}] up {v['n_up']}/5")
stab = {arm: {'n_runs': len(R[arm]), 'failed_or_missing_primary': [k for k in [(t, s) for t in T for s in S] if k not in R[arm]],
              'resumed_runs': sum(1 for m in R[arm].values() if (m.get('resume') or {}).get('resumed_after_epoch'))} for arm in NEW}
json.dump({'primary_seeds': S, 'decisions': dec, 'factorial': FX, 'context': extra, 'stability': stab,
           'rule': 'PREDECLARED_PLAN.md section 4'}, open(os.path.join(a.out, 'decision.json'), 'w'), indent=2)
open(os.path.join(a.out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
