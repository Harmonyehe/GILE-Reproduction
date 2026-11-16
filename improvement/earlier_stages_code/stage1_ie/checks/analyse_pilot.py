"""Read-only analysis of a set of runs: loads every metrics/config/history/checkpoint and prints
selection, target metrics, probes, stability and runtime. Usage: python checks/analyse_pilot.py [root] [target] [seed]"""
import json, os, sys
import numpy as np, pandas as pd, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
root = sys.argv[1] if len(sys.argv) > 1 else 'runs_pilot_colab'
tgt = sys.argv[2] if len(sys.argv) > 2 else '0'
seed = sys.argv[3] if len(sys.argv) > 3 else '0'
B = os.path.join(ROOT, root, 'ucihar')
H = {}
for v in ('released', 'ie_off', 'ie_on'):
    d = os.path.join(B, v, f'target{tgt}', f'seed{seed}')
    m = json.load(open(os.path.join(d, 'metrics.json'))); c = json.load(open(os.path.join(d, 'config.json')))
    h = pd.read_csv(os.path.join(d, 'history.csv')); H[v] = h
    sd = torch.load(os.path.join(d, 'selected_model.pt'), map_location='cpu')
    finite = all(torch.isfinite(t).all().item() for t in sd.values() if t.is_floating_point())
    s, l, k = m['target_at_selected_epoch'], m['target_at_last_epoch'], m['DIAGNOSTIC_ONLY_leaky_best_target_epoch']
    print(f"== {v} | seed {c['seed']} lr {c['lr']} gamma {c['gamma']} false_lr {c['false_optimizer_lr']} | "
          f"{c['env']['gpu']} torch {c['env']['torch']} | ckpt tensors {len(sd)} finite {finite}")
    print(f"   selected epoch {m['selection']['selected_epoch']} | src-val acc {m['selection']['source_val_acc']:.2f} "
          f"ce {m['selection']['source_val_ce_y']:.3f}")
    print(f"   target@selected acc {s['accuracy']:.2f} macroF1 {s['macro_f1']:.2f} wF1 {s['weighted_f1']:.2f} | "
          f"@last acc {l['accuracy']:.2f} macroF1 {l['macro_f1']:.2f} wF1 {l['weighted_f1']:.2f} | "
          f"DIAG leaky ep {k['epoch']} acc {k['accuracy']:.2f} wF1 {k['weighted_f1']:.2f}")
    pr = m['probes_selected_epoch']
    print('   probes', {a: {b: round(x, 1) for b, x in pr[a].items()} for a in pr if isinstance(pr[a], dict)})
    print('   per-class F1 @selected', [round(100 * s['classification_report'][str(i)]['f1-score'], 1) for i in range(6)])
    tl = h.train_loss_per_sample
    print(f"   non-finite losses {int((~np.isfinite(tl)).sum())} | loss ep1/10/50/100/150 "
          f"{[round(tl.iloc[i], 1) for i in (0, 9, 49, 99, 149)]} | mean epoch {m['mean_epoch_seconds']:.2f}s, train {m['train_seconds']:.0f}s")
    va = h.val_acc_mean
    print(f"   src-val acc ep10/50/100/150 {[round(va.iloc[i], 1) for i in (9, 49, 99, 149)]} | max {va.max():.2f} "
          f"@ep {int(va.idxmax()) + 1} | epochs within 1pt of max: {int((va >= va.max() - 1).sum())} | std last 30 {va.tail(30).std():.2f}")
    print(f"   src-val CE ep10/50/100/150 {[round(h.val_ce_y_mean.iloc[i], 3) for i in (9, 49, 99, 149)]}")
    print(f"   cross-acc (last 10 ep) person<-z {h.val_cross_acc_person_from_z.tail(10).mean():.1f} (chance 25) | "
          f"activity<-zd {h.val_cross_acc_activity_from_zd.tail(10).mean():.1f} (chance ~16.7)")
    if v == 'ie_on':
        print(f"   IE/sample ie_d ep1/10/50/150 {[round(h.ie_d_per_sample.iloc[i], 4) for i in (0, 9, 49, 149)]} "
              f"ie_y {[round(h.ie_y_per_sample.iloc[i], 4) for i in (0, 9, 49, 149)]}")
        print(f"   gamma*IE share of loss ep1/50/150: "
              f"{[round(1000 * (h.ie_d_per_sample.iloc[i] + h.ie_y_per_sample.iloc[i]) / tl.iloc[i], 3) for i in (0, 49, 149)]}")
    td = h.target_acc_DIAG
    print(f"   [DIAG] target acc last-20 mean {td.tail(20).mean():.1f} std {td.tail(20).std():.1f} | min/max over ep 50-150 "
          f"{td.iloc[49:].min():.1f}/{td.iloc[49:].max():.1f}")
a, b = H['released'], H['ie_off']
print(f"\nreleased vs ie_off: max |src-val acc diff| {(a.val_acc_mean - b.val_acc_mean).abs().max():.2f}; "
      f"first-3-epoch losses {a.train_loss_per_sample.head(3).round(2).tolist()} vs {b.train_loss_per_sample.head(3).round(2).tolist()}")
