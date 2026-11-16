"""Integrity checks over every returned stage-3 run (no training). Usage: python tools/verify_runs_p3.py <…/ucihar>
V1 arm flags match the arm name; V2 target never among source persons; V3 split/probe leak guards are zero;
V4 environment = Tesla T4 + torch 2.14.1+cu130, cudnn deterministic; V5 code SHA-256 prefixes match the local files;
V6 initial-parameter hash identical across arms for the same seed; V7 history has 150 finite epochs and the selected epoch
is the argmax of source-val accuracy (tie -> min CE); V8 no resumed runs; V9 checkpoint present;
V10 every run's MANIFEST entry matches the file on disk."""
import csv, glob, hashlib, json, math, os, sys
from collections import defaultdict

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); root = sys.argv[1]
NO_CKPT = '--no_ckpt' in sys.argv     # GitHub copy: checkpoints (.pt) are not included; skip V9 and .pt manifest entries
S3 = os.path.join(PKG, 'code', 'stage3')
sha16 = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
local = {'stage3/run_stage3.py': sha16(os.path.join(S3, 'run_stage3.py')), 'stage3/p3_variants.py': sha16(os.path.join(S3, 'p3_variants.py')),
         'model_GILE_ucihar.py': sha16(os.path.join(PKG, 'code', 'model_GILE_ucihar.py')),
         'data_preprocess_ucihar.py': sha16(os.path.join(PKG, 'code', 'data_preprocess_ucihar.py'))}
FLAGS = {'ie_off': (False, False), 'fix': (True, False), 'aug': (False, True), 'fix_aug': (True, True)}
fails, n, inits = [], 0, defaultdict(set)
for mp in sorted(glob.glob(os.path.join(root, '*', 'target*', 'seed*', 'metrics.json'))):
    d = os.path.dirname(mp); m = json.load(open(mp)); c = json.load(open(os.path.join(d, 'config.json'))); n += 1
    tag = f"{m['arm']} t{m['target']} s{m['seed']}"
    def chk(ok, what):
        if not ok: fails.append(f'{tag}: {what}')
    chk((c['layout_fix'], c['augmentation']) == FLAGS[m['arm']], 'V1 flags')
    chk(int(m['target']) not in c['source_persons'] and len(c['source_persons']) == 4, 'V2 target in sources')
    chk(all(s['train_val_overlap_pairs_after_purge'] == 0 and s['train_val_exact_duplicates'] == 0 for s in c['split_stats'].values()), 'V3 split leak')
    lc = m['probes_selected_epoch']['leak_check']; chk(lc['identical_windows_train_test'] == 0 and lc['target_windows_used'] == 0, 'V3 probe leak')
    e = c['env']; chk(e['gpu'] == 'Tesla T4' and e['torch'] == '2.14.1+cu130' and e['cudnn_deterministic'], 'V4 env')
    chk(e['code_sha256_16'] == local, f'V5 code hash {e["code_sha256_16"]}')
    inits[m['seed']].add(c['initial_parameter_hash'])
    h = list(csv.DictReader(open(os.path.join(d, 'history.csv'))))
    chk(len(h) == 150 and all(math.isfinite(float(r['train_loss_per_sample'])) for r in h), 'V7 history')
    best = max(h, key=lambda r: (float(r['val_acc_mean']), -float(r['val_ce_y_mean'])))
    first_best = min(int(r['epoch']) for r in h if (float(r['val_acc_mean']), -float(r['val_ce_y_mean'])) == (float(best['val_acc_mean']), -float(best['val_ce_y_mean'])))
    chk(m['selection']['selected_epoch'] == first_best, f'V7 selection {m["selection"]["selected_epoch"]} vs {first_best}')
    chk(m['resume']['resumed_after_epoch'] is None, 'V8 resumed')
    chk(NO_CKPT or os.path.exists(os.path.join(d, 'selected_model.pt')), 'V9 checkpoint')
chk_init = {s: len(v) for s, v in inits.items()}
if any(v != 1 for v in chk_init.values()): fails.append(f'V6 initial hashes differ within a seed: {chk_init}')
nm = 0
for man in glob.glob(os.path.join(S3, 'manifests', '*.sha256')):
    for line in open(man):
        if not line.strip(): continue
        hsh, rel = line.rstrip('\n').split('  ', 1)
        if NO_CKPT and rel.endswith('.pt'): continue
        nm += 1
        p = os.path.join(S3, *rel.split('/'))
        if not os.path.exists(p) or hashlib.sha256(open(p, 'rb').read()).hexdigest() != hsh: fails.append(f'V10 {rel}')
print(f'{n} runs checked, {nm} manifest entries re-hashed on disk; initial-parameter hashes per seed: {chk_init}')
print('\n'.join(fails) if fails else 'ALL CHECKS PASSED (V1-V10)')
