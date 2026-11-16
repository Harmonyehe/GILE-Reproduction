"""
Person 3 improvement runner for GILE on UCI HAR.  One call = one (variant, target person, seed).

Variants (identical data, split, seed, epochs, hyper-parameters; only the training step differs):
  released : the ORIGINAL train.train_auxi (main Adam lr 1e-4 + "false" Adam lr 1e-3 on all params;
             its IE call is a no-op, so this is mathematically one Adam step at lr 1.1e-3)
  ie_off   : single Adam at lr 1.1e-3, main GILE loss only          (control)
  ie_on    : single Adam at lr 1.1e-3, main loss + gamma * repaired IE (model.loss_function_ie)

Protocol (no target information is used for training or model selection):
  * each SOURCE person is split train/val: per activity, the last val_frac of that activity's windows
    (file order) go to val; then every train window that overlaps a val window (50% sliding-window
    overlap, detected from the actual sensor values) is purged from train. Zero overlap is asserted.
  * epoch selected by mean source-val accuracy (tie -> lower source-val CE_y). Last epoch also reported.
  * target metrics are computed every epoch for logging only; the "leaky" best-target epoch is kept
    ONLY as a diagnostic for comparison with the paper's protocol.
Outputs: <out_root>/ucihar/<variant>/target<t>/seed<k>/{config.json, history.csv, metrics.json,
         log.txt, selected_model.pt, last_model.pt}
"""
import argparse, copy, hashlib, json, os, platform, random, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                      # the released loaders use relative paths ./data/...
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

VARIANTS = ('released', 'ie_off', 'ie_on')
N_PERSONS, N_CLASS, WIN, N_CH = 5, 6, 128, 9

p = argparse.ArgumentParser()
p.add_argument('--variant', required=True, choices=VARIANTS)
p.add_argument('--target', default='0', choices=[str(i) for i in range(N_PERSONS)])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=150)                 # released UCI default
p.add_argument('--lr', type=float, default=None, help='released: main lr (default 1e-4, false-opt 1e-3 fixed); ie_*: default 1.1e-3')
p.add_argument('--gamma', type=float, default=1000.0, help='IE weight (ie_on only); fixed a priori = aux CE multipliers')
p.add_argument('--val_frac', type=float, default=0.2)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--beta_d', type=float, default=1.0)                # main_ucihar.py defaults (Run 2)
p.add_argument('--beta_y', type=float, default=1.0)
p.add_argument('--fake', action='store_true', help='synthetic random data (smoke test only)')
p.add_argument('--fake_n', type=int, default=200, help='windows per person in --fake mode')
p.add_argument('--out_root', default=None, help='default: runs (real) or runs_smoke (fake)')
p.add_argument('--no_probe', action='store_true')
p.add_argument('--no_ckpt', action='store_true')
p.add_argument('--no_last_ckpt', action='store_true', help='save only selected_model.pt (storage); no effect on training')
p.add_argument('--overwrite', action='store_true')
args = p.parse_args()

if args.lr is None:
    args.lr = 1e-4 if args.variant == 'released' else 1.1e-3
FALSE_LR = 1e-3                                                     # train.py line 108 (released only)
out_root = args.out_root or ('runs_smoke' if args.fake else 'runs')
run_dir = os.path.join(HERE, out_root, 'ucihar', args.variant, f'target{args.target}', f'seed{args.seed}')
if os.path.exists(os.path.join(run_dir, 'metrics.json')) and not args.overwrite:
    print('SKIP (already complete):', run_dir); sys.exit(0)
os.makedirs(run_dir, exist_ok=True)


class Tee:
    def __init__(self, path): self.f = open(path, 'w', encoding='utf-8'); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()
sys.stdout = Tee(os.path.join(run_dir, 'log.txt'))

# ----------------------------------------------------------------------------- seeding / device
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    nn.Module.cuda = lambda self, *a, **k: self                   # GILE.__init__ calls self.cuda()

import data_preprocess_ucihar as prep                              # noqa: E402 (after chdir)
from model_GILE_ucihar import GILE                                 # noqa: E402
from train import train_auxi                                       # noqa: E402  (original, unchanged)

# ----------------------------------------------------------------------------- data
def load_person(pid):
    if args.fake:
        r = np.random.RandomState(1000 + pid)
        X = r.randn(args.fake_n, WIN, N_CH)
        y = r.randint(0, N_CLASS, args.fake_n)
        return X, y, np.full(args.fake_n, pid)
    X, y, d = prep.load_domain_data(str(pid))
    return np.asarray(X, dtype=np.float64), np.asarray(y).astype(int), np.asarray(d).astype(int)


def overlap_maps(X):
    """succ[i] = j if window j starts with the second half of window i (exact values), else -1."""
    half = WIN // 2
    first = {}
    for j in range(len(X)):
        first.setdefault(X[j, :half].tobytes(), []).append(j)
    succ = np.full(len(X), -1)
    for i in range(len(X)):
        js = [j for j in first.get(X[i, half:].tobytes(), []) if j != i]
        if js: succ[i] = js[0]
    pred = np.full(len(X), -1)
    for i, j in enumerate(succ):
        if j >= 0: pred[j] = i
    return succ, pred


def split_person(X, y):
    succ, pred = overlap_maps(X)
    stats = {'n': int(len(y)), 'n_with_overlap_successor': int((succ >= 0).sum()),
             'n_successor_is_next_index': int(sum(1 for i, j in enumerate(succ) if j == i + 1))}
    val = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]                                   # file order
        k = max(1, int(round(args.val_frac * len(idx))))
        val.extend(idx[-k:].tolist())
    val = np.array(sorted(val))
    in_val = np.zeros(len(y), bool); in_val[val] = True
    neighbours = set()
    for v in val:
        for nb in (succ[v], pred[v]):
            if nb >= 0 and not in_val[nb]: neighbours.add(int(nb))
    train = np.array([i for i in range(len(y)) if not in_val[i] and i not in neighbours])
    # verification: no remaining train/val pair shares half a window, and no exact duplicates
    tr = set(train.tolist())
    leaks = sum(1 for v in val for nb in (succ[v], pred[v]) if nb >= 0 and nb in tr)
    vb = {X[v].tobytes() for v in val}
    dups = sum(1 for t in train if X[t].tobytes() in vb)
    assert leaks == 0 and dups == 0, f'split leak: overlaps={leaks} duplicates={dups}'
    stats.update({'n_train': int(len(train)), 'n_val': int(len(val)), 'n_purged': len(neighbours),
                  'train_val_overlap_pairs_after_purge': leaks, 'train_val_exact_duplicates': dups,
                  'train_class_counts': dict(Counter(y[train].tolist())),
                  'val_class_counts': dict(Counter(y[val].tolist()))})
    return train, val, stats


def tensors(X, y, d):
    return TensorDataset(torch.as_tensor(X[..., None], dtype=torch.float32),   # (N,128,9,1) = released loader layout
                         torch.as_tensor(y, dtype=torch.long), torch.as_tensor(d, dtype=torch.long))


def train_loader(X, y, d, seed):
    counts = np.bincount(y, minlength=N_CLASS).astype(float)
    w = np.where(counts[y] > 0, 100.0 / counts[y], 0.0)              # released: weights = 100 / class count
    g = torch.Generator(); g.manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True, generator=g)
    return DataLoader(tensors(X, y, d), batch_size=args.batch_size, sampler=sampler, drop_last=True)


def eval_loader(X, y, d):
    return DataLoader(tensors(X, y, d), batch_size=256, shuffle=False)


sources = [i for i in range(N_PERSONS) if str(i) != args.target]
src_idx = torch.as_tensor(sources, dtype=torch.long, device=DEVICE)
t0 = time.time()
train_loaders, val_loaders, split_stats, probe_parts = [], [], {}, []
for k, pid in enumerate(sources):
    X, y, d = load_person(pid)
    tr, va, st = split_person(X, y)
    split_stats[str(pid)] = st
    print(f'person {pid}: n={st["n"]} train={st["n_train"]} val={st["n_val"]} purged={st["n_purged"]} '
          f'overlap-successors={st["n_with_overlap_successor"]} (next-index={st["n_successor_is_next_index"]})')
    train_loaders.append(train_loader(X[tr], y[tr], d[tr], seed=args.seed * 100 + k))
    val_loaders.append(eval_loader(X[va], y[va], d[va]))
    probe_parts.append((X[tr], y[tr], d[tr], X[va], y[va], d[va]))
Xt, yt, dt = load_person(int(args.target))
target_loader = eval_loader(Xt, yt, dt)
print(f'target person {args.target}: n={len(yt)} | data prep {time.time() - t0:.1f}s')

# ----------------------------------------------------------------------------- model
margs = argparse.Namespace(
    now_model_name='GILE', batch_size=args.batch_size, n_epoch=args.epochs, dataset='ucihar',
    n_feature=N_CH, len_sw=WIN, n_class=N_CLASS, d_AE=50, n_domains=N_PERSONS, x_dim=1152,
    aux_loss_multiplier_y=1000., aux_loss_multiplier_d=1000., beta_d=args.beta_d, beta_x=0.,
    beta_y=args.beta_y, weight_true=1000., weight_false=1000., device=DEVICE, target_domain=args.target)
set_seed(args.seed)                                                 # same init for every variant
model = GILE(margs).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
false_optimizer = torch.optim.Adam(model.parameters(), lr=FALSE_LR) if args.variant == 'released' else None


@torch.no_grad()
def predict(loader):
    model.eval()
    out = {k: [] for k in ('y', 'd', 'y_pred', 'logit_y', 'd_from_z', 'y_from_zd')}
    for x, y, d in loader:
        x = x.to(DEVICE)
        zd = model.qzd(x)[0]; zy = model.qzy(x)[0]                  # posterior means (as classifier())
        ly = model.qy(zy)
        out['y'].append(y.numpy()); out['d'].append(d.numpy())
        out['y_pred'].append(ly.argmax(1).cpu().numpy()); out['logit_y'].append(ly.cpu())
        out['d_from_z'].append(src_idx[model.qd(zy)[:, src_idx].argmax(1)].cpu().numpy())
        out['y_from_zd'].append(model.qy(zd).argmax(1).cpu().numpy())
    r = {k: (torch.cat(v) if k == 'logit_y' else np.concatenate(v)) for k, v in out.items()}
    r['ce_y'] = float(F.cross_entropy(r['logit_y'], torch.as_tensor(r['y'])).item())
    return r


def scores(r, full=False):
    s = {'n': int(len(r['y'])), 'accuracy': 100 * accuracy_score(r['y'], r['y_pred']),
         'macro_f1': 100 * f1_score(r['y'], r['y_pred'], average='macro', labels=list(range(N_CLASS)), zero_division=0),
         'weighted_f1': 100 * f1_score(r['y'], r['y_pred'], average='weighted', zero_division=0)}
    if full:
        s['confusion_matrix'] = confusion_matrix(r['y'], r['y_pred'], labels=list(range(N_CLASS))).tolist()
        s['classification_report'] = classification_report(r['y'], r['y_pred'], labels=list(range(N_CLASS)),
                                                           output_dict=True, zero_division=0)
    return s


# ----------------------------------------------------------------------------- training
history, best = [], {'acc': -1.0, 'ce': float('inf'), 'epoch': 0, 'state': None}
t_train = time.time()
for ep in range(1, args.epochs + 1):
    te = time.time(); model.train()
    if args.variant == 'released':
        loss, _ = train_auxi(margs, train_loaders, DEVICE, model, optimizer, ep, false_optimizer)
        loss = float(loss); ie_d_sum = ie_y_sum = float('nan')
    else:
        tot, n, ie_d_sum, ie_y_sum = 0.0, 0, 0.0, 0.0
        for loader in train_loaders:                                # same person-sequential order as released
            for x, y, d in loader:
                x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
                optimizer.zero_grad()
                if args.variant == 'ie_off':
                    l, _ = model.loss_function(d, x, y)
                else:
                    l, _, ie_d, ie_y = model.loss_function_ie(d, x, y, args.gamma, src_idx)
                    ie_d_sum += float(ie_d); ie_y_sum += float(ie_y)
                l.backward(); optimizer.step()
                tot += float(l); n += y.size(0)
        loss = tot / max(n, 1)
        if n: ie_d_sum /= n; ie_y_sum /= n
    if not np.isfinite(loss):
        raise FloatingPointError(f'non-finite training loss at epoch {ep}')

    vals = [predict(L) for L in val_loaders]
    v_acc = float(np.mean([scores(r)['accuracy'] for r in vals]))
    v_ce = float(np.mean([r['ce_y'] for r in vals]))
    v_mf1 = float(np.mean([scores(r)['macro_f1'] for r in vals]))
    v_dz = float(np.mean([100 * np.mean(r['d_from_z'] == r['d']) for r in vals]))
    v_yzd = float(np.mean([100 * np.mean(r['y_from_zd'] == r['y']) for r in vals]))
    tgt = scores(predict(target_loader))                             # DIAGNOSTIC ONLY - never used for selection
    row = {'epoch': ep, 'train_loss_per_sample': loss, 'ie_d_per_sample': ie_d_sum, 'ie_y_per_sample': ie_y_sum,
           'val_acc_mean': v_acc, 'val_macro_f1_mean': v_mf1, 'val_ce_y_mean': v_ce,
           'val_cross_acc_person_from_z': v_dz, 'val_cross_acc_activity_from_zd': v_yzd,
           'target_acc_DIAG': tgt['accuracy'], 'target_macro_f1_DIAG': tgt['macro_f1'],
           'target_weighted_f1_DIAG': tgt['weighted_f1'], 'epoch_seconds': time.time() - te}
    history.append(row)
    if v_acc > best['acc'] or (v_acc == best['acc'] and v_ce < best['ce']):
        best = {'acc': v_acc, 'ce': v_ce, 'epoch': ep,
                'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    print(f'ep {ep:3d} | loss {loss:10.2f} | val acc {v_acc:6.2f} ce {v_ce:.3f} | cross p<-z {v_dz:5.1f} '
          f'y<-zd {v_yzd:5.1f} | [diag] target acc {tgt["accuracy"]:6.2f} | {row["epoch_seconds"]:.1f}s')
train_seconds = time.time() - t_train

import csv                                                          # noqa: E402
with open(os.path.join(run_dir, 'history.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

# ----------------------------------------------------------------------------- final evaluation
last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
target_last = scores(predict(target_loader), full=True)
model.load_state_dict(best['state'])
target_selected = scores(predict(target_loader), full=True)
leaky = max(history, key=lambda r: r['target_acc_DIAG'])


def probe_features():
    model.eval()
    feats = []
    for Xtr, ytr, dtr, Xva, yva, dva in probe_parts:
        feats.append((Xtr, ytr, dtr, Xva, yva, dva))
    def enc(X):
        with torch.no_grad():
            x = torch.as_tensor(X[..., None], dtype=torch.float32, device=DEVICE)
            zs = [(model.qzy(x[i:i + 512])[0].cpu().numpy(), model.qzd(x[i:i + 512])[0].cpu().numpy())
                  for i in range(0, len(x), 512)]
        return np.concatenate([a for a, _ in zs]), np.concatenate([b for _, b in zs])
    Ztr, Zdtr, Ytr, Dtr, Zva, Zdva, Yva, Dva = [], [], [], [], [], [], [], []
    for Xtr, ytr, dtr, Xva, yva, dva in feats:
        a, b = enc(Xtr); Ztr.append(a); Zdtr.append(b); Ytr.append(ytr); Dtr.append(dtr)
        a, b = enc(Xva); Zva.append(a); Zdva.append(b); Yva.append(yva); Dva.append(dva)
    c = np.concatenate
    return c(Ztr), c(Zdtr), c(Ytr), c(Dtr), c(Zva), c(Zdva), c(Yva), c(Dva)


def run_probes():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import warnings; warnings.filterwarnings('ignore')
    # leakage guard: probe-train windows (source-train split) and probe-test windows (source-val split)
    # must be disjoint, share no half-window overlap (asserted in split_person), and contain no target data.
    tr_b = {w.tobytes() for (Xtr, *_rest) in probe_parts for w in Xtr}
    va_b = [w.tobytes() for (_a, _b, _c, Xva, _e, _f) in probe_parts for w in Xva]
    tg_b = {w.tobytes() for w in Xt}
    n_shared = sum(1 for w in va_b if w in tr_b)
    n_target = sum(1 for w in va_b if w in tg_b) + sum(1 for w in tr_b if w in tg_b)
    assert n_shared == 0 and n_target == 0, f'probe leakage: shared={n_shared} target={n_target}'
    Ztr, Zdtr, Ytr, Dtr, Zva, Zdva, Yva, Dva = probe_features()
    def fit(Xa, ya, Xb, yb):
        out = {'majority_baseline': 100 * float(np.max(np.bincount(yb)) / len(yb))}
        lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                                                            random_state=args.seed))
        for name, m in (('linear', lin), ('mlp', mlp)):
            m.fit(Xa, ya); out[name] = 100 * float(accuracy_score(yb, m.predict(Xb)))
        return out
    return {'person_from_activity_code_z': fit(Ztr, Dtr, Zva, Dva),          # lower = less person leakage
            'activity_from_person_code_zd': fit(Zdtr, Ytr, Zdva, Yva),        # lower = less activity leakage
            'person_from_person_code_zd': fit(Zdtr, Dtr, Zdva, Dva),          # sanity: should be high
            'note': 'fit on source-train codes (posterior means, selected epoch), scored on source-val codes',
            'n_probe_train': int(len(Ytr)), 'n_probe_test': int(len(Yva)),
            'leak_check': {'identical_windows_train_test': n_shared, 'target_windows_used': n_target}}


probes = None if args.no_probe else run_probes()


def file_sha(path):
    with open(path, 'rb') as f: return hashlib.sha256(f.read()).hexdigest()[:16]


env = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
       'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       'cudnn_deterministic': True,
       'code_sha256_16': {f: file_sha(os.path.join(HERE, f)) for f in
                          ('run_experiment.py', 'model_GILE_ucihar.py', 'train.py', 'data_preprocess_ucihar.py')}}
config = {k: v for k, v in vars(args).items()}
config.update({'false_optimizer_lr': FALSE_LR if args.variant == 'released' else None,
               'source_persons': sources, 'split': 'per-activity contiguous tail, overlap-purged',
               'split_stats': split_stats, 'env': env, 'run_dir': run_dir,
               'started_unix': t0})
with open(os.path.join(run_dir, 'config.json'), 'w') as f: json.dump(config, f, indent=2)

metrics = {
    'variant': args.variant, 'target': args.target, 'seed': args.seed, 'epochs': args.epochs,
    'selection': {'criterion': 'max mean source-val accuracy, tie -> min source-val CE_y',
                  'selected_epoch': best['epoch'], 'source_val_acc': best['acc'], 'source_val_ce_y': best['ce']},
    'target_at_selected_epoch': target_selected,                     # PRIMARY result
    'target_at_last_epoch': target_last,                             # secondary
    'DIAGNOSTIC_ONLY_leaky_best_target_epoch': {
        'warning': 'epoch chosen by TARGET accuracy (released protocol) - for comparison with the paper only',
        'epoch': leaky['epoch'], 'accuracy': leaky['target_acc_DIAG'],
        'macro_f1': leaky['target_macro_f1_DIAG'], 'weighted_f1': leaky['target_weighted_f1_DIAG']},
    'final_epoch_cross_accuracy': {'person_from_z_val': history[-1]['val_cross_acc_person_from_z'],
                                   'activity_from_zd_val': history[-1]['val_cross_acc_activity_from_zd']},
    'probes_selected_epoch': probes,
    'train_seconds': train_seconds, 'mean_epoch_seconds': float(np.mean([r['epoch_seconds'] for r in history])),
}
if not args.no_ckpt:
    torch.save(best['state'], os.path.join(run_dir, 'selected_model.pt'))
    if not args.no_last_ckpt:
        torch.save(last_state, os.path.join(run_dir, 'last_model.pt'))
with open(os.path.join(run_dir, 'metrics.json'), 'w') as f: json.dump(metrics, f, indent=2)
s = target_selected
print(f'\nDONE {args.variant} target{args.target} seed{args.seed} | selected epoch {best["epoch"]} | '
      f'target acc {s["accuracy"]:.2f} macroF1 {s["macro_f1"]:.2f} wF1 {s["weighted_f1"]:.2f} | '
      f'train {train_seconds:.0f}s | -> {run_dir}')
if probes:
    print('probe person<-z  :', probes['person_from_activity_code_z'])
    print('probe activity<-zd:', probes['activity_from_person_code_zd'])
