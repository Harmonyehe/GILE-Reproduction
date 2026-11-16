"""Stage 2 (XPC-B) correctness tests. CPU, synthetic data, a few minutes. Writes only to stage2_xpc/runs_smoke_tests/.
Usage: python stage2_xpc/checks/test_stage2.py      -> must end with ALL PASSED"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn

S2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CODE = os.path.dirname(S2)
sys.path.insert(0, CODE); sys.path.insert(0, S2)
os.chdir(CODE)
nn.Module.cuda = lambda self, *a, **k: self      # in-process unit tests run on CPU even on a GPU machine
                                                 # (GILE.__init__ calls self.cuda()); subprocess runs are unaffected
from torch.utils.data import TensorDataset
from xpc import BalancedBatches, make_proj_head, xpc_loss, xpc_masks
from model_GILE_ucihar import GILE

ok = True
def check(name, cond, info=''):
    global ok
    ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, info, flush=True)

torch.manual_seed(0); np.random.seed(0)
# ---------------------------------------------------------------- 1. masks
y = torch.tensor([0, 0, 0, 1, 1, 2]); d = torch.tensor([1, 1, 2, 1, 3, 2])
pos, den = xpc_masks(y, d)
check('positives = same activity AND different person',
      all(bool(pos[i, j]) == (y[i] == y[j] and d[i] != d[j] and i != j) for i in range(6) for j in range(6)))
check('denominator excludes self and same-person-same-activity, keeps everything else',
      all(bool(den[i, j]) == (not (y[i] == y[j] and d[i] == d[j])) for i in range(6) for j in range(6)))
check('pair (0,1) same person+activity is neither positive nor in denominator', not pos[0, 1] and not den[0, 1])

# ---------------------------------------------------------------- 2. loss equals a naive loop
head = make_proj_head(50, 0)
mu = torch.randn(32, 50); yy = torch.randint(0, 6, (32,)); dd = torch.randint(1, 5, (32,))
L, na, npp = xpc_loss(mu, yy, dd, head, 0.1)
u = torch.nn.functional.normalize(head(mu), dim=1); s = (u @ u.t() / 0.1).detach()
terms = []
for i in range(32):
    P = [j for j in range(32) if j != i and yy[j] == yy[i] and dd[j] != dd[i]]
    A = [j for j in range(32) if not (yy[j] == yy[i] and dd[j] == dd[i])]
    if P:
        den_i = torch.logsumexp(s[i, A], 0)
        terms.append(-np.mean([float(s[i, p] - den_i) for p in P]))
check('xpc_loss equals naive per-anchor computation', abs(float(L) - np.mean(terms)) < 1e-4,
      f'({float(L):.5f} vs {np.mean(terms):.5f}, anchors {na})')

# ---------------------------------------------------------------- 3. no positives -> zero, differentiable
mu0 = torch.randn(8, 50, requires_grad=True)
L0, na0, _ = xpc_loss(mu0, torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long), head, 0.1)
L0.backward()
check('single-person batch: no anchors, loss 0, backward works', na0 == 0 and float(L0) == 0.0 and mu0.grad is not None)

# ---------------------------------------------------------------- 4. gradient reaches only qzy (+ head)
margs = argparse.Namespace(now_model_name='GILE', batch_size=64, n_epoch=1, dataset='ucihar', n_feature=9, len_sw=128,
                           n_class=6, d_AE=50, n_domains=5, x_dim=1152, aux_loss_multiplier_y=1000.,
                           aux_loss_multiplier_d=1000., beta_d=1., beta_x=0., beta_y=1., weight_true=1000.,
                           weight_false=1000., device=torch.device('cpu'), target_domain='0')
torch.manual_seed(0); model = GILE(margs); model.train()
x = torch.randn(64, 128, 9, 1); y64 = torch.arange(64) % 6; d64 = 1 + (torch.arange(64) // 16)
h = make_proj_head(50, 0)
Lx, _, _ = xpc_loss(model.qzy(x)[0], y64, d64, h, 0.1); Lx.backward()
nz = {n.split('.')[0] for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
check('XPC gradient reaches only the activity encoder qzy (and head g)',
      nz == {'qzy'} and all(p.grad is not None for p in h.parameters()), f'(modules with grad: {sorted(nz)})')

# ---------------------------------------------------------------- 5. head creation does not move the global RNG
torch.manual_seed(123); a = torch.rand(3); torch.manual_seed(123); _ = make_proj_head(50, 7); b = torch.rand(3)
check('make_proj_head leaves the global CPU RNG stream unchanged', torch.equal(a, b))
if torch.cuda.is_available():
    torch.manual_seed(123); a = torch.rand(3, device='cuda'); torch.manual_seed(123); _ = make_proj_head(50, 7)
    b = torch.rand(3, device='cuda')
    check('make_proj_head leaves the global CUDA RNG stream unchanged', torch.equal(a, b))

# ---------------------------------------------------------------- 6. balanced batches
def ds(X, y, d):
    return TensorDataset(torch.as_tensor(X[..., None], dtype=torch.float32), torch.as_tensor(y), torch.as_tensor(d))
rs = np.random.RandomState(0)
parts = [(rs.randn(n, 128, 9), rs.randint(0, 6, n), np.full(n, pid)) for n, pid in ((270, 1), (241, 2), (255, 3), (199, 4))]
bb = BalancedBatches(parts, seed=3, batch_size=64, n_class=6, make_dataset=ds)
batches = list(bb)
steps_stage1 = sum(n // 64 for n in (270, 241, 255, 199))
check('steps per epoch equal stage-1 count', len(batches) == bb.steps == steps_stage1, f'({len(batches)} vs {steps_stage1})')
check('every batch has 64 windows, exactly 16 from each source person',
      all(b[0].shape[0] == 64 and all(int((b[2] == p).sum()) == 16 for p in (1, 2, 3, 4)) for b in batches))
batches2 = list(bb)                                    # second epoch continues/restarts streams without error
check('second epoch works (streams restart when exhausted)', len(batches2) == steps_stage1)
bb_again = BalancedBatches(parts, seed=3, batch_size=64, n_class=6, make_dataset=ds)
check('balanced batches are seed-reproducible', all(torch.equal(a[0], b[0]) for a, b in zip(batches, list(bb_again))))
xb, yb, db = batches[0]
_, na_b, np_b = xpc_loss(torch.randn(64, 50), yb, db, head, 0.1)
check('balanced batch gives cross-person positives for (nearly) every anchor', na_b >= 60, f'(anchors {na_b}/64, mean positives {np_b:.1f})')

# ---------------------------------------------------------------- 7. person prior becomes person-specific in mixed batches
pz = model.pzd(db)[0]
check('[mechanism] prior p(z_d|d) differs between persons in a balanced batch',
      not torch.allclose(pz[db == 1][0], pz[db == 2][0], atol=1e-4))

# ---------------------------------------------------------------- 8-10. end-to-end fake runs (subprocess)
OUT = os.path.join(S2, 'runs_smoke_tests')
py = sys.executable
def run(script, variant, extra=()):
    r = subprocess.run([py, script, '--variant', variant, '--target', '0', '--seed', '0', '--epochs', '2', '--fake',
                        '--out_root', OUT, '--no_ckpt', '--overwrite', *extra], capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout[-1500:], r.stderr[-3000:])
    return r.returncode
def met(variant): return json.load(open(os.path.join(OUT, 'ucihar', variant, 'target0', 'seed0', 'metrics.json')))
def hist(variant):
    import csv
    return list(csv.DictReader(open(os.path.join(OUT, 'ucihar', variant, 'target0', 'seed0', 'history.csv'))))

rc1 = run(os.path.join(CODE, 'run_experiment.py'), 'ie_off')
m1 = met('ie_off'); h1 = hist('ie_off')
rc2 = run(os.path.join(S2, 'run_stage2.py'), 'ie_off')    # same folder name -> overwrites the stage-1 smoke copy, so read first
m2 = met('ie_off'); h2 = hist('ie_off')
same = rc1 == 0 and rc2 == 0 and all(float(a['train_loss_per_sample']) == float(b['train_loss_per_sample']) and
                                     a['target_acc_DIAG'] == b['target_acc_DIAG'] and a['val_acc_mean'] == b['val_acc_mean']
                                     for a, b in zip(h1, h2)) \
       and m1['target_at_selected_epoch']['accuracy'] == m2['target_at_selected_epoch']['accuracy']
check('stage-2 runner variant ie_off reproduces stage-1 run_experiment.py ie_off exactly (fake data)', same)

rc3 = run(os.path.join(S2, 'run_stage2.py'), 'ie_off_bal')
rc4 = run(os.path.join(S2, 'run_stage2.py'), 'xpc_bal', ('--lam', '0'))
hb, hx0 = hist('ie_off_bal'), hist('xpc_bal')
check('xpc_bal with lam=0 is identical to ie_off_bal (only the XPC term differs)',
      rc3 == 0 and rc4 == 0 and all(float(a['main_loss_per_sample']) == float(b['main_loss_per_sample']) and
                                    a['target_acc_DIAG'] == b['target_acc_DIAG'] for a, b in zip(hb, hx0)))
rc5 = run(os.path.join(S2, 'run_stage2.py'), 'xpc_bal', ('--lam', '1000'))
hx = hist('xpc_bal'); mx = met('xpc_bal')
check('xpc_bal (lam=1000) runs, finite loss, XPC active, probes + leak guard present',
      rc5 == 0 and all(np.isfinite(float(r['train_loss_per_sample'])) for r in hx) and float(hx[-1]['xpc_per_batch']) > 0
      and mx['probes_selected_epoch']['leak_check'] == {'identical_windows_train_test': 0, 'target_windows_used': 0},
      f"(xpc/batch ep1 {float(hx[0]['xpc_per_batch']):.3f}, share {float(hx[0]['xpc_share_of_loss']):.3f})")
check('ie_off_bal differs from person-sequential ie_off (batching really changed)',
      any(a['train_loss_per_sample'] != b['train_loss_per_sample'] for a, b in zip(h2, hb)))

print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
