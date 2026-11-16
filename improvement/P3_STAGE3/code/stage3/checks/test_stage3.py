"""Stage-3 correctness checks (CPU or GPU). Prints PASS/FAIL per check and 'ALL PASSED' at the end.
Usage (from anywhere): python code/stage3/checks/test_stage3.py"""
import json, math, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.dirname(HERE); CODE = os.path.dirname(S3)
sys.path.insert(0, CODE); sys.path.insert(0, S3)
import numpy as np
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    nn.Module.cuda = lambda self, *a, **k: self
from p3_variants import apply_layout_fix, PersonStyleAug, to_image_released, to_image_fixed, _rotations
import argparse
from model_GILE_ucihar import GILE

results = []
def check(name, ok, info=''):
    results.append(bool(ok)); print(('PASS ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''), flush=True)

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
B, T, C = 4, 128, 9

# ---------------------------------------------------------------- C1: what the released .view() does
x = torch.arange(B * T * C, dtype=torch.float32).view(B, T, C, 1)    # value encodes (b, t, c)
dec = lambda v: ((int(v) % (T * C)) // C, (int(v) % (T * C)) % C)   # -> (time, sensor)
rel, fix = to_image_released(x), to_image_fixed(x)
check('C1 both layouts give (B,9,1,128)', rel.shape == fix.shape == (B, C, 1, T), str(tuple(rel.shape)))
ok_fix = all(dec(fix[0, c, 0, t]) == (t, c) for c in range(C) for t in range(T))
check('C2 fixed layout: channel c = sensor c, width t = time t (all 1152 positions)', ok_fix)
row0 = [dec(rel[0, 0, 0, t]) for t in range(6)]
sens_in_kernel = [len({dec(rel[0, c, 0, t + k])[1] for k in range(5)}) for c in range(C) for t in range(T - 4)]
check('C3 released layout mixes sensors: every width-5 conv window spans 5 different sensors',
      min(sens_in_kernel) == 5, f'first positions of channel 0 as (time,sensor): {row0}')
rel_ch_sensors = [len({dec(rel[0, c, 0, t])[1] for t in range(T)}) for c in range(C)]
check('C4 released "channel" holds all 9 sensors and ~14 time steps', rel_ch_sensors == [9] * C,
      f'time span of channel 0: {dec(rel[0,0,0,0])[0]}..{dec(rel[0,0,0,T-1])[0]}')

# ---------------------------------------------------------------- C5-C7: patched model
margs = argparse.Namespace(now_model_name='GILE', batch_size=B, n_epoch=1, dataset='ucihar', n_feature=9, len_sw=128,
                           n_class=6, d_AE=50, n_domains=5, x_dim=1152, aux_loss_multiplier_y=1000.,
                           aux_loss_multiplier_d=1000., beta_d=1., beta_x=0., beta_y=1., weight_true=1000.,
                           weight_false=1000., device=DEV, target_domain='0')
torch.manual_seed(0); m_rel = GILE(margs).to(DEV)
torch.manual_seed(0); m_fix = GILE(margs).to(DEV)
same_init = all(torch.equal(a, b) for a, b in zip(m_rel.state_dict().values(), m_fix.state_dict().values()))
patched = apply_layout_fix(m_fix)
check('C5 patch touches only the encoders, init identical', same_init and patched == ['qzd', 'qzy'], str(patched))
m_rel.eval(); m_fix.eval()
xr = torch.randn(B, T, C, 1, device=DEV)
x_equiv = xr.permute(0, 2, 3, 1).contiguous().view(B, T, C, 1)       # memory such that .view() == correct permute
with torch.no_grad():
    a = m_fix.qzy(xr)[0]; b = m_rel.qzy(x_equiv)[0]; a2 = m_fix.qzd(xr)[0]; b2 = m_rel.qzd(x_equiv)[0]
    c = m_rel.qzy(xr)[0]
check('C6 patched forward == released forward applied to correctly laid-out memory (qzy and qzd)',
      torch.allclose(a, b, atol=0, rtol=0) and torch.allclose(a2, b2, atol=0, rtol=0))
check('C7 patched forward differs from released forward on the same input', not torch.allclose(a, c))
m_fix.train(); m_fix.zero_grad()
d = torch.randint(0, 5, (B,), device=DEV); y = torch.randint(0, 6, (B,), device=DEV)
l, _ = m_fix.loss_function(d, xr, y); l.backward()
g1 = m_fix.qzy.conv1[0].weight.grad; g2 = m_fix.qzd.conv1[0].weight.grad
check('C8 loss finite and gradients reach first conv of qzy and qzd (fixed layout)',
      torch.isfinite(l) and g1 is not None and g1.abs().sum() > 0 and g2.abs().sum() > 0, f'loss {float(l):.1f}')
x_rec = m_fix(d, xr, y)[0]
check('C9 decoder output shape == input shape (reconstruction target unchanged)', x_rec.shape == xr.shape)

# ---------------------------------------------------------------- A1-A6: augmentation
g = torch.Generator(); g.manual_seed(1)
xa = torch.randn(64, T, C, 1, generator=g) * 0.2
xa[..., 6:9, 0] += torch.tensor([1.0, 0.0, 0.0])                     # total_acc = body_acc + gravity
xa[..., 6:9, 0] += xa[..., 0:3, 0]
aug = PersonStyleAug(seed=3)
n = xa.shape[0]
apply, s, axis, ang, na, ngy = aug.draws(n, T)
# rotation only (s=1, no noise, all applied): norms of every triad and of gravity preserved
out = PersonStyleAug.transform(xa, torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.float64), axis, ang,
                               torch.zeros_like(na), torch.zeros_like(ngy))
nrm = lambda v, sl: v[..., sl, 0].norm(dim=-1)
okn = all(torch.allclose(nrm(out, sl), nrm(xa, sl), atol=1e-5) for sl in (slice(0, 3), slice(3, 6), slice(6, 9)))
grav_in = (xa[..., 6:9, 0] - xa[..., 0:3, 0]).norm(dim=-1); grav_out = (out[..., 6:9, 0] - out[..., 0:3, 0]).norm(dim=-1)
check('A1 rotation preserves every triad norm and the gravity magnitude', okn and torch.allclose(grav_in, grav_out, atol=1e-5))
R = _rotations(axis, ang)
check('A2 rotations are proper (R R^T = I, det = +1) and within +-15 deg',
      torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=R.dtype).expand_as(R), atol=1e-10)
      and torch.allclose(torch.det(R), torch.ones(n, dtype=R.dtype), atol=1e-10)
      and float(ang.abs().max()) <= math.radians(15) + 1e-12, f'max angle {math.degrees(float(ang.abs().max())):.2f} deg')
out2 = PersonStyleAug.transform(xa, apply, s, axis, ang, na, ngy)
same_off = torch.equal(out2[~apply], xa[~apply]); changed_on = not torch.allclose(out2[apply], xa[apply])
check('A3 windows not selected are bit-identical; selected windows change', same_off and changed_on,
      f'{int(apply.sum())}/{n} selected')
# scaling only: gravity component unchanged, body parts scaled by s
out3 = PersonStyleAug.transform(xa, torch.ones(n, dtype=torch.bool), s, axis, torch.zeros_like(ang),
                                torch.zeros_like(na), torch.zeros_like(ngy))
check('A4 scaling changes body_acc/body_gyro by s and leaves gravity (total-body) unchanged',
      torch.allclose(out3[..., 0:3, 0], (s[:, None, None] * xa[..., 0:3, 0].double()).float(), atol=1e-5)
      and torch.allclose(out3[..., 6:9, 0] - out3[..., 0:3, 0], xa[..., 6:9, 0] - xa[..., 0:3, 0], atol=1e-5)
      and float(s.min()) >= 0.7 and float(s.max()) <= 1.3)
a1, a2 = PersonStyleAug(5), PersonStyleAug(5)
r1 = a1(xa); r2 = a2(xa.clone() * 0 + 7)                               # different content, same seed
st_equal = torch.equal(a1.g.get_state(), a2.g.get_state())
r3 = PersonStyleAug(5)(xa)
check('A5 deterministic per seed; random draws do not depend on the data', torch.equal(r1, r3) and st_equal)
check('A6 global torch RNG untouched by augmentation',
      (lambda s0: (PersonStyleAug(9)(xa), torch.equal(s0, torch.get_rng_state()))[1])(torch.get_rng_state()))

# ---------------------------------------------------------------- R1-R4: runner (fake data, short)
PY = sys.executable
def run(script, *a):
    r = subprocess.run([PY, script, *a], capture_output=True, text=True)
    if r.returncode not in (0, 3): print(r.stdout[-1500:], r.stderr[-1500:])
    return r.returncode
RUN3 = os.path.join(S3, 'run_stage3.py'); RUN2 = os.path.join(CODE, 'stage2_ref', 'run_stage2.py')
TEST = 'runs_tests'
for root in (os.path.join(S3, TEST), os.path.join(CODE, 'stage2_ref', TEST)):
    if os.path.isdir(root): shutil.rmtree(root)
common = ['--target', '1', '--seed', '2', '--epochs', '3', '--fake', '--out_root', TEST, '--no_last_ckpt']
rc = [run(RUN3, '--arm', a, *common) for a in ('ie_off', 'fix', 'aug', 'fix_aug')]
rc.append(run(RUN2, '--variant', 'ie_off', *common))
check('R0 all short runs completed', rc == [0] * 5, str(rc))
ld = lambda root, arm, f: json.load(open(os.path.join(root, TEST, 'ucihar', arm, 'target1', 'seed2', f)))
m3 = {a: ld(S3, a, 'metrics.json') for a in ('ie_off', 'fix', 'aug', 'fix_aug')}
c3 = {a: ld(S3, a, 'config.json') for a in ('ie_off', 'fix', 'aug', 'fix_aug')}
m2 = ld(os.path.join(CODE, 'stage2_ref'), 'ie_off', 'metrics.json')
import csv
hist = lambda root, arm: list(csv.DictReader(open(os.path.join(root, TEST, 'ucihar', arm, 'target1', 'seed2', 'history.csv'))))
h3, h2 = hist(S3, 'ie_off'), hist(os.path.join(CODE, 'stage2_ref'), 'ie_off')
keys = ('train_loss_per_sample', 'val_acc_mean', 'val_ce_y_mean', 'target_acc_DIAG')
same_hist = all(r3[k] == r2[k] for r3, r2 in zip(h3, h2) for k in keys) and len(h3) == len(h2)
pnum = lambda m: {k: v for k, v in m['probes_selected_epoch'].items() if k != 'note'}   # 'note' text was reworded
same_res = all(m3['ie_off'][k] == m2[k] for k in ('target_at_selected_epoch', 'target_at_last_epoch')) \
    and pnum(m3['ie_off']) == pnum(m2)
check('R1 stage-3 control arm == stage-2 runner ie_off (history, target metrics, probes)', same_hist and same_res)
check('R2 identical initial parameters in all four arms',
      len({c3[a]['initial_parameter_hash'] for a in c3}) == 1, c3['ie_off']['initial_parameter_hash'])
diff = lambda a: hist(S3, a)[-1]['train_loss_per_sample'] != h3[-1]['train_loss_per_sample']
check('R3 fix / aug / fix_aug each change training relative to the control', all(diff(a) for a in ('fix', 'aug', 'fix_aug')))
# resume: fix_aug interrupted after epoch 1 then resumed == uninterrupted
TEST2 = 'runs_tests_resume'
if os.path.isdir(os.path.join(S3, TEST2)): shutil.rmtree(os.path.join(S3, TEST2))
c2 = ['--target', '1', '--seed', '2', '--epochs', '3', '--fake', '--out_root', TEST2, '--no_last_ckpt']
r_a = run(RUN3, '--arm', 'fix_aug', *c2, '--stop_after_epoch', '1'); r_b = run(RUN3, '--arm', 'fix_aug', *c2)
mres = json.load(open(os.path.join(S3, TEST2, 'ucihar', 'fix_aug', 'target1', 'seed2', 'metrics.json')))
hres = list(csv.DictReader(open(os.path.join(S3, TEST2, 'ucihar', 'fix_aug', 'target1', 'seed2', 'history.csv'))))
hun = hist(S3, 'fix_aug')
check('R4 interrupted+resumed fix_aug == uninterrupted (history and target metrics)',
      r_a == 3 and r_b == 0 and mres['resume']['resumed_after_epoch'] == 1
      and all(a[k] == b[k] for a, b in zip(hres, hun) for k in keys)
      and mres['target_at_selected_epoch'] == m3['fix_aug']['target_at_selected_epoch'])
print('device', DEV, '| torch', torch.__version__)
print('ALL PASSED' if all(results) else f'FAILED: {results.count(False)} of {len(results)}')
