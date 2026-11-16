"""Phase-1 correctness tests for the implemented ie_on loss (fake data, CPU, writes nothing).
Run: <gile_cpu python> checks/test_variants.py"""
import os, sys, argparse, copy
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
nn.Module.cuda = lambda self, *a, **k: self
from model_GILE_ucihar import GILE

args = argparse.Namespace(n_feature=9, d_AE=50, n_domains=5, x_dim=1152, n_class=6, aux_loss_multiplier_y=1000.,
                          aux_loss_multiplier_d=1000., beta_d=1., beta_x=0., beta_y=1., weight_true=1000.,
                          weight_false=1000., device=torch.device('cpu'))
torch.manual_seed(0); m = GILE(args); m.train()
B = 64; x = torch.randn(B, 128, 9, 1); y = torch.randint(0, 6, (B,)); d = torch.randint(1, 5, (B,))   # target person 0 absent
src = torch.tensor([1, 2, 3, 4])
ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(f'[{"PASS" if cond else "FAIL"}] {name} {detail}')

# 1. main part of loss_function_ie == released loss_function (same RNG state)
torch.manual_seed(5); l_rel, ce_rel = m.loss_function(d, x, y)
torch.manual_seed(5); l_ie0, ce0, ie_d, ie_y = m.loss_function_ie(d, x, y, 0.0, src)
check('gamma=0: loss_function_ie == loss_function', torch.allclose(l_rel, l_ie0, rtol=0, atol=1e-6) and torch.equal(ce_rel, ce0),
      f'({l_rel.item():.4f} vs {l_ie0.item():.4f})')
torch.manual_seed(5); l_g, _, ie_d2, ie_y2 = m.loss_function_ie(d, x, y, 1000.0, src)
check('gamma=1000: total == main + 1000*(ie_d+ie_y)', torch.allclose(l_g, l_ie0 + 1000 * (ie_d2 + ie_y2), rtol=1e-6))
check('IE terms finite and >= 0', bool(torch.isfinite(ie_d2) and torch.isfinite(ie_y2) and ie_d2 >= 0 and ie_y2 >= 0),
      f'(ie_d={ie_d2.item():.3f}, ie_y={ie_y2.item():.3f})')

# 2. gradient of the IE term alone: only the two encoders
m.zero_grad(); torch.manual_seed(5); _, _, a, b = m.loss_function_ie(d, x, y, 1000.0, src); (a + b).backward()
g = {}
for n, p in m.named_parameters():
    g[n.split('.')[0]] = g.get(n.split('.')[0], 0.0) + (0.0 if p.grad is None else p.grad.abs().sum().item())
print('    |grad| of IE term per module:', {k: f'{v:.3e}' for k, v in g.items()})
check('IE gradient reaches person encoder qzd', g['qzd'] > 0)
check('IE gradient reaches activity encoder qzy', g['qzy'] > 0)
check('IE gradient does NOT touch classifiers qd, qy', g['qd'] == 0 and g['qy'] == 0)
check('IE gradient does NOT touch decoder/priors', g['px'] == 0 and g['pzd'] == 0 and g['pzy'] == 0)

# 3. the held-out person's class is excluded from the person-head term
torch.manual_seed(9); zd = torch.randn(B, 50); zy = torch.randn(B, 50)
base_d, base_y = m.ie_term(zd, zy, src)
mm = copy.deepcopy(m); mm.qd.fc1.bias.data[0] += 50.0           # make target-person logit huge
t_d, _ = mm.ie_term(zd, zy, src)
mm2 = copy.deepcopy(m); mm2.qd.fc1.bias.data[2] += 50.0          # make a SOURCE-person logit huge
s_d, _ = mm2.ie_term(zd, zy, src)
check('target-person logit has no effect on ie_d', torch.allclose(base_d, t_d), f'({base_d.item():.4f} vs {t_d.item():.4f})')
check('source-person logit does affect ie_d', s_d.item() > base_d.item() + 1, f'({base_d.item():.2f} -> {s_d.item():.2f})')

# 4. minimum is exactly at uniform predictions
mz = copy.deepcopy(m)
for h in (mz.qd.fc1, mz.qy.fc1): h.weight.data.zero_(); h.bias.data.zero_()
u_d, u_y = mz.ie_term(zd, zy, src)
check('IE == 0 when cross-predictions are uniform', abs(u_d.item()) < 1e-5 and abs(u_y.item()) < 1e-5)

# 5. descent direction on real model: encoder steps on IE reduce cross confidence
mt = copy.deepcopy(m); opt = torch.optim.Adam(list(mt.qzd.parameters()) + list(mt.qzy.parameters()), lr=1e-3)
def conf(model):
    with torch.no_grad():
        zd_, zy_ = model.qzd(x)[0], model.qzy(x)[0]
        return (F.softmax(model.qd(zy_)[:, src], 1).max(1)[0].mean().item(), F.softmax(model.qy(zd_), 1).max(1)[0].mean().item())
c0 = conf(mt)
for _ in range(30):
    opt.zero_grad(); torch.manual_seed(5); _, _, a, b = mt.loss_function_ie(d, x, y, 1.0, src); (a + b).backward(); opt.step()
c1 = conf(mt)
check('IE steps push cross-predictions toward uniform', c1[0] < c0[0] and c1[1] < c0[1],
      f'(person<-z max-prob {c0[0]:.3f}->{c1[0]:.3f}, uniform 0.25; activity<-zd {c0[1]:.3f}->{c1[1]:.3f}, uniform 0.167)')

# 6. FINDING (not changed): prior p(z_d|d) uses BatchNorm on a one-hot, and every training batch has one person
mp = copy.deepcopy(m); mp.train()
p0 = mp.pzd(torch.zeros(B, dtype=torch.long))[0]; p3 = mp.pzd(torch.full((B,), 3, dtype=torch.long))[0]
mixed = mp.pzd(torch.arange(B) % 5)[0]
check('[finding] train-mode prior p(z_d|d) identical for person 0 and person 3 in single-person batches',
      torch.allclose(p0, p3, atol=1e-4), f'(max diff {(p0 - p3).abs().max().item():.1e}; BatchNorm of a constant one-hot -> constant)')
print('    (for comparison, a mixed-person batch gives different priors per person:',
      not torch.allclose(mixed[0], mixed[3]), ')')
print('\nALL PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
