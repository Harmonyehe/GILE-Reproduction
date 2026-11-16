"""
Independent verification of (a) the released IE loss and (b) the proposed fix.
Fake data, CPU, no training, writes no files. Imports the UNMODIFIED model copy
in improvement-code/ (identical to the team's evaluation-code).

Run:  <gile_cpu python> verify_ie_design.py
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
nn.Module.cuda = lambda self, *a, **k: self          # CPU only: make .cuda() a no-op
from model_GILE_ucihar import GILE

torch.manual_seed(0)
args = argparse.Namespace(n_feature=9, d_AE=50, n_domains=5, x_dim=1152, n_class=6,
                          aux_loss_multiplier_y=1000., aux_loss_multiplier_d=1000.,
                          beta_d=1., beta_x=0., beta_y=1., weight_true=1000., weight_false=1000.,
                          device=torch.device('cpu'))
model = GILE(args); model.train()
B = 64
x = torch.randn(B, 128, 9, 1); y = torch.randint(0, 6, (B,)); d = torch.randint(0, 5, (B,))

def groups_with_grad(prefixes=None):
    out = {}
    for n, p in model.named_parameters():
        top = n.split('.')[0]
        g = 0.0 if p.grad is None else p.grad.abs().sum().item()
        out[top] = out.get(top, 0.0) + g
    return out

def line(t): print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)

# ---------------------------------------------------------------- T1
line('T1  RELEASED IE: is loss_function_false connected to any weight?')
loss_false = model.loss_function_false(args, d, x, y)
print('loss_false value       :', float(loss_false))
print('grad_fn (None = cut)   :', loss_false.grad_fn)
print('is a leaf tensor       :', loss_false.is_leaf)
model.zero_grad(); loss_false.backward()
print('total |grad| per module:', {k: round(v, 6) for k, v in groups_with_grad().items()})
pd_, py_, pdf_, pyf_ = model.classifier(x)
print('classifier() outputs are one-hot 0/1 tensors:',
      bool(((pd_ == 0) | (pd_ == 1)).all()), '| requires_grad:', pd_.requires_grad)

# ---------------------------------------------------------------- proposed loss
def ie_confusion(model, zd, zy, detach_classifier=True):
    """KL(Uniform || softmax(cross-logits)), summed over the batch, for both swaps.
    Gradients reach the encoders through zd, zy. Classifier weights are detached so the
    classifier cannot satisfy the penalty by itself."""
    def head(mod, z):
        W, b = mod.fc1.weight, mod.fc1.bias
        if detach_classifier: W, b = W.detach(), b.detach()
        return F.linear(F.relu(z), W, b)              # same as mod.forward: ReLU -> Linear
    out = 0.
    for logits in (head(model.qd, zy), head(model.qy, zd)):  # person-clf on activity-code, activity-clf on person-code
        K = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        out = out + (-(logp.mean(dim=1)) - torch.log(torch.tensor(float(K)))).sum()
    return out

def encode(model, x):
    zd_loc, zd_sc, _, _ = model.qzd(x); zy_loc, zy_sc, _, _ = model.qzy(x)
    return (torch.distributions.Normal(zd_loc, zd_sc).rsample(),
            torch.distributions.Normal(zy_loc, zy_sc).rsample())

# ---------------------------------------------------------------- T2
line('T2  PROPOSED IE: where do its gradients go?')
model.zero_grad()
zd, zy = encode(model, x)
L_ie = ie_confusion(model, zd, zy)
L_ie.backward()
g = groups_with_grad()
print('L_ie value:', round(float(L_ie), 4))
for k, v in g.items():
    print(f'   {k:5s} total |grad| = {v:.4e}   {"<-- encoder" if k in ("qzd","qzy") else ""}')
assert g['qzd'] > 0 and g['qzy'] > 0, 'encoders must receive IE gradient'
assert g['qd'] == 0 and g['qy'] == 0, 'classifier weights must be frozen for the IE term'

# analytic check: dL/dlogit = (p - 1/K)/... per sample -> compare with autograd
logits = torch.randn(8, 5, requires_grad=True)
lp = F.log_softmax(logits, 1); Lt = (-(lp.mean(1)) - torch.log(torch.tensor(5.))).sum(); Lt.backward()
analytic = F.softmax(logits, 1).detach() - 1. / 5
print('autograd == analytic (p - 1/K):', torch.allclose(logits.grad, analytic, atol=1e-6))

# ---------------------------------------------------------------- T3
line('T3  DIRECTION: does a small step on encoders (IE only) push cross-preds toward uniform?')
def cross_stats():
    with torch.no_grad():
        torch.manual_seed(123); zd, zy = encode(model, x)
        L = ie_confusion(model, zd, zy).item()
        pmax = (F.softmax(model.qd(zy), 1).max(1)[0].mean().item(), F.softmax(model.qy(zd), 1).max(1)[0].mean().item())
    return L, pmax
model.eval()   # BatchNorm-free encoders, but keep sampling deterministic via seed
L0, p0 = cross_stats()
enc_params = list(model.qzd.parameters()) + list(model.qzy.parameters())
opt = torch.optim.SGD(enc_params, lr=1e-3)
for _ in range(20):
    opt.zero_grad(); torch.manual_seed(123); zd, zy = encode(model, x)
    ie_confusion(model, zd, zy).backward(); opt.step()
L1, p1 = cross_stats()
print(f'L_ie: {L0:.4f} -> {L1:.4f}   (should decrease)')
print(f'mean max-prob  person-clf(z_act): {p0[0]:.3f} -> {p1[0]:.3f}   (uniform = {1/5:.3f})')
print(f'mean max-prob  activity-clf(z_pers): {p0[1]:.3f} -> {p1[1]:.3f}   (uniform = {1/6:.3f})')

# ---------------------------------------------------------------- T4
line('T4  STABILITY: proposed vs paper-literal (-CE) vs negative entropy, as logits grow')
target = torch.zeros(1, dtype=torch.long)
print(f'{"scale":>8} | {"KL(U||p)":>10} {"|grad|":>8} | {"-CE":>12} {"|grad|":>8} | {"-H(p)":>8} {"|grad|":>9}')
for s in (1., 10., 100., 1e4):
    base = torch.tensor([[0., 1., 0., 0., 0.]]) * s     # confident on class 1 (true label = 0)
    r = []
    for fn in ('kl', 'negce', 'negent'):
        z = base.clone().requires_grad_(True); lp = F.log_softmax(z, 1)
        if fn == 'kl': L = -(lp.mean()) - torch.log(torch.tensor(5.))
        elif fn == 'negce': L = -F.cross_entropy(z, target)
        else: L = (lp.exp() * lp).sum()
        L.backward(); r += [L.item(), z.grad.norm().item()]
    print(f'{s:8.0f} | {r[0]:10.3f} {r[1]:8.3f} | {r[2]:12.3f} {r[3]:8.3f} | {r[4]:8.3f} {r[5]:9.2e}')
print('KL(U||p): bounded below by 0, gradient never vanishes when confident.')
print('-CE     : unbounded below (-> -inf); optimiser can exploit it forever.')
print('-H(p)   : bounded, but gradient -> 0 when prediction is saturated (weak signal).')

# ---------------------------------------------------------------- T5
line('T5  SCALE: IE term vs main loss at initialisation (same batch)')
model.train(); model.zero_grad()
main, _ = model.loss_function(d, x, y)
zd, zy = encode(model, x)
ie = 1000. * ie_confusion(model, zd, zy)
print(f'main loss = {main.item():.1f}   1000*L_ie = {ie.item():.1f}   ratio = {ie.item()/main.item():.4f}')
print('(At init cross-preds are near-uniform, so L_ie is small; it grows only if leakage appears.)')

# ---------------------------------------------------------------- T6
line('T6  Is released training == ONE Adam step at lr 1e-4 + 1e-3 = 1.1e-3?')
import copy
torch.manual_seed(7); A = GILE(args); A.train(); Bm = copy.deepcopy(A)
optA1 = torch.optim.Adam(A.parameters(), lr=1e-4); optA2 = torch.optim.Adam(A.parameters(), lr=1e-3)
optB = torch.optim.Adam(Bm.parameters(), lr=1.1e-3)
for step in range(10):
    xb = torch.randn(B, 128, 9, 1); yb = torch.randint(0, 6, (B,)); db = torch.randint(0, 5, (B,))
    # released train_auxi order
    optA1.zero_grad(); optA2.zero_grad()
    torch.manual_seed(1000 + step); lo, _ = A.loss_function(db, xb, yb)
    lf = A.loss_function_false(args, db, xb, yb)
    lo.backward(); optA1.step(); lf.backward(); optA2.step()
    # single optimiser
    optB.zero_grad(); torch.manual_seed(1000 + step); lb, _ = Bm.loss_function(db, xb, yb)
    lb.backward(); optB.step()
diff = max((pa - pb).abs().max().item() for pa, pb in zip(A.parameters(), Bm.parameters()))
print(f'after 10 steps: max |param_released - param_single_1.1e-3| = {diff:.2e}  (total lr budget 1.1e-2)')

def run(kind, steps):
    torch.manual_seed(7); m = GILE(args); m.train()
    o1 = torch.optim.Adam(m.parameters(), lr=1e-4); o2 = torch.optim.Adam(m.parameters(), lr=1e-3)
    o = torch.optim.Adam(m.parameters(), lr=1.1e-3)
    g = torch.Generator().manual_seed(99)
    for step in range(steps):
        xb = torch.randn(B, 128, 9, 1, generator=g); yb = torch.randint(0, 6, (B,), generator=g); db = torch.randint(0, 5, (B,), generator=g)
        torch.manual_seed(1000 + step); lo, _ = m.loss_function(db, xb, yb)
        if kind == 'released':                       # exact train_auxi order, with IE call
            o1.zero_grad(); o2.zero_grad(); lo.backward(); o1.step()
            lf = m.loss_function_false(args, db, xb, yb); lf.backward(); o2.step()
        elif kind == 'two_adam_no_ie':               # same two optimisers, IE call removed
            o1.zero_grad(); o2.zero_grad(); lo.backward(); o1.step(); o2.step()
        else:                                        # single Adam at 1.1e-3
            o.zero_grad(); lo.backward(); o.step()
    return [p.detach().clone() for p in m.parameters()]

def mdiff(a, b): return max((x - y).abs().max().item() for x, y in zip(a, b))
for steps in (1, 10):
    R, T, S = run('released', steps), run('two_adam_no_ie', steps), run('single', steps)
    print(f'steps={steps:2d}: (a) released vs two-Adam-without-IE = {mdiff(R, T):.2e}   '
          f'(b) two-Adam vs single-Adam(1.1e-3) = {mdiff(T, S):.2e}')

print('\nALL CHECKS RAN.')
