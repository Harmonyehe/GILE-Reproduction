"""
One-batch check for the GILE reproduction (Person 3).

Question 1: does the Independent Excitation (IE) loss change any model weights?
Question 2: does the encoder read the sensor channels in the right order?

- Uses FAKE random data (no dataset needed).
- Does NOT train and does NOT modify any file in the repo.
- Works on CPU (no GPU needed).

Run:  python ie_gradient_check.py
"""
import os
import sys
import argparse

import torch
import torch.nn as nn

# --- point Python at the team's evaluation-code folder (read only) ---
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(
    HERE, '..', 'GILE-Reproduction-main (1)', 'GILE-Reproduction-main', 'evaluation-code'))
sys.path.insert(0, CODE_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    # The GILE class calls self.cuda(), which crashes without an NVIDIA GPU.
    # For this check only, make .cuda() do nothing.
    nn.Module.cuda = lambda self, *a, **k: self

from model_GILE_ucihar import GILE  # the team's unmodified model file

torch.manual_seed(0)

# Same settings as main_ucihar.py defaults
args = argparse.Namespace(
    n_feature=9, d_AE=50, n_domains=5, x_dim=1152, n_class=6,
    aux_loss_multiplier_y=1000., aux_loss_multiplier_d=1000.,
    beta_d=1., beta_x=0., beta_y=1.,
    weight_true=1000., weight_false=1000., device=DEVICE)

model = GILE(args).to(DEVICE)
model.train()

# One fake batch shaped exactly like the real UCI loader output: (64, 128, 9, 1)
B = 64
x = torch.randn(B, 128, 9, 1, device=DEVICE)
y = torch.randint(0, 6, (B,), device=DEVICE)
d = torch.randint(0, 5, (B,), device=DEVICE)


def count_weights_with_gradient():
    params = list(model.named_parameters())
    hit = [n for n, p in params if p.grad is not None and p.grad.abs().sum().item() > 0]
    return len(hit), len(params)


print('=' * 70)
print('QUESTION 1: does the IE loss reach the model?')
print('=' * 70)

# Test 1: the main loss (control) -- should reach many weights
model.zero_grad(set_to_none=True)
loss_main, _ = model.loss_function(d, x, y)
loss_main.backward()
hit, total = count_weights_with_gradient()
print(f'Test 1  main loss (rebuild + classifiers): {hit} of {total} weight groups changed')

# Test 2: the IE loss on its own -- if IE worked, this would also be > 0
model.zero_grad(set_to_none=True)
loss_ie = model.loss_function_false(args, d, x, y)
loss_ie.backward()
hit_ie, total = count_weights_with_gradient()
print(f'Test 2  IE loss alone: value = {loss_ie.item():.1f}, '
      f'but {hit_ie} of {total} weight groups changed')

# Test 3: replay one real training step from train.py (train_auxi)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)        # network_ucihar.py
false_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # train.py line 108
optimizer.zero_grad()
false_optimizer.zero_grad()
loss_main, _ = model.loss_function(d, x, y)
loss_ie = model.loss_function_false(args, d, x, y)
loss_main.backward()
optimizer.step()
w = model.qzy.fc11[0].weight                     # a weight in the activity encoder
grad_before_ie = w.grad.detach().clone()
weight_before_second_step = w.detach().clone()
loss_ie.backward()
grad_unchanged = torch.equal(grad_before_ie, w.grad)
false_optimizer.step()
moved = (w.detach() - weight_before_second_step).abs().max().item()
print(f'Test 3  after IE backward, gradient identical to main-loss gradient: {grad_unchanged}')
print(f'        second optimiser still moved the weights by up to {moved:.2e}')

print()
if hit_ie == 0 and grad_unchanged:
    print('RESULT: IE has NO effect on the model. The second optimiser just re-applies')
    print('        the main loss with a 10x larger learning rate.')
else:
    print('RESULT: IE DOES reach the model -- tell Claude, the diagnosis was wrong.')

print()
print('=' * 70)
print('QUESTION 2: does the encoder see the 9 sensor channels correctly?')
print('=' * 70)
# Fill a batch so that every value equals its channel number (0..8)
probe = torch.arange(9, dtype=torch.float).view(1, 1, 9, 1).expand(2, 128, 9, 1).contiguous()
seen = probe.view(-1, probe.shape[2], 1, probe.shape[1])   # the exact line used in qzd/qzy.forward
first = seen[0, 0, 0, :18].tolist()
print('What the encoder treats as "channel 0" (first 18 time steps):')
print('   ', [int(v) for v in first])
if all(v == 0 for v in first):
    print('RESULT: channels are read correctly (all zeros).')
else:
    print('RESULT: channels are MIXED -- "channel 0" contains readings from all 9 sensors.')
    print('        The encoder uses .view() where a transpose/permute was needed.')
