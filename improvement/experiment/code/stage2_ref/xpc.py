"""Stage 2 (XPC-B) building blocks: person-balanced batches and the cross-person supervised contrastive loss.
Kept in its own module so checks/test_stage2.py can test them directly. Nothing here touches stage-1 code."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


class BalancedBatches:
    """Person-balanced source batches: every step concatenates batch_size/K windows from each of the K source
    persons. Per person it uses the same class-balanced WeightedRandomSampler as stage 1 (with replacement,
    weights 100/class count, generator seed = seed*100 + k), drawn batch_size/K at a time; a person's stream
    restarts when exhausted. Steps per epoch = sum_p floor(n_p / batch_size) = the stage-1 number, so every
    variant sees the same number of gradient steps and windows per epoch."""
    def __init__(self, parts, seed, batch_size, n_class, make_dataset):
        K = len(parts)
        assert batch_size % K == 0, 'batch size must divide evenly across source persons'
        self.per = batch_size // K
        self.loaders, self.gens = [], []
        for k, (X, y, d) in enumerate(parts):
            counts = np.bincount(y, minlength=n_class).astype(float)
            w = np.where(counts[y] > 0, 100.0 / counts[y], 0.0)
            g = torch.Generator(); g.manual_seed(seed * 100 + k)
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True, generator=g)
            self.loaders.append(DataLoader(make_dataset(X, y, d), batch_size=self.per, sampler=sampler, drop_last=True))
            self.gens.append(g)
        self.steps = sum(len(y) // batch_size for _, y, _ in parts)
        # RESUME SUPPORT (transfer package): for each person stream remember the sampler-generator state at the moment
        # its current iterator was created, and how many batches have been drawn from it. Reading a generator state
        # does not change it, so un-resumed behaviour is identical to the original module.
        self.iters, self.snap, self.consumed = [None] * K, [None] * K, [0] * K
        for k in range(K):
            self._new_iter(k)

    def _new_iter(self, k):
        self.snap[k] = self.gens[k].get_state().clone()
        self.iters[k] = iter(self.loaders[k])
        self.consumed[k] = 0

    def _next(self, k):
        try:
            b = next(self.iters[k])
        except StopIteration:
            self._new_iter(k)
            b = next(self.iters[k])
        self.consumed[k] += 1
        return b

    def state_dict(self):
        return {'snap': [s.clone() for s in self.snap], 'consumed': list(self.consumed)}

    def load_state_dict(self, st):
        """Replay each person stream to the saved position. The replay runs under a forked global RNG, because
        creating a DataLoader iterator draws a base seed from the global torch RNG, and that draw already happened
        (and is reflected in the restored global RNG state) in the original run."""
        with torch.random.fork_rng(devices=[]):
            for k in range(len(self.loaders)):
                self.gens[k].set_state(st['snap'][k])
                self.snap[k] = st['snap'][k].clone()
                self.iters[k] = iter(self.loaders[k])
                for _ in range(st['consumed'][k]):
                    next(self.iters[k])
                self.consumed[k] = st['consumed'][k]

    def __iter__(self):
        for _ in range(self.steps):
            bs = [self._next(k) for k in range(len(self.loaders))]
            yield tuple(torch.cat([b[i] for b in bs]) for i in range(3))


def make_proj_head(z_dim, seed):
    """Training-only projection head g: z -> 64 -> 32, built on the CPU. The CPU and ALL CUDA random states are saved
    and restored around its construction, so creating it does not shift any global random stream: the model's
    sampling noise is identical to stage 1, and ie_off_bal / xpc_bal share identical init and noise.
    (Correction found by the Colab test run: torch.manual_seed also reseeds CUDA, so forking only the CPU RNG
    was not enough on a GPU.)"""
    cuda_devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devs):
        torch.manual_seed(10_000 + seed)
        return nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 32))


def xpc_masks(y, d):
    same_y = y[:, None] == y[None, :]
    same_d = d[:, None] == d[None, :]
    pos = same_y & ~same_d                 # same activity, different person
    denom_ok = ~(same_y & same_d)          # drops self and same-person-same-activity (overlapping near-duplicates)
    return pos, denom_ok


def xpc_loss(mu, y, d, head, tau):
    """Cross-person supervised contrastive loss (SupCon form, Khosla et al. 2020) on the activity-code mean mu.
    L = mean over anchors i with |P(i)|>=1 of  -1/|P(i)| sum_{p in P(i)} log( exp(u_i.u_p/tau) / sum_{a in A(i)} exp(u_i.u_a/tau) )
    Returns (loss, n_anchors, mean positives per anchor)."""
    u = F.normalize(head(mu), dim=1)
    sim = u @ u.t() / tau
    pos, denom_ok = xpc_masks(y, d)
    pos = pos.float()
    log_den = torch.logsumexp(sim.masked_fill(~denom_ok, float('-inf')), dim=1, keepdim=True)
    log_prob = sim - log_den
    n_pos = pos.sum(1)
    has = n_pos > 0
    if not bool(has.any()):
        return mu.sum() * 0.0, 0, 0.0
    per_anchor = -(log_prob * pos).sum(1)[has] / n_pos[has]
    return per_anchor.mean(), int(has.sum()), float(n_pos[has].mean())
