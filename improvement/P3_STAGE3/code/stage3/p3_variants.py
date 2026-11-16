"""Person 3, stage 3: the two isolated changes tested against the stage-1 control (see ../../PREDECLARED_PLAN.md).

C  apply_layout_fix(model): replaces forward() of the encoder INSTANCES qzd / qzy (and qzx if present) with an exact
   copy of the released forward in which the input reshaping
        x.view(-1, x.shape[2], 1, x.shape[1])      # released: reinterprets memory -> sensors and time mixed
   is replaced by
        x.permute(0, 2, 3, 1)                       # (B,128,9,1) -> (B,9,1,128): channel = sensor, width = time
   Nothing else changes (same parameters, same init, same decoder). model_GILE_ucihar.py itself is NOT edited.

B  PersonStyleAug: training-time augmentation in raw physical units (UCI HAR channel order
   [body_acc xyz, body_gyro xyz, total_acc xyz]), each window transformed with probability p = 0.5:
     - movement-intensity scaling s ~ N(1, 0.1^2) clipped to [0.7, 1.3] on body_acc and body_gyro;
       total_acc += (s - 1) * body_acc (gravity component unchanged)
     - one small rotation (axis uniform on the sphere, angle ~ U(-15, +15) deg) applied to all three triads
     - Gaussian noise sigma = 0.01 on body_acc (same noise added to total_acc) and on body_gyro
   It owns a separate torch.Generator and draws a FIXED number of random numbers per window (whether or not the window is
   transformed), so the model initialisation, batch order and VAE sampling noise are identical to the control.
"""
import math
import types

import torch

BA, BG, TA = slice(0, 3), slice(3, 6), slice(6, 9)


# ----------------------------------------------------------------------------------------------- C: layout fix
def to_image_released(x):
    return x.view(-1, x.shape[2], 1, x.shape[1])


def to_image_fixed(x):
    return x.permute(0, 2, 3, 1)                                   # (B, T, C, 1) -> (B, C, 1, T)


def _fixed_forward(self, x):
    # exact copy of qzd/qzx/qzy.forward in model_GILE_ucihar.py except the marked line
    x = x.float()
    x_img = to_image_fixed(x)                                       # <-- the only change (was .view(...))

    out_conv1 = self.conv1(x_img)
    out1, idx1 = self.pool1(out_conv1)

    out_conv2 = self.conv2(out1)
    out2, idx2 = self.pool2(out_conv2)

    out_conv3 = self.conv3(out2)
    out3, idx3 = self.pool3(out_conv3)

    out_conv4 = self.conv4(out3)
    out4, idx4 = self.pool4(out_conv4)

    out = out4.reshape(-1, out4.shape[1] * out4.shape[3])
    size1 = out1.size()
    size2 = out2.size()
    size3 = out3.size()
    size4 = out4.size()

    loc = self.fc11(out)
    scale = self.fc12(out) + 1e-7

    return loc, scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


def apply_layout_fix(model):
    patched = []
    for name in ('qzd', 'qzx', 'qzy'):
        enc = getattr(model, name, None)
        if enc is not None:
            enc.forward = types.MethodType(_fixed_forward, enc)
            patched.append(name)
    return patched


# ----------------------------------------------------------------------------------------------- B: augmentation
def _rotations(axis, angle):
    """Rodrigues: axis (N,3) unit vectors, angle (N,) radians -> (N,3,3)."""
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    z = torch.zeros_like(kx)
    K = torch.stack([torch.stack([z, -kz, ky], 1), torch.stack([kz, z, -kx], 1), torch.stack([-ky, kx, z], 1)], 1)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    eye = torch.eye(3, dtype=axis.dtype).expand(len(angle), 3, 3)
    return eye + s * K + (1 - c) * (K @ K)


class PersonStyleAug:
    P, SCALE_SD, SCALE_CLIP, MAX_DEG, NOISE_SD = 0.5, 0.1, (0.7, 1.3), 15.0, 0.01   # fixed a priori (plan section 1)

    def __init__(self, seed):
        self.g = torch.Generator()
        self.g.manual_seed(seed * 7919 + 17)

    def state_dict(self):
        return {'g': self.g.get_state().clone()}

    def load_state_dict(self, st):
        self.g.set_state(st['g'])

    def draws(self, n, T):
        g = self.g
        apply = torch.rand(n, generator=g) < self.P
        s = (1 + self.SCALE_SD * torch.randn(n, generator=g, dtype=torch.float64)).clamp(*self.SCALE_CLIP)
        axis = torch.randn(n, 3, generator=g, dtype=torch.float64)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ang = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * math.radians(self.MAX_DEG)
        n_acc = self.NOISE_SD * torch.randn(n, T, 3, generator=g, dtype=torch.float64)
        n_gyr = self.NOISE_SD * torch.randn(n, T, 3, generator=g, dtype=torch.float64)
        return apply, s, axis, ang, n_acc, n_gyr

    @staticmethod
    def transform(x, apply, s, axis, ang, n_acc, n_gyr):
        """x: (N, T, 9[, 1]) float tensor (CPU). Returns a new tensor; windows with apply=False are returned unchanged."""
        squeeze = x.dim() == 4
        X = (x[..., 0] if squeeze else x).to(torch.float64)
        ba, bg, ta = X[..., BA], X[..., BG], X[..., TA]
        grav = ta - ba                                              # gravity (+ residual) component, kept as is
        sv = s[:, None, None]
        ba2 = sv * ba + n_acc
        bg2 = sv * bg + n_gyr
        ta2 = grav + ba2                                            # = ta + (s-1)*ba + noise
        R = _rotations(axis, ang)                                   # (N,3,3); rotate row vectors: v @ R^T
        rot = lambda v: v @ R.transpose(1, 2)
        out = torch.cat([rot(ba2), rot(bg2), rot(ta2)], dim=-1)
        out = torch.where(apply[:, None, None], out, X)
        out = out.to(x.dtype)
        return out[..., None] if squeeze else out

    def __call__(self, x):
        d = self.draws(x.shape[0], x.shape[1])
        return self.transform(x.cpu(), *d).to(x.device)
