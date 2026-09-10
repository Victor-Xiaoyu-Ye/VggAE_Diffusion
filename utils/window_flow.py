"""Noise-time flow contract: u=1 Gaussian, u=0 clean; no legacy time aliases."""
from dataclasses import asdict, dataclass
import torch


@dataclass(frozen=True)
class WindowFlow:
    prediction: str = 'x0'
    loss_floor: float = .05
    time_shift: float = 1.

    def __post_init__(self):
        if self.prediction not in ('x0', 'velocity') or not 0 < self.loss_floor <= 1 or self.time_shift <= 0:
            raise ValueError('invalid window flow contract')

    def contract(self):
        return dict(schema='r7-window-flow-v1', path='(1-u)*data+u*noise',
                    time_distribution='logit_normal_0_1', **asdict(self))

    def times(self, n, device):
        u = torch.randn(n, device=device).sigmoid()
        return (self.time_shift*u/(1+(self.time_shift-1)*u)).clamp(1e-5, 1-1e-5)

    def loss(self, output, clean, noise, u):
        if self.prediction == 'velocity':
            return (output.float()-(noise.float()-clean.float())).square().mean()
        scale = u.float().clamp_min(self.loss_floor).view(-1, 1, 1, 1)
        return ((output.float()-clean.float())/scale).square().mean()

    def clean(self, output, noisy, u):
        return output.float() if self.prediction == 'x0' else noisy.float()-u[:, None, None, None]*output.float()

    def velocity(self, output, noisy, u):
        if self.prediction == 'velocity':
            return output.float()
        # The integrator never evaluates u=0. No training-weight clamp here.
        return (noisy.float()-output.float())/u[:, None, None, None]

    @torch.no_grad()
    def sample(self, model, anchor, noise, steps=64, dtype=torch.float32, text=None,
               text_valid=None, method='euler', observer=None):
        if steps < 1 or method not in ('euler', 'heun'):
            raise ValueError('invalid integrator')
        x = noise.float().clone()
        grid = torch.linspace(1, 0, steps+1, device=x.device)
        def evaluate(z, value, index=None):
            u = value.expand(z.shape[0])
            with torch.autocast(z.device.type, dtype=dtype, enabled=dtype != torch.float32):
                out = model(z, u, anchor, text, text_valid)
            if observer is not None and index is not None:
                observer(index, float(value), z, self.clean(out, z, u))
            return self.velocity(out, z, u)
        for i, (left, right) in enumerate(zip(grid[:-1], grid[1:])):
            delta = right-left
            v = evaluate(x, left, i)
            candidate = x+delta*v
            if method == 'heun' and i < steps-1:
                candidate = x+delta*.5*(v+evaluate(candidate, right))
            x = candidate
        if not torch.isfinite(x).all():
            raise RuntimeError('nonfinite free-sampled latents')
        return x
