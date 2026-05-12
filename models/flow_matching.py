"""Optimal Transport Conditional Flow Matching (OT-CFM).

Implements the training objective and sampling procedure:
  Forward:   z_t = (1 - t) * z_0 + t * z_1,   z_0 ~ N(0,I), z_1 = data
  Target:    v = z_1 - z_0
  Loss:      MSE(v_pred(z_t, t), v)
  Sampling:  Euler integration of dz/dt = v_theta(z, t) from t=0 to t=1
"""

import torch
import torch.nn.functional as F
import math


def logit_normal_sampling(shape, device, m: float = 0.0, s: float = 1.0):
    """Sample t ~ LogitNormal(m, s) in (0, 1).

    Used by SD3 / FLUX for better coverage of the time domain.
    """
    u = torch.randn(shape, device=device) * s + m
    return torch.sigmoid(u)


class OTCFM:
    """Optimal Transport Conditional Flow Matching trainer / sampler."""

    def __init__(self, model: torch.nn.Module, sigma_min: float = 1e-5):
        """
        Args:
            model: velocity network  v_theta(z_t, t) -> z_1 - z_0
            sigma_min: small noise floor for numerical stability (unused in
                       pure OT-CFM but kept for API compatibility).
        """
        self.model = model
        self.sigma_min = sigma_min

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor = None,
        text_emb: torch.Tensor = None,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """Compute OT-CFM loss.

        Args:
            x1: target data  [B, ...]  (normalised tokens from encoder).
            cond: optional conditioning tensor, forwarded to model.
            text_emb: optional CLIP text embeddings [B, L, text_dim], forwarded to model.
            return_outputs: if True, return loss plus intermediate tensors used
                by decoder-aware auxiliary losses.

        Returns:
            Scalar MSE loss, or a dict with loss/intermediates.
        """
        # Sample noise
        x0 = torch.randn_like(x1)

        # Sample time ~ LogitNormal(0, 1)
        t = logit_normal_sampling((x1.shape[0],), device=x1.device, m=0.0, s=1.0)
        t = t.to(dtype=x1.dtype)
        # Reshape t to broadcast over all dims except batch
        t_expand = t.view(-1, *([1] * (x1.dim() - 1)))

        # Interpolate
        xt = (1 - t_expand) * x0 + t_expand * x1

        # Target velocity
        v_target = x1 - x0

        # Predict
        v_pred = self.model(xt, t, cond=cond, text_emb=text_emb)

        loss = F.mse_loss(v_pred, v_target)

        if not return_outputs:
            return loss

        x1_pred = xt + (1 - t_expand) * v_pred
        return {
            "loss": loss,
            "t": t,
            "xt": xt,
            "v_pred": v_pred,
            "v_target": v_target,
            "x1_pred": x1_pred,
        }

    # ------------------------------------------------------------------
    # Sampling (Euler ODE solver)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 50,
        cond: torch.Tensor = None,
        text_emb: torch.Tensor = None,
        return_trajectory: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample via Euler integration of the flow ODE.

        Args:
            shape: desired output shape, e.g. (B, T, N, D).
            num_steps: number of Euler steps.
            cond: optional conditioning.
            text_emb: optional CLIP text embeddings [B, L, text_dim].
            return_trajectory: if True, return list of (t, z) tuples.
            device: torch device.
            dtype: torch dtype for the noise.

        Returns:
            z_1 tensor of shape `shape`, or trajectory list.
        """
        # Infer model dtype from first parameter
        model_dtype = next(self.model.parameters()).dtype
        z = torch.randn(shape, device=device, dtype=model_dtype)
        dt = torch.tensor(1.0 / num_steps, device=device, dtype=model_dtype)

        trajectory = [] if return_trajectory else None

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((shape[0],), t_val, device=device, dtype=model_dtype)

            if return_trajectory:
                trajectory.append((t_val, z.clone()))

            v = self.model(z, t, cond=cond, text_emb=text_emb)
            z = z + v * dt

        if return_trajectory:
            trajectory.append((1.0, z.clone()))
            return trajectory

        return z

    @torch.no_grad()
    def sample_midpoint(
        self,
        shape: tuple,
        num_steps: int = 50,
        cond: torch.Tensor = None,
        text_emb: torch.Tensor = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample via midpoint (Heun / improved Euler) method.

        More accurate than Euler but twice the cost per step.
        """
        model_dtype = next(self.model.parameters()).dtype
        z = torch.randn(shape, device=device, dtype=model_dtype)
        dt = torch.tensor(1.0 / num_steps, device=device, dtype=model_dtype)

        for i in range(num_steps):
            t = torch.full((shape[0],), i / num_steps, device=device, dtype=model_dtype)

            v1 = self.model(z, t, cond=cond, text_emb=text_emb)

            z_mid = z + 0.5 * dt * v1
            t_mid = torch.full((shape[0],), (i + 0.5) / num_steps, device=device, dtype=model_dtype)

            v2 = self.model(z_mid, t_mid, cond=cond, text_emb=text_emb)

            z = z + dt * v2

        return z
