"""Option B: diffusion directly on the per-qubit AngleEmbedding rotation
angles (the 30-dim `quantum_input` in QuantumVAEDataReupload.process_latent),
rather than the frozen VAE's flat classical latent (diffusion.euclidean).

Implements the tangent-space noising + SU(2) heat-kernel training target
from:
    Jyotirmai Singh, "Lie Group Diffusion Models for Hardware-Aware Quantum
    Circuit Synthesis," arXiv:2606.29636 (2026).
Equation numbers in comments refer to that paper. See diffusion.su2_math
for the underlying quaternion primitives (composition, exp/log maps,
geodesic angle, and the batched numerical decomposition back to (x,y,z)).

Why this needs different math than diffusion.euclidean: each qubit's
rotation is RX(x).RY(y).RZ(z) -- a composed element of SU(2), not an
unconstrained flat vector. Plain per-angle Gaussian noise does not respect
that manifold: empirically verified (see conversation this module came
from) that the map (x,y,z) -> SU(2) has a genuine gimbal-lock singularity
at y = +-pi/2, where two of the three coordinate directions become
parallel -- an isotropic Gaussian in (x,y,z) is provably NOT isotropic on
the group near that configuration. The construction here (tangent-space
noise + exponential map, Eq. 6-9) never uses a global (x,y,z) chart during
noising/training at all, so it has no such singular region.

The training target (Eq. 13) is the heat-kernel-derived SCORE, not the
raw injected noise -- this is the key way this schedule differs from
diffusion.euclidean's plain "predict the noise" DDPM target, and why this
schedule cannot reuse GaussianDiffusionSchedule's loss.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from .base import LatentDenoiserBase, LatentDiffusionScheduleBase
from .su2_math import quat_conj, quat_exp, quat_from_xyz, quat_log, quat_mult, xyz_from_quat


def _heat_kernel_log_deriv(
    phi: torch.Tensor,
    sigma_sq: torch.Tensor,
    n_terms: int = 64,
    small_sigma_sq_threshold: float = 0.3,
    small_phi_threshold: float = 1e-2,
) -> torch.Tensor:
    """d/dphi log K_{sigma^2}(phi) -- the SU(2) heat-kernel log-derivative
    used to build the training target (Eq. 13).

    Two numerically distinct regimes, per the paper's own note that the
    spectral sum (Eq. 11) needs impractically many terms to converge at
    small sigma^2 (the kernel becomes very peaked):

    - sigma_sq >= small_sigma_sq_threshold: evaluate the spectral sum
      (Eq. 11) directly via autograd w.r.t. phi -- converges with a modest
      number of terms in this regime.
    - sigma_sq <  small_sigma_sq_threshold: use the paper's local
      approximation (Eq. 14): 1/phi - cot(phi) - phi/sigma^2, itself split
      further:
        - phi not near 0: evaluate directly.
        - phi near 0 (< small_phi_threshold): (1/phi - cot(phi)) is a
          0/0-type cancellation in floating point (both terms individually
          diverge); replaced by the paper's Taylor series (Eq. 15):
          phi/3 + phi^3/45 + 2*phi^5/945.

    All branches are computed elementwise via torch.where rather than
    Python control flow, so this runs correctly on a batched tensor of
    phi/sigma_sq values (mixed regimes within one batch, one call).
    """
    # --- Eq. 11 branch: spectral sum, differentiated via autograd ---
    # Explicit enable_grad: this needs its own local gradient (of the
    # spectral sum w.r.t. phi) regardless of the caller's autograd context.
    # Callers legitimately compute this loss inside torch.no_grad() during
    # evaluation (see LatentDiffusionTrainer.prediction_step), which would
    # otherwise silently disable the autograd.grad call below -- the exact
    # same failure mode diffusion.su2_math.xyz_from_quat was fixed for.
    with torch.enable_grad():
        phi_grad = phi.detach().clone().requires_grad_(True)
        m = torch.arange(1, n_terms + 1, device=phi.device, dtype=phi.dtype)
        # broadcast m over phi's shape: (..., 1) * (n_terms,) -> (..., n_terms)
        m_phi = m * phi_grad.unsqueeze(-1)
        sigma_sq_b = sigma_sq.unsqueeze(-1)
        coeffs = m * torch.exp(-0.5 * (m**2 - 1) * sigma_sq_b)
        # sin(m*phi)/sin(phi): safe since this branch is only trusted away from
        # phi ~ 0 (the phi-near-0 case is handled by the eq14/eq15 branch below
        # regardless of sigma_sq, via the final torch.where).
        sin_phi = torch.sin(phi_grad).unsqueeze(-1)
        safe_sin_phi = torch.where(sin_phi.abs() > 1e-6, sin_phi, torch.ones_like(sin_phi))
        terms = coeffs * torch.sin(m_phi) / safe_sin_phi
        kernel = terms.sum(-1)
        log_kernel = torch.log(torch.clamp(kernel, min=1e-30))
        (spectral_deriv,) = torch.autograd.grad(log_kernel.sum(), phi_grad, create_graph=False)
    spectral_deriv = spectral_deriv.detach()

    # --- Eq. 14/15 branch: local approximation for small sigma^2 ---
    safe_phi = torch.where(phi.abs() > small_phi_threshold, phi, torch.ones_like(phi))
    cot_phi = torch.cos(safe_phi) / torch.sin(safe_phi)
    inv_minus_cot_direct = 1.0 / safe_phi - cot_phi
    inv_minus_cot_taylor = phi / 3 + phi**3 / 45 + 2 * phi**5 / 945
    inv_minus_cot = torch.where(
        phi.abs() > small_phi_threshold, inv_minus_cot_direct, inv_minus_cot_taylor
    )
    safe_sigma_sq = torch.clamp(sigma_sq, min=1e-12)
    local_deriv = inv_minus_cot - phi / safe_sigma_sq

    return torch.where(sigma_sq > small_sigma_sq_threshold, spectral_deriv, local_deriv)


class AngleSlotDenoiser(LatentDenoiserBase):
    """Per-qubit-slot denoiser: each of the n_qubits noisy rotations is
    treated as one token, processed by a Transformer encoder (mirrors
    arXiv:2606.29636 Section III.E's skeleton-conditioned token denoiser,
    adapted here to a fixed single skeleton -- this circuit does not have
    a discrete skeleton-choice problem, only the continuous per-qubit
    gates, since the entangling CZ pattern is fixed by the architecture).

    Input per slot: the noisy quaternion (4 real numbers). Output per
    slot: a predicted 3D tangent vector (the training target from Eq. 13).
    """

    def __init__(
        self,
        n_qubits: int = 10,
        n_classes: int = 10,
        hidden: int = 256,
        n_timesteps: int = 1000,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__(n_classes=n_classes, n_timesteps=n_timesteps)
        self.n_qubits = n_qubits
        self.hidden = hidden

        self.quat_proj = nn.Linear(4, hidden)
        self.slot_embed = nn.Embedding(n_qubits, hidden)
        self.time_embed = nn.Embedding(n_timesteps, hidden)
        self.class_embed = nn.Embedding(n_classes + 1, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(hidden, 3)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x_noisy: (batch, n_qubits, 4) noisy quaternions.
        t: (batch,) timesteps. y: (batch,) class labels.
        Returns: (batch, n_qubits, 3) predicted tangent vectors.
        """
        batch, n_qubits, _ = x_noisy.shape
        slot_ids = torch.arange(n_qubits, device=x_noisy.device).unsqueeze(0).expand(batch, -1)

        tokens = self.quat_proj(x_noisy)  # (batch, n_qubits, hidden)
        tokens = tokens + self.slot_embed(slot_ids)
        tokens = tokens + self.time_embed(t).unsqueeze(1)
        tokens = tokens + self.class_embed(y).unsqueeze(1)

        encoded = self.transformer(tokens)
        return self.output_head(encoded)  # (batch, n_qubits, 3)


class SU2HeatKernelSchedule(LatentDiffusionScheduleBase):
    """Tangent-space noising (Eq. 6-8) + SU(2) heat-kernel denoising
    target (Eq. 13), applied independently per qubit slot (Eq. 17-21:
    "the conditional density factorises over slots").
    """

    def __init__(self, n_qubits: int = 10, n_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.T = n_timesteps
        self.n_qubits = n_qubits
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.betas = betas
        self.sigma_sq_cumulative = torch.cumsum(betas, dim=0)  # sigma_t^2 = sum_{s=1}^t beta_s (Eq. 9)

    def to(self, device: torch.device) -> "SU2HeatKernelSchedule":
        self.betas = self.betas.to(device)
        self.sigma_sq_cumulative = self.sigma_sq_cumulative.to(device)
        return self

    def sample_noise(self, x0: torch.Tensor) -> torch.Tensor:
        """x0: (batch, n_qubits, 4) quaternions -> noise: (batch, n_qubits,
        3) tangent vectors -- deliberately NOT the same shape as x0, since
        quaternions are 4-dim but their tangent space (su(2)) is 3-dim."""
        tangent_shape = (*x0.shape[:-1], 3)
        return torch.randn(tangent_shape, device=x0.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x0: (batch, n_qubits, 4) clean quaternions. noise: (batch,
        n_qubits, 3) tangent vectors, already scaled by sqrt(beta_t) by
        the caller via the beta schedule -- see Eq. 6:
            U_{t+1} = U_t . exp(sqrt(beta_t) * xi_t)
        Applied as a single cumulative step from x0 using sigma_t (the
        cumulative variance), matching Eq. 8's product-of-increments
        collapsing to one draw from the known cumulative kernel -- exactly
        analogous to how Euclidean DDPM's q_sample uses the closed-form
        cumulative alpha_bar_t rather than literally looping t individual
        steps.
        """
        sigma_t = self.sigma_sq_cumulative[t].sqrt().unsqueeze(-1).unsqueeze(-1)  # (batch,1,1)
        tangent_step = sigma_t * noise
        increment = quat_exp(tangent_step)
        return quat_mult(x0, increment)

    def get_training_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Eq. 13:
            eps_target_t = -sigma_t * d/dphi log K_{sigma_t^2}(phi_t) * (xi_rel_t / phi_t)
        where xi_rel_t = log(U_0^{-1} U_t), phi_t = |xi_rel_t|.
        """
        x_t = self.q_sample(x0, t, noise)
        rel = quat_mult(quat_conj(x0), x_t)
        xi_rel = quat_log(rel)  # (batch, n_qubits, 3)
        phi = torch.linalg.norm(xi_rel, dim=-1)  # (batch, n_qubits)

        sigma_sq_t = self.sigma_sq_cumulative[t].unsqueeze(-1).expand_as(phi)  # (batch, n_qubits)
        sigma_t = sigma_sq_t.sqrt()

        log_deriv = _heat_kernel_log_deriv(phi, sigma_sq_t)  # (batch, n_qubits)
        safe_phi = torch.where(phi > 1e-8, phi, torch.ones_like(phi))
        direction = xi_rel / safe_phi.unsqueeze(-1)
        return -sigma_t.unsqueeze(-1) * log_deriv.unsqueeze(-1) * direction

    @torch.no_grad()
    def sample(
        self,
        model: AngleSlotDenoiser,
        shape: Tuple[int, ...],
        y: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 1.0,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """Reverse Euler-Maruyama sampler on SU(2)^n_qubits (Eq. 26-28).
        shape: (batch, n_qubits) -- output has an extra trailing 4 for the
        quaternion representation, i.e. final shape (batch, n_qubits, 4).

        Starts from a random point on SU(2) per slot (Haar-ish via a random
        tangent vector from identity -- adequate as an initial distribution
        since T reverse steps drive it to the learned distribution
        regardless of the exact starting law, same as Euclidean DDPM
        starting from unit Gaussian).
        """
        batch, n_qubits = shape
        uncond_y = torch.full_like(y, model.unconditional_token)

        init_tangent = torch.randn(batch, n_qubits, 3, device=device) * math.pi
        x = quat_exp(init_tangent)

        for t_int in reversed(range(self.T)):
            t_batch = torch.full((batch,), t_int, device=device, dtype=torch.long)
            beta_t = self.betas[t_int]
            sigma_t = self.sigma_sq_cumulative[t_int].sqrt()

            eps_cond = model(x, t_batch, y)
            if guidance_scale != 1.0:
                eps_uncond = model(x, t_batch, uncond_y)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = eps_cond

            # Eq. 25/27: convert predicted eps back to a score estimate,
            # then a right-invariant Euler-Maruyama reverse step.
            score_estimate = -eps_pred / torch.clamp(sigma_t, min=1e-12)
            noise_term = torch.randn_like(x[..., :3]) if t_int > 0 else torch.zeros_like(x[..., :3])
            delta = beta_t * score_estimate + eta * beta_t.sqrt() * noise_term
            x = quat_mult(x, quat_exp(delta))

        return x


def angles_to_quaternion_batch(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Convenience re-export: convert a batch of real (x,y,z) triples
    (e.g. from the frozen VAE's project_to_quantum output, reshaped to
    (batch, n_qubits, 3)) into the quaternion representation this module's
    schedule operates on. See su2_math.quat_from_xyz for the underlying
    composition (RZ(z).RY(y).RX(x), matching this circuit's AngleEmbedding
    order -- NOT the paper's own Eq. 3, which is a single-axis rotation)."""
    return quat_from_xyz(x, y, z)


def quaternion_batch_to_angles(q: torch.Tensor, n_steps: int = 200) -> torch.Tensor:
    """Convenience re-export: the inverse of angles_to_quaternion_batch,
    via the batched differentiable fit in su2_math.xyz_from_quat. Returns
    (..., 3) = stacked (x, y, z)."""
    return xyz_from_quat(q, n_steps=n_steps)
