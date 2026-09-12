"""Pure SU(2)/quaternion math -- no torch.nn, no diffusion-specific logic.

Implements the group operations from:
    Jyotirmai Singh, "Lie Group Diffusion Models for Hardware-Aware Quantum
    Circuit Synthesis," arXiv:2606.29636 (2026), Section II.A ("Quaternion
    Representation of Single Qubit Gates").
Equation numbers in comments refer to that paper.

All functions accept and return batched tensors with an arbitrary leading
shape (e.g. [batch, n_qubits]) followed by the quaternion/tangent-vector
dimension -- everything here is written to run as ordinary vectorized
tensor ops (fully GPU-compatible), never a Python loop over individual
elements. Quaternions are (..., 4) real tensors [w, x, y, z]; tangent
vectors are (..., 3) real tensors.
"""
from __future__ import annotations

import torch


def quat_mult(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, batched over any leading shape."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate; for a unit quaternion this is also the inverse."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _elementary_quat(theta: torch.Tensor, axis: int) -> torch.Tensor:
    """q_axis(theta) = (cos(theta/2), ...) with sin(theta/2) in the given
    axis slot (0=x, 1=y, 2=z) -- the paper's Eq. (3) specialized to a
    rotation purely about one coordinate axis."""
    half = theta / 2
    w = torch.cos(half)
    s = torch.sin(half)
    zeros = torch.zeros_like(theta)
    parts = [zeros, zeros, zeros]
    parts[axis] = s
    return torch.stack([w, *parts], dim=-1)


def quat_from_xyz(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Compose RZ(z) . RY(y) . RX(x) -- this circuit's actual per-qubit
    AngleEmbedding order -- into the single SU(2) quaternion that
    diffusion.su2_angles noises/denoises. NOT the same object as the
    paper's own Eq. (3) (which parameterizes one rotation about one axis
    directly); this is the Hamilton-product composition of three such
    elementary rotations, needed because AngleEmbedding applies X, Y, Z
    rotations in sequence rather than a single axis-angle rotation."""
    qx = _elementary_quat(x, axis=0)
    qy = _elementary_quat(y, axis=1)
    qz = _elementary_quat(z, axis=2)
    return quat_mult(qz, quat_mult(qy, qx))


def quat_exp(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """su(2) =~ R^3 tangent vector -> SU(2) group element. Eq. (4):
        exp(v) = (cos|v|, sin(|v|)/|v| * v)   if |v| != 0
                 (1, 0, 0, 0)                  if |v| == 0
    """
    norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    safe_norm = torch.where(norm > eps, norm, torch.ones_like(norm))
    w = torch.cos(norm)
    sinc = torch.where(norm > eps, torch.sin(norm) / safe_norm, torch.ones_like(norm))
    return torch.cat([w, sinc * v], dim=-1)


def quat_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SU(2) group element -> su(2) =~ R^3 tangent vector (principal
    branch). Eq. (5):
        log(q) = (u/|u|) * phi,  phi = arctan2(|u|, w)   if |u| != 0
                 0                                        if |u| == 0
    where q = (w, u), u = (x, y, z).
    """
    w = q[..., 0:1]
    u = q[..., 1:4]
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)
    phi = torch.atan2(u_norm, w)
    safe_norm = torch.where(u_norm > eps, u_norm, torch.ones_like(u_norm))
    direction = torch.where(u_norm > eps, u / safe_norm, torch.zeros_like(u))
    return direction * phi


def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """SU(2) geodesic angle between two quaternions (as points on S^3),
    i.e. |log(q1^{-1} q2)| -- the "phi_t" used throughout the heat-kernel
    construction (Eqs. 9-13). Returns shape (..., ) with the last
    dimension reduced."""
    rel = quat_mult(quat_conj(q1), q2)
    return torch.linalg.norm(quat_log(rel), dim=-1)


def xyz_from_quat(
    q_target: torch.Tensor,
    n_steps: int = 200,
    lr: float = 0.1,
) -> torch.Tensor:
    """Recover (x, y, z) such that RZ(z).RY(y).RX(x) ~= q_target, via a
    single batched differentiable fit (Adam) rather than a closed-form
    Euler-angle extraction. Deliberate choice: closed-form extraction from
    a composed rotation has its own branch-cut/gimbal-lock singularities
    (the same phenomenon this whole module's noising process was built to
    avoid) -- a numerical fit sidesteps that entirely on the way back out,
    the same way the tangent-space construction avoids it on the way in.

    Fully batched: q_target may have any leading shape (..., 4); the fit
    runs as ONE optimizer over all of them simultaneously, not a Python
    loop per element -- this is what makes it GPU-appropriate. Cost is
    fixed at n_steps regardless of how many quaternions are being fit at
    once.
    """
    leading_shape = q_target.shape[:-1]
    device = q_target.device
    angles = torch.zeros(*leading_shape, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([angles], lr=lr)
    q_target_detached = q_target.detach()

    for _ in range(n_steps):
        q_pred = quat_from_xyz(angles[..., 0], angles[..., 1], angles[..., 2])
        # SU(2) double-covers the rotation this circuit implements, so q
        # and -q represent the same physical gate -- fit to whichever sign
        # is closer rather than penalizing a spurious global-phase mismatch.
        diff_pos = torch.sum((q_pred - q_target_detached) ** 2, dim=-1)
        diff_neg = torch.sum((q_pred + q_target_detached) ** 2, dim=-1)
        loss = torch.minimum(diff_pos, diff_neg).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    return angles.detach()
