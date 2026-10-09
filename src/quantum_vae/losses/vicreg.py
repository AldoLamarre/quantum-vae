"""VICReg-style regularization terms (Bardes, Ponce, LeCun 2022).

Pure functions: each takes plain [batch, dim] tensor(s) and returns a
scalar loss. No knowledge of any particular model, encoder, or quantum
circuit -- callers decide what tensor to regularize and how to weight
the result.

variance_loss / covariance_loss together discourage representation
collapse: variance keeps each feature spread out across the batch,
covariance discourages features from collapsing onto one shared
direction. invariance_loss is the third VICReg term (consistency
between two views of the same sample) -- included for callers that
have a paired-view setup; unused by callers that don't.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def variance_loss(x: torch.Tensor, target_std: float, eps: float = 1e-4) -> torch.Tensor:
    """Hinge loss pushing each feature's batch-wise std toward target_std.

    x: [batch, dim]. Returns 0.0 for batch_size <= 1, where a std is not
    meaningful.
    """
    if x.shape[0] <= 1:
        return x.new_zeros(())
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalizes off-diagonal batch covariance between features of x.

    x: [batch, dim]. Discourages features from carrying redundant
    information (collapsing onto a shared direction) even when each
    individually satisfies a variance target. Returns 0.0 for
    batch_size <= 1.
    """
    batch_size, dim = x.shape
    if batch_size <= 1:
        return x.new_zeros(())
    centered = x - x.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (batch_size - 1)
    off_diag_sq_sum = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq_sum / dim


def invariance_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Mean squared error between two views of the same batch of samples.

    Ties the representation to sample identity -- the role reconstruction
    loss already plays in a VAE (unused there); needed by any consumer
    without an existing content-anchoring signal (e.g. a paired-view
    self-supervised setup).
    """
    return F.mse_loss(x1, x2)
