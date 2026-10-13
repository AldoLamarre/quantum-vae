"""Recompute BatchNorm running statistics from training data (precise BN).

Running statistics accumulated with momentum lag behind the current weights.
Re-estimating them as a cumulative average over a fixed set of batches, with
all weights frozen, makes eval-mode outputs consistent with the trained model.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import extract_input_images


def batchnorm_layers(model: nn.Module) -> List[nn.modules.batchnorm._BatchNorm]:
    """BatchNorm layers that track running statistics."""
    return [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats
    ]


def image_batches(
    dataset: Any,
    batch_size: int,
    collate_fn: Optional[Callable] = None,
    seed: int = 0,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Iterator[torch.Tensor]:
    """Shuffled image batches from a dataset, using a private generator."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for batch in loader:
        images = extract_input_images(batch)
        yield images.to(device) if device is not None else images


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    num_batches: int,
    forward: Callable[[torch.Tensor], Any],
) -> int:
    """Reset BatchNorm statistics and re-estimate them over num_batches batches.

    BatchNorm layers run in train mode with momentum=None (cumulative average);
    all other modules stay in eval mode. Returns the number of batches used.
    """
    layers = batchnorm_layers(model)
    if not layers or num_batches <= 0:
        return 0

    was_training = model.training
    momenta = [m.momentum for m in layers]
    saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in layers]
    model.eval()
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
        m.train()

    used = 0
    try:
        for images in batches:
            if used >= num_batches:
                break
            forward(images)
            used += 1
    finally:
        for m, momentum in zip(layers, momenta):
            m.momentum = momentum
        if used == 0:
            for m, (mean, var, count) in zip(layers, saved):
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(count)
        model.train(was_training)
    return used
