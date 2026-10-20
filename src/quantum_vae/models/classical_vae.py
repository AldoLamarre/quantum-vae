"""Classical reference VAE: the same AutoencoderKL with no quantum layer.

bottleneck_dim=None: the latent goes straight to the decoder.
bottleneck_dim=k: a linear bottleneck latent_dim -> k -> latent_dim, the
classical counterpart of a quantum layer with k inputs per image.
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import QuantumVAEBase


class ClassicalVAE(QuantumVAEBase):
    def __init__(self, bottleneck_dim: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoding_strategy = "classical"
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_down: Optional[nn.Linear] = None
        self.bottleneck_up: Optional[nn.Linear] = None
        if bottleneck_dim is not None:
            latent_dim = self.latent_dim()
            if not 1 <= bottleneck_dim <= latent_dim:
                raise ValueError(f"bottleneck_dim={bottleneck_dim} must be in [1, {latent_dim}].")
            # Built eagerly so the optimizer sees them.
            self.bottleneck_down = nn.Linear(latent_dim, bottleneck_dim)
            self.bottleneck_up = nn.Linear(bottleneck_dim, latent_dim)

    def latent_dim(self) -> int:
        cfg = self.config
        side = int(cfg.sample_size) // 2 ** (len(cfg.block_out_channels) - 1)
        return int(cfg.latent_channels) * side * side

    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        if self.bottleneck_down is None:
            return z
        return self.bottleneck_up(self.bottleneck_down(z.flatten(1))).reshape(z.shape)

    def get_latent(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if self.bottleneck_down is None:
            return z.flatten(1)
        return self.bottleneck_down(z.flatten(1))


def build_classical_vae(config: dict) -> ClassicalVAE:
    """Build from a config with a 'model' block (AutoencoderKL kwargs) and
    an optional top-level 'bottleneck_dim'."""
    model_kwargs = dict(config.get("model", {}))
    bottleneck = config.get("bottleneck_dim")
    return ClassicalVAE(bottleneck_dim=int(bottleneck) if bottleneck is not None else None, **model_kwargs)
