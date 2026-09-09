from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional


import torch


from .base import QuantumVAEBase


class AnsatzVAEBase(QuantumVAEBase):
    """Base class for ansatz-style quantum VAE variants."""

    @abstractmethod
    def construct_circuit(self):
        raise NotImplementedError("Subclass must implement construct_circuit().")

    def quantum_trainable_parameters(self) -> Iterable["torch.nn.Parameter"]:
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "qlayer" in name or "project_to_quantum" in name or "project_from_quantum" in name:
                yield param

    def get_pre_quantum_features(
        self,
        sample: "torch.FloatTensor",
        sample_posterior: bool = True,
        generator: Optional["torch.Generator"] = None,
    ) -> "torch.Tensor":
        """Encode raw input up to (but not including) this backbone's own
        quantum layer.

        This is what classifier pipelines built on top of an AnsatzVAEBase
        backbone should consume -- NOT get_latent(). get_latent() already
        runs process_latent(), which calls self.qlayer internally; a
        classifier pipeline that shares this backbone's exact qlayer
        object (rather than building its own) would therefore double-apply
        the quantum layer if it were fed get_latent()'s output instead of
        this method's output.

        Shared here (rather than duplicated per subclass) because every
        AnsatzVAEBase variant follows the same
        encode -> sample -> project_to_quantum contract -- see
        process_latent() in QuantumVAEDataReupload / QuantumVAENeutralAtom
        for the matching first half of that pipeline.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if getattr(self, "project_to_quantum", None) is None or getattr(self, "project_from_quantum", None) is None:
            self.initialize_projections(sample)
        return self.project_to_quantum(z.flatten(1))
