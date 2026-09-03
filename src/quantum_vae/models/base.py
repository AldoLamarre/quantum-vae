from __future__ import annotations

"""Base class for Quantum VAE variants.

All quantum VAE experiments in this paper inherit from QuantumVAEBase,
differing only in their quantum encoding strategy (process_latent).
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

try:
    import torch
    from diffusers import AutoencoderKL
    from diffusers.models.autoencoders.vae import DecoderOutput
    has_torch_diffusers = True
except ImportError:
    torch = None  # type: ignore
    AutoencoderKL = object  # type: ignore
    DecoderOutput = object  # type: ignore
    has_torch_diffusers = False


class QuantumVAEBase(ABC, AutoencoderKL):
    """Abstract base class for Quantum VAE variants.
    
    Uses HuggingFace AutoencoderKL as backbone. All variants share:
    - Standard VAE forward() logic
    - KL divergence computation
    - Encoding via HF encoder
    - Decoding via HF decoder
    
    Subclasses override:
    - __init__: Configure variant-specific quantum components
    - process_latent: Implement quantum encoding strategy
    """
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize variant-specific quantum components.
        
        Must call super().__init__() to initialize AutoencoderKL.
        """
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]]:
        """Forward pass for Quantum VAE.
        
        Args:
            sample: Input tensor [batch_size, channels, height, width]
            sample_posterior: Whether to sample from posterior (vs mode)
            return_dict: Whether to return DecoderOutput dict
            generator: Optional torch generator for reproducibility
        
        Returns:
            (reconstruction, kl_div, z_quantum) tuple or DecoderOutput
            
        Flow:
            1. Encode: sample → posterior distribution
            2. Sample: posterior → latent z
            3. Quantum encode: z → z_quantum (varies by subclass)
            4. Decode: z_quantum → reconstruction
            5. Compute KL divergence
        """
        x = sample
        
        # Standard VAE encoding: get posterior distribution
        posterior = self.encode(x).latent_dist
        
        # Compute KL divergence
        kl_div = self._compute_kl(posterior)
        
        # Sample latent z from posterior
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        
        # Apply quantum encoding (varies by subclass)
        z_quantum = self.process_latent(z)
        
        # Standard VAE decoding
        reconstruction = self.decode(z_quantum).sample
        
        if not return_dict:
            return (reconstruction, kl_div, z_quantum)
        
        return DecoderOutput(sample=reconstruction), kl_div, z_quantum

    def _compute_kl(self, posterior) -> torch.FloatTensor:
        """Compute KL divergence for standard Gaussian VAE.
        
        KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        
        Args:
            posterior: Distribution object with logvar and mean attributes
            
        Returns:
            Scalar KL divergence
        """
        return -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.var)
    
    @abstractmethod
    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Apply quantum encoding to latent space.
        
        This is the core method that distinguishes quantum variants.
        
        Args:
            z: Latent tensor from encoder [batch_size, latent_dim, ...]
            
        Returns:
            Transformed latent tensor for decoder
            
        Subclass examples:
            - QuantumVAEAmplitude: Convert z to complex amplitudes
            - QuantumVAEDataReupload: Project z through quantum circuit
        """
        raise NotImplementedError("Subclass must implement process_latent()")

    @abstractmethod
    def get_latent(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return classifier-facing latent features for this VAE variant."""
        raise NotImplementedError("Subclass must implement get_latent()")
    
    def initialize_projections(self, dummy: torch.FloatTensor) -> None:
        """Optional: Initialize projection layers (e.g., for circuits).
        
        Called once with a dummy batch to determine latent dimensions.
        Override in subclasses that need projection layers.
        
        Args:
            dummy: Dummy input tensor to determine latent shape
        """
        pass
    
    def construct_circuit(self):
        """Optional: Define quantum circuit.
        
        Override in subclasses that use quantum circuits.
        
        Returns:
            Quantum circuit function or None
        """
        pass
    
    def get_quantum_device(self):
        """Optional: Get quantum device configuration.
        
        Override in subclasses that use quantum simulators/hardware.
        
        Returns:
            Quantum device or None
        """
        return None
