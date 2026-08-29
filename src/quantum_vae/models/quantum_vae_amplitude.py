"""QuantumVAE with Amplitude Encoding strategy.

This variant encodes the latent space as complex quantum amplitudes.
The latent vector z is split into real and imaginary parts,
normalized as quantum state amplitudes, then used for decoding.

Paper reference: [your paper figure/table here]
"""

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from typing import Optional

from .base import QuantumVAEBase


class QuantumVAEAmplitude(QuantumVAEBase):
    """Quantum VAE using amplitude encoding of latent space.
    
    Strategy:
        1. Encode input → latent z via HF AutoencoderKL encoder
        2. Split z into real and complex components
        3. Normalize as quantum state amplitudes
        4. Decode from quantum states via HF decoder
    
    The quantum encoding is performed entirely in the latent space,
    without explicit quantum circuits. Useful for understanding
    quantum-classical hybrid models and amplitude manipulation.
    
    Args:
        *args, **kwargs: Passed to AutoencoderKL parent class
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Amplitude variant.
        
        Inherits full AutoencoderKL from parent.
        No additional quantum components needed (amplitude encoding
        is performed directly in latent space).
        """
        super().__init__(*args, **kwargs)
        self.encoding_strategy = "amplitude"
        self.Tomography = False

    def set_Tomo(self, Tomography: bool = False) -> bool:
        self.Tomography = bool(Tomography)
        return self.Tomography
    
    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Encode latent z as complex quantum amplitudes.
        
        Process:
            1. Flatten latent tensor
            2. Split into real and imaginary halves
            3. Create complex tensor
            4. Normalize as amplitudes (unit norm)
            5. Convert back to real representation for decoder
        
        Args:
            z: Latent tensor from encoder, shape [batch_size, latent_channels, ...]
            
        Returns:
            Quantum-encoded latent tensor, same shape as input z
        """
        # Remember original shape for reconstruction
        old_shape = z.shape
        
        # Flatten to [batch_size, flattened_dim]
        z_flattened = z.reshape(z.size(0), -1)
        
        # Split into real and complex halves
        real_values, complex_values = torch.split(
            z_flattened, 
            z_flattened.size(1) // 2, 
            dim=1
        )
        
        # Create complex tensor: real + i*complex
        complex_tensor = torch.complex(real_values, complex_values)
        
        # Normalize as quantum state amplitudes (unit norm per batch element)
        states = torch.nn.functional.normalize(complex_tensor, dim=1)
        
        if not self.Tomography:
            z_quantum = torch.real(torch.square(torch.abs(states)))
            z_quantum = torch.stack([z_quantum, z_quantum], dim=1)
        else:
            z_quantum = torch.stack([states.real, states.imag], dim=1)
        
        # Reshape back to original latent shape
        return z_quantum.reshape(old_shape)

    def get_latent(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        z_flattened = z.reshape(z.size(0), -1)
        real_values, complex_values = torch.split(z_flattened, z_flattened.size(1) // 2, dim=1)
        complex_tensor = torch.complex(real_values, complex_values)
        return torch.nn.functional.normalize(complex_tensor, dim=1)

