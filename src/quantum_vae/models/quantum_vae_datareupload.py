"""QuantumVAE with Data Re-uploading Circuit strategy.

This variant uses a parameterized quantum circuit with data re-uploading
to encode the latent space. The VAE encoder produces latent z, which is
then projected to quantum circuit input size, processed through entangling
layers, and projected back to latent space for decoding.

Paper reference: [your paper figure/table here]
"""

from typing import Optional, Union, Tuple, Any, Callable
from numpy import ndarray
import numpy as np

try:
    import torch
    import torch.nn as nn
    import pennylane as qml
    from diffusers.models.autoencoders.vae import DecoderOutput
    has_quantum_deps = True
except ImportError:
    torch = None  # type: ignore
    nn = object  # type: ignore
    qml = None  # type: ignore
    DecoderOutput = object  # type: ignore
    has_quantum_deps = False

from .ansatz_vae_base import AnsatzVAEBase


class QuantumVAEDataReupload(AnsatzVAEBase):
    """Quantum VAE using data re-uploading quantum circuit.
    
    Strategy:
        1. Encode input → latent z via HF AutoencoderKL encoder
        2. Project z to quantum circuit input dimension
        3. Re-upload z through parameterized entangling circuit
        4. Measure quantum observables (e.g., Pauli-Z)
        5. Project quantum output back to latent dimension
        6. Decode from latent via HF decoder
    
    The quantum circuit is a trainable layer with learnable parameters
    optimized during training. This allows learning quantum representations
    of the latent space.
    
    Args:
        n_qubits: Number of qubits in quantum circuit
        n_quantum_layers: Depth of entangling layers
        Tomography: Whether to use full tomography (not implemented here)
        *args, **kwargs: Passed to AutoencoderKL parent class
    """
    
    def __init__(
        self,
        n_qubits: int = 10,
        n_quantum_layers: int = 10,
        Tomography: bool = False,
        *args,
        **kwargs
    ):
        """Initialize Data Re-uploading variant.
        
        Args:
            n_qubits: Number of qubits for quantum circuit
            n_quantum_layers: Depth of StronglyEntanglingLayers
            Tomography: Flag for tomography mode (reserved for future)
            *args, **kwargs: AutoencoderKL initialization arguments
        """
        super().__init__(*args, **kwargs)
        
        self.encoding_strategy = "data_reupload"
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.Tomography = Tomography
        
        # Quantum setup
        self.wires = np.arange(n_qubits)
        self.dev = qml.device('default.qubit', wires=self.wires)
        
        # Quantum circuit weight shape follows StronglyEntanglingLayers:
        # (n_layers, n_wires, 3)
        self.shapeweight = (self.n_quantum_layers + 1, self.n_qubits, 3)
        
        # Quantum circuit will accept flattened input of size n_qubits * 3
        self.shapeinput = (self.n_quantum_layers + 1, self.n_qubits * 3)
        
        # Quantum circuit layer (will be initialized in construct_circuit)
        self.qlayer = qml.qnn.TorchLayer(
            self.construct_circuit(),
            weight_shapes={"weights": self.shapeweight}
        )
        
        # Projection layers (latent ↔ quantum) - initialized on first forward pass
        self.project_to_quantum: Optional[nn.Linear] = None
        self.project_from_quantum: Optional[nn.Linear] = None
    
    def initialize_projections(self, dummy: torch.FloatTensor) -> None:
        """Initialize projection layers based on latent dimension.
        
        Must be called once with a representative batch before process_latent()
        can be used. This determines the latent space dimension.
        
        Args:
            dummy: Dummy input tensor [batch_size, channels, height, width]
                   Used to determine latent dimension via encoder
        """
        with torch.no_grad():
            # Forward through encoder to get latent dimension
            posterior = self.encode(dummy).latent_dist
            z = posterior.mode()
            latent_dim = z.flatten(1).shape[1]
        
        # Project from latent space to quantum circuit input
        # Input: latent_dim → quantum input (n_qubits * 3 rotation angles)
        self.project_to_quantum = nn.Linear(
            latent_dim,
            self.n_qubits * 3  # One set of (θ, φ, ω) per qubit
        ).to(dummy.device)
        
        # Project from quantum circuit output back to latent space
        # Input: n_qubits expectation values → latent_dim
        self.project_from_quantum = nn.Linear(
            self.n_qubits,
            latent_dim
        ).to(dummy.device)
    
    def construct_circuit(self) -> Callable:
        """Construct the parameterized quantum circuit.
        
        Circuit flow:
            1. Amplitude embedding: encode inputs as quantum state
            2. StronglyEntanglingLayers: parameterized entangling gates
            3. Pauli-Z measurement: extract expectation values
        
        Returns:
            Quantum circuit function compatible with qml.qnn.TorchLayer
        """
        @qml.qnode(
            self.dev,
            interface='torch',
            diff_method="backprop",
        )
        def circuit(inputs, weights):
            """Quantum circuit with data re-uploading.
            
            Args:
                inputs: Flattened angle parameters [n_qubits * 3]
                weights: Learnable entangling layer parameters
                
            Returns:
                List of Pauli-Z expectation values, one per qubit
            """
            # Encode input angles as quantum state via amplitude embedding
            qml.AmplitudeEmbedding(
                inputs,
                pad_with=0.0,
                wires=self.wires,
                normalize=True
            )
            
            # Apply parameterized entangling layers
            qml.StronglyEntanglingLayers(weights, wires=self.wires)
            
            # Measure Pauli-Z expectation value for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in self.wires]
        
        return circuit
    
    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Process latent through quantum circuit with projections.
        
        Flow:
            1. Flatten latent z
            2. Project to quantum input dimension
            3. Run through quantum circuit (parameterized qlayer)
            4. Project quantum output back to latent dimension
            5. Reshape to original latent shape
        
        Args:
            z: Latent tensor from encoder, shape [batch_size, latent_channels, ...]
            
        Returns:
            Quantum-processed latent tensor, same shape as input z
            
        Raises:
            RuntimeError: If initialize_projections() hasn't been called yet
        """
        if self.project_to_quantum is None or self.project_from_quantum is None:
            raise RuntimeError(
                'Projections not initialized. Call initialize_projections() '
                'before process_latent()'
            )
        
        # Remember original shape
        old_shape = z.shape
        
        # Flatten latent: [batch_size, latent_channels, ...] → [batch_size, flattened]
        z_flat = z.flatten(1)
        
        # Project to quantum input dimension: [batch_size, n_qubits * 3]
        quantum_input = self.project_to_quantum(z_flat)
        
        # Process through quantum circuit: [batch_size, n_qubits * 3] → [batch_size, n_qubits]
        quantum_output = self.qlayer(quantum_input)
        
        # Project back to latent dimension: [batch_size, n_qubits] → [batch_size, latent_dim]
        z_quantum_flat = self.project_from_quantum(quantum_output)
        
        # Reshape back to original latent shape
        return z_quantum_flat.reshape(old_shape)

    def get_latent(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if self.project_to_quantum is None or self.project_from_quantum is None:
            self.initialize_projections(sample)
        return self.process_latent(z)
