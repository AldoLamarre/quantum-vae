"""QuantumVAE with Data Re-uploading Circuit strategy.

This variant uses a parameterized quantum circuit with data re-uploading
to encode the latent space. The VAE encoder produces latent z, which is
then projected to quantum circuit input size, re-uploaded through repeated
embed/rotate/entangle layers, and projected back to latent space for
decoding.
"""

from typing import Optional, Union, Tuple, Any, Callable
from numpy import ndarray
import numpy as np


import torch
import torch.nn as nn
import pennylane as qml
from diffusers.models.autoencoders.vae import DecoderOutput
has_quantum_deps = True

from .ansatz_vae_base import AnsatzVAEBase


class QuantumVAEDataReupload(AnsatzVAEBase):
    """Quantum VAE using data re-uploading quantum circuit.
    
    Strategy:
        1. Encode input → latent z via HF AutoencoderKL encoder
        2. Project z to quantum circuit input dimension
        3. Re-upload z through the parameterized circuit, once per layer
        4. Measure quantum observables (e.g., Pauli-Z)
        5. Project quantum output back to latent dimension
        6. Decode from latent via HF decoder
    
    The quantum circuit is a trainable layer with learnable parameters
    optimized during training. This allows learning quantum representations
    of the latent space.
    
    Args:
        n_qubits: Number of qubits in quantum circuit
        n_quantum_layers: Number of data re-uploading layer repetitions
            (each layer re-embeds the input and applies a learned Rot +
            CZ entangling step; one additional final reupload is applied
            after the last layer, with no entangling gates)
        *args, **kwargs: Passed to AutoencoderKL parent class
    """
    
    def __init__(
        self,
        n_qubits: int = 10,
        n_quantum_layers: int = 10,
        *args,
        **kwargs
    ):
        """Initialize Data Re-uploading variant.
        
        Args:
            n_qubits: Number of qubits for quantum circuit
            n_quantum_layers: Number of data re-uploading layer repetitions
            *args, **kwargs: AutoencoderKL initialization arguments
        """
        super().__init__(*args, **kwargs)
        
        self.encoding_strategy = "data_reupload"
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        
        # Quantum setup
        self.wires = np.arange(n_qubits)
        self.dev = qml.device('default.qubit', wires=self.wires)
        
        # weights[qubit_index, layer_index, :]
        self.shapeweight = (self.n_qubits, self.n_quantum_layers + 1, 3)
        
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
        """Construct the parameterized data re-uploading quantum circuit.

        Each layer re-embeds a slice of the projected classical input as
        X/Y/Z rotation angles, applies a learned Rot gate per qubit, then
        entangles with alternating CZ patterns. A final reupload (input plus
        a learned offset) is applied without entangling gates.

        Circuit flow (repeated n_quantum_layers times):
            1. AngleEmbedding of inputs as X, Y, Z rotations
            2. Per-qubit learned Rot(alpha, beta, gamma) gate
            3. CZ entangling layer (alternating "double"/"double_odd" pattern)
        Followed by one final reupload (X/Y/Z AngleEmbedding, input + learned
        offset) with no entangling gates, then Pauli-Z expectation readout.

        Returns:
            Quantum circuit function compatible with qml.qnn.TorchLayer
        """
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            """Data re-uploading circuit.

            Args:
                inputs: [batch, n_qubits * 3] angle parameters, re-used every layer
                weights: [n_qubits, n_quantum_layers + 1, 3] learned Rot angles
                    (weights[:, -1, :] is used for the final, non-entangled reupload)

            Returns:
                List of Pauli-Z expectation values, one per qubit
            """
            for layer in range(self.n_quantum_layers):
                x_idx = 0
                qml.AngleEmbedding(inputs[:, x_idx: x_idx + self.n_qubits], wires=self.wires, rotation="X")
                qml.AngleEmbedding(inputs[:, x_idx + self.n_qubits: x_idx + 2 * self.n_qubits], wires=self.wires, rotation="Y")
                qml.AngleEmbedding(inputs[:, x_idx + 2 * self.n_qubits: x_idx + 3 * self.n_qubits], wires=self.wires, rotation="Z")

                for i, wire in enumerate(self.wires):
                    angles = weights[i, layer, :]
                    qml.Rot(*angles, wires=wire)

                # Alternating CZ entanglement: pairs (0,1),(2,3),... on even
                # layers, pairs (1,2),(3,4),... on odd layers.
                if layer % 2 == 0:
                    for i in range(0, len(self.wires) - 1, 2):
                        qml.CZ(wires=[self.wires[i], self.wires[i + 1]])
                else:
                    for i in range(1, len(self.wires) - 1, 2):
                        qml.CZ(wires=[self.wires[i], self.wires[i + 1]])

            # Final reupload: input + learned offset, no entangling gates
            x_idx = 0
            w = weights[:, self.n_quantum_layers]
            angles_x = torch.add(inputs[:, x_idx: x_idx + self.n_qubits], w[:, 0])
            angles_y = torch.add(inputs[:, x_idx + self.n_qubits: x_idx + 2 * self.n_qubits], w[:, 1])
            angles_z = torch.add(inputs[:, x_idx + 2 * self.n_qubits: x_idx + 3 * self.n_qubits], w[:, 2])
            qml.AngleEmbedding(angles_x, wires=self.wires, rotation="X")
            qml.AngleEmbedding(angles_y, wires=self.wires, rotation="Y")
            qml.AngleEmbedding(angles_z, wires=self.wires, rotation="Z")

            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires]

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
