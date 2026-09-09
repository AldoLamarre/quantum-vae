"""QuantumVAE with a neutral-atom (Rydberg) pulse-level ansatz.

Mirrors quantum_vae_datareupload.py on the VAE side.

Physical picture
-----------------
    H(t) = H_interaction + H_drive(t) + H_local(x)

    H_interaction = sum_{i<j} C6/r_ij^6 * n_i * n_j     (fixed, from geometry)
    H_drive(t)    = Omega0/2 * sum_i sigma_x_i - Delta0 * sum_i n_i
    H_local(x)    = sum_i V_i(x) * n_i,  V_i(x) = V0_i * (1 + lam * x_i)

H_local's functional form comes directly from the collaborating lab's own
linearization of the Rydberg interaction under a small positional
perturbation delta_r (V(delta_r) = V0 * (1 - (6/r0) * delta_r), V0 =
C6/r0^6): the data-encoding sensitivity lam plays the role their own
-6/r0 derivative plays, made trainable instead of fixed by geometry.

Parameter tiers:
    - Fixed hardware constants (NeutralAtomDeviceConfig): n_atoms,
      register_um, r0_um, C6, evolution_time_us, n_segments. Set once,
      never trained, never touched by the classical encoder.
    - Trainable physics parameters (NeutralAtomPulseLayer): Omega0_MHz,
      Delta0_MHz, lam. Few, global, learned like ordinary model weights.
    - Per-sample input x (LatentToLocalField's output): computed fresh
      every forward pass. Not a parameter -- this is the model's input.

Hardware scope, confirmed (not speculative) against the collaborating
lab's experimental-tier access:
    - encoding="local_detuning" (Path A) only. Path B (data modulating
      interatomic distance / interaction strength instead of a per-atom
      field) is deliberately unimplemented -- its per-atom -> per-pair
      combination rule was never validated against the lab's own
      per-pair formulation.
    - n_data_injections=1 only. This is a confirmed hardware limitation:
      the local-detuning capability fixes its spatial pattern for the
      entire program -- only the shared time-envelope can vary -- so a
      genuinely different per-segment re-uploading pattern is not
      physically realizable on the hardware this targets. Single
      injection is final for this device, not a placeholder.

Implementation note: PennyLane's pulse-level ODE solver only supports
interface="jax", not "torch". This module builds the circuit as a JAX
qnode and bridges it into the surrounding torch model with a
torch.autograd.Function using jax.vjp for the backward pass. Requires
jax<0.11,jaxlib<0.11 (PennyLane 0.45.1 still calls jax.core.is_concrete,
removed in jax 0.11 -- see CLAUDE_SETUP.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn

import pennylane as qml
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

has_quantum_deps = True

from .ansatz_vae_base import AnsatzVAEBase


def _build_register(
    n_atoms: int,
    geometry: str = "chain",
    spacing: float = 8.0,
) -> List[List[float]]:
    """Build atom coordinates (in micrometers) for a simple register."""
    if geometry == "chain":
        return [[float(i) * spacing, 0.0] for i in range(n_atoms)]
    if geometry == "grid":
        side = int(np.ceil(np.sqrt(n_atoms)))
        coords = []
        for idx in range(n_atoms):
            row, col = divmod(idx, side)
            coords.append([float(col) * spacing, float(row) * spacing])
        return coords
    raise ValueError(f"Unknown register geometry '{geometry}'. Expected 'chain' or 'grid'.")


@dataclass(frozen=True)
class NeutralAtomDeviceConfig:
    """Fixed hardware constants -- one object per physical register.

    Never trained, never touched by the classical encoder. Set once from
    the actual device geometry.
    """

    n_atoms: int
    register_um: Sequence[Sequence[float]]
    r0_um: float
    C6: float
    evolution_time_us: float
    n_segments: int
    # NOTE: not yet wired to anything. Reserved for future piecewise-constant
    # shaping of the global Omega(t)/Delta(t) drive (standard, baseline
    # hardware capability -- unrelated to data re-uploading, which
    # n_data_injections governs and which is disabled below). Right now
    # Omega0_MHz/Delta0_MHz are single constant scalars for the whole
    # evolution window; n_segments has no effect until that changes.
    n_data_injections: int = 1
    encoding: Literal["local_detuning", "geometry"] = "local_detuning"

    def __post_init__(self):
        if self.encoding != "local_detuning":
            raise NotImplementedError(
                f"encoding='{self.encoding}' is not implemented. The "
                "experimental hardware this project targets uses local "
                "detuning (Path A); geometry-modulated interaction "
                "(Path B) is a simulator-only alternative whose per-atom "
                "-> per-pair combination rule was never validated -- it "
                "is deliberately left unbuilt, not silently approximated."
            )
        if self.n_data_injections != 1:
            raise NotImplementedError(
                f"n_data_injections={self.n_data_injections} is not "
                "supported. This is a confirmed hardware limitation, not "
                "an untested one: the experimental local-detuning "
                "capability fixes its spatial pattern for the entire "
                "program -- only the shared time-envelope can vary -- so "
                "a genuinely different per-segment re-uploading pattern "
                "is not physically realizable on the hardware this "
                "targets. Single injection is final for this device."
            )

    @classmethod
    def from_geometry(
        cls,
        n_atoms: int,
        register_geometry: str = "chain",
        atom_spacing_um: float = 8.0,
        r0_um: Optional[float] = None,
        C6: float = 862690.0,
        evolution_time_us: float = 4.0,
        n_segments: int = 6,
    ) -> "NeutralAtomDeviceConfig":
        """Convenience constructor: build a register from a simple
        chain/grid geometry rather than passing coordinates by hand.
        """
        register = _build_register(n_atoms, geometry=register_geometry, spacing=atom_spacing_um)
        return cls(
            n_atoms=n_atoms,
            register_um=register,
            r0_um=r0_um if r0_um is not None else atom_spacing_um,
            C6=C6,
            evolution_time_us=evolution_time_us,
            n_segments=n_segments,
        )


class _NeutralAtomPulseFunction(torch.autograd.Function):
    """Bridges a batched, JAX-differentiable pulse qnode into torch autograd.

    Forward runs the JAX computation and stashes a jax.vjp closure;
    backward evaluates that closure on the incoming torch gradient. All
    numerics happen in float64 on the JAX side (pulse ODE integration is
    not well behaved in float32) and are cast back to the caller's
    dtype/device.
    """

    @staticmethod
    def forward(ctx, x, Omega0_MHz, Delta0_MHz, lam, batched_qnode):
        ctx.x_device, ctx.x_dtype = x.device, x.dtype
        ctx.Omega0_device, ctx.Omega0_dtype = Omega0_MHz.device, Omega0_MHz.dtype
        ctx.Delta0_device, ctx.Delta0_dtype = Delta0_MHz.device, Delta0_MHz.dtype
        ctx.lam_device, ctx.lam_dtype = lam.device, lam.dtype

        x_jax = jnp.asarray(x.detach().cpu().numpy(), dtype=jnp.float64)
        Omega0_jax = jnp.asarray(Omega0_MHz.detach().cpu().numpy(), dtype=jnp.float64)
        Delta0_jax = jnp.asarray(Delta0_MHz.detach().cpu().numpy(), dtype=jnp.float64)
        lam_jax = jnp.asarray(lam.detach().cpu().numpy(), dtype=jnp.float64)

        out_jax, vjp_fn = jax.vjp(batched_qnode, x_jax, Omega0_jax, Delta0_jax, lam_jax)
        ctx.vjp_fn = vjp_fn

        return torch.from_numpy(np.array(out_jax)).to(device=ctx.x_device, dtype=ctx.x_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_jax = jnp.asarray(grad_output.detach().cpu().numpy(), dtype=jnp.float64)
        d_x, d_Omega0, d_Delta0, d_lam = ctx.vjp_fn(grad_jax)

        d_x_t = torch.from_numpy(np.array(d_x)).to(device=ctx.x_device, dtype=ctx.x_dtype)
        d_Omega0_t = torch.from_numpy(np.array(d_Omega0)).to(device=ctx.Omega0_device, dtype=ctx.Omega0_dtype)
        d_Delta0_t = torch.from_numpy(np.array(d_Delta0)).to(device=ctx.Delta0_device, dtype=ctx.Delta0_dtype)
        d_lam_t = torch.from_numpy(np.array(d_lam)).to(device=ctx.lam_device, dtype=ctx.lam_dtype)
        return d_x_t, d_Omega0_t, d_Delta0_t, d_lam_t, None


class LatentToLocalField(nn.Module):
    """Maps VAE latent -> one local-detuning value (x_i) per atom.

    Default implementation is a plain linear layer. This is the piece
    intended to be replaced later (e.g. with symmetric attention) without
    touching anything downstream -- its contract is fixed: consume the
    flattened latent, produce [batch, n_atoms].
    """

    def __init__(self, latent_dim: int, n_atoms: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, n_atoms)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.proj(z_flat)   # [batch, n_atoms] -- this is x


class NeutralAtomPulseLayer(nn.Module):
    """Trainable neutral-atom pulse program: H_interaction + H_drive + H_local.

    Trainable: Omega0_MHz, Delta0_MHz, lam.
    Not trainable: device (NeutralAtomDeviceConfig, tier-1 constants).
    Input to forward(): x -- the per-atom local field, NOT a parameter.

    Named `qlayer` when attached to the VAE (mirroring
    QuantumVAEDataReupload) so AnsatzVAEBase.quantum_trainable_parameters
    picks up Omega0_MHz/Delta0_MHz/lam automatically.
    """

    def __init__(self, device: NeutralAtomDeviceConfig):
        super().__init__()
        self.device_cfg = device
        self.wires = list(range(device.n_atoms))

        # V0_i = C6 / r0^6 per atom -- fixed, from geometry, not trained.
        V0 = device.C6 / (device.r0_um ** 6)
        self.register_buffer("V0", torch.full((device.n_atoms,), float(V0), dtype=torch.float64))

        # Trainable physics parameters (tier 2).
        self.Omega0_MHz = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.Delta0_MHz = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.lam = nn.Parameter(torch.tensor(1e-4, dtype=torch.float64))

        self._qnode = self._build_qnode()
        self._batched_qnode = jax.vmap(self._qnode, in_axes=(0, None, None, None))

    def _build_qnode(self) -> Callable:
        """Assemble H_interaction + H_drive + H_local exactly as documented
        in the module docstring -- no additional structure hidden here.
        """
        n_atoms = self.device_cfg.n_atoms
        wires = self.wires
        T = self.device_cfg.evolution_time_us
        V0_np = np.asarray(self.V0.numpy(), dtype=np.float64)

        H_interaction = qml.pulse.rydberg_interaction(
            list(self.device_cfg.register_um), wires=wires, interaction_coeff=self.device_cfg.C6
        )

        def amp_fn(p, t):
            return p   # constant global Rabi frequency Omega0 for the evolution window

        def det_fn(p, t):
            return p   # constant global detuning Delta0 for the evolution window

        H_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0.0, detuning=det_fn, wires=wires)

        # H_local: V_i(x) = V0_i * (1 + lam * x_i), one coefficient per atom.
        local_coeffs = [(lambda p, t: p) for _ in range(n_atoms)]
        local_ops = [qml.PauliZ(w) for w in wires]
        H_local = qml.dot(local_coeffs, local_ops)

        H = H_interaction + H_drive + H_local

        dev = qml.device("default.qubit", wires=n_atoms)

        @qml.qnode(dev, interface="jax")
        def circuit(x, Omega0_MHz, Delta0_MHz, lam):
            V0_jax = jnp.asarray(V0_np)
            V_local = V0_jax * (1.0 + lam * x)   # V0(1 + lam*x), the slide-28 formula
            local_params = tuple(V_local[i] for i in range(n_atoms))
            qml.evolve(H)((Omega0_MHz, Delta0_MHz) + local_params, t=T)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        def circuit_stacked(x, Omega0_MHz, Delta0_MHz, lam):
            return jnp.stack(circuit(x, Omega0_MHz, Delta0_MHz, lam))

        return circuit_stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evolve a batch of per-atom local-field inputs under the pulse
        Hamiltonian.

        Args:
            x: [batch, n_atoms] local-detuning input (MHz-scale, before
               the V0(1+lam*x) transform, which happens inside the qnode).

        Returns:
            [batch, n_atoms] Pauli-Z expectation values.
        """
        return _NeutralAtomPulseFunction.apply(
            x, self.Omega0_MHz, self.Delta0_MHz, self.lam, self._batched_qnode
        )

    def extra_repr(self) -> str:
        """Human-readable dump for the collaborating lab's verification --
        same symbols as their own notation, not ML-flavored names.
        """
        return (
            f"n_atoms={self.device_cfg.n_atoms}, "
            f"encoding={self.device_cfg.encoding}, "
            f"n_data_injections={self.device_cfg.n_data_injections}, "
            f"r0_um={self.device_cfg.r0_um}, "
            f"evolution_time_us={self.device_cfg.evolution_time_us}, "
            f"Omega0_MHz={self.Omega0_MHz.item():.6f}, "
            f"Delta0_MHz={self.Delta0_MHz.item():.6f}, "
            f"lam={self.lam.item():.6f}"
        )


class QuantumVAENeutralAtom(AnsatzVAEBase):
    """Quantum VAE using a neutral-atom (Rydberg) pulse-level ansatz.

    Strategy:
        1. Encode input -> latent z via HF AutoencoderKL encoder
        2. Project z to one local-detuning value per atom (x)
        3. Evolve the Rydberg register under interaction + global drive +
           V0(1+lam*x) local field for a fixed time window (qlayer)
        4. Project the resulting per-atom Pauli-Z expectations back to
           the latent dimension
        5. Decode from latent via HF decoder

    Args:
        device: NeutralAtomDeviceConfig -- fixed hardware constants.
        *args, **kwargs: Passed to AutoencoderKL parent class.
    """

    def __init__(self, device: NeutralAtomDeviceConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoding_strategy = "neutral_atom_pulse"
        self.device_cfg = device
        self.qlayer = NeutralAtomPulseLayer(device)
        self.n_qubits = device.n_atoms  # alias: consumed by AnsatzClassifierPipelineBase._measurement_dim()
        self._quantum_torch_device = self._infer_quantum_torch_device()

        self.project_to_quantum: Optional[LatentToLocalField] = None
        self.project_from_quantum: Optional[nn.Linear] = None

    def _infer_quantum_torch_device(self) -> torch.device:
        try:
            return next(self.qlayer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.qlayer = self.qlayer.to(self._quantum_torch_device)
        self._quantum_torch_device = self._infer_quantum_torch_device()
        return module

    def initialize_projections(self, dummy: torch.FloatTensor) -> None:
        """Initialize the classical projection layers based on latent
        dimension. Must be called once with a representative batch before
        process_latent() can be used.

        TODO: project_to_quantum/project_from_quantum are hardcoded to
        LatentToLocalField / nn.Linear here. A future symmetric-attention
        option applies equally to this class and QuantumVAEDataReupload --
        worth solving once, generically, when actually designed, rather
        than speculatively parameterizing now.
        """
        with torch.no_grad():
            posterior = self.encode(dummy).latent_dist
            z = posterior.mode()
            latent_dim = z.flatten(1).shape[1]

        self.project_to_quantum = LatentToLocalField(latent_dim, self.device_cfg.n_atoms).to(dummy.device)
        self.project_from_quantum = nn.Linear(self.device_cfg.n_atoms, latent_dim).to(dummy.device)

    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Process latent through the neutral-atom pulse program.

        Flow:
            1. Flatten latent z
            2. Project to one local-detuning value per atom (x)
            3. Evolve the Rydberg register (qlayer)
            4. Project Pauli-Z expectations back to latent dimension
            5. Reshape to original latent shape

        Raises:
            RuntimeError: If initialize_projections() hasn't been called yet.
        """
        if self.project_to_quantum is None or self.project_from_quantum is None:
            raise RuntimeError(
                "Projections not initialized. Call initialize_projections() "
                "before process_latent()"
            )
        to_quantum_layer = self.project_to_quantum
        from_quantum_layer = self.project_from_quantum

        old_shape = z.shape
        z_flat = z.flatten(1)

        x = to_quantum_layer(z_flat)  # [batch, n_atoms]

        x = x.to(self._quantum_torch_device, dtype=torch.float64)
        readout = self.qlayer(x)
        readout = readout.to(from_quantum_layer.weight.device, dtype=from_quantum_layer.weight.dtype)

        z_quantum_flat = from_quantum_layer(readout)
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

    def construct_circuit(self) -> Callable:
        return self.qlayer._qnode
