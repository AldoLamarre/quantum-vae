"""QuantumVAE with a neutral-atom (Rydberg) pulse-level ansatz.

Mirrors quantum_vae_datareupload.py on the VAE side.

Physical picture
-----------------
    H(t) = H_interaction + H_drive(t) + H_local(x)

    H_interaction = sum_{i<j} C6/r_ij^6 * n_i * n_j     (fixed, from geometry)
    H_drive(t)    = Omega0/2 * sum_i sigma_x_i - Delta0 * sum_i n_i
    H_local(x)    = sum_i V_i(x) * n_i,  V_i(x) = V0_i * (1 + lam * x_i)

H_local's functional form is the standard linearization of the Rydberg
van der Waals interaction under a small positional perturbation delta_r
(V(delta_r) = V0 * (1 - (6/r0) * delta_r), V0 = C6/r0^6): the
data-encoding sensitivity lam plays the role the fixed -6/r0 derivative
plays, made trainable instead of fixed by geometry.

Parameter tiers:
    - Fixed hardware constants (NeutralAtomDeviceConfig): n_atoms,
      register_um, r0_um, C6, evolution_time_us, n_segments. Set once,
      never trained, never touched by the classical encoder.
    - Trainable physics parameters (NeutralAtomPulseLayer): Omega0_MHz,
      Delta0_MHz, lam. Few, global, learned like ordinary model weights.
    - Per-sample input x (LatentToLocalField's output): computed fresh
      every forward pass. Not a parameter -- this is the model's input.

Hardware scope:
    - encoding="local_detuning" (Path A) only. Path B (data modulating
      interatomic distance / interaction strength instead of a per-atom
      field) is deliberately unimplemented -- its per-atom -> per-pair
      combination rule has not been validated against the target
      platform's native per-pair formulation.
    - n_data_injections=1 only, a hardware constraint rather than a
      placeholder: local-detuning capability on the target platform
      fixes its spatial pattern for the entire program -- only the
      shared time-envelope can vary -- so a genuinely different
      per-segment re-uploading pattern is not physically realizable.

Implementation notes:
    - PennyLane's pulse-level ODE solver only supports interface="jax",
      not "torch". This module builds the circuit as a JAX qnode and
      bridges it into the surrounding torch model with a
      torch.autograd.Function using persistent jax.jit-compiled
      forward/backward functions (see _NeutralAtomPulseFunction), handed
      off via DLPack (zero-copy on-device when both frameworks share a
      device).
    - Runs in float32, not float64. Consumer GPUs throttle FP64
      throughput relative to FP32 by a large factor (market
      segmentation, not a technical limit), which makes float64 a poor
      default for a pulse ODE solve meant to run on GPU.
    - Requires jax<0.11,jaxlib<0.11 (PennyLane 0.45.1 still calls
      jax.core.is_concrete, removed in jax 0.11 -- see CLAUDE_SETUP.md).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import Callable, List, Literal, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn

import pennylane as qml
import jax
import jax.numpy as jnp

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
    # "expectation": n_atoms real values, one <Z_i> per atom -- equivalent
    # to "correlators" with correlator_order=1, kept as a separate name for
    # readability and backward compatibility.
    # "correlators": all Z-operator products up to correlator_order, i.e.
    # every <Z_i>, <Z_iZ_j> for order=2, plus triples for order=3, etc.
    # Polynomial in n_atoms (sum_{k=1}^{order} C(n_atoms, k)), not
    # exponential -- a deliberate middle ground between "expectation" and
    # "probability". Same underlying single-shot Z-basis measurements as
    # both; correlators are just products of those same per-shot +-1
    # outcomes, no extra hardware capability needed.
    # "probability": 2**n_atoms values, the full computational-basis
    # distribution. Same underlying shots as the above -- richer readout
    # for free at fixed n_atoms, but scales exponentially, not a substitute
    # for more atoms/clusters at scale (2**24 is not a usable linear-layer
    # width).
    measurement_kind: Literal["expectation", "probability", "correlators"] = "expectation"
    # Only read when measurement_kind="correlators". Must be between 1 and
    # n_atoms inclusive (order 1 = plain expectation values; order=n_atoms
    # is the highest-order correlator possible for this register).
    correlator_order: int = 1
    # Number of independent, non-interacting atom clusters -- a tensor
    # product of n_clusters separate n_atoms-atom registers, all evolving
    # under the SAME shared Omega0_MHz/Delta0_MHz/lam (this is what "one
    # physical register, shared global control fields" forces physically:
    # there is only one Omega(t)/Delta(t)/lam(t) hitting the entire
    # register). Only x (the per-atom local pattern) differs per cluster.
    # Classically this costs O(n_clusters) to simulate, not O(4^(n_atoms*
    # n_clusters)) -- clusters never share a joint state vector. Default 1
    # reproduces every existing config's exact behavior.
    n_clusters: int = 1
    # "global": each cluster's x comes from an unrestricted linear
    # projection of the ENTIRE flattened latent (today's LatentToLocalField,
    # just widened and repeated per cluster) -- no assumption that the
    # latent has spatial structure. "local": partition the latent
    # spatially and route each region to its own cluster, exploiting that
    # e.g. a CIFAR latent is genuinely 4x8x8, not just a flat vector --
    # reserved for later, deliberately unbuilt (NotImplementedError),
    # since it needs LatentToLocalField to become spatially-aware rather
    # than a plain linear layer.
    cluster_routing: Literal["global", "local"] = "global"

    def __post_init__(self):
        if self.measurement_kind not in ("expectation", "probability", "correlators"):
            raise ValueError(
                f"measurement_kind='{self.measurement_kind}' is not "
                "supported. Use 'expectation', 'probability', or 'correlators'."
            )
        if self.measurement_kind == "correlators" and not (1 <= self.correlator_order <= self.n_atoms):
            raise ValueError(
                f"correlator_order={self.correlator_order} must be between "
                f"1 and n_atoms={self.n_atoms} inclusive."
            )
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters={self.n_clusters} must be >= 1.")
        if self.cluster_routing not in ("global", "local"):
            raise ValueError(
                f"cluster_routing='{self.cluster_routing}' is not "
                "supported. Use 'global' or 'local'."
            )
        if self.cluster_routing == "local":
            raise NotImplementedError(
                "cluster_routing='local' (spatially partitioning the "
                "latent across clusters) is not implemented yet -- "
                "deliberately deferred until basic clustering "
                "(cluster_routing='global') is validated. Use 'global' "
                "for now."
            )
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
        measurement_kind: Literal["expectation", "probability", "correlators"] = "expectation",
        correlator_order: int = 1,
        n_clusters: int = 1,
        cluster_routing: Literal["global", "local"] = "global",
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
            measurement_kind=measurement_kind,
            correlator_order=correlator_order,
            n_clusters=n_clusters,
            cluster_routing=cluster_routing,
        )


class _NeutralAtomPulseFunction(torch.autograd.Function):
    """Bridges a batched, JAX-differentiable pulse qnode into torch autograd.

    Forward and backward each call a separate, persistent jax.jit-compiled
    function (see NeutralAtomPulseLayer._jit_forward/_jit_backward) so the
    compiled executable is reused across calls rather than retraced every
    time. Backward recomputes the primal internally via jax.vjp before
    applying the cotangent -- a standard tradeoff that keeps both
    directions covered by the same persistent jit cache.

    Device handoff uses DLPack via the standard __dlpack__/__dlpack_device__
    protocol (jax.numpy.from_dlpack / torch.from_dlpack), which shares the
    underlying buffer directly between torch and jax when both tensors
    live on the same device -- no host-RAM round trip.

    Runs in float32. Consumer GPUs throttle FP64 throughput heavily
    relative to FP32, making float64 a poor choice for this workload;
    jax's default ODE-solver tolerance (tuned for float64) has been
    checked empirically and does not cause instability at float32
    precision. Callers are expected to hand this float32 tensors already
    (NeutralAtomPulseLayer.forward enforces this); this class asserts
    rather than silently casting, since an implicit cast here would
    itself require a copy and defeat the point of the zero-copy path.
    """

    @staticmethod
    def forward(ctx, x, lam, Omega0_MHz, Delta0_MHz, jit_forward, jit_backward):
        ctx.x_device, ctx.x_dtype = x.device, x.dtype
        ctx.lam_device, ctx.lam_dtype = lam.device, lam.dtype
        ctx.Omega0_device, ctx.Omega0_dtype = Omega0_MHz.device, Omega0_MHz.dtype
        ctx.Delta0_device, ctx.Delta0_dtype = Delta0_MHz.device, Delta0_MHz.dtype

        for name, t in (("x", x), ("lam", lam), ("Omega0_MHz", Omega0_MHz), ("Delta0_MHz", Delta0_MHz)):
            if t.dtype != torch.float32:
                raise TypeError(
                    f"_NeutralAtomPulseFunction requires float32 inputs (float64 "
                    f"is dramatically slower on consumer GPUs -- see class "
                    f"docstring), got {name}.dtype={t.dtype}. Cast before "
                    f"calling, e.g. tensor.to(dtype=torch.float32) -- an "
                    "implicit cast here would itself require a copy and defeat "
                    "the point of the DLPack zero-copy path."
                )

        x_jax = jnp.from_dlpack(x.detach().contiguous())
        lam_jax = jnp.from_dlpack(lam.detach().contiguous())
        Omega0_jax = jnp.from_dlpack(Omega0_MHz.detach().contiguous())
        Delta0_jax = jnp.from_dlpack(Delta0_MHz.detach().contiguous())

        # Stashed as plain ctx attributes (not save_for_backward, since
        # these are jax arrays rather than torch tensors) so backward can
        # recompute the primal via jit_backward.
        ctx.jit_backward = jit_backward
        ctx.x_jax, ctx.lam_jax, ctx.Omega0_jax, ctx.Delta0_jax = x_jax, lam_jax, Omega0_jax, Delta0_jax

        out_jax = jit_forward(x_jax, lam_jax, Omega0_jax, Delta0_jax)

        out_torch = torch.from_dlpack(out_jax)
        return out_torch.to(device=ctx.x_device, dtype=ctx.x_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.dtype != torch.float32:
            grad_output = grad_output.to(dtype=torch.float32)
        grad_jax = jnp.from_dlpack(grad_output.detach().contiguous())

        _, (d_x, d_lam, d_Omega0, d_Delta0) = ctx.jit_backward(
            ctx.x_jax, ctx.lam_jax, ctx.Omega0_jax, ctx.Delta0_jax, grad_jax
        )

        d_x_t = torch.from_dlpack(d_x).to(device=ctx.x_device, dtype=ctx.x_dtype)
        d_lam_t = torch.from_dlpack(d_lam).to(device=ctx.lam_device, dtype=ctx.lam_dtype)
        d_Omega0_t = torch.from_dlpack(d_Omega0).to(device=ctx.Omega0_device, dtype=ctx.Omega0_dtype)
        d_Delta0_t = torch.from_dlpack(d_Delta0).to(device=ctx.Delta0_device, dtype=ctx.Delta0_dtype)
        return d_x_t, d_lam_t, d_Omega0_t, d_Delta0_t, None, None


class LatentToLocalField(nn.Module):
    """Maps VAE latent -> (x, lam) packed into one tensor.

    x: n_atoms * n_clusters values -- the per-atom local-detuning pattern
       for every cluster, "global" routing (cluster_routing="global"):
       an unrestricted linear projection of the entire flattened latent,
       no assumption that the latent has spatial structure. All clusters'
       x columns come from this same linear layer, just different output
       columns -- there is no per-cluster specialization in the mapping
       itself yet ("local" routing, spatially partitioning the latent
       across clusters, is reserved for later).
    lam: n_segments values, a per-segment gate on how strongly x matters
         at each pulse-shaping window (the pseudo-reuploading channel).
         ONE shared lam for every cluster within a sample -- physically
         forced, since all clusters sit under the same shared time-
         envelope (see NeutralAtomDeviceConfig.n_clusters).

    Returned as ONE concatenated [batch, n_atoms*n_clusters + n_segments]
    tensor, not a tuple -- this keeps the single-tensor-in/single-tensor-out
    contract AnsatzClassifierPipelineBase (shared with the gate-model
    classifier) already assumes; NeutralAtomPulseLayer splits it back
    apart internally. Default implementation is a plain linear layer;
    this is the piece intended to be replaced later (e.g. with symmetric
    attention, or cluster_routing="local"/"attention") without touching
    anything downstream.
    """

    def __init__(self, latent_dim: int, n_atoms: int, n_segments: int, n_clusters: int = 1):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_segments = n_segments
        self.n_clusters = n_clusters
        self.proj = nn.Linear(latent_dim, n_atoms * n_clusters + n_segments)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.proj(z_flat)   # [batch, n_atoms*n_clusters + n_segments]


class NeutralAtomPulseLayer(nn.Module):
    """Trainable neutral-atom pulse program: H_interaction + H_drive + H_local.

    H(t) = H_interaction + H_drive(t) + H_local(x, lam, t)
        H_drive(t) = Omega0(t)/2 * sum_i sigma_x_i - Delta0(t) * sum_i n_i
        H_local(x, lam, t) = sum_i V0_i * (1 + lam(t) * x_i) * n_i

    Omega0(t) and Delta0(t) are piecewise-constant over n_segments
    windows, each with its own independently trained value
    (Omega0_MHz[k], Delta0_MHz[k]) -- pure pulse shaping, shared across
    every sample, no different in kind from your collaborator's own
    slide-12 pulse schedule.

    lam(t) is also piecewise-constant over the same n_segments windows,
    but lam itself comes from the ENCODER (LatentToLocalField's output),
    not from a trainable parameter here -- the same x gets re-exposed to
    a different, per-sample lam_k at each window, structurally the pulse
    analogue of gate-model data re-uploading ("encode -> trainable layer
    -> encode -> trainable layer -> ..."), except what varies per
    repetition is the (per-sample) weight on x, not x itself. x's
    spatial pattern across atoms never changes -- only its shared,
    time-varying, per-sample scale does, which is what keeps this legal
    under local detuning's hardware constraint (spatial pattern fixed
    for the whole program, only the shared envelope may vary in time).

    Trainable: Omega0_MHz, Delta0_MHz (each n_segments-length vectors).
    NOT trainable here: lam -- it's an encoder output, passed into
    forward() alongside x, not a qlayer parameter.
    Not trainable: device (NeutralAtomDeviceConfig, tier-1 constants).

    Named `qlayer` when attached to the VAE (mirroring
    QuantumVAEDataReupload) so AnsatzVAEBase.quantum_trainable_parameters
    picks up Omega0_MHz/Delta0_MHz automatically.
    """

    def __init__(self, device: NeutralAtomDeviceConfig):
        super().__init__()
        self.device_cfg = device
        self.wires = list(range(device.n_atoms))
        self.n_atoms = device.n_atoms
        self.n_segments = device.n_segments
        self.n_clusters = device.n_clusters

        # V0_i = C6 / r0^6 per atom -- fixed, from geometry, not trained.
        V0 = device.C6 / (device.r0_um ** 6)
        self.register_buffer("V0", torch.full((device.n_atoms,), float(V0), dtype=torch.float32))

        # Trainable physics parameters (tier 2): per-segment pulse shape,
        # shared across every sample. lam is NOT here -- see class docstring.
        self.Omega0_MHz = nn.Parameter(torch.ones(device.n_segments, dtype=torch.float32))
        self.Delta0_MHz = nn.Parameter(torch.zeros(device.n_segments, dtype=torch.float32))

        self._qnode = self._build_qnode()
        # x and lam are both batched (one pair per sample); Omega0/Delta0
        # are shared, the same pulse shape applied to every sample in the batch.
        self._batched_qnode = jax.vmap(self._qnode, in_axes=(0, 0, None, None))

        # Persistent, jit-compiled forward/backward, built once and never
        # recreated so the compiled XLA executable is reused across calls.
        # jax.vjp's returned closure is a fresh Python object every call
        # and is not cached across calls even under jax.jit (jit's cache
        # keys on the function object itself). Wrapping the entire
        # forward-then-differentiate computation in one persistent
        # jax.jit avoids this: backward recomputes the primal internally,
        # but that recomputation is itself compiled once and reused.
        self._jit_forward = jax.jit(self._batched_qnode)

        def _forward_and_vjp(x, lam, Omega0_MHz, Delta0_MHz, cotangent):
            out, vjp_fn = jax.vjp(self._batched_qnode, x, lam, Omega0_MHz, Delta0_MHz)
            grads = vjp_fn(cotangent)
            return out, grads

        self._jit_backward = jax.jit(_forward_and_vjp)

    def _build_qnode(self) -> Callable:
        """Assemble H_interaction + H_drive + H_local exactly as documented
        in the class docstring -- no additional structure hidden here.

        Builds exactly ONE cluster's circuit. Clustering (n_clusters > 1)
        never appears in this method at all: every cluster is an
        identical replica of this same single-cluster circuit, and
        forward() below routes multiple clusters through the SAME
        jax.vmap by flattening (batch, cluster) into one leading axis
        before calling it -- confirmed empirically that nesting a second
        vmap here instead (one over clusters, one over batch) breaks
        gradients through the closure-captured lam (a real JAX/PennyLane
        interaction quirk with this ODE-based ParametrizedEvolution
        backend, not a design choice), so this method's job is only ever
        "build one cluster's physics," never "know how many clusters
        exist."
        """
        n_atoms = self.device_cfg.n_atoms
        wires = self.wires
        T = self.device_cfg.evolution_time_us
        V0_np = np.asarray(self.V0.numpy(), dtype=np.float32)
        measurement_kind = self.device_cfg.measurement_kind
        correlator_order = 1 if measurement_kind == "expectation" else self.device_cfg.correlator_order

        H_interaction = qml.pulse.rydberg_interaction(
            list(self.device_cfg.register_um), wires=wires, interaction_coeff=self.device_cfg.C6
        )

        def amp_fn(p, t):
            return qml.pulse.pwc((0, T))(p, t)   # piecewise-constant Omega0 over n_segments windows

        def det_fn(p, t):
            return qml.pulse.pwc((0, T))(p, t)   # piecewise-constant Delta0 over n_segments windows

        H_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0.0, detuning=det_fn, wires=wires)

        dev = qml.device("default.qubit", wires=n_atoms)

        def _correlator_ops(order: int) -> list:
            # All Z-operator products up to `order`: every <Z_i> (order 1),
            # plus every <Z_iZ_j> pair if order>=2, plus triples if
            # order>=3, etc. "expectation" is just this at order=1 -- same
            # ops, same ordering, so it's a genuine special case of this
            # function, not a separately-maintained code path.
            ops = []
            for k in range(1, order + 1):
                for combo in itertools.combinations(wires, k):
                    op = qml.PauliZ(combo[0])
                    for w in combo[1:]:
                        op = op @ qml.PauliZ(w)
                    ops.append(op)
            return ops

        @qml.qnode(dev, interface="jax")
        def circuit(x, lam, Omega0_MHz, Delta0_MHz):
            # H_local's per-atom coefficient: V0_i * (1 + lam(t) * x_i).
            # lam is captured directly from this function's own argument
            # (a traced JAX value, differentiable exactly like any other),
            # not threaded through qml.evolve's params list -- that list
            # requires every entry to collapse to one homogeneous array,
            # and (x_i, lam) would mix a scalar with an n_segments-length
            # vector in one slot, which PennyLane rejects. x_i alone
            # still goes through qml.evolve's params normally.
            def make_local_coeff(i):
                def f(p, t):
                    lam_t = qml.pulse.pwc((0, T))(lam, t)
                    return V0_np[i] * (1.0 + lam_t * p)
                return f

            local_coeffs = [make_local_coeff(i) for i in range(n_atoms)]
            local_ops = [qml.PauliZ(w) for w in wires]
            H_local = qml.dot(local_coeffs, local_ops)
            H = H_interaction + H_drive + H_local

            local_params = tuple(x[i] for i in range(n_atoms))
            qml.evolve(H)((Omega0_MHz, Delta0_MHz) + local_params, t=T)
            if measurement_kind == "probability":
                # Full computational-basis distribution, 2**n_atoms values.
                # Hardware-native: the same single-shot Z-basis fluorescence
                # detection as the expectation-value readout, just reporting
                # the full outcome histogram instead of only per-atom means.
                return qml.probs(wires=wires)
            # "expectation" (order=1) or "correlators" (order>=1): all
            # Z-operator products up to correlator_order.
            return [qml.expval(op) for op in _correlator_ops(correlator_order)]

        if measurement_kind == "probability":
            def circuit_stacked(x, lam, Omega0_MHz, Delta0_MHz):
                return circuit(x, lam, Omega0_MHz, Delta0_MHz)
        else:
            def circuit_stacked(x, lam, Omega0_MHz, Delta0_MHz):
                return jnp.stack(circuit(x, lam, Omega0_MHz, Delta0_MHz))

        return circuit_stacked

    def _per_cluster_measurement_dim(self) -> int:
        if self.device_cfg.measurement_kind == "probability":
            return 2 ** self.device_cfg.n_atoms
        order = 1 if self.device_cfg.measurement_kind == "expectation" else self.device_cfg.correlator_order
        return sum(comb(self.device_cfg.n_atoms, k) for k in range(1, order + 1))

    @property
    def measurement_dim(self) -> int:
        """Total width of this layer's output: n_clusters times whatever
        a single cluster produces (n_atoms for expectation, a polynomial
        sum-of-binomials for correlators, or 2**n_atoms for probability).
        """
        return self.device_cfg.n_clusters * self._per_cluster_measurement_dim()

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """Evolve a batch of (x, lam) pairs under the pulse Hamiltonian,
        across all n_clusters independent registers.

        Args:
            combined: [batch, n_atoms*n_clusters + n_segments] -- x (first
               n_atoms*n_clusters columns, cluster-major: cluster 0's
               n_atoms columns, then cluster 1's, etc.) and lam (last
               n_segments columns, shared across every cluster within a
               sample) packed into one tensor, exactly as
               LatentToLocalField produces. Cast to float32 if not
               already -- a no-op in the common case, since float32 is
               both torch's default and this layer's required dtype.

        Returns:
            [batch, measurement_dim] readout, all clusters concatenated
            in the same cluster-major order as the input. Pauli-Z
            correlators up to correlator_order (measurement_kind=
            "expectation" or "correlators") or LOG-probabilities
            (measurement_kind="probability" -- see the log-transform
            below for why raw probabilities aren't returned directly).
        """
        if combined.dtype != torch.float32:
            original_dtype = combined.dtype
            combined = combined.to(dtype=torch.float32)
        else:
            original_dtype = None

        batch_size = combined.shape[0]
        n_clusters = self.n_clusters
        n_atoms = self.n_atoms

        x = combined[:, : n_atoms * n_clusters].contiguous()
        lam = combined[:, n_atoms * n_clusters :].contiguous()

        # The vmap'd qnode only ever sees a flat collection of
        # independent simulations -- it has no notion of "cluster" at
        # all (see _build_qnode's docstring for why a second, nested
        # vmap over an explicit cluster axis was tried and rejected:
        # it silently breaks gradients through the closure-captured
        # lam). Flattening (batch, cluster) into one leading axis here,
        # entirely in torch before ever crossing into JAX, reuses the
        # exact single-vmap pattern already validated for the
        # non-clustered (n_clusters=1) case. lam is the same for every
        # cluster within a sample, so repeat_interleave duplicates each
        # sample's lam n_clusters times consecutively, matching x's
        # cluster-major flat ordering exactly.
        x_flat = x.reshape(batch_size * n_clusters, n_atoms)
        lam_flat = lam.repeat_interleave(n_clusters, dim=0)

        out_flat = _NeutralAtomPulseFunction.apply(
            x_flat, lam_flat, self.Omega0_MHz, self.Delta0_MHz, self._jit_forward, self._jit_backward
        )
        # [batch*n_clusters, per_cluster_dim] -> [batch, n_clusters*per_cluster_dim],
        # clusters concatenated in the same cluster-major order as the input.
        out = out_flat.reshape(batch_size, n_clusters * out_flat.shape[-1])

        if self.device_cfg.measurement_kind == "probability":
            # Raw probabilities are constrained to a simplex (>=0, sum to
            # 1) -- early in training the vast majority of the
            # 2**n_atoms entries are near-zero and sharply skewed
            # (increasing one entry mechanically forces others down). A
            # freshly-initialized linear layer (project_from_quantum)
            # implicitly assumes roughly zero-centered, well-scaled
            # input; feeding it a heavily-skewed, strictly-positive
            # vector produces poor early gradient flow regardless of how
            # much raw information the vector carries -- confirmed
            # directly: a real trained checkpoint showed gradient
            # attenuating ~113x crossing this exact layer in probability
            # mode, vs. essentially no attenuation in expectation mode
            # (which is naturally spread across [-1, 1]) on an otherwise
            # comparable, working checkpoint. Log-probabilities spread
            # the near-zero entries out over a much wider effective
            # range -- standard practice wherever probabilities feed a
            # downstream linear layer. Small epsilon avoids log(0) =
            # -inf for basis states with negligible amplitude. Applied
            # elementwise, so this is correct regardless of n_clusters --
            # equivalent to log-transforming each cluster's probability
            # vector independently before concatenating.
            out = torch.log(out + 1e-8)
        if original_dtype is not None:
            # Cast back to the caller's original dtype -- this layer's
            # float32 requirement is an implementation detail, not
            # something callers should have to account for.
            out = out.to(dtype=original_dtype)
        return out

    def extra_repr(self) -> str:
        """Human-readable dump for verification against experimental data --
        same symbols as standard physics notation (Omega, Delta), not
        ML-flavored parameter names. lam is not listed here -- it's an
        encoder output (varies per sample), not a qlayer parameter.
        """
        def _fmt_vec(t: torch.Tensor) -> str:
            return "[" + ", ".join(f"{v:.6f}" for v in t.detach().tolist()) + "]"

        return (
            f"n_atoms={self.device_cfg.n_atoms}, "
            f"n_clusters={self.device_cfg.n_clusters}, "
            f"cluster_routing={self.device_cfg.cluster_routing}, "
            f"n_segments={self.device_cfg.n_segments}, "
            f"measurement_kind={self.device_cfg.measurement_kind}"
            + (f"(order={self.device_cfg.correlator_order})" if self.device_cfg.measurement_kind == "correlators" else "")
            + f", measurement_dim={self.measurement_dim}, "
            f"encoding={self.device_cfg.encoding}, "
            f"n_data_injections={self.device_cfg.n_data_injections}, "
            f"r0_um={self.device_cfg.r0_um}, "
            f"evolution_time_us={self.device_cfg.evolution_time_us}, "
            f"Omega0_MHz={_fmt_vec(self.Omega0_MHz)}, "
            f"Delta0_MHz={_fmt_vec(self.Delta0_MHz)}"
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

        self.project_to_quantum = LatentToLocalField(
            latent_dim, self.device_cfg.n_atoms, self.device_cfg.n_segments, self.device_cfg.n_clusters
        ).to(dummy.device)
        self.project_from_quantum = nn.Linear(self.qlayer.measurement_dim, latent_dim).to(dummy.device)

    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Process latent through the neutral-atom pulse program.

        Flow:
            1. Flatten latent z
            2. Project to (x, lam) -- the per-atom pattern and the
               per-segment gate on how strongly it matters at each
               pulse-shaping window (see LatentToLocalField)
            3. Evolve the Rydberg register (qlayer)
            4. Project Pauli-Z expectations/probabilities back to latent
               dimension
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

        combined = to_quantum_layer(z_flat)  # [batch, n_atoms + n_segments] -- (x, lam)

        combined = combined.to(self._quantum_torch_device, dtype=torch.float32)
        readout = self.qlayer(combined)
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
