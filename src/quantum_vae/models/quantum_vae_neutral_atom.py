"""QuantumVAE with a neutral-atom (Rydberg) pulse-level ansatz.

Hamiltonian
-----------
    H(t) = H_interaction + H_drive(t) + H_local(x)

    H_interaction = sum_{i<j} C6/r_ij^6 * n_i * n_j
    H_drive(t)    = Omega(t)/2 * sum_i sigma_x_i - Delta(t) * sum_i n_i
    H_local(x)    = sum_i V_i(x) * n_i,  V_i(x) = V0 * (1 + lam * x_i)

Coefficients are in MHz and multiplied by 2*pi, n_i = (I - Z_i) / 2
(PennyLane's Rydberg conventions), and V0 = C6 / r0^6.

Parameters:
    - Hardware constants (NeutralAtomDeviceConfig): fixed.
    - Pulse shape Omega(t), Delta(t) (NeutralAtomPulseLayer): trainable,
      produced by the encoder, or both; see omega_delta_source.
    - x and lam: produced per sample by the encoder. lam is one scalar per
      sample, constant over the evolution:
      lam = lam_offset + lam_span * tanh(lam_encoder), or
      lam_offset + lam_encoder when lam_span is None.

Omega, Delta and V_i are soft-bounded (sigmoid/tanh) to the limits of the
selected device profile (DEVICE_PROFILES).

Only encoding="local_detuning" with n_data_injections=1 is supported: the
local-detuning pattern is fixed for the whole pulse program.

The time evolution is delegated to the backend selected by
evolution_backend: "pennylane" (ODE solve in JAX, bridged to torch),
"torch_expm", "torch_split" or "qadence". The simulation runs in float32.
The PennyLane backend requires jax<0.11.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from math import comb
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn

has_quantum_deps = True

from .ansatz_vae_base import AnsatzVAEBase
from ..losses.quantum_regularizers import connected_correlators, mean_state_fidelity

# Aquila (QuEra) limits in 2*pi MHz. Kept for backward compatibility;
# DEVICE_PROFILES is the source of truth.
AQUILA_OMEGA_MAX_MHZ = 2.5  # global Rabi frequency: one-sided, >= 0
AQUILA_DELTA_MAX_MHZ = 20.0  # global detuning: symmetric, [-20, 20]
AQUILA_LOCAL_DETUNING_MAX_MHZ = 25.0  # per-atom local field magnitude
AQUILA_OMEGA_SLEW_MAX_MHZ_PER_US = 250.0 / (2.0 * np.pi)  # 250 rad/us^2 (arXiv:2306.11727)
# Braket aquila.properties (rydbergGlobal); confirm against the live device before hardware runs.
AQUILA_DELTA_SLEW_MAX_MHZ_PER_US = 6000.0 / (2.0 * np.pi)  # 6e15 rad/s^2
AQUILA_TIME_STEP_MIN_US = 0.05  # timeDeltaMin 50 ns

# Pasqal WeightedAnalogDevice (pulser-core 1.9.1), converted from rad/us to
# MHz. AnalogDevice has no DMM (local detuning) and is not offered.
# Check against the calibrated QPU specs before a hardware run.
PASQAL_DMM_C6_MHZ_UM6 = 1_948_281.6328864507  # Rydberg level 75
PASQAL_DMM_OMEGA_MAX_MHZ = 2.0  # global Rabi frequency: one-sided, >= 0
PASQAL_DMM_DELTA_MAX_MHZ = 10.0  # global detuning: symmetric, [-10, 10]
PASQAL_DMM_LOCAL_DETUNING_MAX_MHZ = 10.0  # per atom, detuning in [-10, 0]
PASQAL_DMM_TOTAL_LOCAL_DETUNING_MAX_MHZ = 1000.0  # summed over atoms
PASQAL_DMM_MIN_ATOM_DISTANCE_UM = 5.0
PASQAL_DMM_MAX_RADIAL_DISTANCE_UM = 80.0
PASQAL_DMM_MAX_SEQUENCE_US = 6.0
PASQAL_DMM_MAX_ATOMS = 256
PASQAL_DMM_CLOCK_PERIOD_NS = 4

# Sharpness (1/MHz) of the softplus floor for one-sided local fields.
LOCAL_FIELD_FLOOR_SHARPNESS = 4.0


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware constants of one QPU (MHz, um, us), shared by all backends.

    local_field_one_sided: local detuning has one sign only (Pasqal DMM),
    i.e. V_i >= 0 with H_local = +2*pi*V_i*n_i. Limits set to None are not
    validated.
    """

    name: str
    C6: float
    omega_max_mhz: float
    delta_max_mhz: float
    local_detuning_max_mhz: float
    local_field_one_sided: bool
    total_local_detuning_max_mhz: Optional[float] = None
    min_atom_distance_um: Optional[float] = None
    max_radial_distance_um: Optional[float] = None
    max_sequence_us: Optional[float] = None
    max_atoms: Optional[int] = None
    clock_period_ns: Optional[int] = None
    omega_slew_max_mhz_per_us: Optional[float] = None
    delta_slew_max_mhz_per_us: Optional[float] = None
    # Minimum spacing of waveform time points.
    time_step_min_us: Optional[float] = None


DEVICE_PROFILES = {
    "aquila": DeviceProfile(
        name="aquila",
        C6=862690.0,
        omega_max_mhz=AQUILA_OMEGA_MAX_MHZ,
        delta_max_mhz=AQUILA_DELTA_MAX_MHZ,
        local_detuning_max_mhz=AQUILA_LOCAL_DETUNING_MAX_MHZ,
        local_field_one_sided=False,
        omega_slew_max_mhz_per_us=AQUILA_OMEGA_SLEW_MAX_MHZ_PER_US,
        delta_slew_max_mhz_per_us=AQUILA_DELTA_SLEW_MAX_MHZ_PER_US,
        time_step_min_us=AQUILA_TIME_STEP_MIN_US,
    ),
    "pasqal_dmm": DeviceProfile(
        name="pasqal_dmm",
        C6=PASQAL_DMM_C6_MHZ_UM6,
        omega_max_mhz=PASQAL_DMM_OMEGA_MAX_MHZ,
        delta_max_mhz=PASQAL_DMM_DELTA_MAX_MHZ,
        local_detuning_max_mhz=PASQAL_DMM_LOCAL_DETUNING_MAX_MHZ,
        local_field_one_sided=True,
        total_local_detuning_max_mhz=PASQAL_DMM_TOTAL_LOCAL_DETUNING_MAX_MHZ,
        min_atom_distance_um=PASQAL_DMM_MIN_ATOM_DISTANCE_UM,
        max_radial_distance_um=PASQAL_DMM_MAX_RADIAL_DISTANCE_UM,
        max_sequence_us=PASQAL_DMM_MAX_SEQUENCE_US,
        max_atoms=PASQAL_DMM_MAX_ATOMS,
        clock_period_ns=PASQAL_DMM_CLOCK_PERIOD_NS,
    ),
}


def get_device_profile(name: str) -> DeviceProfile:
    try:
        return DEVICE_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"device_profile='{name}' is not supported. Use one of {sorted(DEVICE_PROFILES)}."
        ) from None


def correlator_index_sets(n_atoms: int, order: int) -> List[tuple]:
    """Atom index tuples of all Z-correlators up to `order`, in readout order."""
    return [
        combo
        for k in range(1, order + 1)
        for combo in itertools.combinations(range(n_atoms), k)
    ]


def local_field_mhz(x, lam, V0, bounded: bool, xp, profile: Optional[DeviceProfile] = None):
    """Per-atom local field V0 * (1 + lam * x) in MHz, optionally soft-bounded.

    `xp` is the torch or jax.numpy module. `profile` defaults to Aquila.
    Symmetric profiles: max * tanh(v / max). One-sided profiles apply a
    softplus floor at 0 first, so v is in [0, max].
    """
    profile = profile or DEVICE_PROFILES["aquila"]
    v = V0 * (1.0 + lam * x)
    if bounded:
        vmax = profile.local_detuning_max_mhz
        if profile.local_field_one_sided:
            beta = LOCAL_FIELD_FLOOR_SHARPNESS
            v = xp.logaddexp(v * 0.0, beta * v) / beta
        v = vmax * xp.tanh(v / vmax)
    return v


def windowed_pulse_basis(n_points: int, n_steps: int, duration_us: float, ramp_us: float):
    """Linear interpolation matrix [n_points, n_steps] from control points
    at segment centres to step midpoints (held constant outside the first
    and last centre), and the Omega window [n_steps]: sin^2 rise and fall
    of length ramp_us, 1 in between, 0 at both ends.
    """
    t = (torch.arange(n_steps, dtype=torch.float64) + 0.5) * duration_us / n_steps
    centres = (torch.arange(n_points, dtype=torch.float64) + 0.5) * duration_us / n_points
    interp = torch.zeros(n_points, n_steps, dtype=torch.float64)
    if n_points == 1:
        interp[0] = 1.0
    else:
        pos = ((t - centres[0]) / (centres[1] - centres[0])).clamp(0.0, n_points - 1.0)
        left = pos.floor().long().clamp(max=n_points - 2)
        frac = pos - left
        cols = torch.arange(n_steps)
        interp[left, cols] = 1.0 - frac
        interp[left + 1, cols] += frac
    edge = torch.minimum(t, duration_us - t)
    window = torch.where(edge < ramp_us, torch.sin(0.5 * np.pi * edge / ramp_us) ** 2, torch.ones_like(t))
    return interp.to(torch.float32), window.to(torch.float32)


def _build_register(
    n_atoms: int,
    geometry: str = "chain",
    spacing: float = 8.0,
    shape: Optional[Sequence[int]] = None,
) -> List[List[float]]:
    """Build atom coordinates (in micrometers) for a simple register.

    shape: (rows, cols) of a grid register; default is the smallest square.
    """
    if geometry == "chain":
        return [[float(i) * spacing, 0.0] for i in range(n_atoms)]
    if geometry == "grid" and shape is not None:
        rows, cols = (int(v) for v in shape)
        if rows * cols != n_atoms:
            raise ValueError(f"register_shape={tuple(shape)} does not hold n_atoms={n_atoms}.")
        return [[float(col) * spacing, float(row) * spacing] for row in range(rows) for col in range(cols)]
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
    """Hardware constants and simulation options of one register. Not trained."""

    n_atoms: int
    register_um: Sequence[Sequence[float]]
    r0_um: float
    C6: float
    evolution_time_us: float
    # Control points of Omega(t) and Delta(t); independent waveforms on the device.
    omega_segments: int
    delta_segments: int
    n_data_injections: int = 1
    encoding: Literal["local_detuning", "geometry"] = "local_detuning"
    # "trainable": Omega/Delta are model weights shared by all samples.
    # "encoder": produced per sample by the encoder.
    # "trainable_plus_encoder": shared trainable pulse plus a bounded
    # per-sample correction from the encoder, summed before the bound.
    omega_delta_source: Literal["trainable", "encoder", "trainable_plus_encoder"] = "trainable"
    # "expectation": <Z_i> per atom (correlators of order 1).
    # "correlators": all Z products up to correlator_order.
    # "probability": the 2**n_atoms computational-basis distribution.
    measurement_kind: Literal["expectation", "probability", "correlators"] = "expectation"
    # Used with measurement_kind="correlators"; between 1 and n_atoms.
    correlator_order: int = 1
    # Number of independent, non-interacting clusters of n_atoms atoms.
    # All share Omega(t), Delta(t) and lam; only x differs per cluster.
    n_clusters: int = 1
    # False disables the soft bounds on Omega, Delta and V_i (ablation only).
    bound_pulse_params: bool = True
    # "global": x of every cluster is a linear projection of the whole
    # latent. "local" (spatial routing) is not implemented.
    cluster_routing: Literal["global", "local"] = "global"
    # ODE tolerances of the PennyLane backend; None uses the solver default.
    ode_rtol: Optional[float] = None
    ode_atol: Optional[float] = None
    # "batchnorm_gated": BatchNorm1d on the projected readout, blended in
    # through a learnable gate. None disables it.
    post_quantum_norm: Optional[Literal["batchnorm_gated"]] = "batchnorm_gated"
    # True: skip project_from_quantum and reshape the readout into the
    # latent shape, zero-padded if smaller. measurement_dim > latent_dim
    # raises.
    skip_quantum_projection: bool = False
    # "linear": LatentToLocalField / nn.Linear.
    # "graph": hypergraph message-passing projections
    # (gnn_hypergraph_interface.py). Not available with
    # omega_delta_source="encoder" or measurement_kind="probability";
    # n_clusters > 1 also needs cluster_grid and atom_token_mode="spatial".
    projection_kind: Literal["linear", "graph"] = "linear"
    # Hidden width of the graph projections.
    graph_d_model: int = 24
    # Std of the Gaussian noise added per segment to the trainable pulse at
    # init, to break the symmetry between segments.
    pulse_init_noise_std: float = 0.0
    # Initial global Omega (MHz). Larger clusters scramble at 1 MHz over 4 us.
    omega0_init_mhz: float = 1.0
    # Graph projections only. Adds fixed Fourier position features from the
    # register coordinates to the correlator embeddings, the decoder's atom
    # nodes and project_to_quantum's atom queries.
    use_fourier_hyperedge_pos: bool = False
    # Per-atom width of the Fourier position features; a multiple of 2
    # (chain) or 4 (grid).
    fourier_atom_dim: int = 4
    # lam = lam_offset + lam_span * tanh(lam_encoder) when lam_span is set,
    # otherwise lam_offset + lam_encoder. lam_offset > lam_span keeps lam
    # bounded away from zero.
    lam_offset: float = 1e-4
    lam_span: Optional[float] = None
    # Graph projections only. "attention": learned queries cross-attend the
    # latent grid. "spatial": each atom reads the latent region at its
    # register position.
    atom_token_mode: Literal["attention", "spatial"] = "attention"
    # Graph projections only: residual connection around each hypergraph layer.
    graph_residual: bool = False
    # Initial value of the post-quantum BatchNorm gate (0.0 = identity at init).
    post_quantum_gate_init: float = 0.0
    # Graph projections only. "batchnorm": normalize the x logits across the
    # batch and scale them by a trainable gain before the sigmoid;
    # pre_quantum_gain is its initial value.
    pre_quantum_norm: Optional[str] = None
    pre_quantum_gain: float = 0.4
    # Time-evolution backend. "pennylane": qml.evolve ODE solve (JAX).
    # "torch_expm": exact piecewise-constant evolution via matrix
    # exponentials in torch; ode_rtol/ode_atol are ignored.
    # "qadence": same Hamiltonian via Qadence/pyqtorch, for evaluation.
    # "torch_split": Strang split-step evolution (neutral_atom_split.py),
    # for larger clusters where matrix exponentials are too costly.
    evolution_backend: Literal["pennylane", "torch_expm", "torch_split", "qadence"] = "pennylane"
    # Target hardware: C6 default, pulse bounds and register/timing checks.
    device_profile: Literal["aquila", "pasqal_dmm"] = "aquila"
    # Graph projections only. (rows, cols) placement of clusters on the
    # latent grid, row-major in cluster order. Required when n_clusters > 1.
    cluster_grid: Optional[Tuple[int, int]] = None
    # Graph projections only. "pooled": attention-pool all atoms into one
    # vector. "spatial": each atom writes its own latent patch.
    graph_output: Literal["pooled", "spatial"] = "pooled"
    # Graph projections only, spatial output with atoms sharing latent cells.
    # "resample": bilinear resampling of the atom grid. "merge":
    # concatenate the atoms of each cell and map them with a shared layer.
    graph_cell_write: Literal["resample", "merge"] = "resample"
    # "piecewise": constant segments (requires omega_segments ==
    # delta_segments). "windowed": control points, linear in between, Omega
    # multiplied by a window that is zero at t=0 and t=T, simulated on
    # pulse_steps steps.
    pulse_shape: Literal["piecewise", "windowed"] = "piecewise"
    pulse_steps: int = 80
    # Length of the window's rise and fall (windowed only).
    omega_ramp_us: float = 0.15
    # Bound c on the per-sample correction c * tanh(a)
    # (omega_delta_source="trainable_plus_encoder").
    pulse_delta_scale: float = 2.0
    # Finite-shot readout: correlators estimated from this many sampled
    # bitstrings per cluster (straight-through gradient of the exact value).
    # None: exact expectation values. torch_expm and torch_split only.
    readout_shots: Optional[int] = None
    # torch_split only: Strang substeps per simulation step.
    split_substeps: int = 4

    @property
    def profile(self) -> DeviceProfile:
        return get_device_profile(self.device_profile)

    @property
    def simulation_steps(self) -> int:
        """Number of piecewise-constant steps the evolvers simulate."""
        return self.pulse_steps if self.pulse_shape == "windowed" else self.omega_segments

    def max_omega_segments(self) -> Optional[int]:
        """Largest omega_segments for which the windowed Omega(t) respects the
        profile's slew limit, or None if the profile has no slew limit."""
        profile = self.profile
        if profile.omega_slew_max_mhz_per_us is None:
            return None
        rate = profile.omega_slew_max_mhz_per_us / profile.omega_max_mhz - np.pi / (2.0 * self.omega_ramp_us)
        return int(np.floor(self.evolution_time_us * rate + 1e-9))

    def max_delta_segments(self) -> Optional[int]:
        """Largest delta_segments for which a full-range step between adjacent
        control points respects the profile's Delta slew limit."""
        profile = self.profile
        if profile.delta_slew_max_mhz_per_us is None:
            return None
        rate = profile.delta_slew_max_mhz_per_us / (2.0 * profile.delta_max_mhz)
        return int(np.floor(self.evolution_time_us * rate + 1e-9))

    def max_segments_for_time_step(self) -> Optional[int]:
        """Largest number of control points whose waveform time points
        (0, segment centres, T) respect the profile's minimum time step."""
        step = self.profile.time_step_min_us
        if step is None:
            return None
        return int(np.floor(self.evolution_time_us / (2.0 * step) + 1e-9))

    def _validate_pulse_shape(self) -> None:
        if self.pulse_shape not in ("piecewise", "windowed"):
            raise ValueError(
                f"pulse_shape='{self.pulse_shape}' is not supported. Use 'piecewise' or 'windowed'."
            )
        if self.pulse_delta_scale <= 0.0:
            raise ValueError(f"pulse_delta_scale={self.pulse_delta_scale} must be > 0.")
        for name in ("omega_segments", "delta_segments"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}={getattr(self, name)} must be >= 1.")
        if self.pulse_shape != "windowed":
            if self.omega_segments != self.delta_segments:
                raise ValueError(
                    f"pulse_shape='{self.pulse_shape}' needs omega_segments == delta_segments "
                    f"(got {self.omega_segments} and {self.delta_segments}); use 'windowed'."
                )
            return
        if not self.bound_pulse_params:
            raise ValueError("pulse_shape='windowed' requires bound_pulse_params=True.")
        n_max = max(self.omega_segments, self.delta_segments)
        if self.pulse_steps < n_max:
            raise ValueError(f"pulse_steps={self.pulse_steps} must be >= the number of control points ({n_max}).")
        if not 0.0 < self.omega_ramp_us <= self.evolution_time_us / 2.0:
            raise ValueError(
                f"omega_ramp_us={self.omega_ramp_us} must be in (0, evolution_time_us / 2]."
            )
        if self.evolution_time_us / self.pulse_steps > self.omega_ramp_us / 2.0:
            warnings.warn(
                f"pulse_steps={self.pulse_steps} resolves the {self.omega_ramp_us} us ramp with "
                "fewer than two steps.",
                stacklevel=3,
            )
        max_segments = self.max_omega_segments()
        if max_segments is None:
            warnings.warn(
                f"device_profile='{self.device_profile}' has no Omega slew limit; the windowed "
                "pulse is not checked against the hardware slew rate.",
                stacklevel=3,
            )
        elif self.omega_segments > max_segments:
            raise ValueError(
                f"omega_segments={self.omega_segments} can exceed {self.device_profile}'s Omega slew limit "
                f"with omega_ramp_us={self.omega_ramp_us}; the maximum is {max_segments}."
            )
        max_segments = self.max_delta_segments()
        if max_segments is not None and self.delta_segments > max_segments:
            raise ValueError(
                f"delta_segments={self.delta_segments} can exceed {self.device_profile}'s Delta slew limit; "
                f"the maximum is {max_segments}."
            )
        max_segments = self.max_segments_for_time_step()
        if max_segments is not None:
            for name in ("omega_segments", "delta_segments"):
                if getattr(self, name) > max_segments:
                    raise ValueError(
                        f"{name}={getattr(self, name)} puts waveform points closer than "
                        f"{self.device_profile}'s minimum time step; the maximum is {max_segments}."
                    )

    def __post_init__(self):
        if self.evolution_backend not in ("pennylane", "torch_expm", "torch_split", "qadence"):
            raise ValueError(
                f"evolution_backend='{self.evolution_backend}' is not supported. "
                "Use 'pennylane', 'torch_expm', 'torch_split' or 'qadence'."
            )
        self._validate_against_profile(get_device_profile(self.device_profile))
        if self.readout_shots is not None:
            if int(self.readout_shots) < 1:
                raise ValueError(f"readout_shots={self.readout_shots} must be >= 1.")
            if self.evolution_backend not in ("torch_expm", "torch_split"):
                raise ValueError("readout_shots requires evolution_backend='torch_expm' or 'torch_split'.")
        if not 0.0 < self.omega0_init_mhz < get_device_profile(self.device_profile).omega_max_mhz:
            raise ValueError(f"omega0_init_mhz={self.omega0_init_mhz} must be in (0, omega_max).")
        if self.split_substeps < 1:
            raise ValueError(f"split_substeps={self.split_substeps} must be >= 1.")
        if self.lam_span is not None:
            if self.lam_span <= 0.0:
                raise ValueError(f"lam_span={self.lam_span} must be > 0.")
            if self.lam_offset <= self.lam_span:
                raise ValueError(
                    f"lam_offset={self.lam_offset} must exceed lam_span={self.lam_span} "
                    "so lam cannot reach zero."
                )
        if self.atom_token_mode not in ("attention", "spatial"):
            raise ValueError(
                f"atom_token_mode='{self.atom_token_mode}' is not supported. "
                "Use 'attention' or 'spatial'."
            )
        if self.projection_kind != "graph" and (
            self.atom_token_mode != "attention" or self.graph_residual
        ):
            raise ValueError(
                "atom_token_mode='spatial' and graph_residual=True require projection_kind='graph'."
            )
        if self.pre_quantum_norm not in (None, "batchnorm"):
            raise ValueError(
                f"pre_quantum_norm='{self.pre_quantum_norm}' is not supported. Use None or 'batchnorm'."
            )
        if self.pre_quantum_norm is not None and self.projection_kind != "graph":
            raise ValueError("pre_quantum_norm requires projection_kind='graph'.")
        if self.pre_quantum_gain <= 0.0:
            raise ValueError(f"pre_quantum_gain={self.pre_quantum_gain} must be > 0.")
        if self.omega_delta_source not in ("trainable", "encoder", "trainable_plus_encoder"):
            raise ValueError(
                f"omega_delta_source='{self.omega_delta_source}' is not "
                "supported. Use 'trainable', 'encoder' or 'trainable_plus_encoder'."
            )
        if self.omega_delta_source == "trainable_plus_encoder" and self.projection_kind != "graph":
            raise ValueError("omega_delta_source='trainable_plus_encoder' requires projection_kind='graph'.")
        self._validate_pulse_shape()
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
        if self.projection_kind not in ("linear", "graph"):
            raise ValueError(
                f"projection_kind='{self.projection_kind}' is not "
                "supported. Use 'linear' or 'graph'."
            )
        if self.cluster_grid is not None:
            grid = tuple(int(v) for v in self.cluster_grid)
            if len(grid) != 2 or min(grid) < 1:
                raise ValueError(f"cluster_grid={self.cluster_grid} must be two positive integers.")
            object.__setattr__(self, "cluster_grid", grid)
            if grid[0] * grid[1] != self.n_clusters:
                raise ValueError(f"cluster_grid={grid} does not match n_clusters={self.n_clusters}.")
            if self.projection_kind != "graph":
                raise ValueError("cluster_grid requires projection_kind='graph'.")
        if self.graph_output not in ("pooled", "spatial"):
            raise ValueError(
                f"graph_output='{self.graph_output}' is not supported. Use 'pooled' or 'spatial'."
            )
        if self.graph_output != "pooled" and self.projection_kind != "graph":
            raise ValueError("graph_output='spatial' requires projection_kind='graph'.")
        if self.graph_cell_write not in ("resample", "merge"):
            raise ValueError(
                f"graph_cell_write='{self.graph_cell_write}' is not supported. Use 'resample' or 'merge'."
            )
        if self.graph_cell_write == "merge" and self.graph_output != "spatial":
            raise ValueError("graph_cell_write='merge' requires graph_output='spatial'.")
        if self.projection_kind == "graph":
            if self.n_clusters > 1:
                if self.cluster_grid is None:
                    raise ValueError("projection_kind='graph' with n_clusters > 1 requires cluster_grid.")
                if self.atom_token_mode != "spatial":
                    raise ValueError("projection_kind='graph' with n_clusters > 1 requires atom_token_mode='spatial'.")
            if self.omega_delta_source == "encoder":
                raise NotImplementedError(
                    "projection_kind='graph' supports omega_delta_source="
                    "'trainable' or 'trainable_plus_encoder'."
                )
            if self.measurement_kind == "probability":
                raise ValueError(
                    "projection_kind='graph' does not support "
                    "measurement_kind='probability' -- correlator "
                    "hyperedges are not defined for a probability readout."
                )
        if self.use_fourier_hyperedge_pos and self.projection_kind != "graph":
            raise ValueError(
                "use_fourier_hyperedge_pos=True requires projection_kind='graph'."
            )
        if self.fourier_atom_dim < 1:
            raise ValueError(f"fourier_atom_dim={self.fourier_atom_dim} must be >= 1.")
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

    def _validate_against_profile(self, profile: DeviceProfile) -> None:
        """Raise on registers/timings the hardware cannot run; warn on off-clock segments."""
        n_physical = self.n_atoms * self.n_clusters
        if profile.max_atoms is not None and n_physical > profile.max_atoms:
            raise ValueError(
                f"n_atoms*n_clusters={n_physical} exceeds {profile.name}'s "
                f"max_atoms={profile.max_atoms}."
            )
        pos = np.asarray(self.register_um, dtype=float)
        if profile.min_atom_distance_um is not None and len(pos) > 1:
            diffs = pos[:, None, :] - pos[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)[np.triu_indices(len(pos), k=1)]
            if dists.min() < profile.min_atom_distance_um - 1e-9:
                raise ValueError(
                    f"Minimum atom distance {dists.min():.3f} um is below "
                    f"{profile.name}'s min_atom_distance_um={profile.min_atom_distance_um}."
                )
        if profile.max_radial_distance_um is not None:
            # Measured from the register centroid.
            radial = np.linalg.norm(pos - pos.mean(axis=0), axis=-1).max()
            if radial > profile.max_radial_distance_um + 1e-9:
                raise ValueError(
                    f"Register radius {radial:.3f} um (from centroid) exceeds "
                    f"{profile.name}'s max_radial_distance_um={profile.max_radial_distance_um}."
                )
        if profile.max_sequence_us is not None and self.evolution_time_us > profile.max_sequence_us + 1e-12:
            raise ValueError(
                f"evolution_time_us={self.evolution_time_us} exceeds "
                f"{profile.name}'s max_sequence_us={profile.max_sequence_us}."
            )
        if (
            profile.total_local_detuning_max_mhz is not None
            and n_physical * profile.local_detuning_max_mhz > profile.total_local_detuning_max_mhz
        ):
            raise ValueError(
                f"{n_physical} atoms at up to {profile.local_detuning_max_mhz} MHz local "
                f"detuning each can exceed {profile.name}'s total DMM budget of "
                f"{profile.total_local_detuning_max_mhz} MHz."
            )
        if profile.clock_period_ns is not None:
            seg_ns = self.evolution_time_us * 1000.0 / self.simulation_steps
            if abs(seg_ns / profile.clock_period_ns - round(seg_ns / profile.clock_period_ns)) > 1e-6:
                warnings.warn(
                    f"Segment duration {seg_ns:.3f} ns is not a multiple of {profile.name}'s "
                    f"{profile.clock_period_ns} ns clock period.",
                    stacklevel=3,
                )

    @classmethod
    def from_geometry(
        cls,
        n_atoms: int,
        register_geometry: str = "chain",
        atom_spacing_um: float = 8.0,
        r0_um: Optional[float] = None,
        C6: Optional[float] = None,
        evolution_time_us: float = 4.0,
        omega_segments: int = 6,
        delta_segments: int = 6,
        measurement_kind: Literal["expectation", "probability", "correlators"] = "expectation",
        correlator_order: int = 1,
        n_clusters: int = 1,
        cluster_routing: Literal["global", "local"] = "global",
        omega_delta_source: Literal["trainable", "encoder", "trainable_plus_encoder"] = "trainable",
        bound_pulse_params: bool = True,
        ode_rtol: Optional[float] = None,
        ode_atol: Optional[float] = None,
        post_quantum_norm: Optional[Literal["batchnorm_gated"]] = "batchnorm_gated",
        skip_quantum_projection: bool = False,
        projection_kind: Literal["linear", "graph"] = "linear",
        graph_d_model: int = 24,
        pulse_init_noise_std: float = 0.0,
        omega0_init_mhz: float = 1.0,
        use_fourier_hyperedge_pos: bool = False,
        fourier_atom_dim: int = 4,
        lam_offset: float = 1e-4,
        lam_span: Optional[float] = None,
        atom_token_mode: Literal["attention", "spatial"] = "attention",
        graph_residual: bool = False,
        post_quantum_gate_init: float = 0.0,
        pre_quantum_norm: Optional[str] = None,
        pre_quantum_gain: float = 0.4,
        evolution_backend: Literal["pennylane", "torch_expm", "torch_split", "qadence"] = "pennylane",
        device_profile: Literal["aquila", "pasqal_dmm"] = "aquila",
        cluster_grid: Optional[Sequence[int]] = None,
        graph_output: Literal["pooled", "spatial"] = "pooled",
        graph_cell_write: Literal["resample", "merge"] = "resample",
        pulse_shape: Literal["piecewise", "windowed"] = "piecewise",
        pulse_steps: int = 80,
        omega_ramp_us: float = 0.15,
        pulse_delta_scale: float = 2.0,
        readout_shots: Optional[int] = None,
        split_substeps: int = 4,
        register_shape: Optional[Sequence[int]] = None,
    ) -> "NeutralAtomDeviceConfig":
        """Build the config from a chain or grid geometry.

        C6=None uses the device profile's C6; a different explicit C6 warns.
        """
        profile = get_device_profile(device_profile)
        if C6 is None:
            C6 = profile.C6
        elif not np.isclose(C6, profile.C6, rtol=1e-6):
            warnings.warn(
                f"C6={C6} overrides device_profile='{device_profile}' C6={profile.C6}.",
                stacklevel=2,
            )
        register = _build_register(n_atoms, geometry=register_geometry, spacing=atom_spacing_um, shape=register_shape)
        return cls(
            n_atoms=n_atoms,
            register_um=register,
            r0_um=r0_um if r0_um is not None else atom_spacing_um,
            C6=C6,
            evolution_time_us=evolution_time_us,
            omega_segments=omega_segments,
            delta_segments=delta_segments,
            measurement_kind=measurement_kind,
            correlator_order=correlator_order,
            n_clusters=n_clusters,
            cluster_routing=cluster_routing,
            omega_delta_source=omega_delta_source,
            bound_pulse_params=bound_pulse_params,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            post_quantum_norm=post_quantum_norm,
            skip_quantum_projection=skip_quantum_projection,
            projection_kind=projection_kind,
            graph_d_model=graph_d_model,
            pulse_init_noise_std=pulse_init_noise_std,
            omega0_init_mhz=omega0_init_mhz,
            use_fourier_hyperedge_pos=use_fourier_hyperedge_pos,
            fourier_atom_dim=fourier_atom_dim,
            lam_offset=lam_offset,
            lam_span=lam_span,
            atom_token_mode=atom_token_mode,
            graph_residual=graph_residual,
            post_quantum_gate_init=post_quantum_gate_init,
            pre_quantum_norm=pre_quantum_norm,
            pre_quantum_gain=pre_quantum_gain,
            evolution_backend=evolution_backend,
            device_profile=device_profile,
            cluster_grid=tuple(cluster_grid) if cluster_grid is not None else None,
            graph_output=graph_output,
            graph_cell_write=graph_cell_write,
            pulse_shape=pulse_shape,
            pulse_steps=pulse_steps,
            omega_ramp_us=omega_ramp_us,
            pulse_delta_scale=pulse_delta_scale,
            readout_shots=readout_shots,
            split_substeps=split_substeps,
        )


class _NeutralAtomPulseFunction(torch.autograd.Function):
    """Bridges the batched JAX pulse qnode into torch autograd.

    Forward and backward call persistent jax.jit-compiled functions; backward
    recomputes the primal through jax.vjp. Tensors are exchanged via DLPack
    and must be float32.
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
                    f"_NeutralAtomPulseFunction requires float32 inputs, "
                    f"got {name}.dtype={t.dtype}."
                )

        import jax.numpy as jnp

        x_jax = jnp.from_dlpack(x.detach().contiguous())
        lam_jax = jnp.from_dlpack(lam.detach().contiguous())
        Omega0_jax = jnp.from_dlpack(Omega0_MHz.detach().contiguous())
        Delta0_jax = jnp.from_dlpack(Delta0_MHz.detach().contiguous())

        # jax arrays cannot go through save_for_backward.
        ctx.jit_backward = jit_backward
        ctx.x_jax, ctx.lam_jax, ctx.Omega0_jax, ctx.Delta0_jax = x_jax, lam_jax, Omega0_jax, Delta0_jax

        out_jax = jit_forward(x_jax, lam_jax, Omega0_jax, Delta0_jax)

        out_torch = torch.from_dlpack(out_jax)
        return out_torch.to(device=ctx.x_device, dtype=ctx.x_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        import jax.numpy as jnp

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
    """Linear map from the flattened latent to one packed tensor
    [x, lam(, omega_seg, delta_seg)].

    x: n_atoms * n_clusters local-detuning values, cluster-major.
    lam: one scalar per sample, shared by all clusters.
    omega_seg, delta_seg: omega_segments and delta_segments values, shared by
        all clusters;
        present when omega_delta_source="encoder".
    """

    def __init__(
        self,
        latent_dim: int,
        n_atoms: int,
        omega_segments: int,
        delta_segments: int,
        n_clusters: int = 1,
        omega_delta_source: Literal["trainable", "encoder"] = "trainable",
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.omega_segments = omega_segments
        self.delta_segments = delta_segments
        self.n_clusters = n_clusters
        self.omega_delta_source = omega_delta_source
        out_dim = n_atoms * n_clusters + 1
        if omega_delta_source == "encoder":
            out_dim += omega_segments + delta_segments  # omega_seg, delta_seg
        self.proj = nn.Linear(latent_dim, out_dim)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.proj(z_flat)


class PennyLaneEvolver(nn.Module):
    """Time evolution via PennyLane's pulse ODE solver (JAX), bridged into torch.

    Called as evolver(x, lam, Omega0_MHz, Delta0_MHz) with already-bounded
    pulse values; returns [batch, per_cluster_dim] readout. Holds no
    parameters or buffers, so the layer's state_dict is unaffected.
    """

    def __init__(self, device: NeutralAtomDeviceConfig, V0: torch.Tensor):
        super().__init__()
        self.device_cfg = device
        self.wires = list(range(device.n_atoms))
        self.omega_delta_source = device.omega_delta_source
        self._V0_np = np.asarray(V0.detach().cpu().numpy(), dtype=np.float32)

        import jax

        self.qnode = self._build_qnode()
        # x and lam are batched; Omega/Delta only when encoder-sourced.
        omega_delta_axis = 0 if self.omega_delta_source in ("encoder", "trainable_plus_encoder") else None
        self._batched_qnode = jax.vmap(self.qnode, in_axes=(0, 0, omega_delta_axis, omega_delta_axis))

        # Built once so the compiled executables are reused. The backward
        # function recomputes the primal inside the same jit.
        self._jit_forward = jax.jit(self._batched_qnode)

        def _forward_and_vjp(x, lam, Omega0_MHz, Delta0_MHz, cotangent):
            out, vjp_fn = jax.vjp(self._batched_qnode, x, lam, Omega0_MHz, Delta0_MHz)
            grads = vjp_fn(cotangent)
            return out, grads

        self._jit_backward = jax.jit(_forward_and_vjp)

    def _build_qnode(self) -> Callable:
        """Build the qnode of a single cluster.

        Clusters are folded into the batch axis by the caller; a nested vmap
        over clusters breaks the gradient with respect to lam.
        """
        import jax.numpy as jnp
        import pennylane as qml

        n_atoms = self.device_cfg.n_atoms
        wires = self.wires
        T = self.device_cfg.evolution_time_us
        ode_kwargs = {}
        if self.device_cfg.ode_rtol is not None:
            ode_kwargs["rtol"] = self.device_cfg.ode_rtol
        if self.device_cfg.ode_atol is not None:
            ode_kwargs["atol"] = self.device_cfg.ode_atol
        V0_np = self._V0_np
        bound_pulse_params = self.device_cfg.bound_pulse_params
        measurement_kind = self.device_cfg.measurement_kind
        correlator_order = 1 if measurement_kind == "expectation" else self.device_cfg.correlator_order
        profile = self.device_cfg.profile

        H_interaction = qml.pulse.rydberg_interaction(
            list(self.device_cfg.register_um), wires=wires, interaction_coeff=self.device_cfg.C6
        )

        def amp_fn(p, t):
            return qml.pulse.pwc((0, T))(p, t)   # piecewise-constant Omega0 over the simulation steps

        def det_fn(p, t):
            return qml.pulse.pwc((0, T))(p, t)   # piecewise-constant Delta0 over the simulation steps

        H_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0.0, detuning=det_fn, wires=wires)

        dev = qml.device("default.qubit", wires=n_atoms)

        def _correlator_ops(order: int) -> list:
            ops = []
            for combo in correlator_index_sets(n_atoms, order):
                op = qml.PauliZ(wires[combo[0]])
                for w in combo[1:]:
                    op = op @ qml.PauliZ(wires[w])
                ops.append(op)
            return ops

        @qml.qnode(dev, interface="jax")
        def circuit(x, lam, Omega0_MHz, Delta0_MHz):
            # H_local = sum_i 2*pi * V_i(x) * n_i with n_i = (I - Z_i) / 2. The
            # identity part is a global phase, leaving -pi * V_i on Z_i. lam is
            # captured from the enclosing scope; x_i is an evolve parameter.
            def make_local_coeff(i):
                def f(p, t):
                    v_i = local_field_mhz(p, lam, V0_np[i], bound_pulse_params, jnp, profile)
                    return -np.pi * v_i
                return f

            local_coeffs = [make_local_coeff(i) for i in range(n_atoms)]
            local_ops = [qml.PauliZ(w) for w in wires]
            H_local = qml.dot(local_coeffs, local_ops)
            H = H_interaction + H_drive + H_local

            local_params = tuple(x[i] for i in range(n_atoms))
            qml.evolve(H, **ode_kwargs)((Omega0_MHz, Delta0_MHz) + local_params, t=T)
            if measurement_kind == "probability":
                # Computational-basis distribution, 2**n_atoms values.
                return qml.probs(wires=wires)
            # Z products up to correlator_order.
            return [qml.expval(op) for op in _correlator_ops(correlator_order)]

        if measurement_kind == "probability":
            def circuit_stacked(x, lam, Omega0_MHz, Delta0_MHz):
                return circuit(x, lam, Omega0_MHz, Delta0_MHz)
        else:
            def circuit_stacked(x, lam, Omega0_MHz, Delta0_MHz):
                return jnp.stack(circuit(x, lam, Omega0_MHz, Delta0_MHz))

        return circuit_stacked

    def forward(self, x, lam, Omega0_MHz, Delta0_MHz) -> torch.Tensor:
        return _NeutralAtomPulseFunction.apply(
            x, lam, Omega0_MHz, Delta0_MHz, self._jit_forward, self._jit_backward
        )


def _build_evolver(device: NeutralAtomDeviceConfig, V0: torch.Tensor) -> nn.Module:
    """Select the time-evolution module from device.evolution_backend."""
    if device.evolution_backend == "torch_expm":
        from .neutral_atom_expm import ExpmEvolver

        return ExpmEvolver(device, V0)
    if device.evolution_backend == "torch_split":
        from .neutral_atom_split import SplitStepEvolver

        return SplitStepEvolver(device, V0)
    if device.evolution_backend == "qadence":
        from .neutral_atom_qadence import QadenceEvolver

        return QadenceEvolver(device, V0)
    return PennyLaneEvolver(device, V0)


class NeutralAtomPulseLayer(nn.Module):
    """Neutral-atom pulse program: H_interaction + H_drive(t) + H_local(x, lam).

    Omega(t) and Delta(t) are defined at omega_segments and delta_segments
    control points. With omega_delta_source="trainable" they are parameters of
    this layer
    (Omega0_MHz, Delta0_MHz) shared by all samples; with "encoder" they are
    read per sample from the input; "trainable_plus_encoder" adds a bounded
    per-sample correction to the shared pulse. lam is always an input.

    Raw pulse values are soft-bounded to the device profile's limits.
    """

    def __init__(self, device: NeutralAtomDeviceConfig):
        super().__init__()
        self.device_cfg = device
        self.wires = list(range(device.n_atoms))
        self.n_atoms = device.n_atoms
        self.omega_segments = device.omega_segments
        self.delta_segments = device.delta_segments
        self.n_clusters = device.n_clusters
        self.omega_delta_source = device.omega_delta_source

        # V0 = C6 / r0^6.
        V0 = device.C6 / (device.r0_um ** 6)
        self.register_buffer("V0", torch.full((device.n_atoms,), float(V0), dtype=torch.float32))

        # Fixed offset added to the encoder's lam, so x acts on the circuit at init.
        self.register_buffer("lam_pulseqpu", torch.tensor(device.lam_offset, dtype=torch.float32))
        self.lam_span = device.lam_span

        # Per-segment pulse parameters shared by all samples.
        if self.omega_delta_source in ("trainable", "trainable_plus_encoder"):
            omega0_target = device.omega0_init_mhz
            if device.bound_pulse_params:
                # Logit of target / omega_max, so the bounded value starts at target.
                omega0_p = omega0_target / device.profile.omega_max_mhz
                omega0_init = float(np.log(omega0_p / (1.0 - omega0_p)))
            else:
                # No sigmoid downstream -- init directly at the target.
                omega0_init = omega0_target
            noise_std = device.pulse_init_noise_std
            omega0_vals = torch.full((device.omega_segments,), omega0_init, dtype=torch.float32)
            delta0_vals = torch.zeros(device.delta_segments, dtype=torch.float32)
            if noise_std > 0.0:
                # Break the symmetry between segments.
                omega0_vals = omega0_vals + torch.randn(device.omega_segments) * noise_std
                delta0_vals = delta0_vals + torch.randn(device.delta_segments) * noise_std
            self.Omega0_MHz = nn.Parameter(omega0_vals)
            self.Delta0_MHz = nn.Parameter(delta0_vals)
        else:
            self.Omega0_MHz = None
            self.Delta0_MHz = None

        if device.pulse_shape == "windowed":
            omega_interp, window = windowed_pulse_basis(
                device.omega_segments, device.pulse_steps, device.evolution_time_us, device.omega_ramp_us
            )
            delta_interp, _ = windowed_pulse_basis(
                device.delta_segments, device.pulse_steps, device.evolution_time_us, device.omega_ramp_us
            )
            self.register_buffer("omega_interp", omega_interp, persistent=False)
            self.register_buffer("delta_interp", delta_interp, persistent=False)
            self.register_buffer("omega_window", window, persistent=False)

        self.evolver = _build_evolver(device, self.V0)

    def _per_cluster_measurement_dim(self) -> int:
        if self.device_cfg.measurement_kind == "probability":
            return 2 ** self.device_cfg.n_atoms
        order = 1 if self.device_cfg.measurement_kind == "expectation" else self.device_cfg.correlator_order
        return sum(comb(self.device_cfg.n_atoms, k) for k in range(1, order + 1))

    @property
    def measurement_dim(self) -> int:
        """Output width: n_clusters times the per-cluster readout size."""
        return self.device_cfg.n_clusters * self._per_cluster_measurement_dim()

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        """Evolve all clusters and return the readout.

        Args:
            combined: [batch, n_atoms * n_clusters + 1 (+ omega_segments +
                delta_segments)]:
                x (cluster-major), lam, and omega_seg/delta_seg when the
                pulse is encoder-sourced, as packed by LatentToLocalField.

        Returns:
            [batch, measurement_dim]: Z correlators up to correlator_order,
            or log-probabilities for measurement_kind="probability".
        """
        if combined.dtype != torch.float32:
            original_dtype = combined.dtype
            combined = combined.to(dtype=torch.float32)
        else:
            original_dtype = None

        batch_size = combined.shape[0]
        n_clusters = self.n_clusters
        n_atoms = self.n_atoms

        n_om, n_de = self.omega_segments, self.delta_segments
        x = combined[:, : n_atoms * n_clusters].contiguous()
        lam = combined[:, n_atoms * n_clusters : n_atoms * n_clusters + 1].squeeze(-1).contiguous()

        # Fold clusters into the batch axis. lam and encoder-sourced pulses are
        # shared by the clusters of a sample, hence repeat_interleave.
        x_flat = x.reshape(batch_size * n_clusters, n_atoms)
        lam_flat = lam.repeat_interleave(n_clusters, dim=0)
        if self.lam_span is not None:
            # Bounded to [lam_offset - lam_span, lam_offset + lam_span].
            lam_flat = self.lam_pulseqpu + self.lam_span * torch.tanh(lam_flat)
        else:
            lam_flat = lam_flat + self.lam_pulseqpu

        if self.omega_delta_source in ("encoder", "trainable_plus_encoder"):
            offset = n_atoms * n_clusters + 1
            omega_seg = combined[:, offset : offset + n_om].contiguous()
            delta_seg = combined[:, offset + n_om : offset + n_om + n_de].contiguous()
            if self.omega_delta_source == "trainable_plus_encoder":
                scale = self.device_cfg.pulse_delta_scale
                omega_seg = self.Omega0_MHz + scale * torch.tanh(omega_seg)
                delta_seg = self.Delta0_MHz + scale * torch.tanh(delta_seg)
            Omega0_MHz = omega_seg.repeat_interleave(n_clusters, dim=0)
            Delta0_MHz = delta_seg.repeat_interleave(n_clusters, dim=0)
        else:
            Omega0_MHz = self.Omega0_MHz
            Delta0_MHz = self.Delta0_MHz

        Omega0_bounded, Delta0_bounded = self._bounded_omega_delta(Omega0_MHz, Delta0_MHz)

        out_flat = self.evolver(x_flat, lam_flat, Omega0_bounded, Delta0_bounded)
        # [batch * n_clusters, d] -> [batch, n_clusters * d], cluster-major.
        out = out_flat.reshape(batch_size, n_clusters * out_flat.shape[-1])

        if self.device_cfg.measurement_kind == "probability":
            # Log-probabilities are better conditioned than raw probabilities
            # as input to the linear projection.
            out = torch.log(out + 1e-8)
        if original_dtype is not None:
            # Return the caller's dtype.
            out = out.to(dtype=original_dtype)
        return out

    def _bounded_omega_delta(self, Omega0_MHz: torch.Tensor, Delta0_MHz: torch.Tensor):
        """Soft-bound raw Omega/Delta to the device profile's drive limits."""
        if not self.device_cfg.bound_pulse_params:
            return Omega0_MHz, Delta0_MHz
        profile = self.device_cfg.profile
        omega = profile.omega_max_mhz * torch.sigmoid(Omega0_MHz)
        delta = profile.delta_max_mhz * torch.tanh(Delta0_MHz)
        if self.device_cfg.pulse_shape == "windowed":
            # Control points -> simulation steps; Omega is zero at t=0 and t=T.
            omega = (omega @ self.omega_interp) * self.omega_window
            delta = delta @ self.delta_interp
        return omega, delta

    def extra_repr(self) -> str:
        """Configuration and bounded pulse values in physical units.

        Encoder-sourced pulses vary per sample and are not listed.
        """
        def _fmt_vec(t: torch.Tensor) -> str:
            return "[" + ", ".join(f"{v:.6f}" for v in t.detach().tolist()) + "]"

        if self.omega_delta_source in ("trainable", "trainable_plus_encoder"):
            omega_bounded, delta_bounded = self._bounded_omega_delta(self.Omega0_MHz, self.Delta0_MHz)
            omega_delta_str = (
                f"Omega0_MHz={_fmt_vec(omega_bounded)}, "
                f"Delta0_MHz={_fmt_vec(delta_bounded)}"
            )
        else:
            omega_delta_str = "Omega0_MHz=<encoder output>, Delta0_MHz=<encoder output>"

        return (
            f"n_atoms={self.device_cfg.n_atoms}, "
            f"n_clusters={self.device_cfg.n_clusters}, "
            f"cluster_routing={self.device_cfg.cluster_routing}, "
            f"omega_segments={self.device_cfg.omega_segments}, "
            f"delta_segments={self.device_cfg.delta_segments}, "
            f"measurement_kind={self.device_cfg.measurement_kind}"
            + (f"(order={self.device_cfg.correlator_order})" if self.device_cfg.measurement_kind == "correlators" else "")
            + f", measurement_dim={self.measurement_dim}, "
            f"encoding={self.device_cfg.encoding}, "
            f"omega_delta_source={self.omega_delta_source}, "
            f"bound_pulse_params={self.device_cfg.bound_pulse_params}, "
            f"skip_quantum_projection={self.device_cfg.skip_quantum_projection}, "
            f"n_data_injections={self.device_cfg.n_data_injections}, "
            f"r0_um={self.device_cfg.r0_um}, "
            f"lam_offset={self.device_cfg.lam_offset}, "
            f"lam_span={self.device_cfg.lam_span}, "
            f"evolution_time_us={self.device_cfg.evolution_time_us}, "
            f"evolution_backend={self.device_cfg.evolution_backend}, "
            f"device_profile={self.device_cfg.device_profile}, "
            f"C6={self.device_cfg.C6}, "
            + omega_delta_str
        )


class QuantumVAENeutralAtom(AnsatzVAEBase):
    """Quantum VAE with a neutral-atom (Rydberg) pulse-level ansatz.

        1. Encode the input to a latent z (AutoencoderKL encoder).
        2. Project z to per-atom local detunings x, a scale lam and,
           optionally, a per-sample pulse shape.
        3. Evolve the register (qlayer) and read out Z correlators.
        4. Project the readout back to the latent shape.
        5. Decode (AutoencoderKL decoder).

    Args:
        device: NeutralAtomDeviceConfig.
        *args, **kwargs: passed to AutoencoderKL.
    """

    def __init__(self, device: NeutralAtomDeviceConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoding_strategy = "neutral_atom_pulse"
        self.device_cfg = device
        self.qlayer = NeutralAtomPulseLayer(device)
        self.n_qubits = device.n_atoms  # alias used by the classifier pipelines
        self._quantum_torch_device = self._infer_quantum_torch_device()

        self.project_to_quantum: Optional[nn.Module] = None
        self.project_from_quantum: Optional[nn.Module] = None
        self.register_layout = None
        self.post_quantum_bn: Optional[nn.BatchNorm1d] = None
        self.post_quantum_gate: Optional[nn.Parameter] = None
        self.quantum_pad: int = 0
        # Raw readout of the last process_latent() call, before projection
        # and normalization. Read by the trainer's regularizers.
        self._last_quantum_readout: Optional[torch.Tensor] = None

    @property
    def last_quantum_state(self) -> Optional[torch.Tensor]:
        """Final cluster states of the last forward call, [B, n_clusters, 2**n_atoms],
        or None when the evolution backend does not expose them."""
        state = getattr(self.qlayer.evolver, "last_state", None)
        if state is None:
            return None
        return state.reshape(-1, self.device_cfg.n_clusters, state.shape[-1])

    def _infer_quantum_torch_device(self) -> torch.device:
        # The qlayer has no parameters when the pulse is encoder-sourced.
        try:
            return next(self.qlayer.parameters()).device
        except StopIteration:
            return self.qlayer.V0.device

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self._quantum_torch_device = self._infer_quantum_torch_device()
        return module

    def initialize_projections(self, dummy: torch.FloatTensor) -> None:
        """Build the projection layers from the latent shape of `dummy`.

        Must be called once before process_latent().
        """
        with torch.no_grad():
            posterior = self.encode(dummy).latent_dist
            z = posterior.mode()
            latent_dim = z.flatten(1).shape[1]

        if self.device_cfg.projection_kind == "graph":
            # torch_geometric is only needed for projection_kind="graph".
            from .gnn_hypergraph_interface import GraphLatentToLocalField, RegisterLayout
            if z.dim() != 4:
                raise ValueError(
                    f"projection_kind='graph' requires a spatial [B,C,H,W] "
                    f"latent (got shape {tuple(z.shape)}) -- the atom-token "
                    "extractor cross-attends into the latent grid."
                )
            effective_order = (
                1 if self.device_cfg.measurement_kind == "expectation"
                else self.device_cfg.correlator_order
            )
            layout = RegisterLayout(
                self.device_cfg.n_atoms,
                effective_order,
                self.device_cfg.register_um,
                n_clusters=self.device_cfg.n_clusters,
                cluster_grid=self.device_cfg.cluster_grid,
            )
            self.register_layout = layout
            self.project_to_quantum = GraphLatentToLocalField(
                self.device_cfg.n_atoms,
                effective_order,
                latent_channels=z.shape[1],
                d_model=self.device_cfg.graph_d_model,
                measurement_kind=self.device_cfg.measurement_kind,
                register_um=self.device_cfg.register_um,
                use_fourier_pos=self.device_cfg.use_fourier_hyperedge_pos,
                atom_token_mode=self.device_cfg.atom_token_mode,
                residual=self.device_cfg.graph_residual,
                pre_quantum_norm=self.device_cfg.pre_quantum_norm,
                pre_quantum_gain=self.device_cfg.pre_quantum_gain,
                layout=layout,
                pulse_points=(
                    (self.device_cfg.omega_segments, self.device_cfg.delta_segments)
                    if self.device_cfg.omega_delta_source == "trainable_plus_encoder" else None
                ),
            ).to(dummy.device)
        else:
            self.project_to_quantum = LatentToLocalField(
                latent_dim,
                self.device_cfg.n_atoms,
                self.device_cfg.omega_segments,
                self.device_cfg.delta_segments,
                self.device_cfg.n_clusters,
                omega_delta_source=self.device_cfg.omega_delta_source,
            ).to(dummy.device)

        if self.device_cfg.skip_quantum_projection:
            measurement_dim = self.qlayer.measurement_dim
            if measurement_dim > latent_dim:
                raise ValueError(
                    f"skip_quantum_projection=True requires measurement_dim "
                    f"({measurement_dim}) <= latent_dim ({latent_dim}) -- "
                    "truncating would discard already-computed correlators. "
                    "Reduce n_atoms/correlator_order/n_clusters, or disable "
                    "skip_quantum_projection to use a learned projection instead."
                )
            self.quantum_pad = latent_dim - measurement_dim
            if self.quantum_pad > 0:
                warnings.warn(
                    f"skip_quantum_projection: measurement_dim={measurement_dim} "
                    f"< latent_dim={latent_dim}, padding with {self.quantum_pad} zeros.",
                    stacklevel=2,
                )
            self.project_from_quantum = None
        elif self.device_cfg.projection_kind == "graph":
            from .gnn_hypergraph_interface import GraphQuantumToDecoderVector
            effective_order = (
                1 if self.device_cfg.measurement_kind == "expectation"
                else self.device_cfg.correlator_order
            )
            self.project_from_quantum = GraphQuantumToDecoderVector(
                self.device_cfg.n_atoms,
                effective_order,
                out_dim=latent_dim,
                d_model=self.device_cfg.graph_d_model,
                measurement_kind=self.device_cfg.measurement_kind,
                register_um=self.device_cfg.register_um,
                use_fourier_pos=self.device_cfg.use_fourier_hyperedge_pos,
                fourier_atom_dim=self.device_cfg.fourier_atom_dim,
                residual=self.device_cfg.graph_residual,
                layout=self.register_layout,
                output_mode=self.device_cfg.graph_output,
                latent_shape=tuple(z.shape[1:]),
                cell_write=self.device_cfg.graph_cell_write,
            ).to(dummy.device)
        else:
            self.project_from_quantum = nn.Linear(self.qlayer.measurement_dim, latent_dim).to(dummy.device)

        if self.device_cfg.post_quantum_norm == "batchnorm_gated":
            self.post_quantum_bn = nn.BatchNorm1d(latent_dim).to(dummy.device)
            self.post_quantum_gate = nn.Parameter(
                torch.full((1,), float(self.device_cfg.post_quantum_gate_init), device=dummy.device)
            )

    def process_latent(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """Run the latent through the pulse program.

        Projects z to the pulse inputs, evolves the register and maps the
        readout back to the latent shape: through project_from_quantum, or by
        zero-padding and reshaping when skip_quantum_projection is set. The
        gated post-quantum BatchNorm, when enabled, is applied before the
        final reshape.

        Raises:
            RuntimeError: if initialize_projections() has not been called.
        """
        skip_projection = self.device_cfg.skip_quantum_projection
        if self.project_to_quantum is None or (not skip_projection and self.project_from_quantum is None):
            raise RuntimeError(
                "Projections not initialized. Call initialize_projections() "
                "before process_latent()"
            )
        # The quantum interface (encoder head, pulse evolution, decoder head,
        # post-quantum norm) always runs in float32, also under autocast.
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self._process_latent_fp32(z.float())

    def _process_latent_fp32(self, z: torch.FloatTensor) -> torch.FloatTensor:
        skip_projection = self.device_cfg.skip_quantum_projection
        to_quantum_layer = self.project_to_quantum

        old_shape = z.shape
        z_flat = z.flatten(1)

        if self.device_cfg.projection_kind == "graph":
            combined = to_quantum_layer.pack(z)  # spatial latent in, unflattened
        else:
            combined = to_quantum_layer(z_flat)  # (x, lam[, omega_seg, delta_seg])

        combined = combined.to(self._quantum_torch_device, dtype=torch.float32)
        readout = self.qlayer(combined)
        self._last_quantum_readout = readout

        if skip_projection:
            readout = readout.to(z.device, dtype=z.dtype)
            if self.quantum_pad > 0:
                readout = nn.functional.pad(readout, (0, self.quantum_pad))
            z_quantum_flat = readout
        else:
            from_quantum_layer = self.project_from_quantum
            ref_param = next(from_quantum_layer.parameters())
            readout = readout.to(ref_param.device, dtype=ref_param.dtype)
            z_quantum_flat = from_quantum_layer(readout)

        if self.post_quantum_bn is not None:
            normed = self.post_quantum_bn(z_quantum_flat)
            z_quantum_flat = z_quantum_flat + self.post_quantum_gate * (normed - z_quantum_flat)

        return z_quantum_flat.reshape(old_shape)

    @torch.no_grad()
    def cluster_diagnostics(self, sample: torch.FloatTensor, dead_ratio: float = 0.05) -> dict:
        """Per-cluster spread across the batch of x and of the readout.

        A cluster counts as dead when its readout spread is below dead_ratio
        times the median spread over clusters.
        """
        n_atoms, n_clusters = self.device_cfg.n_atoms, self.device_cfg.n_clusters
        was_training = self.training
        self.eval()
        try:
            combined = self.get_pre_quantum_features(sample, sample_posterior=False)
            combined = combined.to(self._quantum_torch_device, dtype=torch.float32)
            readout = self.qlayer(combined)
        finally:
            self.train(was_training)
        batch = combined.shape[0]
        x_std = combined[:, : n_atoms * n_clusters].reshape(batch, n_clusters, n_atoms).std(0).mean(-1)
        readout_std = readout.reshape(batch, n_clusters, -1).std(0).mean(-1)
        median = readout_std.median().clamp_min(1e-12)
        gain = getattr(self.project_to_quantum, "pre_quantum_gain", None)
        stats = {
            "cluster_x_std_min": float(x_std.min()),
            "cluster_x_std_mean": float(x_std.mean()),
            "cluster_readout_std_min": float(readout_std.min()),
            "cluster_readout_std_mean": float(readout_std.mean()),
            "cluster_readout_std_argmin": int(readout_std.argmin()),
            "cluster_dead": int((readout_std < dead_ratio * median).sum()),
            **({"pre_quantum_gain": gain} if gain is not None else {}),
        }
        if self.device_cfg.measurement_kind == "correlators" and self.device_cfg.correlator_order >= 2:
            connected = connected_correlators(readout, n_atoms, n_clusters, self.device_cfg.correlator_order)
            stats["cluster_connected_std_mean"] = float(connected.std(0).mean())
        state = self.last_quantum_state
        if state is not None:
            stats["state_fidelity_mean"] = float(mean_state_fidelity(state))
        return stats

    def get_pre_quantum_features(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Encoder output packed for the qlayer.

        The graph projection takes the spatial latent instead of the flat one.
        """
        if self.device_cfg.projection_kind != "graph":
            return super().get_pre_quantum_features(sample, sample_posterior, generator)

        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if self.project_to_quantum is None or self.project_from_quantum is None:
            self.initialize_projections(sample)
        return self.project_to_quantum.pack(z)

    def get_latent(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        if self.project_to_quantum is None:
            self.initialize_projections(sample)
        return self.process_latent(z)

    def construct_circuit(self) -> Callable:
        """The pulse program's evolution, as called by the qlayer."""
        return self.qlayer.evolver
