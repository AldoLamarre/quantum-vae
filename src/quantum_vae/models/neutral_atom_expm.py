"""Matrix-exponential time evolution for the neutral-atom pulse layer.

The pulse Hamiltonian is constant within each of the n_segments windows
(piecewise-constant Omega0/Delta0, time-independent interaction and local
field), so the evolution is exact:

    |psi(T)> = U_S ... U_1 |0...0>,   U_k = expm(-i H_k T / S)

    H_k = 2*pi * sum_{i<j} C6/r_ij^6 n_i n_j
          + pi * Omega0_k * sum_i sigma_x_i
          - 2*pi * Delta0_k * sum_i n_i
          - pi * sum_i V_i(x) Z_i

C6 and the bounds on Omega0/Delta0/V_i come from device.device_profile.

Same conventions as PennyLane's rydberg_interaction/rydberg_drive (MHz with
an explicit 2*pi, n = (I - Z)/2, wire 0 = most significant bit) and the same
local term as PennyLaneEvolver. Pure torch: no JAX, no ODE tolerance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .quantum_vae_neutral_atom import (
    NeutralAtomDeviceConfig,
    correlator_index_sets,
    local_field_mhz,
)


class ExpmEvolver(nn.Module):
    """Exact piecewise-constant pulse evolution via batched matrix exponentials.

    Called as evolver(x, lam, Omega0_MHz, Delta0_MHz) with already-bounded
    pulse values: x [B, n_atoms], lam [B], Omega0/Delta0 [S] or [B, S].
    Returns [B, per_cluster_dim] readout. All buffers are non-persistent,
    so the layer's state_dict matches the PennyLane backend.
    """

    def __init__(self, device: NeutralAtomDeviceConfig, V0: torch.Tensor):
        super().__init__()
        self.device_cfg = device
        n = device.n_atoms
        dim = 2 ** n
        self.n_segments = device.simulation_steps
        self.dt = device.evolution_time_us / device.simulation_steps
        self.bounded = device.bound_pulse_params
        self.profile = device.profile
        self.measurement_kind = device.measurement_kind

        # Computational basis bits, wire 0 = most significant bit.
        shifts = torch.arange(n - 1, -1, -1)
        bits = (torch.arange(dim)[:, None] >> shifts) & 1  # [dim, n]
        bits_f = bits.to(torch.float64)
        z = 1.0 - 2.0 * bits_f  # Z eigenvalues

        pos = torch.tensor(device.register_um, dtype=torch.float64)
        interaction = torch.zeros(dim, dtype=torch.float64)
        for i in range(n):
            for j in range(i + 1, n):
                r = torch.linalg.norm(pos[i] - pos[j])
                interaction += device.C6 / r ** 6 * bits_f[:, i] * bits_f[:, j]
        interaction *= 2.0 * math.pi

        sigma_x_sum = torch.zeros(dim, dim, dtype=torch.float64)
        basis = torch.arange(dim)
        for i in range(n):
            flipped = basis ^ (1 << (n - 1 - i))
            sigma_x_sum[basis, flipped] += 1.0

        self.register_buffer("V0", V0.detach().clone().to(torch.float32), persistent=False)
        self.register_buffer("z", z.to(torch.float32), persistent=False)
        self.register_buffer("interaction_diag", interaction.to(torch.float32), persistent=False)
        self.register_buffer("number_diag", bits_f.sum(1).to(torch.float32), persistent=False)
        self.register_buffer("sigma_x_sum", sigma_x_sum.to(torch.float32), persistent=False)

        if self.measurement_kind != "probability":
            order = 1 if self.measurement_kind == "expectation" else device.correlator_order
            combos = correlator_index_sets(n, order)
            signs = torch.stack([z[:, list(c)].prod(dim=1) for c in combos], dim=1)  # [dim, M]
            self.register_buffer("readout_signs", signs.to(torch.float32), persistent=False)

    def forward(self, x, lam, Omega0_MHz, Delta0_MHz) -> torch.Tensor:
        batch = x.shape[0]
        S = self.n_segments
        omega = Omega0_MHz.expand(batch, S) if Omega0_MHz.dim() == 1 else Omega0_MHz
        delta = Delta0_MHz.expand(batch, S) if Delta0_MHz.dim() == 1 else Delta0_MHz

        v = local_field_mhz(x, lam[:, None], self.V0, self.bounded, torch, self.profile)  # [B, n]
        static_diag = self.interaction_diag - math.pi * (v @ self.z.T)  # [B, dim]

        psi = None
        for k in range(S):
            diag = static_diag - 2.0 * math.pi * delta[:, k : k + 1] * self.number_diag
            H = math.pi * omega[:, k, None, None] * self.sigma_x_sum + torch.diag_embed(diag)
            U = torch.linalg.matrix_exp((-1j * self.dt) * H.to(torch.complex64))
            # Initial state |0...0>: first step is U's first column.
            psi = U[:, :, 0] if psi is None else (U @ psi[..., None]).squeeze(-1)

        probs = psi.real ** 2 + psi.imag ** 2
        if self.measurement_kind == "probability":
            return probs
        return probs @ self.readout_signs
