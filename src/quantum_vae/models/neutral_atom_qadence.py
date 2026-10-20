"""Qadence (pyqtorch) time evolution for the neutral-atom pulse layer.

Builds the same piecewise-constant Hamiltonian as ExpmEvolver,

    H_k = 2*pi * sum_{i<j} C6/r_ij^6 n_i n_j
          + pi * Omega0_k * sum_i X_i
          - 2*pi * Delta0_k * sum_i n_i
          - pi * sum_i V_i(x) Z_i,

as one HamEvo block per segment, with V_i, Omega0_k and Delta0_k passed as
batched feature parameters. Intended for evaluation and cross-checks: it is
slower and uses more memory than ExpmEvolver.
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


def _import_qadence():
    """Import qadence without changing torch's default dtype (it sets float64 on import)."""
    previous = torch.get_default_dtype()
    try:
        import qadence
    finally:
        torch.set_default_dtype(previous)
    return qadence


class QadenceEvolver(nn.Module):
    """Pulse evolution with Qadence; same call signature and outputs as ExpmEvolver.

    The Qadence model is kept outside the module registry, so state_dict and
    parameters() match the other backends and checkpoints load unchanged.
    """

    def __init__(self, device: NeutralAtomDeviceConfig, V0: torch.Tensor):
        super().__init__()
        qd = _import_qadence()
        from qadence import (
            BackendName,
            DiffMode,
            FeatureParameter,
            HamEvo,
            N,
            QuantumCircuit,
            QuantumModel,
            X,
            Z,
            add,
            chain,
        )

        self.device_cfg = device
        self.profile = device.profile
        self.bounded = device.bound_pulse_params
        self.measurement_kind = device.measurement_kind
        n = device.n_atoms
        self.n_atoms = n
        self.n_segments = device.simulation_steps
        dt = device.evolution_time_us / device.simulation_steps

        pos = torch.tensor(device.register_um, dtype=torch.float64)
        interaction = add(*[
            2.0 * math.pi * device.C6 / float(torch.linalg.norm(pos[i] - pos[j])) ** 6 * (N(i) @ N(j))
            for i in range(n) for j in range(i + 1, n)
        ])
        local = add(*[-math.pi * FeatureParameter(f"v{i}") * Z(i) for i in range(n)])
        sum_x = add(*[X(i) for i in range(n)])
        sum_n = add(*[N(i) for i in range(n)])
        segments = [
            HamEvo(
                interaction + local
                + math.pi * FeatureParameter(f"omega{k}") * sum_x
                - 2.0 * math.pi * FeatureParameter(f"delta{k}") * sum_n,
                dt,
            )
            for k in range(self.n_segments)
        ]
        model = QuantumModel(
            QuantumCircuit(n, chain(*segments)),
            backend=BackendName.PYQTORCH,
            diff_mode=DiffMode.AD,
        )
        # Not registered as a submodule: see class docstring.
        object.__setattr__(self, "_qadence_model", model)
        self._qadence_device = torch.device("cpu")

        self.register_buffer("V0", V0.detach().clone().to(torch.float32), persistent=False)
        if self.measurement_kind != "probability":
            dim = 2 ** n
            shifts = torch.arange(n - 1, -1, -1)
            bits = (torch.arange(dim)[:, None] >> shifts) & 1
            z = 1.0 - 2.0 * bits.to(torch.float32)
            order = 1 if self.measurement_kind == "expectation" else device.correlator_order
            signs = torch.stack(
                [z[:, list(c)].prod(dim=1) for c in correlator_index_sets(n, order)], dim=1
            )
            self.register_buffer("readout_signs", signs, persistent=False)

    def _model_on(self, device: torch.device):
        if self._qadence_device != device:
            self._qadence_model.to(device)
            self._qadence_device = device
        return self._qadence_model

    def forward(self, x, lam, Omega0_MHz, Delta0_MHz) -> torch.Tensor:
        batch = x.shape[0]
        S = self.n_segments
        omega = Omega0_MHz.expand(batch, S) if Omega0_MHz.dim() == 1 else Omega0_MHz
        delta = Delta0_MHz.expand(batch, S) if Delta0_MHz.dim() == 1 else Delta0_MHz
        v = local_field_mhz(x, lam[:, None], self.V0, self.bounded, torch, self.profile)  # [B, n]

        values = {f"v{i}": v[:, i] for i in range(self.n_atoms)}
        values.update({f"omega{k}": omega[:, k] for k in range(S)})
        values.update({f"delta{k}": delta[:, k] for k in range(S)})
        psi = self._model_on(x.device).run(values=values)  # [B, 2**n], complex128

        probs = (psi.real ** 2 + psi.imag ** 2).to(x.dtype)
        if self.measurement_kind == "probability":
            return probs
        return probs @ self.readout_signs
