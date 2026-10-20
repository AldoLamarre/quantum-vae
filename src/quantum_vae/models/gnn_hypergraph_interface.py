"""Hypergraph-based replacement for LatentToLocalField / project_from_quantum.

GraphLatentToLocalField: encoder latent -> per-atom x, global lam.
GraphQuantumToDecoderVector: correlator readout -> decoder-facing vector.

Hyperedges are built from atom subsets (order 1..correlator_order), in the
same enumeration order as _correlator_ops in quantum_vae_neutral_atom.py.
Batched via block-diagonal graph construction (no Python loop over batch).

RegisterLayout is the single source of truth for atom indexing, latent
routing and the hyperedge/readout order across clusters.
"""
import itertools
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn.aggr import AttentionalAggregation


def validate_measurement_kind(measurement_kind: str) -> None:
    """Rejects measurement_kind="probability"; see raised message."""
    if measurement_kind == "probability":
        raise ValueError(
            "Hypergraph modules are built from atom subsets (order-k "
            "correlators) and do not apply to measurement_kind='probability'."
        )


def build_hyperedge_index(n_atoms: int, order: int):
    """hyperedge_index [2, nnz] plus each hyperedge id's atom subset,
    enumerated k=1..order (matches _correlator_ops's ordering).
    """
    node_idx, edge_idx, subsets = [], [], []
    hid = 0
    for k in range(1, order + 1):
        for subset in itertools.combinations(range(n_atoms), k):
            for atom in subset:
                node_idx.append(atom)
                edge_idx.append(hid)
            subsets.append(subset)
            hid += 1
    return torch.tensor([node_idx, edge_idx], dtype=torch.long), subsets


def batch_hyperedge_index(hyperedge_index, n_atoms, n_hyperedges, batch_size, device):
    """Block-diagonal batching: B disjoint copies of the same hyperedge
    structure, offset so no hyperedge spans two samples.
    """
    node_idx, edge_idx = hyperedge_index.to(device)
    offset_n = torch.arange(batch_size, device=device).view(-1, 1) * n_atoms
    offset_e = torch.arange(batch_size, device=device).view(-1, 1) * n_hyperedges
    batched_node_idx = (node_idx.unsqueeze(0) + offset_n).reshape(-1)
    batched_edge_idx = (edge_idx.unsqueeze(0) + offset_e).reshape(-1)
    return torch.stack([batched_node_idx, batched_edge_idx], dim=0)


def _effective_register_dims(coords: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Drop coordinate columns that are constant across all atoms (e.g. y=0
    on a 1D chain register) -- keeps the Fourier encoding from spending
    budget on a spatial dimension that carries no information for this
    geometry. A 2D grid register keeps both columns.
    """
    keep = coords.std(dim=0) > eps
    if not keep.any():
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[0] = True
    return coords[:, keep]


def fourier_atom_features(register_um: Sequence[Sequence[float]], dim_per_atom: int) -> torch.Tensor:
    """Fixed (non-trainable) sinusoidal position features, one row per atom,
    built from the register's real physical coordinates -- same idea as
    transformer positional encoding, applied to the atoms' actual (x, y)
    layout instead of a sequence index.

    Constant coordinate columns (e.g. y=0 on a 1D chain) are dropped
    automatically, so dim_per_atom only needs to be a multiple of
    2 * effective_spatial_dim: 2 for a chain, 4 for a 2D grid.
    """
    coords = torch.as_tensor(register_um, dtype=torch.float32)
    coords = _effective_register_dims(coords)
    n_dims = coords.shape[1]
    if dim_per_atom % (2 * n_dims) != 0:
        raise ValueError(
            f"dim_per_atom={dim_per_atom} must be a multiple of "
            f"2*{n_dims}={2 * n_dims} for this register (effective spatial "
            f"dim={n_dims})."
        )
    n_freq = dim_per_atom // (2 * n_dims)
    span = (coords.max(dim=0).values - coords.min(dim=0).values).clamp(min=1.0)
    # One frequency band per effective spatial dim, geometrically spaced
    # from one cycle across the register's own span down to a finer scale.
    freqs = torch.stack(
        [torch.logspace(0, 2, n_freq) / span[d] for d in range(n_dims)], dim=0
    )  # (n_dims, n_freq)
    args = coords.unsqueeze(-1) * freqs.unsqueeze(0)  # (n_atoms, n_dims, n_freq)
    feats = torch.cat([args.sin(), args.cos()], dim=-1)  # (n_atoms, n_dims, 2*n_freq)
    return feats.reshape(coords.shape[0], -1)  # (n_atoms, dim_per_atom)


def build_hyperedge_slot_index(subsets, correlator_order: int, n_atoms: int) -> torch.Tensor:
    """(n_hyperedges, correlator_order) atom-index table for concatenation-
    based hyperedge identity encoding. Real members use their atom index;
    unused slots (for hyperedges below the max order) point at `n_atoms`,
    a sentinel row reserved for a learned pad vector.
    """
    slot_idx = torch.full((len(subsets), correlator_order), n_atoms, dtype=torch.long)
    for hid, subset in enumerate(subsets):
        for slot, atom in enumerate(subset):
            slot_idx[hid, slot] = atom
    return slot_idx


class LatentToAtomTokens(nn.Module):
    """n_atoms learned queries cross-attend into the encoder's spatial
    latent grid (kept unflattened, so 2D locality reaches the attention).

    atom_pos (optional): fixed Fourier features of each atom's real
    register position, added to the learned query so "which atom" carries
    physical identity from step 0 instead of depending on training to
    differentiate n_atoms initially-similar learned vectors.

    Raw sin/cos output is O(1) in magnitude (mean_abs ~0.6, measured),
    while atom_queries is initialized at *0.02 (mean_abs ~0.016) -- adding
    them directly would make atom_pos dominate ~40x over, effectively
    replacing the learned query with a fixed one at init. pos_scale is a
    learnable multiplier on atom_pos, itself initialized at 0.02 to match
    atom_queries' own scale, so the two start out comparable and training
    can grow the position term's influence once the rest of the network
    is ready to use it, rather than it dominating from step 0.
    """

    def __init__(self, n_atoms: int, latent_channels: int, d_model: int, n_heads: int = 4,
                 atom_pos: torch.Tensor = None):
        super().__init__()
        self.atom_queries = nn.Parameter(torch.randn(n_atoms, d_model) * 0.02)
        self.kv_proj = nn.Linear(latent_channels, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_tokens = nn.LayerNorm(latent_channels)
        if atom_pos is not None:
            self.register_buffer("atom_pos", atom_pos)
            self.pos_scale = nn.Parameter(torch.tensor(0.02))
        else:
            self.atom_pos = None

    def forward(self, z_spatial: torch.Tensor):
        B, C, H, W = z_spatial.shape
        tokens = z_spatial.flatten(2).transpose(1, 2)
        tokens = self.norm_tokens(tokens)
        kv = self.kv_proj(tokens)
        queries = self.atom_queries if self.atom_pos is None else self.atom_queries + self.pos_scale * self.atom_pos
        q = queries.unsqueeze(0).expand(B, -1, -1)
        atom_tokens, attn_weights = self.attn(q, kv, kv, need_weights=True)
        return atom_tokens, attn_weights


def register_grid_index(register_um: Sequence[Sequence[float]]):
    """(n_rows, n_cols, row_idx, col_idx) placing each atom on the grid
    spanned by the register's distinct y and x coordinates.
    """
    coords = torch.as_tensor(register_um, dtype=torch.float32)
    xs = torch.unique(coords[:, 0])
    ys = torch.unique(coords[:, 1])
    col_idx = torch.argmin((coords[:, 0:1] - xs.unsqueeze(0)).abs(), dim=1)
    row_idx = torch.argmin((coords[:, 1:2] - ys.unsqueeze(0)).abs(), dim=1)
    return len(ys), len(xs), row_idx, col_idx


class RegisterLayout:
    """Global atom indexing for n_clusters copies of one register.

    Atoms are numbered cluster-major (global index = cluster * n_atoms + atom),
    matching the qlayer's x and readout layout. Clusters are placed row-major
    on cluster_grid. Hyperedges are listed per cluster in _correlator_ops
    order, so hyperedge i is readout column i. The list is append-only:
    cross-cluster hyperedges can be added later without moving indices.

    coords_um are layout coordinates for position features and latent
    routing only; the physics simulates each cluster's register_um.
    """

    def __init__(self, n_atoms: int, correlator_order: int,
                 register_um: Optional[Sequence[Sequence[float]]] = None,
                 n_clusters: int = 1, cluster_grid: Optional[Tuple[int, int]] = None):
        if cluster_grid is None:
            if n_clusters != 1:
                raise ValueError("cluster_grid is required when n_clusters > 1.")
            cluster_grid = (1, 1)
        cluster_rows, cluster_cols = (int(v) for v in cluster_grid)
        if cluster_rows * cluster_cols != n_clusters:
            raise ValueError(
                f"cluster_grid={tuple(cluster_grid)} does not match n_clusters={n_clusters}."
            )
        if register_um is None and n_clusters != 1:
            raise ValueError("register_um is required when n_clusters > 1.")

        self.n_atoms = n_atoms
        self.n_clusters = n_clusters
        self.cluster_grid = (cluster_rows, cluster_cols)
        self.correlator_order = correlator_order
        self.n_nodes = n_atoms * n_clusters

        _, local_subsets = build_hyperedge_index(n_atoms, correlator_order)
        self.per_cluster_hyperedges = len(local_subsets)
        self.subsets = [
            tuple(c * n_atoms + a for a in subset)
            for c in range(n_clusters) for subset in local_subsets
        ]
        singleton = {subset[0]: h for h, subset in enumerate(self.subsets) if len(subset) == 1}
        self.node_readout_idx = torch.tensor([singleton[g] for g in range(self.n_nodes)], dtype=torch.long)
        self._build_hyperedge_tensors()

        self.coords_um = None
        if register_um is not None:
            reg = torch.as_tensor(register_um, dtype=torch.float32)
            atom_rows, atom_cols, row_idx, col_idx = register_grid_index(register_um)
            self.atom_grid = (atom_rows, atom_cols)
            self.grid_shape = (cluster_rows * atom_rows, cluster_cols * atom_cols)
            cluster_r = torch.arange(n_clusters) // cluster_cols
            cluster_c = torch.arange(n_clusters) % cluster_cols
            self.cell_row = (cluster_r.view(-1, 1) * atom_rows + row_idx.view(1, -1)).reshape(-1)
            self.cell_col = (cluster_c.view(-1, 1) * atom_cols + col_idx.view(1, -1)).reshape(-1)
            cells = self.cell_row * self.grid_shape[1] + self.cell_col
            if torch.unique(cells).numel() != self.n_nodes:
                raise ValueError("Two atoms map to the same latent cell; use a register with distinct grid positions.")
            if n_clusters == 1:
                self.coords_um = reg
            else:
                if n_atoms > 1:
                    d = torch.cdist(reg, reg)
                    nn_um = d[d > 0].min()
                else:
                    nn_um = torch.tensor(1.0)
                step = (reg.max(dim=0).values - reg.min(dim=0).values) + nn_um
                offset = torch.stack([cluster_c.float() * step[0], cluster_r.float() * step[1]], dim=-1)
                self.coords_um = (offset.unsqueeze(1) + reg.unsqueeze(0)).reshape(-1, 2)

    def _build_hyperedge_tensors(self) -> None:
        self.n_hyperedges = len(self.subsets)
        node_idx = [a for subset in self.subsets for a in subset]
        edge_idx = [h for h, subset in enumerate(self.subsets) for _ in subset]
        self.hyperedge_index = torch.tensor([node_idx, edge_idx], dtype=torch.long)
        self.slot_idx = build_hyperedge_slot_index(self.subsets, self.correlator_order, self.n_nodes)

    def append_hyperedges(self, subsets: Sequence[Sequence[int]]) -> None:
        """Append extra hyperedges (e.g. cross-cluster pairs) after the
        per-cluster ones. Existing indices are unchanged; the readout must
        provide one extra column per appended hyperedge, in this order.
        Call before building modules from this layout.
        """
        existing = set(self.subsets)
        for subset in subsets:
            subset = tuple(int(a) for a in subset)
            if not 2 <= len(subset) <= self.correlator_order:
                raise ValueError(
                    f"Hyperedge {subset} must have between 2 and correlator_order={self.correlator_order} atoms."
                )
            if len(set(subset)) != len(subset) or min(subset) < 0 or max(subset) >= self.n_nodes:
                raise ValueError(f"Hyperedge {subset} has invalid atom indices.")
            if subset in existing:
                raise ValueError(f"Hyperedge {subset} already exists.")
            existing.add(subset)
            self.subsets.append(subset)
        self._build_hyperedge_tensors()

    @property
    def cell_index(self) -> torch.Tensor:
        """Flat latent-grid cell of each global atom."""
        return self.cell_row * self.grid_shape[1] + self.cell_col


class _BatchedIndexCache:
    """Caches the block-diagonal hyperedge index per (batch size, device)."""

    def __init__(self, hyperedge_index: torch.Tensor, n_nodes: int, n_hyperedges: int):
        self.hyperedge_index = hyperedge_index
        self.n_nodes = n_nodes
        self.n_hyperedges = n_hyperedges
        self._cache = {}

    def __call__(self, batch_size: int, device) -> torch.Tensor:
        key = (batch_size, str(device))
        if key not in self._cache:
            self._cache[key] = batch_hyperedge_index(
                self.hyperedge_index, self.n_nodes, self.n_hyperedges, batch_size, device
            )
        return self._cache[key]


def _pop_derived_buffers(module: nn.Module, state_dict: dict, prefix: str,
                         names: Sequence[str], prefix_ok: Sequence[str] = ()) -> None:
    """Remove layout-derived buffers saved by older checkpoints.

    These buffers are rebuilt from the config and are not saved. A saved
    copy must match the rebuilt one; names in prefix_ok may be a leading
    slice of it (hyperedges appended since the checkpoint). Matching float
    buffers keep the saved values, so older checkpoints reproduce exactly.
    """
    for name in names:
        saved = state_dict.pop(prefix + name, None)
        current = getattr(module, name, None)
        if saved is None or current is None:
            continue
        saved = saved.to(device=current.device, dtype=current.dtype)
        if name in prefix_ok:
            ok = (saved.shape[1:] == current.shape[1:] and saved.shape[0] <= current.shape[0]
                  and torch.equal(saved, current[: saved.shape[0]]))
        else:
            ok = saved.shape == current.shape and torch.allclose(saved, current, rtol=1e-5, atol=1e-6)
        if not ok:
            raise RuntimeError(
                f"Checkpoint buffer '{prefix + name}' does not match the configured register layout."
            )
        if current.is_floating_point():
            with torch.no_grad():
                current.copy_(saved)


def resample_grid(grid: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinear resampling of [B, C, H, W] to size, antialiased when shrinking."""
    return F.interpolate(grid, size=size, mode="bilinear", align_corners=False, antialias=True)


class LatentToAtomPatches(nn.Module):
    """Each atom reads the latent region at its register position: the
    latent grid is average-pooled to the register's (rows, cols) layout,
    and each atom takes its own cell. Gives atoms distinct inputs from
    initialization, unlike learned-query cross-attention. With a layout,
    the grid spans all clusters (cluster grid x atom grid).

    If the atom grid is larger than the latent on either axis, the latent
    is bilinearly resampled to the atom grid instead, so several atoms can
    read the same region.
    """

    def __init__(self, register_um: Sequence[Sequence[float]], latent_channels: int, d_model: int,
                 layout: Optional[RegisterLayout] = None):
        super().__init__()
        if layout is None:
            self.n_rows, self.n_cols, row_idx, col_idx = register_grid_index(register_um)
        else:
            (self.n_rows, self.n_cols), row_idx, col_idx = layout.grid_shape, layout.cell_row, layout.cell_col
        self.register_buffer("row_idx", row_idx.clone(), persistent=False)
        self.register_buffer("col_idx", col_idx.clone(), persistent=False)
        self.norm_tokens = nn.LayerNorm(latent_channels)
        self.proj = nn.Linear(latent_channels, d_model)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _pop_derived_buffers(self, state_dict, prefix, ("row_idx", "col_idx"))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def sample_cells(self, z_spatial: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, n_atoms, C]: the latent values each atom reads."""
        height, width = z_spatial.shape[-2:]
        if self.n_rows <= height and self.n_cols <= width:
            pooled = F.adaptive_avg_pool2d(z_spatial, (self.n_rows, self.n_cols))
        else:
            pooled = resample_grid(z_spatial, (self.n_rows, self.n_cols))
        return pooled[:, :, self.row_idx, self.col_idx].transpose(1, 2)

    def forward(self, z_spatial: torch.Tensor):
        return self.proj(self.norm_tokens(self.sample_cells(z_spatial))), None


def time_features(times: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed sinusoidal features [len(times), dim] of times in [0, 1]."""
    n_freq = dim // 2
    freqs = torch.logspace(0, 2, n_freq) * math.pi
    args = times[:, None] * freqs[None, :]
    feats = torch.cat([args.sin(), args.cos()], dim=-1)
    if feats.shape[1] < dim:
        feats = torch.cat([feats, times[:, None]], dim=-1)
    return feats


class PulseQueryHead(nn.Module):
    """Per-sample pulse corrections at n_points control points.

    One query per control point (fixed time features plus a learned offset)
    cross-attends the atom tokens; a shared MLP maps each query to raw
    (Omega, Delta) corrections. The last layer is zero-initialized, so the
    corrections start at zero. Output [B, 2 * n_points]: Omega, then Delta.
    """

    def __init__(self, d_model: int, n_points: int):
        super().__init__()
        n_heads = 4 if d_model % 4 == 0 else 1
        times = (torch.arange(n_points, dtype=torch.float32) + 0.5) / n_points
        self.register_buffer("time_embed", time_features(times, d_model), persistent=False)
        self.query_offset = nn.Parameter(torch.zeros(n_points, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, atom_tokens: torch.Tensor) -> torch.Tensor:
        B = atom_tokens.shape[0]
        q = (self.time_embed + self.query_offset).unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, atom_tokens, atom_tokens)
        a = self.mlp(out)  # [B, n_points, 2]
        return torch.cat([a[..., 0], a[..., 1]], dim=-1)


class GraphLatentToLocalField(nn.Module):
    """Cross-attention extraction + hypergraph message passing, producing
    x_i (per-atom local-detuning weight, in [0, 1]) and lam (pooled global
    scale). Drop-in replacement for LatentToLocalField's (x, lam) output;
    supports omega_delta_source="trainable" only.

    With a multi-cluster layout, x covers all n_atoms * n_clusters atoms
    (cluster-major), graph weights are shared across clusters and lam is
    pooled over the whole register. Multi-cluster requires
    atom_token_mode="spatial".

    pulse_points (optional): adds a PulseQueryHead producing per-sample
    pulse corrections at that many control points (see pack()).
    """

    def __init__(self, n_atoms: int, correlator_order: int, latent_channels: int,
                 d_model: int = 24, measurement_kind: str = "correlators",
                 register_um: Sequence[Sequence[float]] = None,
                 use_fourier_pos: bool = False, atom_token_mode: str = "attention",
                 residual: bool = False, pre_quantum_norm: Optional[str] = None,
                 pre_quantum_gain: float = 0.4, layout: Optional[RegisterLayout] = None,
                 pulse_points: Optional[int] = None):
        super().__init__()
        validate_measurement_kind(measurement_kind)
        if layout is None:
            layout = RegisterLayout(n_atoms, correlator_order, register_um)
        if layout.n_clusters > 1 and atom_token_mode != "spatial":
            raise ValueError("n_clusters > 1 requires atom_token_mode='spatial'.")
        self.layout = layout
        n_atoms = layout.n_nodes
        self.n_atoms = n_atoms
        self.residual = residual
        # Optional batch normalization of the x logits: fixes the across-sample
        # spread of each atom's local field to pre_quantum_gain before the sigmoid.
        if pre_quantum_norm not in (None, "batchnorm"):
            raise ValueError(f"pre_quantum_norm='{pre_quantum_norm}' is not supported.")
        self.x_norm = nn.BatchNorm1d(n_atoms, affine=False) if pre_quantum_norm == "batchnorm" else None
        self.pre_quantum_gain = float(pre_quantum_gain)
        self.x_shift = nn.Parameter(torch.zeros(n_atoms)) if self.x_norm is not None else None
        if atom_token_mode == "spatial":
            if register_um is None:
                raise ValueError("atom_token_mode='spatial' requires register_um.")
            self.extractor = LatentToAtomPatches(register_um, latent_channels, d_model, layout=layout)
        else:
            atom_pos = None
            if use_fourier_pos:
                if register_um is None:
                    raise ValueError("use_fourier_pos=True requires register_um.")
                atom_pos = fourier_atom_features(register_um, d_model)
            self.extractor = LatentToAtomTokens(n_atoms, latent_channels, d_model, atom_pos=atom_pos)
        self.hyperedge_index, self.subsets = layout.hyperedge_index, list(layout.subsets)
        self.n_hyperedges = layout.n_hyperedges
        self._batched_index = _BatchedIndexCache(self.hyperedge_index, self.n_atoms, self.n_hyperedges)
        self.hconv1 = HypergraphConv(d_model, d_model, use_attention=True, heads=1)
        self.hconv2 = HypergraphConv(d_model, d_model, use_attention=True, heads=1)

        gate_nn = nn.Linear(d_model, 1)
        nn.init.normal_(gate_nn.weight, std=0.02)
        nn.init.zeros_(gate_nn.bias)
        self.hyperedge_pool = AttentionalAggregation(gate_nn=gate_nn)
        self.x_layer = nn.Linear(d_model, 1)
        self.lam_pool_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.lam_pool_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.lam_layer = nn.Linear(d_model, 1)
        self.pulse_head = PulseQueryHead(d_model, pulse_points) if pulse_points is not None else None

    def forward(self, z_spatial: torch.Tensor):
        x, lam, extract_attn, _ = self._encode(z_spatial)
        return x, lam, extract_attn

    def pack(self, z_spatial: torch.Tensor) -> torch.Tensor:
        """Qlayer input: [x, lam] plus, with a pulse head, the raw pulse
        corrections [Omega_1..Omega_Q, Delta_1..Delta_Q]."""
        x, lam, _, h = self._encode(z_spatial)
        parts = [x, lam.unsqueeze(-1)]
        if self.pulse_head is not None:
            parts.append(self.pulse_head(h))
        return torch.cat(parts, dim=-1)

    def _encode(self, z_spatial: torch.Tensor):
        B = z_spatial.shape[0]
        atom_tokens, extract_attn = self.extractor(z_spatial)

        batched_index = self._batched_index(B, atom_tokens.device)
        batched_node_idx, batched_edge_idx = batched_index
        n_edges_total = B * self.n_hyperedges

        h = atom_tokens.reshape(B * self.n_atoms, -1)

        attr1 = self.hyperedge_pool(h[batched_node_idx], batched_edge_idx, dim_size=n_edges_total)
        out = torch.relu(self.hconv1(h, batched_index, hyperedge_attr=attr1))
        h = h + out if self.residual else out
        attr2 = self.hyperedge_pool(h[batched_node_idx], batched_edge_idx, dim_size=n_edges_total)
        out = torch.relu(self.hconv2(h, batched_index, hyperedge_attr=attr2))
        h = h + out if self.residual else out

        h = h.reshape(B, self.n_atoms, -1)
        x_logit = self.x_layer(h).squeeze(-1)
        if self.x_norm is not None:
            x_logit = self.pre_quantum_gain * self.x_norm(x_logit) + self.x_shift
        x = torch.sigmoid(x_logit)

        q = self.lam_pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.lam_pool_attn(q, h, h)
        lam = self.lam_layer(pooled.squeeze(1)).squeeze(-1)

        return x, lam, extract_attn, h


def _merge_plan(cell_index: torch.Tensor, n_atoms: int, grid: Tuple[int, int], latent_hw: Tuple[int, int]):
    """Per axis, either several atoms merge into one latent cell (atom grid a
    multiple of the latent) or one merged cell writes a patch (latent a
    multiple of the atom grid). Returns (merge factors, patch factors,
    merge_idx [n_cells, merge_r * merge_c] of atom indices; n_atoms = empty).
    """
    merge, patch = [], []
    for g, size in zip(grid, latent_hw):
        if g >= size and g % size == 0:
            merge.append(g // size)
            patch.append(1)
        elif g < size and size % g == 0:
            merge.append(1)
            patch.append(size // g)
        else:
            raise ValueError(
                f"cell_write='merge' needs each atom grid axis to be a multiple or divisor of the "
                f"latent size; got atom grid {grid} and latent {latent_hw}. Use cell_write='resample'."
            )
    (mr, mc), (gr, gc) = merge, grid
    lookup = torch.full((gr * gc,), n_atoms, dtype=torch.long)
    lookup[cell_index] = torch.arange(n_atoms)
    rows = (torch.arange(gr // mr).view(-1, 1, 1, 1) * mr + torch.arange(mr).view(1, 1, -1, 1))
    cols = (torch.arange(gc // mc).view(1, -1, 1, 1) * mc + torch.arange(mc).view(1, 1, 1, -1))
    merge_idx = lookup[(rows * gc + cols).reshape(-1, mr * mc)]
    return tuple(merge), tuple(patch), merge_idx


class GraphQuantumToDecoderVector(nn.Module):
    """Correlator readout -> graph -> decoder-facing vector. Drop-in
    replacement for project_from_quantum's nn.Linear.

    readout[i] must match build_hyperedge_index's subset order -- verified
    against the real qlayer's _correlator_ops (same nested-loop structure
    over wires=range(n_atoms)).

    use_fourier_pos (opt-in): concatenates each hyperedge's own member-atom
    Fourier position features to its scalar readout value before
    hyperedge_proj, instead of projecting the bare scalar alone. Without
    this, every hyperedge's embedding is confined to hyperedge_proj's
    single weight column (a 1D line in d_model-space, positioned only by
    the scalar value) -- the attention in hconv1/hconv2 has no way to tell
    hyperedges apart by which atoms they cover, only by value. This gives
    it that missing identity. Member atoms are concatenated (not summed)
    into fixed correlator_order slots, padded with a learned pad_token for
    hyperedges below the max order, so no two distinct atom subsets can
    ever collide onto the same identity vector.

    output_mode="pooled" (default): attention-pool all atoms into one vector,
    then out_layer -> out_dim. output_mode="spatial": each atom writes its
    own latent patch at the cell it is routed to (requires latent_shape and
    a register layout); output width does not depend on the hyperedge count.
    When the latent is not an integer multiple of the atom grid, cell_write
    decides how atoms sharing a latent cell are combined:
      "resample": each atom writes one value per channel and the atom grid
        is bilinearly resampled to the latent size (low-pass: suppresses
        differences between neighbouring atoms).
      "merge": the features of the atoms sharing a cell are concatenated
        and mapped to the cell by one shared linear layer (lossless).
        Requires each axis of the atom grid to be a multiple or a divisor
        of the latent size.
    """

    def __init__(self, n_atoms: int, correlator_order: int, out_dim: int,
                 d_model: int = 24, measurement_kind: str = "correlators",
                 register_um: Sequence[Sequence[float]] = None,
                 use_fourier_pos: bool = False, fourier_atom_dim: int = 4,
                 residual: bool = False, layout: Optional[RegisterLayout] = None,
                 output_mode: str = "pooled", latent_shape: Optional[Tuple[int, int, int]] = None,
                 cell_write: str = "resample"):
        super().__init__()
        validate_measurement_kind(measurement_kind)
        if output_mode not in ("pooled", "spatial"):
            raise ValueError(f"output_mode='{output_mode}' is not supported. Use 'pooled' or 'spatial'.")
        if cell_write not in ("resample", "merge"):
            raise ValueError(f"cell_write='{cell_write}' is not supported. Use 'resample' or 'merge'.")
        if layout is None:
            layout = RegisterLayout(n_atoms, correlator_order, register_um)
        self.layout = layout
        n_atoms = layout.n_nodes
        self.n_atoms = n_atoms
        self.residual = residual
        self.output_mode = output_mode
        self.hyperedge_index, self.subsets = layout.hyperedge_index, list(layout.subsets)
        self.n_hyperedges = layout.n_hyperedges
        self._batched_index = _BatchedIndexCache(self.hyperedge_index, self.n_atoms, self.n_hyperedges)
        self.register_buffer("node_readout_idx", layout.node_readout_idx, persistent=False)

        self.use_fourier_pos = use_fourier_pos
        if use_fourier_pos:
            if register_um is None:
                raise ValueError("use_fourier_pos=True requires register_um.")
            identity_dim = correlator_order * fourier_atom_dim
            min_d_model = 1 + identity_dim
            if d_model < min_d_model:
                raise ValueError(
                    f"d_model={d_model} is too small for correlator_order="
                    f"{correlator_order} with fourier_atom_dim={fourier_atom_dim} "
                    f"(needs d_model >= {min_d_model} = 1 + correlator_order * "
                    "fourier_atom_dim, or hyperedge_proj's output rank bottlenecks "
                    "and compresses/loses part of the identity encoding). Increase "
                    "d_model, or lower fourier_atom_dim/correlator_order."
                )
            atom_pos_small = fourier_atom_features(layout.coords_um, fourier_atom_dim)
            self.register_buffer("atom_pos_small", atom_pos_small, persistent=False)
            self.pad_token = nn.Parameter(torch.zeros(fourier_atom_dim))
            self.register_buffer("slot_idx", layout.slot_idx.clone(), persistent=False)
            self.hyperedge_proj = nn.Linear(1 + identity_dim, d_model)
        else:
            self.hyperedge_proj = nn.Linear(1, d_model)

        self.hconv1 = HypergraphConv(d_model, d_model, use_attention=True, heads=1)
        self.hconv2 = HypergraphConv(d_model, d_model, use_attention=True, heads=1)

        if output_mode == "pooled":
            self.pool_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.out_layer = nn.Linear(d_model, out_dim)
        else:
            if latent_shape is None or layout.coords_um is None:
                raise ValueError("output_mode='spatial' requires latent_shape and a register layout.")
            channels, height, width = (int(v) for v in latent_shape)
            grid_rows, grid_cols = layout.grid_shape
            if channels * height * width != out_dim:
                raise ValueError(f"latent_shape={tuple(latent_shape)} does not match out_dim={out_dim}.")
            self.latent_shape = (channels, height, width)
            self.grid_shape = (grid_rows, grid_cols)
            self.merge = None
            if height % grid_rows == 0 and width % grid_cols == 0:
                self.patch = (height // grid_rows, width // grid_cols)
            elif cell_write == "merge":
                self.merge, self.patch, merge_idx = _merge_plan(
                    layout.cell_index, n_atoms, (grid_rows, grid_cols), (height, width)
                )
                self.register_buffer("merge_idx", merge_idx, persistent=False)
            else:
                self.patch = None
            patch_cells = 1 if self.patch is None else self.patch[0] * self.patch[1]
            merged = 1 if self.merge is None else self.merge[0] * self.merge[1]
            self.register_buffer("cell_index", layout.cell_index.clone(), persistent=False)
            self.cell_layer = nn.Linear(d_model * merged, channels * patch_cells)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _pop_derived_buffers(self, state_dict, prefix, ("atom_pos_small", "slot_idx"), prefix_ok=("slot_idx",))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def atoms_to_latent(self, atom_out: torch.Tensor) -> torch.Tensor:
        """[B, n_atoms, C*ph*pw] (or [B, n_atoms, C] when resampling) ->
        [B, C*H*W] in z.flatten(1) order. Cells without an atom stay zero.
        With merging, atom_out is per merged cell: [B, n_cells, C*ph*pw]."""
        B = atom_out.shape[0]
        channels, height, width = self.latent_shape
        grid_rows, grid_cols = self.grid_shape
        if self.merge is not None:
            ph, pw = self.patch
            cells = atom_out.reshape(B, height // ph, width // pw, channels, ph, pw)
            return cells.permute(0, 3, 1, 4, 2, 5).reshape(B, channels * height * width)
        if self.patch is None:
            cells = atom_out.new_zeros(B, grid_rows * grid_cols, channels)
            cells[:, self.cell_index] = atom_out
            grid = cells.transpose(1, 2).reshape(B, channels, grid_rows, grid_cols)
            return resample_grid(grid, (height, width)).reshape(B, channels * height * width)
        ph, pw = self.patch
        cells = atom_out.new_zeros(B, grid_rows * grid_cols, channels * ph * pw)
        cells[:, self.cell_index] = atom_out
        cells = cells.reshape(B, grid_rows, grid_cols, channels, ph, pw)
        return cells.permute(0, 3, 1, 4, 2, 5).reshape(B, channels * height * width)

    def write_latent(self, h: torch.Tensor) -> torch.Tensor:
        """Spatial output: atom features [B, n_atoms, d] -> [B, C*H*W]."""
        if self.merge is not None:
            # Concatenate the atoms of each merged cell; empty slots are zero.
            h_pad = torch.cat([h, h.new_zeros(h.shape[0], 1, h.shape[-1])], dim=1)
            h = h_pad[:, self.merge_idx].reshape(h.shape[0], self.merge_idx.shape[0], -1)
        return self.atoms_to_latent(self.cell_layer(h))

    def forward(self, readout: torch.Tensor):
        B = readout.shape[0]
        batched_index = self._batched_index(B, readout.device)

        if self.use_fourier_pos:
            # (n_atoms+1, atom_dim): real atom rows plus one pad row.
            atom_pos_table = torch.cat([self.atom_pos_small, self.pad_token.unsqueeze(0)], dim=0)
            identity = atom_pos_table[self.slot_idx].reshape(self.n_hyperedges, -1)
            identity = identity.unsqueeze(0).expand(B, -1, -1).reshape(B * self.n_hyperedges, -1)
            hyperedge_attr = self.hyperedge_proj(
                torch.cat([readout.reshape(-1, 1), identity], dim=-1)
            )
        else:
            hyperedge_attr = self.hyperedge_proj(readout.reshape(-1, 1))

        h = hyperedge_attr.reshape(B, self.n_hyperedges, -1)[:, self.node_readout_idx, :]
        h = h.reshape(B * self.n_atoms, -1)

        out = torch.relu(self.hconv1(h, batched_index, hyperedge_attr=hyperedge_attr))
        h = h + out if self.residual else out
        out = torch.relu(self.hconv2(h, batched_index, hyperedge_attr=hyperedge_attr))
        h = h + out if self.residual else out

        h = h.reshape(B, self.n_atoms, -1)
        if self.output_mode == "spatial":
            return self.write_latent(h)
        q = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, h, h)
        return self.out_layer(pooled.squeeze(1))
