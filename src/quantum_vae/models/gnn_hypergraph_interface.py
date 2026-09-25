"""Hypergraph-based replacement for LatentToLocalField / project_from_quantum.

GraphLatentToLocalField: encoder latent -> per-atom x, global lam.
GraphQuantumToDecoderVector: correlator readout -> decoder-facing vector.

Hyperedges are built from atom subsets (order 1..correlator_order), in the
same enumeration order as _correlator_ops in quantum_vae_neutral_atom.py.
Batched via block-diagonal graph construction (no Python loop over batch).
"""
import itertools
from typing import Sequence

import torch
import torch.nn as nn
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


class GraphLatentToLocalField(nn.Module):
    """Cross-attention extraction + hypergraph message passing, producing
    x_i (per-atom local-detuning weight, in [0, 1]) and lam (pooled global
    scale). Drop-in replacement for LatentToLocalField's (x, lam) output;
    supports n_clusters=1 and omega_delta_source="trainable" only.
    """

    def __init__(self, n_atoms: int, correlator_order: int, latent_channels: int,
                 d_model: int = 24, measurement_kind: str = "correlators",
                 register_um: Sequence[Sequence[float]] = None,
                 use_fourier_pos: bool = False):
        super().__init__()
        validate_measurement_kind(measurement_kind)
        self.n_atoms = n_atoms
        atom_pos = None
        if use_fourier_pos:
            if register_um is None:
                raise ValueError("use_fourier_pos=True requires register_um.")
            atom_pos = fourier_atom_features(register_um, d_model)
        self.extractor = LatentToAtomTokens(n_atoms, latent_channels, d_model, atom_pos=atom_pos)
        self.hyperedge_index, self.subsets = build_hyperedge_index(n_atoms, correlator_order)
        self.n_hyperedges = len(self.subsets)
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

    def forward(self, z_spatial: torch.Tensor):
        B = z_spatial.shape[0]
        atom_tokens, extract_attn = self.extractor(z_spatial)

        batched_index = batch_hyperedge_index(
            self.hyperedge_index, self.n_atoms, self.n_hyperedges, B, atom_tokens.device
        )
        batched_node_idx, batched_edge_idx = batched_index
        n_edges_total = B * self.n_hyperedges

        h = atom_tokens.reshape(B * self.n_atoms, -1)

        attr1 = self.hyperedge_pool(h[batched_node_idx], batched_edge_idx, dim_size=n_edges_total)
        h = torch.relu(self.hconv1(h, batched_index, hyperedge_attr=attr1))
        attr2 = self.hyperedge_pool(h[batched_node_idx], batched_edge_idx, dim_size=n_edges_total)
        h = torch.relu(self.hconv2(h, batched_index, hyperedge_attr=attr2))

        h = h.reshape(B, self.n_atoms, -1)
        x = torch.sigmoid(self.x_layer(h)).squeeze(-1)

        q = self.lam_pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.lam_pool_attn(q, h, h)
        lam = self.lam_layer(pooled.squeeze(1)).squeeze(-1)

        return x, lam, extract_attn


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
    """

    def __init__(self, n_atoms: int, correlator_order: int, out_dim: int,
                 d_model: int = 24, measurement_kind: str = "correlators",
                 register_um: Sequence[Sequence[float]] = None,
                 use_fourier_pos: bool = False, fourier_atom_dim: int = 4):
        super().__init__()
        validate_measurement_kind(measurement_kind)
        self.n_atoms = n_atoms
        self.hyperedge_index, self.subsets = build_hyperedge_index(n_atoms, correlator_order)
        self.n_hyperedges = len(self.subsets)

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
            atom_pos_small = fourier_atom_features(register_um, fourier_atom_dim)
            self.register_buffer("atom_pos_small", atom_pos_small)
            self.pad_token = nn.Parameter(torch.zeros(fourier_atom_dim))
            slot_idx = build_hyperedge_slot_index(self.subsets, correlator_order, n_atoms)
            self.register_buffer("slot_idx", slot_idx)
            self.hyperedge_proj = nn.Linear(1 + identity_dim, d_model)
        else:
            self.hyperedge_proj = nn.Linear(1, d_model)

        self.hconv1 = HypergraphConv(d_model, d_model, use_attention=True, heads=1)
        self.hconv2 = HypergraphConv(d_model, d_model, use_attention=True, heads=1)

        self.pool_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.out_layer = nn.Linear(d_model, out_dim)

    def forward(self, readout: torch.Tensor):
        B = readout.shape[0]
        batched_index = batch_hyperedge_index(
            self.hyperedge_index, self.n_atoms, self.n_hyperedges, B, readout.device
        )

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

        h = hyperedge_attr.reshape(B, self.n_hyperedges, -1)[:, :self.n_atoms, :]
        h = h.reshape(B * self.n_atoms, -1)

        h = torch.relu(self.hconv1(h, batched_index, hyperedge_attr=hyperedge_attr))
        h = torch.relu(self.hconv2(h, batched_index, hyperedge_attr=hyperedge_attr))

        h = h.reshape(B, self.n_atoms, -1)
        q = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, h, h)
        return self.out_layer(pooled.squeeze(1))
