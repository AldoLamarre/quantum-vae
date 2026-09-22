"""Hypergraph-based replacement for LatentToLocalField / project_from_quantum.

GraphLatentToLocalField: encoder latent -> per-atom x, global lam.
GraphQuantumToDecoderVector: correlator readout -> decoder-facing vector.

Hyperedges are built from atom subsets (order 1..correlator_order), in the
same enumeration order as _correlator_ops in quantum_vae_neutral_atom.py.
Batched via block-diagonal graph construction (no Python loop over batch).
"""
import itertools

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


class LatentToAtomTokens(nn.Module):
    """n_atoms learned queries cross-attend into the encoder's spatial
    latent grid (kept unflattened, so 2D locality reaches the attention).
    """

    def __init__(self, n_atoms: int, latent_channels: int, d_model: int, n_heads: int = 4):
        super().__init__()
        self.atom_queries = nn.Parameter(torch.randn(n_atoms, d_model) * 0.02)
        self.kv_proj = nn.Linear(latent_channels, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_tokens = nn.LayerNorm(latent_channels)

    def forward(self, z_spatial: torch.Tensor):
        B, C, H, W = z_spatial.shape
        tokens = z_spatial.flatten(2).transpose(1, 2)
        tokens = self.norm_tokens(tokens)
        kv = self.kv_proj(tokens)
        q = self.atom_queries.unsqueeze(0).expand(B, -1, -1)
        atom_tokens, attn_weights = self.attn(q, kv, kv, need_weights=True)
        return atom_tokens, attn_weights


class GraphLatentToLocalField(nn.Module):
    """Cross-attention extraction + hypergraph message passing, producing
    x_i (per-atom local-detuning weight, in [0, 1]) and lam (pooled global
    scale). Drop-in replacement for LatentToLocalField's (x, lam) output;
    supports n_clusters=1 and omega_delta_source="trainable" only.
    """

    def __init__(self, n_atoms: int, correlator_order: int, latent_channels: int,
                 d_model: int = 24, measurement_kind: str = "correlators"):
        super().__init__()
        validate_measurement_kind(measurement_kind)
        self.n_atoms = n_atoms
        self.extractor = LatentToAtomTokens(n_atoms, latent_channels, d_model)
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
    """

    def __init__(self, n_atoms: int, correlator_order: int, out_dim: int,
                 d_model: int = 24, measurement_kind: str = "correlators"):
        super().__init__()
        validate_measurement_kind(measurement_kind)
        self.n_atoms = n_atoms
        self.hyperedge_index, self.subsets = build_hyperedge_index(n_atoms, correlator_order)
        self.n_hyperedges = len(self.subsets)

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

        hyperedge_attr = self.hyperedge_proj(readout.reshape(-1, 1))
        h = hyperedge_attr.reshape(B, self.n_hyperedges, -1)[:, :self.n_atoms, :]
        h = h.reshape(B * self.n_atoms, -1)

        h = torch.relu(self.hconv1(h, batched_index, hyperedge_attr=hyperedge_attr))
        h = torch.relu(self.hconv2(h, batched_index, hyperedge_attr=hyperedge_attr))

        h = h.reshape(B, self.n_atoms, -1)
        q = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(q, h, h)
        return self.out_layer(pooled.squeeze(1))
