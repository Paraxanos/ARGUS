"""Attention-based evidence subgraph extraction — architecture doc sec 4.4:
"extract an evidence subgraph: the k-hop neighborhood restricted to the top
attention-weighted edges." Reuses models/anomaly.py's already-trained shared
encoder (architecture sec 4.3's stated design) rather than training a
separate model — the encoder + its input graph tensors are persisted to
disk by detectors/cli.py right after training (save_encoder_checkpoint) and
reloaded here, so this module never needs a second training pass, and
extraction happens at fusion time over the small (~2,000-alert) final list,
not the full ~343k-node population.

METHOD: the shared encoder is 2 layers, so a node's final embedding depends
on its 2-hop neighborhood. Layer 2's attention weights explain each 1-hop
neighbor's direct contribution to the final embedding; layer 1's (computed
on the same nodes, one message-passing step earlier) explain each of THOSE
neighbors' own incoming contributions — i.e. the target's 2-hop
neighborhood. Both are captured in one extra (no-backprop) forward pass via
argus.models.encoder.TemporalHeteroGATEncoder.forward_with_attention. For a
target node, this module walks both layers' attention weights inbound to
that node (and then inbound to its top 1-hop neighbors), keeps the top N by
weight per hop — the "top attention-weighted edges" the architecture doc
asks for — and reports which specific node contributed each one, so an
investigator sees not just a confidence number but which neighbors the
model itself weighted most heavily.

PERFORMANCE (measured, not assumed — see the Phase 4 commit message): the
forward pass over the WHOLE graph is the expensive part, but is identical
for every target node — only the neighbor lookup varies. Computing it fresh
per node (an earlier version of this module did exactly that) turned "a few
hundred alerts" into "a few hundred full-graph forward passes," measured
directly to add several minutes to what should have been a sub-second
lookup. build_attention_context() runs that forward pass ONCE; the returned
AttentionContext is then reused by extract_attention_evidence() for every
alert — a plain dict/index-map lookup, not a model call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from argus.models.encoder import HEADS, TIME2VEC_DIM, TemporalHeteroGATEncoder

DEFAULT_TOP_N = 5


@dataclass
class AttentionNeighbor:
    node_id: str
    node_type: str
    relation: str
    weight: float
    hop: int  # 1 (directly attended-to by the target) or 2 (attended-to by a hop-1 neighbor)
    connects_to: str  # target_node_id for hop 1; the hop-1 neighbor's node_id for hop 2


@dataclass
class AttentionEvidence:
    target_node_id: str
    neighbors: list[AttentionNeighbor] = field(default_factory=list)

    @property
    def nodes(self) -> list[str]:
        return sorted({self.target_node_id, *(n.node_id for n in self.neighbors)})

    @property
    def edges(self) -> list[tuple[str, str]]:
        return sorted({(n.node_id, n.connects_to) for n in self.neighbors})


def save_encoder_checkpoint(
    encoder: TemporalHeteroGATEncoder,
    data: HeteroData,
    node_ids: dict[str, list[str]],
    edge_types: list[tuple[str, str, str]],
    temporal_edge_types: set[tuple[str, str, str]],
    hidden_dim: int,
    embedding_dim: int,
    path: Path,
    heads: int = HEADS,
    time2vec_dim: int = TIME2VEC_DIM,
) -> None:
    """Persists exactly what's needed to rebuild the trained encoder and
    re-run forward_with_attention later — NOT the full _GraphAutoencoder
    (its decoders are only needed for anomaly scoring, already done by the
    time this is called; evidence extraction only ever needs the encoder).
    heads/time2vec_dim must match training-time values exactly or
    load_state_dict fails on a shape mismatch (verified directly — an
    earlier version of this function omitted them, silently reconstructing
    the encoder with this module's *default* values instead).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "data": data,
            "node_ids": node_ids,
            "edge_types": edge_types,
            "temporal_edge_types": temporal_edge_types,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "heads": heads,
            "time2vec_dim": time2vec_dim,
        },
        path,
    )


def load_encoder_checkpoint(path: Path) -> tuple[TemporalHeteroGATEncoder, HeteroData, dict[str, list[str]]]:
    # weights_only=False: this checkpoint intentionally carries a HeteroData
    # object (arbitrary Python, not just tensors) alongside the state dict —
    # produced by this same codebase, not an untrusted third-party file.
    checkpoint = torch.load(path, weights_only=False)
    encoder = TemporalHeteroGATEncoder(
        checkpoint["edge_types"],
        checkpoint["temporal_edge_types"],
        hidden_dim=checkpoint["hidden_dim"],
        out_dim=checkpoint["embedding_dim"],
        heads=checkpoint["heads"],
        time2vec_dim=checkpoint["time2vec_dim"],
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    return encoder, checkpoint["data"], checkpoint["node_ids"]


@dataclass
class AttentionContext:
    """Everything extract_attention_evidence needs, computed ONCE (one
    full-graph forward pass) and reused across every alert — see module
    docstring's PERFORMANCE section.
    """

    attention: dict[int, dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]]
    index_maps: dict[str, dict[str, int]]
    reverse_index: dict[str, dict[int, str]]


def build_attention_context(
    encoder: TemporalHeteroGATEncoder, data: HeteroData, node_ids: dict[str, list[str]]
) -> AttentionContext:
    encoder.eval()
    with torch.no_grad():
        _, attention = encoder.forward_with_attention(data)
    index_maps = {nt: {nid: i for i, nid in enumerate(ids)} for nt, ids in node_ids.items()}
    reverse_index = {nt: {i: nid for nid, i in m.items()} for nt, m in index_maps.items()}
    return AttentionContext(attention=attention, index_maps=index_maps, reverse_index=reverse_index)


def extract_attention_evidence(
    context: AttentionContext,
    target_node_id: str,
    target_node_type: str,
    top_n: int = DEFAULT_TOP_N,
) -> AttentionEvidence:
    attention = context.attention
    index_maps = context.index_maps
    reverse_index = context.reverse_index

    target_idx = index_maps.get(target_node_type, {}).get(target_node_id)
    if target_idx is None:
        return AttentionEvidence(target_node_id=target_node_id)

    def top_incoming(layer: int, node_type: str, local_idx: int, connects_to: str, hop: int) -> list[AttentionNeighbor]:
        # A node can send a message into the same destination via more than
        # one relation (e.g. Wallet -> Transaction via both FUNDS and the
        # synthetic rev_PAYS ToUndirected adds) — keyed by node_id so the
        # SAME physical neighbor doesn't occupy two of the top_n slots (or
        # get hop-2-expanded twice) under two different relation labels.
        best_by_node: dict[str, AttentionNeighbor] = {}
        for (src_type, relation, dst_type), (edge_index, alpha) in attention[layer].items():
            if dst_type != node_type:
                continue
            mask = edge_index[1] == local_idx
            if not bool(mask.any()):
                continue
            for pos, weight in zip(edge_index[0][mask].tolist(), alpha[mask].tolist()):
                src_id = reverse_index.get(src_type, {}).get(pos)
                if src_id is None:
                    continue
                if src_id not in best_by_node or weight > best_by_node[src_id].weight:
                    best_by_node[src_id] = AttentionNeighbor(
                        node_id=src_id, node_type=src_type, relation=relation, weight=float(weight),
                        hop=hop, connects_to=connects_to,
                    )
        candidates = sorted(best_by_node.values(), key=lambda c: c.weight, reverse=True)
        return candidates[:top_n]

    hop1 = top_incoming(layer=2, node_type=target_node_type, local_idx=target_idx, connects_to=target_node_id, hop=1)
    neighbors = list(hop1)

    for n1 in hop1:
        n1_idx = index_maps.get(n1.node_type, {}).get(n1.node_id)
        if n1_idx is None:
            continue
        neighbors.extend(top_incoming(layer=1, node_type=n1.node_type, local_idx=n1_idx, connects_to=n1.node_id, hop=2))

    return AttentionEvidence(target_node_id=target_node_id, neighbors=neighbors)
