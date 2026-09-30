"""Seeded Personalized PageRank risk head.

Propagates over BOTH CO_SPEND (ER pass 1) and SAME_ENTITY (ER pass 2,
argus.er.embed_cluster) edges — the architecture doc's full spec. Pass 1
alone produces zero CO_SPEND edges on this dataset (diagnosed in
docs/WRITEUP.md's Phase 2 section: the only transactions with multiple
genuinely-distinct input wallets are CoinJoin rounds, which pass 1 must skip
to avoid over-merging, and the change-address heuristic never links two
DIFFERENT wallets here), so before pass 2 runs, propagation over an
all-CO_SPEND, zero-edge subgraph is a no-op beyond the seed set itself.
Pass 2's SAME_ENTITY edges are what actually give this head graph structure
to propagate over on this dataset — see docs/contracts.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import igraph as ig

DAMPING = 0.85  # standard PageRank damping factor
DISTANCE_DECAY = 0.5  # explicit per-hop score multiplier, on top of PPR's own implicit decay


@dataclass
class RiskScore:
    node_id: str
    score: float
    seed_distance: int  # hops to nearest seed over CO_SPEND/SAME_ENTITY edges; 0 for a seed itself
    nearest_seed: str
    path: list[str]  # nearest_seed -> ... -> node_id


ENTITY_LINK_EDGE_TYPES = ("CO_SPEND", "SAME_ENTITY")


def _entity_link_subgraph(g: ig.Graph) -> ig.Graph:
    """An UNDIRECTED subgraph containing only CO_SPEND + SAME_ENTITY edges —
    both are symmetric "same entity" relationships, so PPR/BFS should walk
    either direction.

    delete_vertices=True prunes away every node with zero edges of these
    types rather than keeping the full graph's ~340k vertices around — with
    few/no such edges (e.g. before ER pass 2 has run) that turns a
    mathematically trivial computation (every seed keeps its own weight,
    nothing else gets any) into an expensive/pathological one:
    personalized_pagerank and an all-pairs distance query both scale with
    vertex count, and were observed to consume multiple GB of RAM and hang
    when run over the full disconnected graph instead of the pruned one.
    """
    edge_ids = [e.index for e in g.es if e["type"] in ENTITY_LINK_EDGE_TYPES]
    sub = g.subgraph_edges(edge_ids, delete_vertices=True)
    return sub.as_undirected(mode="collapse")


def compute_risk_scores(g: ig.Graph, seed_wallet_ids: list[str], damping: float = DAMPING) -> list[RiskScore]:
    """damping exposed as a parameter (default DAMPING, unchanged behavior):
    real-data finding (ARGUS dataset Track M benchmarking report,
    2026-09-30) — a sparse real seed set (17 wallets) needs propagation to
    reach farther than this dataset's own default was tuned for. Higher
    damping = lower restart probability = more hops before the walk resets
    to a seed, i.e. wider propagation from the same seed set.
    """
    seed_ids = set(seed_wallet_ids)
    sub = _entity_link_subgraph(g)
    names = sub.vs["name"] if sub.vcount() > 0 else []
    name_to_index = {name: i for i, name in enumerate(names)}

    raw: list[RiskScore] = []

    # A seed with no CO_SPEND/SAME_ENTITY edges at all was pruned out of
    # `sub` entirely — it trivially gets its own full weight at distance 0,
    # no graph computation needed.
    for wallet_id in seed_ids - set(names):
        raw.append(RiskScore(node_id=wallet_id, score=1.0, seed_distance=0, nearest_seed=wallet_id, path=[wallet_id]))

    seed_indices = sorted({name_to_index[w] for w in seed_ids if w in name_to_index})
    if seed_indices:
        pagerank = sub.personalized_pagerank(reset_vertices=seed_indices, damping=damping, directed=False)
        max_pr = max(pagerank) if pagerank else 0.0
        if max_pr > 0:
            pagerank = [p / max_pr for p in pagerank]

        # Multi-source BFS distance from the seed set, and which seed achieves
        # it per target — needed for the SEED_DIST reason code, the
        # distance-decay term, and the "top contributing path" evidence.
        dist_rows = sub.distances(source=seed_indices, target=None, mode="all")

        for v in range(sub.vcount()):
            best_dist = float("inf")
            best_seed_row = None
            for row_idx, seed_idx in enumerate(seed_indices):
                d = dist_rows[row_idx][v]
                if d < best_dist:
                    best_dist, best_seed_row = d, seed_idx
            if best_seed_row is None or best_dist == float("inf"):
                continue  # unreachable from any seed over CO_SPEND/SAME_ENTITY — no risk signal to report

            distance = int(best_dist)
            decayed_score = pagerank[v] * (DISTANCE_DECAY**distance)

            if distance == 0:
                path_ids = [names[v]]
            else:
                vpath = sub.get_shortest_paths(best_seed_row, to=v, mode="all", output="vpath")[0]
                path_ids = [names[i] for i in vpath]

            raw.append(
                RiskScore(
                    node_id=names[v],
                    score=decayed_score,
                    seed_distance=distance,
                    nearest_seed=names[best_seed_row],
                    path=path_ids,
                )
            )

    max_final = max((r.score for r in raw), default=0.0)
    if max_final > 0:
        for r in raw:
            r.score = r.score / max_final

    return raw


def risk_score_rows(scores: list[RiskScore]) -> list[dict]:
    rows = []
    for r in scores:
        reason_code = f"SEED_DIST={r.seed_distance}"
        evidence_json = json.dumps(
            {"seed_distance": r.seed_distance, "nearest_seed": r.nearest_seed, "path": r.path}
        )
        rows.append({"node_id": r.node_id, "score": r.score, "reason_code": reason_code, "evidence_json": evidence_json})
    return rows
