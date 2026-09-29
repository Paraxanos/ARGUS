"""Embedding-similarity pattern detector — architecture doc sec 4.3's Pattern
head, second half: "Embedding-similarity search catches near-variants the
hard-coded pattern missed." The explicit structural detectors
(detectors/peeling.py, detectors/coinjoin.py) run first and are the primary,
high-precision signal; this module's job is recall on variants that broke
some structural assumption (e.g. a peeling chain with one hop's amount
continuity perturbed past CONTINUITY_TOLERANCE, or a CoinJoin round with more
than one output nudged off the modal denomination) but whose participants
still LOOK like the confirmed instances in embedding space.

METHOD: reuses argus.models.anomaly's already-trained shared encoder
embeddings (no second training pass — see that module's docstring) rather
than training one here. For each pattern type, the confirmed structural
detections' own wallets/transactions form a REFERENCE set; every other node
of the same type is a CANDIDATE scored by its cosine similarity to the
nearest reference embedding. A candidate close to ANY single confirmed
instance is flagged — not close to a centroid/average, since a pattern's
members can play structurally different roles (e.g. a peeling chain's
"sender" vs. "receiver" wallets) that a single average would blur together.

This appends MORE rows to scores_pattern.parquet alongside the classical
detectors' rows (per detectors/scores.py's own docstring) — never replaces
or deduplicates them; a node already a confirmed reference member is
excluded from the candidate pool (a wallet can't be a "near variant" of
itself).

THRESHOLDING — measured, not assumed (same standard as every other finding
in this repo): a single fixed cosine cutoff does not work here. Measured
directly on the full dataset, candidate-vs-reference similarity is wildly
different across (pattern_type, node_type) pairs — Transaction-vs-CoinJoin
is well separated (median similarity ~0.04, only ~0.1% of candidates reach
0.95: a CoinJoin transaction's many-in/many-out shape is genuinely rare),
but Wallet-vs-peeling is compressed (median already ~0.95, i.e. HALF the
population would clear a flat 0.95 bar) — the same root cause diagnosed in
argus.models.sage/embed_cluster: most wallets are structurally-ordinary
single-hop spenders, so their embeddings cluster tightly regardless of
whether they're near a peeling chain. An uncapped flat threshold measured
142,351 matches out of ~343k nodes (41%) before this fix — the same class of
failure as ER pass 2's uncapped HDBSCAN run, and fixed the same way: gate on
being a genuine OUTLIER RELATIVE TO ITS OWN CANDIDATE POPULATION (z-score,
exactly argus.models.anomaly's own methodology) in addition to an absolute
floor, rather than trusting one global cutoff to mean the same thing for
every embedding distribution shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from argus.detectors.coinjoin import CoinjoinRound
from argus.detectors.peeling import PeelingChain

# Absolute floor: cosine similarity is in [-1, 1]; below this it's not
# "alike" in any meaningful sense regardless of population statistics.
MIN_ABSOLUTE_SIMILARITY = 0.95

# A candidate must ALSO be a genuine outlier within its own candidate
# population (see module docstring) — z > 3 is a conservative "clearly
# unlike the bulk" bar, not a rigorous significance test.
Z_THRESHOLD = 3.0

# Below this many candidates, population statistics (mean/std) are too
# unstable to trust — fall back to the absolute floor alone. Real pipeline
# runs have tens of thousands of candidates; this only matters for small
# hand-built graphs (tests, or a tiny real dataset).
MIN_CANDIDATES_FOR_ADAPTIVE_THRESHOLD = 30

PEELING = "peeling"
COINJOIN = "coinjoin"


@dataclass
class PatternSimMatch:
    node_id: str
    node_type: str
    pattern_type: str  # PEELING | COINJOIN
    similarity: float  # cosine, in [-1, 1]
    nearest_reference_id: str


def _reference_ids(chains: list[PeelingChain], rounds: list[CoinjoinRound]) -> dict[str, dict[str, set[str]]]:
    """pattern_type -> node_type -> set of confirmed member ids."""
    refs: dict[str, dict[str, set[str]]] = {
        PEELING: {"Wallet": set(), "Transaction": set()},
        COINJOIN: {"Wallet": set(), "Transaction": set()},
    }
    for chain in chains:
        refs[PEELING]["Wallet"].update(chain.wallets)
        refs[PEELING]["Transaction"].update(chain.txids)
    for r in rounds:
        refs[COINJOIN]["Wallet"].update(r.input_wallets)
        refs[COINJOIN]["Wallet"].update(r.output_wallets)
        refs[COINJOIN]["Transaction"].add(r.txid)
    return refs


def _cosine_sim_matrix(candidates: np.ndarray, references: np.ndarray) -> np.ndarray:
    c = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-12)
    r = references / (np.linalg.norm(references, axis=1, keepdims=True) + 1e-12)
    return c @ r.T  # (n_candidates, n_references)


def _flag_mask(best_sim: np.ndarray, min_absolute_similarity: float) -> np.ndarray:
    """See module docstring's THRESHOLDING section: an absolute floor alone
    means wildly different things across (pattern_type, node_type) pairs, so
    a candidate must also be a genuine outlier within its OWN candidate
    population — unless that population is too small to trust the
    statistics, in which case the absolute floor is all there is.
    """
    if len(best_sim) < MIN_CANDIDATES_FOR_ADAPTIVE_THRESHOLD:
        return best_sim >= min_absolute_similarity
    std = best_sim.std()
    if std < 1e-8:
        return np.zeros(len(best_sim), dtype=bool)  # every candidate equally (dis)similar -- no signal to act on
    z = (best_sim - best_sim.mean()) / std
    return (best_sim >= min_absolute_similarity) & (z > Z_THRESHOLD)


def detect_pattern_similarity(
    chains: list[PeelingChain],
    rounds: list[CoinjoinRound],
    embeddings: dict[str, np.ndarray],
    node_ids: dict[str, list[str]],
    min_absolute_similarity: float = MIN_ABSOLUTE_SIMILARITY,
) -> list[PatternSimMatch]:
    refs = _reference_ids(chains, rounds)
    # A node already a confirmed member of EITHER pattern type is never a
    # candidate — it can't be a "near variant" of a pattern it's already a
    # confirmed part of (of the same or the other type).
    all_reference_ids = {
        node_type: refs[PEELING][node_type] | refs[COINJOIN][node_type] for node_type in ("Wallet", "Transaction")
    }

    matches: list[PatternSimMatch] = []
    for node_type in ("Wallet", "Transaction"):
        ids = node_ids.get(node_type, [])
        emb = embeddings.get(node_type)
        if not ids or emb is None or emb.shape[0] == 0:
            continue
        id_to_row = {nid: i for i, nid in enumerate(ids)}

        candidate_ids = [nid for nid in ids if nid not in all_reference_ids[node_type]]
        if not candidate_ids:
            continue
        candidate_rows = np.array([id_to_row[nid] for nid in candidate_ids])
        candidate_emb = emb[candidate_rows]

        for pattern_type in (PEELING, COINJOIN):
            reference_ids = sorted(refs[pattern_type][node_type] & set(ids))
            if not reference_ids:
                continue
            reference_rows = np.array([id_to_row[nid] for nid in reference_ids])
            reference_emb = emb[reference_rows]

            sim = _cosine_sim_matrix(candidate_emb, reference_emb)
            best_idx = sim.argmax(axis=1)
            best_sim = sim[np.arange(len(candidate_ids)), best_idx]

            flagged = _flag_mask(best_sim, min_absolute_similarity)
            for i in np.flatnonzero(flagged):
                matches.append(
                    PatternSimMatch(
                        node_id=candidate_ids[i],
                        node_type=node_type,
                        pattern_type=pattern_type,
                        similarity=float(best_sim[i]),
                        nearest_reference_id=reference_ids[best_idx[i]],
                    )
                )
    return matches


def pattern_sim_rows(matches: list[PatternSimMatch]) -> list[dict]:
    rows = []
    for m in matches:
        reason_code = f"PATTERN_SIM_{m.pattern_type.upper()}={m.similarity:.2f}"
        evidence_json = json.dumps(
            {
                "pattern_type": m.pattern_type,
                "similarity": m.similarity,
                "nearest_reference_id": m.nearest_reference_id,
                "node_type": m.node_type,
            }
        )
        # (similarity + 1) / 2: cosine similarity's [-1, 1] range mapped
        # monotonically onto every other score head's contracted [0, 1]
        # (docs/contracts.md) — order-preserving, so the SIMILARITY_THRESHOLD
        # gate above is unaffected by this rescaling.
        score = (m.similarity + 1.0) / 2.0
        rows.append({"node_id": m.node_id, "score": score, "reason_code": reason_code, "evidence_json": evidence_json})
    return rows
