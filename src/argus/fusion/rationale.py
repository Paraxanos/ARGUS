"""Rationale-templating engine — architecture doc sec 4.4: turns a flagged
node's reason_codes (plus, when available, an attention-based evidence
subgraph from fusion/evidence.py) into a human-checkable explanation and its
supporting evidence nodes/edges.

Replaces argus.fusion.blend's former inline `_rationale`/`_extract_evidence`
(moved and generalized here — blend.py imports this module, not the other
way around) with a REGISTRY: each detector's reason_code prefix maps to an
(evidence_extractor, clause_renderer) pair. Adding a new detector's
explanation later means calling register_template() once, not editing an
if/elif chain in blend.py or here.
"""
from __future__ import annotations

from typing import Callable

# z above this is "worth a rationale clause" (~93rd percentile under a
# normal assumption) — a defensible "notable" bar, not a rigorous
# significance test. Every alert's components.anomaly score is shown
# regardless; this only gates whether the RATIONALE TEXT mentions it, so an
# unremarkable z-score doesn't produce a misleading "flagged as anomalous"
# sentence for a node that wasn't.
ANOMALY_NOTABLE_Z = 1.5

EvidenceExtractor = Callable[[str, dict], tuple[list[str], list[tuple[str, str]]]]
ClauseRenderer = Callable[[str, dict], "str | None"]


def _peel_evidence(node_id: str, evidence: dict) -> tuple[list[str], list[tuple[str, str]]]:
    wallets = evidence.get("wallets", [])
    txids = evidence.get("txids", [])
    return [*wallets, *txids], list(zip(wallets[:-1], wallets[1:]))


def _peel_clause(reason_code: str, evidence: dict) -> str | None:
    return f"part of a peeling chain with {reason_code.split('=')[1]} hops"


def _coinjoin_evidence(node_id: str, evidence: dict) -> tuple[list[str], list[tuple[str, str]]]:
    txid = evidence["txid"]
    inputs = evidence.get("input_wallets", [])
    outputs = evidence.get("output_wallets", [])
    edges = [(w, txid) for w in inputs] + [(txid, w) for w in outputs]
    return [txid, *inputs, *outputs], edges


def _coinjoin_clause(reason_code: str, evidence: dict) -> str | None:
    return f"a participant in a CoinJoin round with {reason_code.split('=')[1]} participants"


def _risk_evidence(node_id: str, evidence: dict) -> tuple[list[str], list[tuple[str, str]]]:
    path = evidence.get("path", [])
    return path, list(zip(path[:-1], path[1:]))


def _risk_clause(reason_code: str, evidence: dict) -> str | None:
    n = reason_code.split("=")[1]
    return "a known-illicit seed wallet" if n == "0" else f"{n} hop(s) from a known-illicit seed"


def _anomaly_evidence(node_id: str, evidence: dict) -> tuple[list[str], list[tuple[str, str]]]:
    # A reconstruction-error anomaly is a property of the flagged node
    # itself, not a multi-node structure — nothing to add beyond the node
    # already seeded into the alert's own evidence set. (An attention-based
    # explanation of WHY is exactly what fusion/evidence.py adds instead —
    # see build_rationale's attention_evidence parameter.)
    return [], []


def _anomaly_clause(reason_code: str, evidence: dict) -> str | None:
    z = float(reason_code.split("=")[1])
    return f"a statistical anomaly relative to its population (z-score {z:.2f})" if z > ANOMALY_NOTABLE_Z else None


def _pattern_sim_evidence(node_id: str, evidence: dict) -> tuple[list[str], list[tuple[str, str]]]:
    ref = evidence.get("nearest_reference_id")
    return ([ref], [(node_id, ref)]) if ref else ([], [])


def _pattern_sim_clause(reason_code: str, evidence: dict) -> str | None:
    pattern_type = evidence.get("pattern_type", "pattern")
    ref = evidence.get("nearest_reference_id", "an unspecified reference")
    similarity = evidence.get("similarity")
    sim_str = f"{similarity:.2f}" if similarity is not None else "an unspecified"
    return f"an embedding-similarity near-variant of a confirmed {pattern_type} instance ({ref}, cosine similarity {sim_str})"


_REGISTRY: dict[str, tuple[EvidenceExtractor, ClauseRenderer]] = {
    "PEEL_CHAIN_HOPS": (_peel_evidence, _peel_clause),
    "COINJOIN_ROUND_N": (_coinjoin_evidence, _coinjoin_clause),
    "SEED_DIST": (_risk_evidence, _risk_clause),
    "ANOMALY_ZSCORE": (_anomaly_evidence, _anomaly_clause),
    "PATTERN_SIM_PEELING": (_pattern_sim_evidence, _pattern_sim_clause),
    "PATTERN_SIM_COINJOIN": (_pattern_sim_evidence, _pattern_sim_clause),
}


def register_template(prefix: str, evidence_extractor: EvidenceExtractor, clause_renderer: ClauseRenderer) -> None:
    """Extensibility point for a future detector's reason_code prefix —
    register it here rather than editing this module's or blend.py's
    existing logic. A prefix already in the registry is overwritten
    (last-registered wins), matching a plain dict's own semantics.
    """
    _REGISTRY[prefix] = (evidence_extractor, clause_renderer)


def _lookup(reason_code: str) -> tuple[EvidenceExtractor, ClauseRenderer] | None:
    for prefix, pair in _REGISTRY.items():
        if reason_code.startswith(prefix):
            return pair
    return None


def extract_evidence(node_id: str, reason_code: str, evidence: dict) -> tuple[list[str], list[tuple[str, str]]]:
    entry = _lookup(reason_code)
    return entry[0](node_id, evidence) if entry else ([], [])


def render_clause(reason_code: str, evidence: dict) -> str | None:
    entry = _lookup(reason_code)
    return entry[1](reason_code, evidence) if entry else reason_code  # unrecognized: show verbatim, don't drop silently


def build_rationale(
    node_id: str,
    final_score: float,
    node_reasons: list[tuple[str, dict]],
    attention_neighbors: list | None = None,
) -> str:
    """attention_neighbors, when given, is fusion/evidence.py's
    AttentionEvidence.neighbors (or any object exposing .node_id/.weight) —
    kept as a loosely-typed sequence rather than importing fusion.evidence
    here, so this module has no hard dependency on torch/PyG and stays
    trivially unit-testable.
    """
    clauses: list[str] = []
    seen_codes: set[str] = set()
    for rc, ev in sorted(node_reasons, key=lambda pair: pair[0]):
        if rc in seen_codes:
            continue
        seen_codes.add(rc)
        clause = render_clause(rc, ev)
        if clause:
            clauses.append(clause)

    if attention_neighbors:
        top = max(attention_neighbors, key=lambda n: n.weight)
        clauses.append(
            f"the model's own attention weighted {top.node_id} most heavily among "
            f"{len(attention_neighbors)} evidence-subgraph neighbors (weight {top.weight:.2f})"
        )

    body = "; ".join(clauses) if clauses else "no specific structural signal"
    return f"{node_id} flagged (confidence {final_score:.2f}). {body[0].upper() + body[1:]}."
