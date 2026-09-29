"""Calibrated fusion of the three score heads into artifacts/alerts.json.

scores_anomaly.parquet is now a real model (argus.models.anomaly's graph
autoencoder, Dev B Phase 2) — see docs/contracts.md and that module's
docstring for the training/scoring design.

Rationale/evidence generation is argus.fusion.rationale's job (Dev B Phase
4's templating engine) — this module builds the raw alert (scores,
reason_codes) and delegates to it, rather than duplicating that logic
inline as an earlier version of this file did.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from argus.fusion.rationale import build_rationale, extract_evidence

# Fallback fill for a node with no anomaly row at all (shouldn't happen in
# practice — argus.models.anomaly scores every node in the graph — kept as a
# defensive default): 0.5 is "no information either way" on the [0,1] scale.
NEUTRAL_ANOMALY_SCORE = 0.5

ILLICIT_TYPES = ("ransomware", "darknet", "mixer")

# Below this many labeled nodes (or with only one class present), a 3-feature
# logistic regression is more likely to overfit/degenerate than to calibrate
# anything meaningful — fall back to a fixed blend instead.
MIN_LABELED_NODES = 30

# Fallback weights when calibration can't be trained. Pattern and risk carry
# real structural signal; anomaly is a known-constant placeholder, so it gets
# a small, harmless residual weight — a constant term added to every score
# doesn't change any two nodes' RELATIVE ranking, but the weight is already
# correct and nonzero for the day a real anomaly score replaces the constant.
FALLBACK_WEIGHTS = {"pattern": 0.5, "risk": 0.4, "anomaly": 0.1}

# Tuned against the real fused-score distribution, which turned out bimodal:
# nodes flagged ONLY by a pattern detector (real peeling/coinjoin structure)
# cluster near 0.67; nodes flagged ONLY by the risk head (currently just the
# seed set itself, per risk_ppr.py's documented CO_SPEND=0 limitation)
# cluster near 0.9998 — with nothing in between. A threshold ABOVE ~0.67
# would perversely keep the seed-echo cluster while dropping the real
# structural detections, which is backwards. Set below the low cluster, this
# threshold keeps the entire fusion universe (every node with ANY pattern or
# risk signal) as the alert list — ~2,249 of ~343,000 total graph nodes
# (~0.65%) on the full dataset, already a curated, reasonably-sized watchlist
# by construction (component_table only considers nodes with SOME signal).
ALERT_THRESHOLD = 0.6


def component_table(scores_pattern: pd.DataFrame, scores_risk: pd.DataFrame, scores_anomaly: pd.DataFrame) -> pd.DataFrame:
    """One row per node with ANY pattern or risk signal. Anomaly alone is
    never a reason to consider a node — every one of the graph's ~340k
    nodes gets an anomaly score (argus.models.anomaly scores the full
    population), so including it in the universe would blow that up to
    ~340k trivial rows; it only ever narrows or ranks within the
    pattern/risk-flagged set, matching the pre-Phase-2 placeholder's
    behavior exactly (same fusion universe either way).
    """
    pattern_max = scores_pattern.groupby("node_id")["score"].max().rename("pattern")
    risk_max = scores_risk.groupby("node_id")["score"].max().rename("risk")
    universe = pattern_max.index.union(risk_max.index)

    anomaly_by_node = scores_anomaly.set_index("node_id")["score"]  # exactly one row per node, by construction

    table = pd.DataFrame(index=universe)
    table["pattern"] = pattern_max.reindex(universe).fillna(0.0)
    table["risk"] = risk_max.reindex(universe).fillna(0.0)
    table["anomaly"] = anomaly_by_node.reindex(universe).fillna(NEUTRAL_ANOMALY_SCORE)
    return table.reset_index(names="node_id")


def _labeled_rows(table: pd.DataFrame, ground_truth_entities: pd.DataFrame) -> pd.DataFrame:
    labels = ground_truth_entities[["wallet_id", "entity_type"]].rename(columns={"wallet_id": "node_id"})
    labels["label"] = labels["entity_type"].isin(ILLICIT_TYPES).astype(int)
    return table.merge(labels[["node_id", "label"]], on="node_id", how="inner")


def _fit_calibrated_blend(labeled: pd.DataFrame) -> LogisticRegression | None:
    if len(labeled) < MIN_LABELED_NODES or labeled["label"].nunique() < 2:
        return None
    model = LogisticRegression()
    model.fit(labeled[["pattern", "risk", "anomaly"]].to_numpy(), labeled["label"].to_numpy())
    return model


def compute_final_scores(table: pd.DataFrame, ground_truth_entities: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Trains a calibrated logistic blend on synthetic labels from
    ground_truth/entities.parquet when enough labeled nodes exist (>=
    MIN_LABELED_NODES, both classes present); otherwise falls back to a fixed
    weighted average. Both paths are implemented and exercised by tests —
    which one runs depends on the data, not a flag.
    """
    labeled = _labeled_rows(table, ground_truth_entities)
    model = _fit_calibrated_blend(labeled)

    table = table.copy()
    if model is not None:
        table["final_score"] = model.predict_proba(table[["pattern", "risk", "anomaly"]].to_numpy())[:, 1]
        method = "calibrated_logistic"
    else:
        table["final_score"] = (
            FALLBACK_WEIGHTS["pattern"] * table["pattern"]
            + FALLBACK_WEIGHTS["risk"] * table["risk"]
            + FALLBACK_WEIGHTS["anomaly"] * table["anomaly"]
        )
        method = "fallback_weighted_average"
    return table, method


def build_alerts(
    final_table: pd.DataFrame,
    scores_pattern: pd.DataFrame,
    scores_risk: pd.DataFrame,
    scores_anomaly: pd.DataFrame,
    threshold: float,
    attention_evidence_fn=None,
) -> list[dict]:
    """attention_evidence_fn, when given, is called as
    attention_evidence_fn(node_id) for every alert and should return an
    fusion.evidence.AttentionEvidence (or None) — accepted duck-typed here
    (only .nodes/.edges/.neighbors are used) rather than imported, so this
    module has no hard dependency on torch/PyG and stays importable/testable
    without them, matching argus.fusion.rationale's own design choice.
    Omitted entirely (the default), alerts are built exactly as before
    Dev B Phase 4 — existing callers/tests need no changes.
    """
    # scores_pattern/scores_risk only ever cover nodes with real structural
    # signal (thousands of rows) — pre-indexing them in full is cheap.
    reasons_by_node: dict[str, list[tuple[str, dict]]] = {}
    for source in (scores_pattern, scores_risk):
        for row in source.itertuples(index=False):
            reasons_by_node.setdefault(row.node_id, []).append((row.reason_code, json.loads(row.evidence_json)))

    # scores_anomaly covers the FULL graph population (~340k rows on the
    # full dataset) — only look up the handful of flagged nodes below rather
    # than pre-parsing JSON for the whole table.
    anomaly_by_node = scores_anomaly.set_index("node_id")

    flagged = final_table[final_table["final_score"] >= threshold].sort_values("final_score", ascending=False)

    alerts = []
    for i, row in enumerate(flagged.itertuples(index=False)):
        node_reasons = list(reasons_by_node.get(row.node_id, []))
        if row.node_id in anomaly_by_node.index:
            a = anomaly_by_node.loc[row.node_id]
            node_reasons.append((a["reason_code"], json.loads(a["evidence_json"])))

        nodes_ev: set[str] = {row.node_id}
        edges_ev: set[tuple[str, str]] = set()
        for rc, ev in node_reasons:
            ns, es = extract_evidence(row.node_id, rc, ev)
            nodes_ev.update(ns)
            edges_ev.update(es)

        attention = attention_evidence_fn(row.node_id) if attention_evidence_fn else None
        attention_neighbors = None
        if attention is not None and attention.neighbors:
            nodes_ev.update(attention.nodes)
            edges_ev.update(attention.edges)
            attention_neighbors = attention.neighbors

        alerts.append(
            {
                "alert_id": f"alert_{i:05d}",
                "node_id": row.node_id,
                "final_score": float(row.final_score),
                "components": {"pattern": float(row.pattern), "risk": float(row.risk), "anomaly": float(row.anomaly)},
                "evidence": {"nodes": sorted(nodes_ev), "edges": sorted(list(e) for e in edges_ev)},
                "rationale": build_rationale(row.node_id, row.final_score, node_reasons, attention_neighbors),
            }
        )
    return alerts


def write_alerts(alerts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)
