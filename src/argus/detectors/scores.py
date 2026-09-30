"""Converts peeling/coinjoin detector output into scores_pattern.parquet-shaped
rows: node_id, score, reason_code, evidence_json — per docs/contracts.md's hard
rule that every head emits reason_code + evidence_json, no exceptions. The
same write_scores() writer is reused for scores_risk.parquet (identical
4-column schema) — see argus.detectors.risk_ppr.

This artifact currently contains ONLY classical (structural) detections from
argus.detectors.peeling and argus.detectors.coinjoin. Dev B's
detectors/pattern_sim.py (embedding-similarity variant) would append MORE rows
to this SAME artifact later — not implemented in this repo. A node touched by
more than one detection (e.g. two different chains) gets one row per
detection, not a collapsed/deduplicated single row — evidence is never merged
away.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from argus.detectors.coinjoin import CoinjoinRound
from argus.detectors.peeling import PeelingChain


def peeling_chain_rows(chains: list[PeelingChain], behaviour: dict[str, dict] | None = None) -> list[dict]:
    """behaviour (argus.detectors.peeling.chain_behaviour_scores), when given, replaces each chain's structural
    confidence with its behavioural score and records the cash-out / service-source evidence."""
    rows = []
    for chain in chains:
        reason_code = f"PEEL_CHAIN_HOPS={chain.hop_count}"
        b = (behaviour or {}).get(chain.chain_id)
        evidence = {"chain_id": chain.chain_id, "hop_count": chain.hop_count, "txids": chain.txids, "wallets": chain.wallets}
        if b is not None:
            evidence.update({"swept_fraction": b["swept_fraction"], "service_source": b["service_source"]})
        evidence_json = json.dumps(evidence)
        score = b["score"] if b is not None else chain.confidence
        for node_id in (*chain.wallets, *chain.txids):
            rows.append({"node_id": node_id, "score": score, "reason_code": reason_code, "evidence_json": evidence_json})
    return rows


def coinjoin_round_rows(rounds: list[CoinjoinRound], participant_factor: float = 1.0) -> list[dict]:
    """participant_factor scales the score: taking part in a CoinJoin is a privacy practice and a mixing SIGNAL,
    not evidence of crime on its own (it becomes strong only together with taint from a known-bad source, which is
    fusion's job). 1.0 keeps the previous behaviour."""
    rows = []
    for r in rounds:
        reason_code = f"COINJOIN_ROUND_N={r.participant_count}"
        evidence_json = json.dumps(
            {
                "txid": r.txid,
                "participant_count": r.participant_count,
                "denomination": r.denomination,
                "input_wallets": r.input_wallets,
                "output_wallets": r.output_wallets,
            }
        )
        score = r.equal_output_fraction * participant_factor
        node_ids = {r.txid, *r.input_wallets, *r.output_wallets}
        for node_id in node_ids:
            rows.append({"node_id": node_id, "score": score, "reason_code": reason_code, "evidence_json": evidence_json})
    return rows


def write_scores(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["node_id", "score", "reason_code", "evidence_json"])
    df.to_parquet(path, index=False)
