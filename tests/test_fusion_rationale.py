"""argus.fusion.rationale: the templating engine — reason_code -> evidence
extraction and clause rendering, plus the registry's extensibility point.
"""
from __future__ import annotations

from dataclasses import dataclass

from argus.fusion import rationale
from argus.fusion.rationale import build_rationale, extract_evidence, register_template, render_clause


def test_peel_chain_evidence_and_clause():
    evidence = {"wallets": ["w1", "w2", "w3"], "txids": ["tx1", "tx2"]}
    nodes, edges = extract_evidence("w2", "PEEL_CHAIN_HOPS=3", evidence)
    assert set(nodes) == {"w1", "w2", "w3", "tx1", "tx2"}
    assert edges == [("w1", "w2"), ("w2", "w3")]
    assert render_clause("PEEL_CHAIN_HOPS=3", evidence) == "part of a peeling chain with 3 hops"


def test_coinjoin_evidence_and_clause():
    evidence = {"txid": "tx_cj", "input_wallets": ["a", "b"], "output_wallets": ["c", "d"]}
    nodes, edges = extract_evidence("a", "COINJOIN_ROUND_N=4", evidence)
    assert set(nodes) == {"tx_cj", "a", "b", "c", "d"}
    assert set(edges) == {("a", "tx_cj"), ("b", "tx_cj"), ("tx_cj", "c"), ("tx_cj", "d")}
    assert render_clause("COINJOIN_ROUND_N=4", evidence) == "a participant in a CoinJoin round with 4 participants"


def test_risk_seed_distance_evidence_and_clause():
    evidence = {"path": ["seed1", "mid1", "target"]}
    nodes, edges = extract_evidence("target", "SEED_DIST=2", evidence)
    assert nodes == ["seed1", "mid1", "target"]
    assert edges == [("seed1", "mid1"), ("mid1", "target")]
    assert render_clause("SEED_DIST=2", evidence) == "2 hop(s) from a known-illicit seed"
    assert render_clause("SEED_DIST=0", {}) == "a known-illicit seed wallet"


def test_anomaly_clause_gated_by_notable_z_threshold():
    assert render_clause("ANOMALY_ZSCORE=0.30", {}) is None  # unremarkable -- no misleading clause
    clause = render_clause("ANOMALY_ZSCORE=4.47", {})
    assert clause == "a statistical anomaly relative to its population (z-score 4.47)"
    # Never a multi-node structure regardless of z -- see the module docstring.
    assert extract_evidence("w0", "ANOMALY_ZSCORE=4.47", {}) == ([], [])


def test_pattern_sim_evidence_and_clause():
    evidence = {"pattern_type": "coinjoin", "similarity": 0.97, "nearest_reference_id": "tx_ref", "node_type": "Transaction"}
    nodes, edges = extract_evidence("tx_candidate", "PATTERN_SIM_COINJOIN=0.97", evidence)
    assert nodes == ["tx_ref"]
    assert edges == [("tx_candidate", "tx_ref")]
    clause = render_clause("PATTERN_SIM_COINJOIN=0.97", evidence)
    assert "tx_ref" in clause and "0.97" in clause and "coinjoin" in clause


def test_unrecognized_reason_code_shown_verbatim_not_dropped():
    assert extract_evidence("x", "SOME_FUTURE_CODE=1", {}) == ([], [])
    assert render_clause("SOME_FUTURE_CODE=1", {}) == "SOME_FUTURE_CODE=1"


def test_build_rationale_combines_multiple_reason_codes_deduplicated():
    node_reasons = [
        ("COINJOIN_ROUND_N=5", {"txid": "tx", "input_wallets": [], "output_wallets": []}),
        ("COINJOIN_ROUND_N=5", {"txid": "tx", "input_wallets": [], "output_wallets": []}),  # duplicate, must not double-render
        ("ANOMALY_ZSCORE=0.10", {}),  # unremarkable -- must not appear
    ]
    text = build_rationale("wX", 0.83, node_reasons)
    assert text.startswith("wX flagged (confidence 0.83).")
    assert text.count("CoinJoin round") == 1
    assert "z-score" not in text


def test_build_rationale_with_no_reasons_says_so():
    text = build_rationale("wX", 0.61, [])
    assert "specific structural signal" in text  # sentence-cased to "No specific..." — check case-insensitively


@dataclass
class _FakeAttentionNeighbor:
    node_id: str
    weight: float


def test_build_rationale_appends_attention_clause_when_provided():
    neighbors = [_FakeAttentionNeighbor("tx_a", 0.42), _FakeAttentionNeighbor("tx_b", 0.91)]
    text = build_rationale("wX", 0.75, [], attention_neighbors=neighbors)
    assert "tx_b" in text  # the highest-weight neighbor, not tx_a
    assert "0.91" in text
    assert "2 evidence-subgraph neighbors" in text


def test_register_template_is_a_real_extensibility_point():
    calls = []

    def fake_evidence(node_id, evidence):
        calls.append((node_id, evidence))
        return ["extra_node"], [(node_id, "extra_node")]

    def fake_clause(reason_code, evidence):
        return "a brand new detector's finding"

    register_template("MY_NEW_DETECTOR", fake_evidence, fake_clause)
    try:
        nodes, edges = extract_evidence("n1", "MY_NEW_DETECTOR=1", {"k": "v"})
        assert nodes == ["extra_node"]
        assert edges == [("n1", "extra_node")]
        assert render_clause("MY_NEW_DETECTOR=1", {}) == "a brand new detector's finding"
        assert calls == [("n1", {"k": "v"})]
    finally:
        del rationale._REGISTRY["MY_NEW_DETECTOR"]  # don't leak state into other tests
