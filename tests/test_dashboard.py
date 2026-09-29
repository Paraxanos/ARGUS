"""argus.dashboard.app: the Streamlit dashboard's testable logic (data
loading, the alerts table shape, and the pyvis evidence-graph renderer).
`main()`'s actual Streamlit UI calls are exercised manually (this repo has
no browser-automation test harness) — see the Phase 5 commit message for
that verification — but every pure function here has a real, direct test,
same standard as every other module in this repo.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from argus.dashboard.app import (
    _alerts_dataframe,
    _strip_external_cdn_links,
    load_alerts,
    load_node_types,
    render_evidence_graph,
)


def _sample_alert(alert_id: str = "alert_00000", node_id: str = "wallet_1") -> dict:
    return {
        "alert_id": alert_id,
        "node_id": node_id,
        "final_score": 0.87,
        "components": {"pattern": 0.9, "risk": 0.5, "anomaly": 0.6},
        "evidence": {"nodes": [node_id, "tx_1", "ip_1"], "edges": [[node_id, "tx_1"], ["tx_1", "ip_1"]]},
        "rationale": f"{node_id} flagged (confidence 0.87). Part of a peeling chain with 3 hops.",
    }


def test_load_alerts_reads_real_json(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    alerts = [_sample_alert()]
    (artifacts / "alerts.json").write_text(json.dumps(alerts), encoding="utf-8")

    loaded = load_alerts(str(tmp_path))
    assert loaded == alerts


def test_load_alerts_missing_file_returns_empty_list(tmp_path):
    assert load_alerts(str(tmp_path)) == []


def test_load_node_types_reads_real_parquet(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    df = pd.DataFrame(
        {"node_id": ["w1", "tx1", "ip1"], "node_type": ["Wallet", "Transaction", "IP"], "f_dummy": [0.0, 1.0, 2.0]}
    )
    df.to_parquet(artifacts / "node_features.parquet", index=False)

    node_types = load_node_types(str(tmp_path))
    assert node_types == {"w1": "Wallet", "tx1": "Transaction", "ip1": "IP"}


def test_load_node_types_missing_file_returns_empty_dict(tmp_path):
    assert load_node_types(str(tmp_path)) == {}


def test_alerts_dataframe_has_expected_columns_and_values():
    alerts = [_sample_alert("alert_00000", "wallet_1"), _sample_alert("alert_00001", "wallet_2")]
    df = _alerts_dataframe(alerts)

    assert list(df.columns) == ["alert_id", "node_id", "final_score", "pattern", "risk", "anomaly", "evidence_size"]
    assert len(df) == 2
    assert df.iloc[0]["evidence_size"] == 3  # len(["wallet_1", "tx_1", "ip_1"])
    assert df.iloc[0]["pattern"] == 0.9


def test_strip_external_cdn_links_removes_cdn_tags_but_keeps_inline_content():
    html = (
        '<html><head>'
        '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css" rel="stylesheet">'
        '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/js/bootstrap.bundle.min.js"></script>'
        '<script>var visNetworkInlineCode = 1;</script>'
        '</head><body><div id="mynetwork"></div></body></html>'
    )
    cleaned = _strip_external_cdn_links(html)

    assert "cdn.jsdelivr.net" not in cleaned
    assert "visNetworkInlineCode" in cleaned  # inline (non-CDN) script survives
    assert 'id="mynetwork"' in cleaned  # the graph container survives


def test_strip_external_cdn_links_raises_if_external_reference_survives():
    html = '<script src="https://unstripped.example.com/lib.js"></script>'
    # Deliberately not matched by the stripping regex's tag shape, to prove
    # the internal safety assertion actually fires rather than silently
    # passing bad output through.
    broken_pattern_html = html.replace("<script", "<scriptx")
    with pytest.raises(AssertionError):
        _strip_external_cdn_links(broken_pattern_html)


def test_render_evidence_graph_includes_all_nodes_and_no_external_urls():
    node_types = {"wallet_1": "Wallet", "tx_1": "Transaction", "ip_1": "IP"}
    html = render_evidence_graph(
        "alert_00000", "wallet_1", ["wallet_1", "tx_1", "ip_1"], [["wallet_1", "tx_1"], ["tx_1", "ip_1"]],
        json.dumps(node_types),
    )

    for node_id in ("wallet_1", "tx_1", "ip_1"):
        assert node_id in html
    assert "cdn.jsdelivr.net" not in html
    assert "cdnjs.cloudflare.com" not in html


def test_render_evidence_graph_ignores_edges_referencing_nodes_outside_the_evidence_set():
    """An edge to a node not in `nodes` (shouldn't happen given
    fusion/blend.py's own construction, but defensive) must not crash pyvis
    with an unknown-node reference.
    """
    node_types = {"wallet_1": "Wallet", "tx_1": "Transaction"}
    html = render_evidence_graph(
        "alert_00000", "wallet_1", ["wallet_1", "tx_1"], [["wallet_1", "tx_1"], ["tx_1", "not_in_node_list"]],
        json.dumps(node_types),
    )
    assert "wallet_1" in html and "tx_1" in html
