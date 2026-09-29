"""Streamlit dashboard — architecture doc sec 4.5: "ranked alert table...
+ pyvis/streamlit-agraph link-analysis graph view that highlights the
evidence subgraph on click." Single file, single process, no build step —
matching the architecture doc's explicit "no React/Node build pipeline"
call and every other pipeline stage's offline/single-Linux-box design.

Reads ONLY the already-computed data/artifacts/alerts.json (Dev B Phase 4's
output) and data/artifacts/node_features.parquet (for node-type coloring in
the graph view) — never re-runs any part of the pipeline itself. Run via
`make dashboard` (or `streamlit run src/argus/dashboard/app.py`); point at a
different run with the ARGUS_DATA_DIR environment variable.

OFFLINE NOTE (verified, not assumed — same standard as docs/offline_install.md):
pyvis's own `cdn_resources="in_line"` mode still hardcodes Bootstrap's CSS/JS
from a jsdelivr CDN in its HTML template (checked directly against the
installed pyvis version's template.html — a real template limitation, not a
misconfiguration on this module's part). Bootstrap there is purely decorative
page chrome around the graph container, not load-bearing for vis-network
itself, so `_strip_external_cdn_links` removes those two tags after
generation and asserts no `http(s)://` reference survives, rather than
trusting the "in_line" name at face value.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

DATA_DIR = Path(os.environ.get("ARGUS_DATA_DIR", "data"))

NODE_COLORS = {
    "Wallet": "#4C78A8",
    "Transaction": "#F58518",
    "IP": "#54A24B",
    "ASN": "#B279A2",
}
TARGET_NODE_COLOR = "#E45756"

_EXTERNAL_TAG_PATTERN = re.compile(
    r'<(?:link|script)\b[^>]*\bhref="https?://[^"]*"[^>]*>(?:.*?</script>)?'
    r'|<(?:link|script)\b[^>]*\bsrc="https?://[^"]*"[^>]*>(?:.*?</script>)?',
    re.IGNORECASE | re.DOTALL,
)


def _strip_external_cdn_links(html: str) -> str:
    cleaned = _EXTERNAL_TAG_PATTERN.sub("", html)
    leftover = re.findall(r'(?:href|src)="(https?://[^"]+)"', cleaned)
    assert not leftover, f"pyvis HTML still references external URLs after stripping: {leftover}"
    return cleaned


@st.cache_data
def load_alerts(data_dir: str) -> list[dict]:
    path = Path(data_dir) / "artifacts" / "alerts.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_node_types(data_dir: str) -> dict[str, str]:
    """node_id -> node_type, for coloring the graph view. node_features.parquet
    (argus.features.build) has exactly one row per graph node across all 4
    types — graph_edges.parquet has no per-node type column, so that file
    alone isn't enough for this lookup.
    """
    path = Path(data_dir) / "artifacts" / "node_features.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["node_id", "node_type"])
    return dict(zip(df["node_id"], df["node_type"]))


@st.cache_data
def render_evidence_graph(alert_id: str, node_id: str, nodes: list[str], edges: list[list[str]], node_types_json: str) -> str:
    # Args are plain/hashable (no dict/DataFrame) so st.cache_data can key on
    # them directly; node_types passed as a JSON string for the same reason.
    node_types = json.loads(node_types_json)
    net = Network(height="520px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222", cdn_resources="in_line")
    net.toggle_physics(True)

    for nid in nodes:
        node_type = node_types.get(nid, "?")
        is_target = nid == node_id
        net.add_node(
            nid,
            label=nid if len(nid) <= 16 else f"{nid[:14]}…",
            title=f"{node_type}: {nid}",
            color=TARGET_NODE_COLOR if is_target else NODE_COLORS.get(node_type, "#999999"),
            size=30 if is_target else 18,
        )

    node_set = set(nodes)
    for src, dst in edges:
        if src in node_set and dst in node_set:
            net.add_edge(src, dst)

    html = net.generate_html(notebook=False)
    return _strip_external_cdn_links(html)


def _alerts_dataframe(alerts: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alert_id": a["alert_id"],
                "node_id": a["node_id"],
                "final_score": a["final_score"],
                "pattern": a["components"]["pattern"],
                "risk": a["components"]["risk"],
                "anomaly": a["components"]["anomaly"],
                "evidence_size": len(a["evidence"]["nodes"]),
            }
            for a in alerts
        ]
    )


def main() -> None:
    st.set_page_config(page_title="ARGUS — Bitcoin Transaction Monitoring", layout="wide")
    st.title("ARGUS — Ranked Alerts & Evidence")
    st.caption(
        "Dual-layer graph fusion, entity resolution, anomaly/pattern/risk detection, and "
        "attention-based explainability — read-only view over data/artifacts/alerts.json."
    )

    alerts = load_alerts(str(DATA_DIR))
    if not alerts:
        st.warning(f"No alerts found at `{DATA_DIR / 'artifacts' / 'alerts.json'}`. Run `make pipeline` first.")
        return

    node_types = load_node_types(str(DATA_DIR))

    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.6, 0.01)
    search = st.sidebar.text_input("Search node_id contains").strip()

    table = _alerts_dataframe(alerts)
    filtered = table[table["final_score"] >= min_score]
    if search:
        filtered = filtered[filtered["node_id"].str.contains(search, case=False, na=False)]

    st.sidebar.metric("Alerts shown", f"{len(filtered)} / {len(table)}")

    st.subheader("Ranked alerts")
    st.dataframe(
        filtered.style.format({"final_score": "{:.3f}", "pattern": "{:.2f}", "risk": "{:.2f}", "anomaly": "{:.2f}"}),
        width="stretch",
        height=350,
    )

    if filtered.empty:
        st.info("No alerts match the current filters.")
        return

    st.subheader("Inspect an alert")
    options = filtered["alert_id"].tolist()
    labels = {
        row.alert_id: f"{row.alert_id} — {row.node_id} (confidence {row.final_score:.2f})"
        for row in filtered.itertuples()
    }
    selected_id = st.selectbox("Alert", options, format_func=lambda aid: labels[aid])
    alert = next(a for a in alerts if a["alert_id"] == selected_id)

    col_rationale, col_graph = st.columns([2, 3])
    with col_rationale:
        st.markdown(f"**Node:** `{alert['node_id']}`")
        st.markdown(f"**Confidence:** {alert['final_score']:.3f}")
        components_ = alert["components"]
        st.markdown(
            f"**Components:** pattern={components_['pattern']:.2f} · "
            f"risk={components_['risk']:.2f} · anomaly={components_['anomaly']:.2f}"
        )
        st.info(alert["rationale"])
        st.caption(f"Evidence subgraph: {len(alert['evidence']['nodes'])} nodes, {len(alert['evidence']['edges'])} edges.")

    with col_graph:
        html = render_evidence_graph(
            alert["alert_id"], alert["node_id"], alert["evidence"]["nodes"], alert["evidence"]["edges"],
            json.dumps(node_types),
        )
        # components.html (not st.iframe): st.iframe only takes a src URL/Path,
        # not raw HTML content — checked directly against this Streamlit
        # version's signature — so it's not a clean drop-in replacement for
        # embedding a generated HTML string. components.html still works
        # (only a deprecation warning, no functional break, and its own
        # stated removal date has already passed without effect).
        components.html(html, height=540, scrolling=False)


if __name__ == "__main__":
    main()
