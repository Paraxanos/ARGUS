"""Tests for the real-data generalisation changes: equal-output CoinJoin rule, address-type change heuristic,
money-flow taint, label-free rank fusion, held-out blend model and per-entity alert dedupe."""
from __future__ import annotations

import igraph as ig
import pandas as pd

from argus.detectors.risk_ppr import combine_risk, compute_flow_taint
from argus.er.union_find import is_coinjoin_like, resolve_entities, script_class, type_change_output
from argus.fusion.blend import apply_blend_model, compute_rank_scores, dedupe_by_entity, fit_blend_model

P2WPKH_A = "bc1q" + "a" * 38
P2WPKH_B = "bc1q" + "b" * 38
P2PKH_C = "1" + "C" * 33
P2TR_D = "bc1p" + "d" * 58


def test_script_class_from_address_encoding():
    assert script_class(P2WPKH_A) == "p2wpkh"
    assert script_class("bc1q" + "e" * 58) == "p2wsh"
    assert script_class(P2TR_D) == "p2tr"
    assert script_class(P2PKH_C) == "p2pkh"
    assert script_class("3" + "F" * 33) == "p2sh"


def test_equal_output_coinjoin_with_change_is_detected():
    # 5 participants, 5 equal outputs + 3 change outputs: the strict all-equal rule misses it, the general rule does not
    inputs = [f"in{i}" for i in range(5)]
    outputs = [0.01] * 5 + [0.0031, 0.0047, 0.0012]
    assert is_coinjoin_like(inputs, outputs)


def test_ordinary_batch_payment_is_not_coinjoin():
    # one owner (2 inputs), several different amounts
    assert not is_coinjoin_like(["a", "b"], [0.1, 0.2, 0.3, 0.4])
    # many inputs but outputs all different
    assert not is_coinjoin_like([f"i{i}" for i in range(6)], [0.1, 0.2, 0.3, 0.4, 0.5])


def test_type_change_fires_only_on_a_type_mismatch_with_a_fresh_output():
    # inputs P2WPKH; outputs one P2WPKH (change) + one P2PKH (payment)
    assert type_change_output([P2WPKH_A], [P2WPKH_B, P2PKH_C], seen_outputs=set()) == P2WPKH_B
    # both outputs share the input type -> ambiguous -> no inference
    assert type_change_output([P2WPKH_A], [P2WPKH_B, "bc1q" + "z" * 38], seen_outputs=set()) is None
    # candidate already seen as an output earlier -> not a fresh change address
    assert type_change_output([P2WPKH_A], [P2WPKH_B, P2PKH_C], seen_outputs={P2WPKH_B}) is None
    # three outputs -> no inference
    assert type_change_output([P2WPKH_A], [P2WPKH_B, P2PKH_C, P2TR_D], seen_outputs=set()) is None


def test_resolve_entities_type_change_is_opt_in():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "input_addresses": [[P2WPKH_A]], "output_addresses": [[P2WPKH_B, P2PKH_C]], "output_amounts": [[0.4, 0.1]],
    })
    uf_off, links_off = resolve_entities(df)
    assert links_off == []
    uf_on, links_on = resolve_entities(df, type_change=True)
    assert uf_on.find(P2WPKH_A) == uf_on.find(P2WPKH_B)
    assert uf_on.find(P2WPKH_A) != uf_on.find(P2PKH_C)


def _flow_graph(txs):
    """txs: list of (txid, timestamp, [(wallet, amount)], [(wallet, amount)])"""
    names, types = [], []
    idx = {}

    def v(name, typ):
        if name not in idx:
            idx[name] = len(names)
            names.append(name)
            types.append(typ)
        return idx[name]

    edges, etypes, amounts, stamps = [], [], [], []
    for txid, ts, ins, outs in txs:
        t = v(txid, "Transaction")
        ip = v(f"ip-{txid}", "IP")
        edges.append((t, ip)); etypes.append("BROADCAST_VIA"); amounts.append(None); stamps.append(pd.Timestamp(ts))
        for w, a in ins:
            edges.append((v(w, "Wallet"), t)); etypes.append("FUNDS"); amounts.append(a); stamps.append(None)
        for w, a in outs:
            edges.append((t, v(w, "Wallet"))); etypes.append("PAYS"); amounts.append(a); stamps.append(None)
    g = ig.Graph(directed=True)
    g.add_vertices(len(names))
    g.vs["name"], g.vs["type"] = names, types
    g.add_edges(edges)
    g.es["type"], g.es["amount"], g.es["timestamp"] = etypes, amounts, stamps
    return g


def test_flow_taint_moves_forward_in_time_and_follows_co_spent_inputs():
    g = _flow_graph([
        ("t0", "2026-01-01 00:00", [("victim", 1.0)], [("seed", 1.0)]),            # before: victim pays the seed
        ("t1", "2026-01-01 01:00", [("seed", 1.0), ("own2", 1.0)], [("x", 1.9)]),  # ordinary tx: one owner
        ("t2", "2026-01-01 02:00", [("x", 1.9)], [("y", 1.8)]),
    ])
    scores = {r.node_id: r.score for r in compute_flow_taint(g, ["seed"])}
    assert "victim" not in scores                        # taint never flows backwards to a payer
    assert abs(scores["x"] - 0.9) < 1e-9                 # full taint of the owner, one hop of decay
    assert abs(scores["own2"] - 0.9) < 1e-9              # co-spent input = same owner
    assert abs(scores["y"] - 0.81) < 1e-9
    assert "t1" in scores and "t2" in scores             # transactions are scored too


def test_flow_taint_haircut_applies_only_to_coinjoin_shaped_transactions():
    ins = [("seed", 0.0105)] + [(f"p{i}", 0.0105) for i in range(4)]       # 5 participants, 1 tainted
    outs = [(f"o{i}", 0.01) for i in range(5)]                              # 5 equal outputs
    g = _flow_graph([("cj", "2026-01-01 00:00", ins, outs)])
    scores = {r.node_id: r.score for r in compute_flow_taint(g, ["seed"])}
    assert abs(scores["o0"] - 0.9 * 0.2) < 1e-9          # value share 1/5, one hop of decay
    assert "p1" not in scores                            # other participants are not the seed's owner


def test_flow_taint_does_not_pass_through_extreme_degree_service_wallets():
    txs = [("t_dep", "2026-01-01 00:00", [("seed", 1.0)], [("exchange", 1.0)])]
    # the exchange pays many customers: by far the highest degree in this run
    for i in range(3000):
        txs.append((f"w{i}", f"2026-01-01 01:{i % 60:02d}", [("exchange", 0.01)], [(f"cust{i}", 0.009)]))
    for i in range(3000):   # a background of ordinary low-degree wallets
        txs.append((f"b{i}", "2026-01-01 00:30", [(f"a{i}", 0.5)], [(f"c{i}", 0.49)]))
    scores = {r.node_id: r.score for r in compute_flow_taint(_flow_graph(txs), ["seed"])}
    assert scores.get("exchange", 0) > 0                 # the service itself is tainted (it received the funds)
    assert not any(k.startswith("cust") for k in scores)  # but it does not taint its unrelated customers


def test_combine_risk_keeps_the_stronger_signal_per_node():
    from argus.detectors.risk_ppr import RiskScore
    a = [RiskScore("n", 0.2, 1, "s", ["s", "n"])]
    b = [RiskScore("n", 0.7, 2, "s", ["s", "t", "n"]), RiskScore("m", 0.1, 3, "s", ["s", "m"])]
    out = {r.node_id: r.score for r in combine_risk(a, b)}
    assert out == {"n": 0.7, "m": 0.1}


def _table():
    return pd.DataFrame({"node_id": ["a", "b", "c", "d"], "pattern": [0.0, 0.9, 0.0, 0.5],
                         "risk": [0.8, 0.0, 0.0, 0.4], "anomaly": [0.5, 0.49, 0.51, 0.5]})


def test_rank_blend_needs_no_labels_and_ignores_score_scale():
    t1, method = compute_rank_scores(_table())
    scaled = _table()
    scaled["risk"] *= 1000          # a different score scale must not change the ranking
    t2, _ = compute_rank_scores(scaled)
    assert method == "rank_blend"
    assert list(t1.sort_values("final_score")["node_id"]) == list(t2.sort_values("final_score")["node_id"])
    assert t1.set_index("node_id").loc["c", "final_score"] < t1.set_index("node_id").loc["a", "final_score"]


def test_held_out_model_round_trip_applies_without_refitting():
    rows = []
    for i in range(40):
        illicit = i % 4 == 0
        rows.append({"node_id": f"w{i}", "pattern": 0.8 if illicit else 0.1, "risk": 0.9 if illicit else 0.05, "anomaly": 0.5})
    dev = pd.DataFrame(rows)
    gt = pd.DataFrame({"wallet_id": dev["node_id"], "entity_type": ["ransomware" if i % 4 == 0 else "licit" for i in range(40)]})
    model = fit_blend_model(dev, gt)
    assert model is not None and model["features"] == ["pattern", "risk", "anomaly"]
    scored, method = apply_blend_model(_table(), model)
    assert method == "held_out_logistic" and scored["final_score"].between(0, 1).all()


def test_dedupe_keeps_one_alert_per_entity_and_all_transactions():
    t = pd.DataFrame({"node_id": ["w1", "w2", "w3", "tx1"], "final_score": [0.9, 0.8, 0.7, 0.6]})
    ents = pd.DataFrame({"wallet_id": ["w1", "w2", "w3"], "entity_id": ["E1", "E1", "E2"]})
    out = dedupe_by_entity(t, ents)
    assert list(out["node_id"]) == ["w1", "w3", "tx1"]
