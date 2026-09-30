import random

import pandas as pd
import pytest

from argus.detectors.coinjoin import detect_coinjoin_rounds
from argus.detectors.pattern_sim import detect_pattern_similarity, pattern_sim_rows
from argus.detectors.peeling import detect_peeling_chains
from argus.detectors.risk_ppr import compute_risk_scores, risk_score_rows
from argus.detectors.scores import coinjoin_round_rows, peeling_chain_rows
from argus.features.build import compute_node_features
from argus.fusion.blend import ALERT_THRESHOLD, build_alerts, component_table, compute_final_scores
from argus.fusion.evidence import (
    build_attention_context,
    extract_attention_evidence,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from argus.graph.build import build_graph
from argus.ingest.pipeline import run_ingest
from argus.models.anomaly import anomaly_score_rows, train_and_score_anomalies
from argus.synth.config import SynthConfig
from argus.synth.corrupt import inject_corruption
from argus.synth.entities import generate_entities
from argus.synth.export import write_csv
from argus.synth.patterns import generate_illicit_patterns
from argus.synth.seeds import generate_seed_wallets
from argus.synth.transactions import generate_transactions
from argus.synth.wallets import generate_wallets


def _pipeline_outputs(tmp_path, seed: int = 41):
    cfg = SynthConfig(
        random_seed=seed,
        num_entities=1000,
        num_wallets=15000,
        num_transactions=15000,
        ip_noise=0.05,
        heuristic_break_rate=0.1,
        mixer_fraction=0.05,
    )
    rng = random.Random(cfg.random_seed)
    entities = generate_entities(cfg, rng)
    wallets = generate_wallets(cfg, rng, entities)
    baseline = generate_transactions(cfg, rng, entities, wallets)
    pattern_rows, _ = generate_illicit_patterns(cfg, rng, entities, wallets)
    corrupted_baseline = inject_corruption(baseline, rng)
    all_rows = corrupted_baseline + pattern_rows
    seed_wallets = generate_seed_wallets(rng, entities, wallets)

    csv_path = tmp_path / "transactions.csv"
    write_csv(all_rows, csv_path)
    df = run_ingest(csv_path, "csv", tmp_path / "rejects.log")
    g = build_graph(df)

    chains = detect_peeling_chains(g)
    rounds = detect_coinjoin_rounds(g)

    empty_resolved_entities = pd.DataFrame(columns=["wallet_id", "entity_id"])
    node_features = compute_node_features(g, df, empty_resolved_entities)
    anomaly_result = train_and_score_anomalies(g, node_features, max_epochs=10)
    scores_anomaly = pd.DataFrame(
        anomaly_score_rows(anomaly_result.scores), columns=["node_id", "score", "reason_code", "evidence_json"]
    )

    sim_matches = detect_pattern_similarity(chains, rounds, anomaly_result.embeddings, anomaly_result.node_ids)
    scores_pattern = pd.DataFrame(
        peeling_chain_rows(chains) + coinjoin_round_rows(rounds) + pattern_sim_rows(sim_matches),
        columns=["node_id", "score", "reason_code", "evidence_json"],
    )

    risk_scores = compute_risk_scores(g, seed_wallets)
    scores_risk = pd.DataFrame(
        risk_score_rows(risk_scores), columns=["node_id", "score", "reason_code", "evidence_json"]
    )

    ground_truth_entities_path = tmp_path / "gt_entities.parquet"
    pd.DataFrame(
        [{"wallet_id": w.wallet_id, "entity_id": w.entity_id} for w in wallets]
    ).merge(
        pd.DataFrame([{"entity_id": e.entity_id, "entity_type": e.entity_type} for e in entities]),
        on="entity_id",
    ).to_parquet(ground_truth_entities_path, index=False)
    ground_truth_entities = pd.read_parquet(ground_truth_entities_path)

    return scores_pattern, scores_risk, scores_anomaly, ground_truth_entities, anomaly_result


# Module-scoped: _pipeline_outputs generates a 15k-wallet dataset and trains
# the shared encoder (~10 epochs) — deterministic (fixed seed) and expensive
# enough (~5 min) that both tests below sharing ONE run, rather than each
# paying for their own, is a real, measured difference (this file's runtime
# roughly halves), not a style preference.
@pytest.fixture(scope="module")
def pipeline_outputs(tmp_path_factory):
    return _pipeline_outputs(tmp_path_factory.mktemp("fusion_pipeline"))


def test_alerts_have_evidence_rationale_and_valid_scores(pipeline_outputs):
    scores_pattern, scores_risk, scores_anomaly, ground_truth_entities, _anomaly_result = pipeline_outputs

    table = component_table(scores_pattern, scores_risk, scores_anomaly)
    final_table, method = compute_final_scores(table, ground_truth_entities)
    assert method in ("calibrated_logistic", "fallback_weighted_average")

    alerts = build_alerts(final_table, scores_pattern, scores_risk, scores_anomaly, ALERT_THRESHOLD)
    assert alerts

    for alert in alerts:
        assert 0.0 <= alert["final_score"] <= 1.0
        assert alert["evidence"]["nodes"]
        assert alert["rationale"]
        assert alert["node_id"] in alert["evidence"]["nodes"]

    scores = [a["final_score"] for a in alerts]
    assert scores == sorted(scores, reverse=True)


def test_alerts_include_attention_evidence_via_checkpoint_round_trip(pipeline_outputs, tmp_path):
    """Full Phase 4 integration: save the trained encoder exactly as
    detectors/cli.py does, reload it exactly as fusion/cli.py does, and
    confirm at least one real alert ends up with an attention-derived
    rationale clause and extra evidence nodes/edges beyond what the
    classical/anomaly reason_codes alone would produce.
    """
    scores_pattern, scores_risk, scores_anomaly, ground_truth_entities, anomaly_result = pipeline_outputs

    table = component_table(scores_pattern, scores_risk, scores_anomaly)
    final_table, _method = compute_final_scores(table, ground_truth_entities)

    baseline_alerts = build_alerts(final_table, scores_pattern, scores_risk, scores_anomaly, ALERT_THRESHOLD)

    ckpt_path = tmp_path / "encoder_checkpoint.pt"
    save_encoder_checkpoint(
        anomaly_result.encoder, anomaly_result.data, anomaly_result.node_ids, anomaly_result.edge_types,
        anomaly_result.temporal_edge_types, anomaly_result.hidden_dim, anomaly_result.embedding_dim, ckpt_path,
        heads=anomaly_result.heads, time2vec_dim=anomaly_result.time2vec_dim,
    )
    encoder, data, node_ids = load_encoder_checkpoint(ckpt_path)
    id_to_type = {node_id: node_type for node_type, ids in node_ids.items() for node_id in ids}
    context = build_attention_context(encoder, data, node_ids)  # ONE forward pass, reused for every alert below

    def attention_evidence_fn(node_id: str):
        node_type = id_to_type.get(node_id)
        return extract_attention_evidence(context, node_id, node_type) if node_type else None

    augmented_alerts = build_alerts(
        final_table, scores_pattern, scores_risk, scores_anomaly, ALERT_THRESHOLD, attention_evidence_fn
    )

    assert len(augmented_alerts) == len(baseline_alerts)
    by_id_baseline = {a["node_id"]: a for a in baseline_alerts}
    grew = [
        a for a in augmented_alerts
        if len(a["evidence"]["nodes"]) > len(by_id_baseline[a["node_id"]]["evidence"]["nodes"])
    ]
    assert grew, "at least one alert's evidence should grow once attention-based neighbors are available"
    assert any("attention weighted" in a["rationale"] for a in grew)


def test_build_alerts_top_k_caps_a_low_prevalence_flood():
    """Real-data finding (ARGUS dataset Track M benchmarking report,
    2026-09-30): at ~0.2% real-world illicit prevalence, ALERT_THRESHOLD
    alone passed ~78% of the scored universe (31,059 of 40,016 nodes) —
    calibrated against synthetic data's much higher prevalence. Reproduce
    that shape locally (no real dataset needed, just the marker: many
    benign nodes clustered just above threshold, a few genuinely high
    scorers) and confirm top_k bounds the queue regardless of how many
    nodes clear the threshold, while never dropping the highest scorers.
    """
    n_benign = 998
    benign_ids = [f"benign_{i}" for i in range(n_benign)]
    benign_scores = [0.60 + 0.15 * (i / n_benign) for i in range(n_benign)]  # all clear ALERT_THRESHOLD=0.6
    illicit_ids = ["illicit_0", "illicit_1"]
    illicit_scores = [0.97, 0.99]

    final_table = pd.DataFrame(
        {
            "node_id": benign_ids + illicit_ids,
            "final_score": benign_scores + illicit_scores,
            "pattern": [0.5] * (n_benign + 2),
            "risk": [0.5] * (n_benign + 2),
            "anomaly": [0.5] * (n_benign + 2),
        }
    )
    empty_scores = pd.DataFrame(columns=["node_id", "score", "reason_code", "evidence_json"])

    flooded = build_alerts(final_table, empty_scores, empty_scores, empty_scores, ALERT_THRESHOLD)
    assert len(flooded) == n_benign + 2  # reproduces the flood: everyone clears the threshold

    capped = build_alerts(final_table, empty_scores, empty_scores, empty_scores, ALERT_THRESHOLD, top_k=10)
    assert len(capped) == 10
    capped_scores = [a["final_score"] for a in capped]
    assert capped_scores == sorted(capped_scores, reverse=True)
    assert {"illicit_0", "illicit_1"}.issubset({a["node_id"] for a in capped})


def test_component_table_anomaly_admission_widens_universe_without_flooding():
    """Real-data finding (ARGUS dataset Track M benchmarking report,
    2026-09-30): with only 17 real seeds, risk propagation + pattern hits
    together covered just 14.7% of illicit targets — most of the anomaly
    head's own signal (which covers every node) never had a chance to
    matter, because component_table never admits a node on anomaly alone.
    Reproduce the marker locally: a few "quiet illicit" nodes with no
    pattern/risk hit but an elevated anomaly score, among a large all-normal
    population — confirm they're missed by default and captured once
    admitted, without the all-normal population flooding the universe too.
    """
    quiet_illicit = [f"quiet_{i}" for i in range(3)]
    normal_population = [f"normal_{i}" for i in range(500)]

    # Explicit dtypes, not pd.DataFrame(columns=[...]): an empty frame with
    # inferred object dtype triggers a pandas FutureWarning on the later
    # reindex().fillna(0.0) — real detector output always has proper dtypes,
    # so this is purely a hand-built-empty-fixture wrinkle, not a production
    # code path this test needs to exercise.
    empty_str_score = pd.DataFrame({"node_id": pd.Series(dtype=str), "score": pd.Series(dtype=float)})
    scores_pattern = empty_str_score.assign(reason_code=pd.Series(dtype=str), evidence_json=pd.Series(dtype=str))
    scores_risk = scores_pattern.copy()
    scores_anomaly = pd.DataFrame(
        {
            "node_id": quiet_illicit + normal_population,
            "score": [0.95] * len(quiet_illicit) + [0.5] * len(normal_population),
            "reason_code": ["ANOMALY_ZSCORE=3.00"] * len(quiet_illicit) + ["ANOMALY_ZSCORE=0.00"] * len(normal_population),
            "evidence_json": ["{}"] * (len(quiet_illicit) + len(normal_population)),
        }
    )

    default_table = component_table(scores_pattern, scores_risk, scores_anomaly)
    assert default_table.empty, "quiet illicit nodes are missed by default — no pattern/risk hit"

    widened_table = component_table(scores_pattern, scores_risk, scores_anomaly, anomaly_admission_threshold=0.9)
    assert set(widened_table["node_id"]) == set(quiet_illicit)
    assert "normal_0" not in set(widened_table["node_id"]), "widening must not flood in the all-normal population"


def test_fallback_blend_when_insufficient_labels():
    table = pd.DataFrame(
        {"node_id": ["a", "b", "c"], "pattern": [0.9, 0.1, 0.5], "risk": [0.8, 0.2, 0.5], "anomaly": [0.5, 0.5, 0.5]}
    )
    ground_truth_entities = pd.DataFrame({"wallet_id": [], "entity_type": []})  # no labels at all

    final_table, method = compute_final_scores(table, ground_truth_entities)

    assert method == "fallback_weighted_average"
    expected = 0.5 * table["pattern"] + 0.4 * table["risk"] + 0.1 * table["anomaly"]
    assert (final_table["final_score"] - expected).abs().max() < 1e-9
