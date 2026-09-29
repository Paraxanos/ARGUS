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


def test_fallback_blend_when_insufficient_labels():
    table = pd.DataFrame(
        {"node_id": ["a", "b", "c"], "pattern": [0.9, 0.1, 0.5], "risk": [0.8, 0.2, 0.5], "anomaly": [0.5, 0.5, 0.5]}
    )
    ground_truth_entities = pd.DataFrame({"wallet_id": [], "entity_type": []})  # no labels at all

    final_table, method = compute_final_scores(table, ground_truth_entities)

    assert method == "fallback_weighted_average"
    expected = 0.5 * table["pattern"] + 0.4 * table["risk"] + 0.1 * table["anomaly"]
    assert (final_table["final_score"] - expected).abs().max() < 1e-9
