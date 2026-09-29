from __future__ import annotations

# argus.models.anomaly (torch/torch_geometric, transitively via
# argus.models.encoder) MUST be imported before numpy/pandas/sklearn/igraph
# in this process — see src/argus/er/cli.py's matching comment and
# tests/conftest.py for the full explanation (a Windows-only DLL-init crash,
# bisected during Dev B Phase 1).
from argus.models.anomaly import anomaly_score_rows, train_and_score_anomalies  # noqa: E402, isort:skip

from pathlib import Path

import pandas as pd
import typer

from argus.detectors.coinjoin import detect_coinjoin_rounds
from argus.detectors.pattern_sim import detect_pattern_similarity, pattern_sim_rows
from argus.detectors.peeling import detect_peeling_chains
from argus.detectors.risk_ppr import compute_risk_scores, risk_score_rows
from argus.detectors.scores import coinjoin_round_rows, peeling_chain_rows, write_scores
from argus.fusion.evidence import save_encoder_checkpoint
from argus.graph.export import read_graph_pickle

app = typer.Typer()


@app.command()
def run(data_dir: Path = typer.Option(Path("data"), "--data-dir")) -> None:
    g = read_graph_pickle(data_dir / "artifacts" / "graph.pkl")
    node_features = pd.read_parquet(data_dir / "artifacts" / "node_features.parquet")

    chains = detect_peeling_chains(g)
    rounds = detect_coinjoin_rounds(g)
    classical_rows = peeling_chain_rows(chains) + coinjoin_round_rows(rounds)

    # Trains the ONE shared encoder instance for this run (architecture doc
    # sec 4.3) — its embeddings feed pattern_sim below directly, no second
    # training pass. See argus.models.anomaly's module docstring.
    anomaly_result = train_and_score_anomalies(g, node_features)
    write_scores(anomaly_score_rows(anomaly_result.scores), data_dir / "artifacts" / "scores_anomaly.parquet")

    # Persists the trained encoder (not the full autoencoder — its decoders
    # are only needed for the scoring already done above) so fusion/cli.py
    # can extract attention-based evidence (Dev B Phase 4) for the final,
    # much smaller alert list without a third training pass. A None encoder
    # means train_and_score_anomalies hit its degenerate-empty-graph branch —
    # nothing to checkpoint.
    if anomaly_result.encoder is not None:
        save_encoder_checkpoint(
            anomaly_result.encoder,
            anomaly_result.data,
            anomaly_result.node_ids,
            anomaly_result.edge_types,
            anomaly_result.temporal_edge_types,
            anomaly_result.hidden_dim,
            anomaly_result.embedding_dim,
            data_dir / "artifacts" / "encoder_checkpoint.pt",
            heads=anomaly_result.heads,
            time2vec_dim=anomaly_result.time2vec_dim,
        )

    sim_matches = detect_pattern_similarity(chains, rounds, anomaly_result.embeddings, anomaly_result.node_ids)
    sim_rows = pattern_sim_rows(sim_matches)
    pattern_rows = classical_rows + sim_rows
    write_scores(pattern_rows, data_dir / "artifacts" / "scores_pattern.parquet")

    seeds = pd.read_parquet(data_dir / "ground_truth" / "seeds.parquet")
    risk_scores = compute_risk_scores(g, seeds["wallet_id"].tolist())
    risk_rows = risk_score_rows(risk_scores)
    write_scores(risk_rows, data_dir / "artifacts" / "scores_risk.parquet")

    typer.echo(
        f"peeling_chains={len(chains)} coinjoin_rounds={len(rounds)} pattern_sim_matches={len(sim_matches)} "
        f"pattern_score_rows={len(pattern_rows)} anomaly_score_rows={len(anomaly_result.scores)} "
        f"risk_score_rows={len(risk_rows)} seeds={len(seeds)}"
    )


if __name__ == "__main__":
    app()
