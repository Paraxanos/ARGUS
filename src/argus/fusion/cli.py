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

from argus.detectors.scores import write_scores
from argus.fusion.blend import ALERT_THRESHOLD, build_alerts, component_table, compute_final_scores, write_alerts
from argus.graph.export import read_graph_pickle

app = typer.Typer()


@app.command()
def run(data_dir: Path = typer.Option(Path("data"), "--data-dir")) -> None:
    g = read_graph_pickle(data_dir / "artifacts" / "graph.pkl")
    node_features = pd.read_parquet(data_dir / "artifacts" / "node_features.parquet")

    anomaly_scores = train_and_score_anomalies(g, node_features)
    write_scores(anomaly_score_rows(anomaly_scores), data_dir / "artifacts" / "scores_anomaly.parquet")

    scores_pattern = pd.read_parquet(data_dir / "artifacts" / "scores_pattern.parquet")
    scores_risk = pd.read_parquet(data_dir / "artifacts" / "scores_risk.parquet")
    scores_anomaly = pd.read_parquet(data_dir / "artifacts" / "scores_anomaly.parquet")
    ground_truth_entities = pd.read_parquet(data_dir / "ground_truth" / "entities.parquet")

    table = component_table(scores_pattern, scores_risk, scores_anomaly)
    final_table, method = compute_final_scores(table, ground_truth_entities)
    alerts = build_alerts(final_table, scores_pattern, scores_risk, scores_anomaly, ALERT_THRESHOLD)
    write_alerts(alerts, data_dir / "artifacts" / "alerts.json")

    scores = final_table["final_score"]
    typer.echo(
        f"method={method} fusion_universe={len(final_table)} threshold={ALERT_THRESHOLD} alerts={len(alerts)} "
        f"score_min={scores.min():.4f} score_max={scores.max():.4f} score_median={scores.median():.4f}"
    )


if __name__ == "__main__":
    app()
