from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from argus.fusion.blend import ALERT_THRESHOLD, build_alerts, component_table, compute_final_scores, write_alerts

app = typer.Typer()


@app.command()
def run(data_dir: Path = typer.Option(Path("data"), "--data-dir")) -> None:
    # scores_anomaly.parquet is produced by argus.detectors.cli's `run`
    # (Dev B Phase 2/3: the shared encoder trains there, once, so
    # detectors/pattern_sim.py's embedding-similarity search can reuse the
    # same trained embeddings instead of paying for a second training pass —
    # see argus.models.anomaly's module docstring). This stage is pure
    # fusion: read the three already-computed score heads, blend, done.
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
