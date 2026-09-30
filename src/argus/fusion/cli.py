from __future__ import annotations

# argus.fusion.evidence (torch/torch_geometric, transitively via
# argus.models.encoder) MUST be imported before numpy/pandas/sklearn/igraph
# in this process — see src/argus/er/cli.py's matching comment and
# tests/conftest.py for the full explanation (a Windows-only DLL-init crash,
# bisected during Dev B Phase 1).
from argus.fusion.evidence import build_attention_context, extract_attention_evidence, load_encoder_checkpoint  # noqa: E402, isort:skip

import json
from pathlib import Path

import pandas as pd
import typer

from argus.fusion.blend import (
    ALERT_THRESHOLD,
    DEFAULT_TOP_K,
    apply_blend_model,
    build_alerts,
    component_table,
    compute_final_scores,
    RANK_WEIGHTS,
    compute_rank_scores,
    dedupe_by_entity,
    fit_blend_model,
    write_alerts,
)

app = typer.Typer()


@app.command()
def run(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", help="Cap the alert queue at this many highest-scoring nodes; 0 means unbounded (threshold alone)."),
    anomaly_admission_threshold: float = typer.Option(
        0.0, "--anomaly-admission-threshold",
        help="Additionally admit into the scored universe any node whose anomaly score alone "
             "exceeds this cutoff. 0.0 disables it (preserves existing universe exactly, since "
             "anomaly scores are always > 0). Real-data widening lever — see "
             "docs/WRITEUP.md's benchmarking findings.",
    ),
    fusion_mode: str = typer.Option(
        "rank", "--fusion-mode",
        help="rank = label-free percentile-rank blend (default); model = apply a blend fitted on a separate dev run "
             "(--fusion-model); in-sample = the original blend fitted on THIS run's ground truth (leaks labels into "
             "scores; kept only for comparison).",
    ),
    fusion_model: Path = typer.Option(None, "--fusion-model", help="JSON blend model saved by --save-fusion-model on a dev run."),
    save_fusion_model: Path = typer.Option(
        None, "--save-fusion-model",
        help="Fit the logistic blend on THIS run's ground truth and save it as JSON (use on a dev run only).",
    ),
    rank_weights: str = typer.Option(
        None, "--rank-weights",
        help="Override the rank blend's weights, e.g. risk=0.55,pattern=0.45,anomaly=0 (choose on a dev set only).",
    ),
    dedupe_entities: bool = typer.Option(True, "--dedupe-entities/--no-dedupe-entities",
                                         help="One alert per resolved entity (its top-scoring wallet)."),
) -> None:
    if fusion_mode not in ("rank", "model", "in-sample"):
        raise typer.BadParameter("--fusion-mode must be rank, model or in-sample")
    # scores_anomaly.parquet is produced by argus.detectors.cli's `run`
    # (Dev B Phase 2/3: the shared encoder trains there, once, so
    # detectors/pattern_sim.py's embedding-similarity search can reuse the
    # same trained embeddings instead of paying for a second training pass —
    # see argus.models.anomaly's module docstring). This stage is pure
    # fusion: read the three already-computed score heads, blend, done.
    scores_pattern = pd.read_parquet(data_dir / "artifacts" / "scores_pattern.parquet")
    scores_risk = pd.read_parquet(data_dir / "artifacts" / "scores_risk.parquet")
    scores_anomaly = pd.read_parquet(data_dir / "artifacts" / "scores_anomaly.parquet")
    gt_path = data_dir / "ground_truth" / "entities.parquet"
    ground_truth_entities = pd.read_parquet(gt_path) if gt_path.exists() else pd.DataFrame(columns=["wallet_id", "entity_type"])

    # 0.0 is the CLI's "disabled" sentinel, translated to None here — a
    # literal 0.0 passed straight to component_table would admit nearly
    # every node (sigmoid anomaly scores are always > 0.0), exactly the
    # flood the anomaly-admission design is meant to fix, not cause.
    effective_admission_threshold = anomaly_admission_threshold if anomaly_admission_threshold > 0.0 else None
    table = component_table(scores_pattern, scores_risk, scores_anomaly, effective_admission_threshold)
    if save_fusion_model is not None:
        model = fit_blend_model(table, ground_truth_entities)
        if model is None:
            raise typer.BadParameter("not enough labelled nodes in this run to fit a blend model")
        save_fusion_model.parent.mkdir(parents=True, exist_ok=True)
        save_fusion_model.write_text(json.dumps(model, indent=2))
        typer.echo(f"saved fusion model to {save_fusion_model}: {model}")
    if fusion_mode == "rank":
        weights = dict(RANK_WEIGHTS)
        if rank_weights:
            weights.update({k.strip(): float(v) for k, v in (kv.split("=") for kv in rank_weights.split(","))})
        final_table, method = compute_rank_scores(table, {k: v for k, v in weights.items() if v > 0})
        threshold = 0.0            # ranks carry no absolute meaning; the top-k queue is the gate
    elif fusion_mode == "model":
        if fusion_model is None:
            raise typer.BadParameter("--fusion-mode model needs --fusion-model")
        final_table, method = apply_blend_model(table, json.loads(fusion_model.read_text()))
        threshold = 0.0
    else:
        final_table, method = compute_final_scores(table, ground_truth_entities)
        threshold = ALERT_THRESHOLD
    if dedupe_entities:
        ent_path = data_dir / "artifacts" / "entities.parquet"
        final_table = dedupe_by_entity(final_table, pd.read_parquet(ent_path) if ent_path.exists() else None)

    # Dev B Phase 4: attention-based evidence extraction, reusing the SAME
    # trained encoder detectors/cli.py already persisted (no third training
    # pass) — only for the final, small alert list, not the full ~343k-node
    # population. A missing checkpoint (e.g. detectors/cli.py hit its
    # degenerate-empty-graph branch) degrades gracefully: alerts are built
    # exactly as they were pre-Phase-4.
    attention_evidence_fn = None
    checkpoint_path = data_dir / "artifacts" / "encoder_checkpoint.pt"
    if checkpoint_path.exists():
        encoder, encoder_data, node_ids = load_encoder_checkpoint(checkpoint_path)
        id_to_type = {node_id: node_type for node_type, ids in node_ids.items() for node_id in ids}
        # ONE full-graph forward pass, reused for every alert below — see
        # fusion/evidence.py's PERFORMANCE note; computing it per-alert
        # instead was measured to add several minutes for no reason.
        attention_context = build_attention_context(encoder, encoder_data, node_ids)

        def attention_evidence_fn(node_id: str):
            node_type = id_to_type.get(node_id)
            if node_type is None:
                return None
            return extract_attention_evidence(attention_context, node_id, node_type)

    effective_top_k = top_k if top_k > 0 else None
    alerts = build_alerts(
        final_table, scores_pattern, scores_risk, scores_anomaly, threshold, attention_evidence_fn,
        top_k=effective_top_k,
    )
    write_alerts(alerts, data_dir / "artifacts" / "alerts.json")

    scores = final_table["final_score"]
    typer.echo(
        f"method={method} fusion_universe={len(final_table)} threshold={threshold} top_k={effective_top_k} "
        f"alerts={len(alerts)} score_min={scores.min():.4f} score_max={scores.max():.4f} "
        f"score_median={scores.median():.4f} attention_evidence={'yes' if attention_evidence_fn else 'no'}"
    )


if __name__ == "__main__":
    app()
