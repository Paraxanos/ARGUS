from __future__ import annotations

# argus.models.sage (torch/torch_geometric) MUST be imported before
# numpy/pandas/sklearn/igraph in this process — on Windows, importing torch
# AFTER those (which happens if this import is left in normal alphabetical
# order below) reproducibly crashes with "OSError: [WinError 1114] A
# dynamic link library (DLL) initialization routine failed ... c10.dll"
# the moment torch is imported, apparently an OpenBLAS/MKL runtime-init
# conflict between this environment's OpenBLAS-linked scipy/sklearn wheels
# and torch's bundled runtime. Importing torch first avoids it entirely;
# verified empirically, not a guess — see the Dev B Phase 1 commit message
# for the bisection. Do not reorder this without re-testing on Windows.
from argus.models.sage import train_wallet_embeddings, wallet_embeddings_frame  # noqa: E402, isort:skip

from pathlib import Path

import pandas as pd
import typer

from argus.er.embed_cluster import (
    add_same_entity_edges,
    drop_high_fanin_broadcasters,
    reconcile_with_pass1,
    run_hdbscan,
    run_with_big_stack,
    write_pass2_outputs,
)
from argus.er.evaluate import evaluate_pairwise
from argus.er.union_find import add_co_spend_edges, resolve_entities, write_entities_parquet
from argus.graph.export import export_edges, read_graph_pickle, write_graph_pickle

app = typer.Typer()


def _ground_truth(data_dir: Path) -> pd.DataFrame | None:
    """Ground truth is used here for printed metrics only, never to fit anything; real
    (unlabelled) data has none, which must not stop the pipeline."""
    path = data_dir / "ground_truth" / "entities.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _metrics_text(predicted: pd.DataFrame, ground_truth: pd.DataFrame | None) -> str:
    if ground_truth is None or ground_truth.empty:
        return "precision=n/a recall=n/a f1=n/a (no ground truth)"
    m = evaluate_pairwise(predicted, ground_truth)
    return (f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} "
            f"tp_pairs={m['tp_pairs']} predicted_pairs={m['predicted_pairs']} true_pairs={m['true_pairs']}")


@app.command()
def resolve(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    type_change: bool = typer.Option(
        False, "--type-change",
        help="Also apply the address-type change heuristic (argus.er.union_find.type_change_output).",
    ),
) -> None:
    canonical_path = data_dir / "canonical" / "transactions.parquet"
    df = pd.read_parquet(canonical_path)

    uf, links = resolve_entities(df, type_change=type_change)

    graph_path = data_dir / "artifacts" / "graph.pkl"
    g = read_graph_pickle(graph_path)
    g = add_co_spend_edges(g, links)
    write_graph_pickle(g, graph_path)
    export_edges(g, data_dir / "artifacts" / "graph_edges.parquet")

    entities_path = data_dir / "artifacts" / "entities.parquet"
    write_entities_parquet(uf, entities_path)

    predicted = pd.read_parquet(entities_path)
    typer.echo(f"clusters={len(uf.groups())} co_spend_links={len(links)} {_metrics_text(predicted, _ground_truth(data_dir))}")


@app.command()
def embed(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    min_cluster_size: int = typer.Option(3, "--min-cluster-size"),
    max_epochs: int = typer.Option(30, "--max-epochs"),
    seed: int = typer.Option(0, "--seed"),
    protect_cospend_clusters: bool = typer.Option(
        False, "--protect-cospend-clusters",
        help="Never let pass 2 split a pass-1 entity with >1 wallet. Real-data guard (see "
             "docs/WRITEUP.md's benchmarking findings) — turn on only once pass 1's own precision "
             "is independently confirmed high on this dataset; leave off for this repo's own "
             "synthetic data, where pass 1 is diagnosed as vacuous (0 co-spend links).",
    ),
    min_merge_probability: float = typer.Option(
        0.0, "--min-merge-probability",
        help="Skip a pass-2 merge unless every wallet in the triggering hdbscan cluster has "
             "hdbscan_prob at or above this floor. 0.0 preserves existing behavior exactly.",
    ),
    drop_broadcaster_outliers: bool = typer.Option(
        False, "--drop-broadcaster-outliers",
        help="Drop BROADCAST_VIA edges into population-outlier-fan-in IP nodes (e.g. a shared "
             "light-wallet server) before embedding training only — real-data over-merge guard, "
             "inert no-op on this repo's own synthetic data.",
    ),
    skip: bool = typer.Option(
        False, "--skip",
        help="Skip pass 2 entirely: pass 1's entities.parquet and graph stay as they are, and no "
             "SAME_ENTITY edges are added. Use when pass 2 does not beat pass 1 on a held-out dev set.",
    ),
) -> None:
    """ER pass 2: heterogeneous GraphSAGE wallet embeddings (argus.models.sage)
    + HDBSCAN clustering, reconciled against pass 1's entities.parquet (merge
    over-separated pass-1 clusters, split over-merged ones — see
    argus.er.embed_cluster's module docstring). Run after `resolve` (pass 1)
    and `features build` (its node_features.parquet is this pass's embedding
    input) — see the Makefile's `er2` target.
    """
    if skip:
        pass1_entities = pd.read_parquet(data_dir / "artifacts" / "entities.parquet")
        typer.echo(f"pass2=skipped (pass-1 entities kept) {_metrics_text(pass1_entities, _ground_truth(data_dir))}")
        return

    graph_path = data_dir / "artifacts" / "graph.pkl"
    g = read_graph_pickle(graph_path)
    node_features = pd.read_parquet(data_dir / "artifacts" / "node_features.parquet")
    pass1_entities = pd.read_parquet(data_dir / "artifacts" / "entities.parquet")

    g_for_embedding = drop_high_fanin_broadcasters(g) if drop_broadcaster_outliers else g
    training = train_wallet_embeddings(g_for_embedding, node_features, max_epochs=max_epochs, seed=seed)
    wallet_embeddings = wallet_embeddings_frame(training)

    # Real-data guard (see argus.er.embed_cluster.run_with_big_stack's own
    # docstring): always on, not opt-in like the guards above -- it changes
    # nothing about the result, only where HDBSCAN.fit's C-level recursion
    # runs, so there's no behavior to preserve-by-default for.
    hdbscan_df = run_with_big_stack(lambda: run_hdbscan(wallet_embeddings, min_cluster_size=min_cluster_size))
    result = reconcile_with_pass1(
        pass1_entities, hdbscan_df,
        protect_multiwallet_pass1_entities=protect_cospend_clusters,
        min_merge_probability=min_merge_probability,
    )

    entities_path = data_dir / "artifacts" / "entities.parquet"
    log_path = data_dir / "artifacts" / "er_pass2_log.parquet"
    write_pass2_outputs(result, entities_path, log_path)

    g = add_same_entity_edges(g, result.same_entity_links)
    write_graph_pickle(g, graph_path)
    export_edges(g, data_dir / "artifacts" / "graph_edges.parquet")

    n_merges = int((result.log["action"] == "merge").sum()) if not result.log.empty else 0
    n_splits = int((result.log["action"] == "split").sum()) if not result.log.empty else 0
    n_touched = int((result.entities["source"] == "pass2").sum())

    loss_prefix = f"final_training_loss={training.losses[-1]:.4f} " if training.losses else ""
    typer.echo(
        f"{loss_prefix}wallets_embedded={len(wallet_embeddings)} merges={n_merges} splits={n_splits} "
        f"wallets_touched=pass2:{n_touched} same_entity_links={len(result.same_entity_links)} "
        f"{_metrics_text(result.entities, _ground_truth(data_dir))}"
    )


if __name__ == "__main__":
    app()
