"""Isolation-forest anomaly scorer — a label-free, explainable alternative to the graph-autoencoder anomaly head.

Why: on real-shaped data the graph autoencoder's scores sit in a narrow band around 0.5 for every node and their
ranking was not stable across training runs (the benchmark on the ARGUS dataset measured an AUC of 0.58 in one run
and 0.21 in another on the same data), so they carried no usable signal.

Design (nothing here is fitted to any labelled dataset):
  * one IsolationForest per node type (a wallet is only comparable with wallets, a transaction with transactions),
    over argus.features.build's per-node features, log1p-scaled (counts and amounts are heavy-tailed);
  * an ensemble of ENSEMBLE_SEEDS forests, each node's score = mean within-type percentile rank of its isolation
    score, so the output is stable across seeds by construction and means the same thing at any scale;
  * the evidence names the features on which the node deviates most from its type's median (robust z-scores), so
    an analyst can see why it was flagged.
The per-seed rank correlation is reported so instability would be visible, not hidden.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ENSEMBLE_SEEDS = (0, 1, 2)
N_ESTIMATORS = 200
SCORED_TYPES = ("Wallet", "Transaction")
TOP_EVIDENCE_FEATURES = 3


@dataclass
class IForestResult:
    rows: list[dict]
    seed_rank_correlation: float   # minimum pairwise Spearman correlation between the ensemble members' rankings


def score_nodes(node_features: pd.DataFrame) -> IForestResult:
    feats = [c for c in node_features.columns if c.startswith("f_")]
    rows: list[dict] = []
    min_corr = 1.0
    for node_type, part in node_features.groupby("node_type"):
        if node_type not in SCORED_TYPES or len(part) < 50:
            continue
        X = part[feats].astype(float).fillna(0.0)
        X = np.sign(X) * np.log1p(np.abs(X))
        usable = [c for c in feats if X[c].std() > 0]
        if not usable:
            continue
        Xu = X[usable].to_numpy()
        ranks = []
        for seed in ENSEMBLE_SEEDS:
            forest = IsolationForest(n_estimators=N_ESTIMATORS, random_state=seed).fit(Xu)
            iso = -forest.score_samples(Xu)                         # higher = more isolated = more anomalous
            ranks.append(pd.Series(iso).rank(pct=True).to_numpy())
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                min_corr = min(min_corr, float(np.corrcoef(ranks[i], ranks[j])[0, 1]))
        score = np.mean(ranks, axis=0)

        med = X[usable].median()
        mad = (X[usable] - med).abs().median().replace(0, np.nan)
        robust_z = ((X[usable] - med) / (1.4826 * mad)).fillna(0.0)
        top = robust_z.abs().to_numpy().argsort(axis=1)[:, ::-1][:, :TOP_EVIDENCE_FEATURES]
        for k, (node_id, s) in enumerate(zip(part["node_id"], score)):
            dev = {usable[c]: round(float(robust_z.iat[k, c]), 2) for c in top[k]}
            rows.append({
                "node_id": node_id, "score": float(s), "reason_code": f"IFOREST_PCTL={s:.2f}",
                "evidence_json": json.dumps({"node_type": node_type, "percentile": round(float(s), 4),
                                             "top_deviating_features": dev}),
            })
    return IForestResult(rows=rows, seed_rank_correlation=round(min_corr, 4))
