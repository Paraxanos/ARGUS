# ARGUS Data Contracts

These are the exact data contracts between pipeline stages. Do not add, remove, or reinterpret
any column without updating this document and getting explicit sign-off — downstream stages are
written against these schemas verbatim.

## `canonical/transactions.parquet`

Producer: `ingest`. One row per transaction event.

- `txid` (string, primary key)
- `timestamp` (datetime, UTC)
- `src_ip` (string)
- `dst_ip` (string)
- `src_port` (int)
- `dst_port` (int)
- `geo_country` (string)
- `asn` (int)
- `input_addresses` (array<string>)
- `input_amounts` (array<float>)
- `output_addresses` (array<string>)
- `output_amounts` (array<float>)
- `fee` (float)
- `script_type` (string)

## Graph schema

Artifacts: `artifacts/graph.pkl` + `artifacts/graph_edges.parquet`. Producer: `graph`.

**Nodes:** `Wallet`, `Transaction`, `IP`, `ASN`

**Edges:**

| Edge | Direction | Meaning | Attrs |
|---|---|---|---|
| `FUNDS` | `Wallet -> Tx` | tx input | `amount` |
| `PAYS` | `Tx -> Wallet` | tx output | `amount` |
| `BROADCAST_VIA` | `Tx -> IP` | first-seen relay | `timestamp`, `port` |
| `RESOLVES_TO` | `IP -> ASN` | geo enrichment | static |
| `CO_SPEND` | `Wallet <-> Wallet` | ER pass 1, Dev A | `confidence` |
| `SAME_ENTITY` | `Wallet <-> Wallet` | ER pass 2, Dev B (`er/embed_cluster.py`) | `confidence` ∈ [0,1] |

This repo produces every edge type in the table: `FUNDS`, `PAYS`, `BROADCAST_VIA`,
`RESOLVES_TO` (from graph build), `CO_SPEND` (ER pass 1), and `SAME_ENTITY` (ER pass 2, added
via `add_same_entity_edges` — same `add_wallet_link_edges` machinery `CO_SPEND` uses, see
`er/union_find.py`). See `artifacts/entities.parquet` and `artifacts/scores_risk.parquet` below
for the measured, diagnosed quality of the `SAME_ENTITY` edges on this dataset.

## `ground_truth/entities.parquet`

Producer: `synth`.

- `wallet_id`
- `entity_id`
- `entity_type` (licit/ransomware/darknet/mixer/exchange)

## `ground_truth/seeds.parquet`

Producer: `synth`. Known-illicit seed set, ~5-10% of illicit wallets.

- `wallet_id`

## `ground_truth/patterns.parquet`

Producer: `synth`.

- `pattern_id`
- `type` (peeling/coinjoin)
- `txids[]`
- `wallets[]`

## `artifacts/entities.parquet`

Producer: ER pass 1 (`er/cli.py resolve`), then overwritten in place by ER pass 2
(`er/cli.py embed`) once it has run — pass 2 reconciles against whatever is currently on disk,
it never re-derives pass 1 from scratch. Run order matters: `resolve` must run before `embed`
(the Makefile's `er` -> `features` -> `er2` chain enforces this).

- `wallet_id`
- `entity_id` (pass 1's `resolved_<wallet>` id, or a pass-2 merged/split id — see
  `er/embed_cluster.py`'s module docstring for the exact algorithm)
- `source` (`"pass1"` for a wallet pass 2 left untouched, `"pass2"` for one it merged or split)
- `conf` (pass 1's fixed 1.0/0.85, or — for a `"pass2"` row — the HDBSCAN membership
  probability from `run_hdbscan`)
- `merge_split_log_ref` (`null`, or a `log_id` into `artifacts/er_pass2_log.parquet`)

## `artifacts/er_pass2_log.parquet`

Producer: `er/embed_cluster.py`, via `write_pass2_outputs`. One row per merge or split action
pass 2 applied — every `entities.parquet` row with `source="pass2"` has its `merge_split_log_ref`
pointing here, per this doc's hard rule below.

- `log_id`
- `action` (`"merge"` or `"split"`)
- `hdbscan_cluster` (the HDBSCAN label that triggered a merge; `null` for a split)
- `pass1_entities` (JSON-encoded list — the pass-1 entity ids involved)
- `wallets` (JSON-encoded list — every wallet covered by this action)
- `evidence_json`

## `artifacts/node_features.parquet`

Producer: `features`.

- `node_id`
- `node_type`
- `f_*` numeric columns
- no NaNs

## `artifacts/scores_pattern.parquet`

Producer: `detectors/peeling.py` + `detectors/coinjoin.py` (classical, structural) and, as of
Dev B Phase 3, `detectors/pattern_sim.py` (embedding-similarity recall extension) — all three
append rows to the same artifact (`detectors/scores.py`'s own long-standing docstring), never
replacing or deduplicating each other's detections.

- `node_id`
- `score` ∈ [0,1] (pattern_sim: `(cosine_similarity + 1) / 2`, order-preserving)
- `reason_code` (classical: `PEEL_CHAIN_HOPS=n` / `COINJOIN_ROUND_N=n`; pattern_sim:
  `PATTERN_SIM_PEELING=<sim>` / `PATTERN_SIM_COINJOIN=<sim>`)
- `evidence_json`

**pattern_sim measured result (full 200k-tx dataset — verified, not assumed):** 102 additional
matches, **all** `PATTERN_SIM_COINJOIN` on `Transaction`-type nodes, zero on `Wallet`-type nodes
and zero `PATTERN_SIM_PEELING` matches of either node type. This is the correct, diagnosed
outcome, not underperformance: candidate-vs-reference cosine similarity was measured directly
per (pattern_type, node_type) pair before shipping, and only Transaction-vs-CoinJoin is
genuinely well-separated (median similarity ~0.04 — a CoinJoin transaction's many-in/many-out
shape is a rare, distinctive local topology); every other pairing is compressed (median already
~0.94-0.95), the same root cause diagnosed in `er/embed_cluster.py`'s docstring (most
wallets/transactions are structurally-ordinary single-hop activity, so their embeddings cluster
tightly regardless of true pattern membership). See `detectors/pattern_sim.py`'s module
docstring for the full measurement and the adaptive z-score gate (on top of an absolute
similarity floor) this finding required — an uncapped flat cosine threshold measured 142,351
matches out of ~343k nodes (41% of the entire graph) before that fix, the same class of failure
as ER pass 2's uncapped HDBSCAN run and fixed on the same principle: a candidate must be a
genuine outlier *relative to its own candidate population*, not just above one global constant.

## `artifacts/scores_risk.parquet`

Producer: `detectors/risk_ppr.py`.

- `node_id`
- `score` ∈ [0,1]
- `reason_code` (`SEED_DIST=n`)
- `evidence_json`

**Status (updated, Dev B Phase 1):** `detectors/risk_ppr.py` now propagates over BOTH
`CO_SPEND` + `SAME_ENTITY` edges, the architecture doc's full spec — no longer a scope
limitation. ER pass 1 still produces zero `CO_SPEND` edges on this dataset (unchanged; see the
Phase 2 ER precision/recall diagnosis), but ER pass 2 (`er/embed_cluster.py`) now supplies
`SAME_ENTITY` edges, so this head does propagate beyond the seed set (measured: 928 seeds ->
1,616 scored rows on the full dataset, via 7,846 `SAME_ENTITY` edges). **However**, per
`er/embed_cluster.py`'s own MEASURED RESULT diagnosis, those `SAME_ENTITY` edges carry weak
signal on this dataset (pass 2's pairwise recall against ground truth stays ~0), so this
head's precision@50 is correspondingly not meaningfully better than the pre-pass-2 baseline —
propagation is happening, but over largely noisy edges. Root cause and fix path are documented
in `er/embed_cluster.py`'s module docstring, not repeated here.

## `artifacts/scores_anomaly.parquet`

Producer: `fusion/cli.py`, via `argus.models.anomaly.train_and_score_anomalies` (Dev B Phase
2) — a real graph autoencoder, not a placeholder. Shares `argus.models.encoder`'s temporal
heterogeneous GAT-v2 encoder (architecture doc sec 4.3) with a per-node-type linear decoder,
trained to reconstruct each node's own input features; reconstruction error is z-scored within
its node type and squashed to [0,1] via sigmoid. Covers **every** node in the graph (not just
pattern/risk-flagged ones) — see `fusion/blend.py`'s `component_table` for why that's safe
(anomaly alone never expands the fusion universe).

- `node_id`
- `score` ∈ [0,1] (`sigmoid(z_score)`)
- `reason_code` (`ANOMALY_ZSCORE=<z_score, 2dp>`)
- `evidence_json` (`z_score`, `raw_reconstruction_error`, `node_type`, `population_mean_error`,
  `population_std_error`)

**Measured result (full 200k-tx dataset — verified, not assumed, same standard as every other
finding in this repo):** AUC-ROC 0.510 / precision@50 0.30 against `ground_truth/entities.parquet`'s
illicit labels — essentially chance-level ranking overall, with modest lift in the very top
scores. The mechanism itself is independently verified correct: a hand-built fixture with one
deliberately planted feature-space outlier among 20 near-identical "normal" nodes is scored
correctly and by a wide margin (`tests/test_models_anomaly.py`). The gap between that and the
real-dataset result is a real, diagnosable finding, not a bug: this head is *unsupervised*
reconstruction-error anomaly detection — "statistically unusual" — not a classifier for "is this
labeled ransomware/darknet/mixer," and those aren't the same thing here. Illicit entities in
this generator aren't necessarily feature-space outliers on the current `f_*` schema (many
behave in ordinary-looking ways by construction), while legitimate high-volume entities
(exchanges) can be structural outliers without being illicit — so high reconstruction error and
the illicit label only partially overlap. Not fixed here: a supervised or semi-supervised
variant, or features more specifically discriminative of the illicit campaign types, would be
the next step, out of this phase's scope.

## `artifacts/encoder_checkpoint.pt`

Producer: `detectors/cli.py`, via `fusion.evidence.save_encoder_checkpoint` (Dev B Phase 4).
Persists the trained shared encoder (state dict) plus its full input `HeteroData` and
`node_ids` — NOT the full `_GraphAutoencoder` (its decoders are only needed for the anomaly
scoring `detectors/cli.py` already finished by the time this is written). Reloaded by
`fusion/cli.py` for attention-based evidence extraction, so no third training pass is needed —
see `fusion/evidence.py`'s module docstring. ~76 MB on the full dataset (dominated by the
persisted `HeteroData`, not the encoder weights themselves) — a real, documented disk-space
cost of this design, not accidental bloat.

## `artifacts/alerts.json`

Producer: `fusion`.

- `alert_id`
- `node_id`
- `final_score`
- `components` `{pattern, risk, anomaly}`
- `evidence` `{nodes[], edges[]}` — as of Dev B Phase 4, includes attention-based neighbors from
  `fusion/evidence.py` when `encoder_checkpoint.pt` exists, alongside each reason_code's own
  evidence (`fusion/rationale.py`)
- `rationale` — as of Phase 4, generated by `fusion/rationale.py`'s templating engine (a
  registry of reason_code-prefix -> evidence/clause functions, replacing an earlier inline
  version in `fusion/blend.py`), and includes an attention clause naming the single
  highest-weight evidence-subgraph neighbor when available

**Measured result (full 200k-tx dataset):** 2,095 of 2,096 alerts (99.95%) carry a real
attention-derived clause naming a specific neighbor node (e.g. an IP address the model
weighted most heavily). This is a genuine capability increase over pre-Phase-4 alerts, most
visible for `ANOMALY_ZSCORE`-only flags: `fusion/rationale.py`'s anomaly evidence extractor
deliberately returns no multi-node evidence of its own (a reconstruction-error anomaly is a
property of the node itself), so before Phase 4 those alerts had NO structural evidence beyond
the flagged node — attention-based extraction is what fills that specific gap.

**Consumer (Dev B Phase 5):** `dashboard/app.py` reads this file plus `node_features.parquet`
(for graph node-type coloring) — read-only, produces no artifact of its own, so no new contract
section is needed here.

## Hard rule

Every head must emit `reason_code` + `evidence_json` — no exceptions, no retrofitting later. This
is a stated hard rule from the project plan.
