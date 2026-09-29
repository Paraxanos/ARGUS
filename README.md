# ARGUS — SIH26146 / NTRO Bitcoin Transaction Monitoring

Team Doomsbyte. Dev A's classical pipeline plus Dev B's Phase 1 (ER pass 2: GraphSAGE +
HDBSCAN), Phase 2 (shared temporal GAT-v2 encoder + graph autoencoder anomaly detection), Phase
3 (embedding-similarity pattern detector), and Phase 4 (attention-based evidence extraction +
rationale-templating engine). Still **not** included:

- The Streamlit **dashboard**.

See `docs/WRITEUP.md` for the full write-up (Dev A's sections plus Dev B's Phases 1-4; the
dashboard is marked TODO, not drafted).

## What this repo does

Synthetic Bitcoin transaction data generation → ingestion (CSV/JSON/XML) → typed dual-layer
graph construction → classical entity resolution pass 1 (Union-Find) → node feature engineering
→ entity resolution pass 2 (heterogeneous GraphSAGE embeddings + HDBSCAN clustering, merging or
splitting pass 1's clusters) → classical pattern detection (peeling chains, CoinJoin) + a shared
GAT-v2 encoder feeding both graph-autoencoder anomaly detection and embedding-similarity
pattern-recall extension → seeded risk propagation over `CO_SPEND` + `SAME_ENTITY` edges
(Personalized PageRank) → score fusion with attention-based evidence extraction and a
templated, human-checkable rationale per alert → `alerts.json`. Everything runs fully offline
at runtime — see `docs/offline_install.md`.

## Install

Requires Python 3.12 (see `docs/offline_install.md` for why not the latest interpreter) and
GNU Make.

```
make env
```

This creates a `.venv` and installs everything in `requirements.txt`. Internet access is
needed for this step only (downloading packages) — see `CLAUDE.md`'s offline requirement and
`docs/offline_install.md` for the verification that nothing after this point makes a network
call.

## Running the pipeline

```
make pipeline
```

Runs the full chain: `data → ingest → graph → er → features → er2 → detect → fusion`, producing
`data/artifacts/alerts.json` as the final output. Each stage can also be run individually
(`make data`, `make ingest`, `make graph`, `make er`, `make features`, `make er2`, `make detect`,
`make fusion`) — each depends on the previous stage's output via the Makefile. `er2` is ER pass
2 (`argus.er.cli embed`): GraphSAGE + HDBSCAN, reconciled against pass 1's `entities.parquet` —
see `docs/WRITEUP.md`'s "Entity resolution pass 2" section for its measured result and a
diagnosed root cause for why it doesn't yet meaningfully improve on pass 1 on this dataset.
`detect` trains the shared GAT-v2 encoder once (`argus.models.anomaly`) and uses it for both the
anomaly head and the embedding-similarity pattern detector (`argus.detectors.pattern_sim`, Dev B
Phase 3 — no second training pass), alongside the classical peeling/CoinJoin/risk detectors, and
persists the trained encoder (`data/artifacts/encoder_checkpoint.pt`, ~76 MB) — see
`docs/WRITEUP.md`'s "Anomaly detection" and "Pattern-similarity detection" sections. `fusion`
reloads that checkpoint to extract attention-based evidence for the final alert list (Dev B
Phase 4, `argus.fusion.evidence` + `argus.fusion.rationale` — no third training pass) — see
"Explainable evidence extraction". All training happens in `er2` and `detect` (~60-100s for
`er2`; `detect` ~4-4.5 minutes at this scale, CPU-only, dominated by encoder training, not the
detectors themselves); `fusion` adds ~1.5 minutes for evidence extraction (checkpoint load + one
attention-capturing forward pass), no longer the sub-10-second pure score-reading it briefly
was between Phases 2 and 4.

`configs/default.yaml` controls the generated scale and noise/difficulty knobs (`ip_noise`,
`heuristic_break_rate`, `mixer_fraction`); see `docs/WRITEUP.md`'s "Scale tested" section for
what was actually run.

## The three input formats

`make data` (via `src/argus/synth/`) generates one synthetic dataset and exports the **same**
underlying transactions to three raw formats under `data/raw/`:

- `transactions.csv`
- `transactions.json`
- `transactions.xml`

This exists to exercise `src/argus/ingest/`'s streaming parsers (`csv.DictReader`, `ijson`,
`lxml.iterparse`) against realistically messy multi-format input — including a deliberate
~0.5% data-quality corruption rate (bad checksums, negative amounts, unparseable timestamps),
which ingestion must reject to `data/artifacts/rejects.log`, never silently drop. All three
formats are asserted to parse to row-for-row identical canonical output
(`tests/test_ingest.py::test_cross_format_equality`).

## Reproducing metrics

```
make eval
```

Prints a single summary table reproducing every metric from the ER, pattern-detector, and
risk-head phases against the current `make pipeline` artifacts, then runs the dual-layer vs
on-chain-only ablation (swept across `ip_noise`), saving `docs/ablation_results.csv` and
`docs/ablation_plot.png`. See `docs/WRITEUP.md` for the actual numbers and their
interpretation — including an honest negative result on the ablation (see below).

## Tests

```
make test
```

## Known, documented limitations (not bugs)

- **ER pass 1 currently has 0.0 recall** on this dataset — a verified, diagnosed finding, not
  an oversight. See `docs/WRITEUP.md`'s "Entity resolution" section.
- **ER pass 2 (GraphSAGE + HDBSCAN) runs correctly but stays near-0 recall on this dataset too**
  — the algorithm itself is verified correct against a hand-built fixture; the real-dataset
  weakness has a diagnosed root cause (an `IP`-node graph-topology limitation, not a bug in the
  ER code) with a concrete, out-of-scope-for-this-phase fix identified. See `docs/WRITEUP.md`'s
  "Entity resolution pass 2" section and `src/argus/er/embed_cluster.py`'s module docstring.
- **The risk head now propagates beyond the seed set** via pass 2's `SAME_ENTITY` edges (it no
  longer reduces to a no-op), but inherits pass 2's weak signal above — not a separate issue in
  `src/argus/detectors/risk_ppr.py`. See `docs/contracts.md`.
- **The dual-layer vs on-chain-only ablation shows no measurable difference** — verified: none
  of this repo's classical detectors read `BROADCAST_VIA`/`RESOLVES_TO` edges or the
  cross-layer `f_*` features. See `docs/WRITEUP.md`'s "Ablation" section. (Unaffected by ER pass
  2 or the anomaly head, both distinct, non-classical stages — the ablation covers the classical
  pipeline only, per its own scope.)
- **The anomaly head (graph autoencoder, `models/anomaly.py`) is a real, verified-correct
  model** — a planted feature-space outlier is detected with a wide margin on a hand-built
  fixture — **but stays near chance-level (AUC-ROC 0.51) at ranking the real dataset's specific
  illicit-entity-type labels**, a diagnosed gap between "statistically unusual" and "one of the
  labeled illicit types," not a bug. See `docs/WRITEUP.md`'s "Anomaly detection" section and
  `docs/contracts.md`'s `scores_anomaly.parquet` section.
- **The pattern-similarity detector (`detectors/pattern_sim.py`) only finds signal for
  CoinJoin-transaction near-variants, not peeling chains or wallet-level near-variants** — the
  same diagnosed embedding-collapse limitation as ER pass 2 and the anomaly head (most
  wallets/transactions are structurally ordinary, so embeddings can't discriminate); measured,
  not assumed, via the actual similarity distribution per pattern/node-type pairing before
  shipping. An uncapped flat threshold measured a 41%-of-graph false-positive flood before an
  adaptive population-relative z-score gate fixed it. See `docs/WRITEUP.md`'s
  "Pattern-similarity detection" section and `docs/contracts.md`'s `scores_pattern.parquet`
  section.
- **`fusion/evidence.py`'s attention extraction had a real, caught-before-shipping performance
  bug**: an early version ran a full-graph forward pass per alert, measured to push `fusion`'s
  runtime for a 15k-wallet test dataset past 9 minutes. Fixed by separating the one-time forward
  pass (`build_attention_context`) from the per-alert lookup (`extract_attention_evidence`);
  `fusion` now runs in ~1.5 minutes on the full 200k-tx dataset. See `docs/WRITEUP.md`'s
  "Explainable evidence extraction" section.
