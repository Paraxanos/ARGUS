# ARGUS — SIH26146 / NTRO Bitcoin Transaction Monitoring

Team Doomsbyte. Dev A's classical pipeline plus Dev B's Phase 1 (ER pass 2: GraphSAGE +
HDBSCAN). Still **not** included:

- Dev B's remaining ML work: the temporal hetero GAT-v2 detection encoder, graph autoencoder
  anomaly detection, embedding-similarity pattern detector, attention-based evidence extractor,
  full rationale-templating engine.
- The Streamlit **dashboard**.

See `docs/WRITEUP.md` for the full write-up (Dev A's sections plus Dev B's Phase 1; the rest of
Dev B's are marked TODO, not drafted).

## What this repo does

Synthetic Bitcoin transaction data generation → ingestion (CSV/JSON/XML) → typed dual-layer
graph construction → classical entity resolution pass 1 (Union-Find) → node feature engineering
→ entity resolution pass 2 (heterogeneous GraphSAGE embeddings + HDBSCAN clustering, merging or
splitting pass 1's clusters) → classical pattern detection (peeling chains, CoinJoin) → seeded
risk propagation over `CO_SPEND` + `SAME_ENTITY` edges (Personalized PageRank) → score fusion →
`alerts.json`. Everything runs fully offline at runtime — see `docs/offline_install.md`.

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
  2, which is a distinct, non-classical stage.)
- **`artifacts/scores_anomaly.parquet` is a documented placeholder** (fixed 0.5 for every
  node), not a real anomaly model — Dev B's `models/anomaly.py` is out of scope here.
