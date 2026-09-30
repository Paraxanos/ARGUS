# ARGUS — SIH26146 / NTRO Bitcoin Transaction Monitoring

Team Doomsbyte. Dev A's classical pipeline plus Dev B's Phase 1 (ER pass 2: GraphSAGE +
HDBSCAN), Phase 2 (shared temporal GAT-v2 encoder + graph autoencoder anomaly detection), Phase
3 (embedding-similarity pattern detector), Phase 4 (attention-based evidence extraction +
rationale-templating engine), and Phase 5 (Streamlit dashboard). All Dev B phases from the
project plan are now complete.

See `docs/WRITEUP.md` for the full write-up (Dev A's sections plus Dev B's Phases 1-5).

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

## Running the dashboard

```
make dashboard
```

Opens the read-only Streamlit dashboard (`src/argus/dashboard/app.py`) over the pipeline's
already-computed `data/artifacts/alerts.json`: a ranked, filterable alert table plus an
interactive pyvis link-analysis view of each alert's evidence subgraph. It never re-runs any
pipeline stage — run `make pipeline` first, or `make demo` to run both in sequence. Point it at
a different run with `ARGUS_DATA_DIR=<path> make dashboard`. Fully offline: pyvis's HTML output
is generated with `cdn_resources="in_line"` plus a verified post-generation strip of the two
Bootstrap CDN tags its template still hardcodes in that mode — see `docs/WRITEUP.md`'s
"Dashboard" section.

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
- **pyvis's `cdn_resources="local"`/`"in_line"` modes both still reference at least one external
  CDN** — verified directly against the installed pyvis version's own HTML template, not
  assumed. The dashboard works around this with a post-generation strip plus a self-verifying
  assertion that no `http(s)://` resource reference survives. The dashboard's own UI was
  verified via a direct `main()` call and an HTTP 200 from the running server, not visually in a
  browser — no browser-automation harness was available in this environment. See
  `docs/WRITEUP.md`'s "Dashboard" section.
- **`streamlit run` blocks on an interactive email prompt and calls out to a telemetry endpoint
  on a machine with no pre-existing `~/.streamlit/credentials.toml`** — a real bug caught by
  reading Streamlit's own source, not assumed, since this dev machine's own prior Streamlit use
  masked it locally. Fixed: `make dashboard` now passes `--server.headless=true
  --browser.gatherUsageStats=false`, verified against a scratch fake-`HOME` with no `.streamlit`
  directory at all. See `docs/WRITEUP.md`'s "Dashboard" section.
- **A teammate's real-data benchmarking run (2026-09-30) found the fixed 0.6 alert threshold
  floods the queue at realistic (~0.2%) illicit prevalence** (31,059 alerts from 40,016 scored
  nodes) — confirmed by reading `fusion/blend.py`'s own calibration comment, which admits the
  threshold was tuned to synthetic data's much higher prevalence. Fixed: `build_alerts()` gained
  a `top_k` cap (`fusion/cli.py --top-k`, default 500) applied after the threshold floor,
  verified against a hand-built low-prevalence fixture (no real data needed). Four more findings
  from the same report (ER pass-2 over-merging, a too-narrow scored universe, an uninformative
  anomaly head, and peeling false positives on real transaction shapes) are diagnosed but not yet
  fixed — see `docs/WRITEUP.md`'s "Real-data benchmarking findings" section for the full
  diagnosis plan.
- **Two more findings from the same report are now fixed (opt-in, off by default):** ER pass 2
  was dropping real-data precision from 0.960 to 0.209 by splitting pass 1's mostly-correct
  clusters and over-merging wallets sharing broadcast infrastructure — `er/cli.py embed` gained
  `--protect-cospend-clusters`, `--min-merge-probability`, and `--drop-broadcaster-outliers`. And
  only 14.7% of illicit targets ever entered the scored universe — `fusion/cli.py
  --anomaly-admission-threshold` and `detectors/cli.py --risk-damping` widen it. All four flags
  default to the exact pre-fix behavior; verified against hand-built fixtures reproducing each
  marker, no real data touched. See `docs/WRITEUP.md`'s "Real-data benchmarking findings"
  section.
- **The anomaly head's real-data flatness and ER pass 2's real-data loss blowup were tested
  against two hypotheses (pure scale, degree skew) and NEITHER held up** — reported honestly
  rather than forcing an unverified fix. `models/anomaly.py` gained per-epoch loss tracking
  (`AnomalyResult.losses`, matching `models/sage.py`'s existing convention) so any future run can
  directly tell whether training converged. See `docs/WRITEUP.md`'s "Real-data benchmarking
  findings" section for the concrete, non-sensitive handoff ask for the benchmarking team's next
  re-run.
- **A real-shaped IP address (outside this generator's synthetic pool) or IPv6 used to crash the
  entire ingest run** — fixed: `ingest/geoip.py` now supports real MaxMind-format databases
  (`ARGUS_GEOIP_ASN_MMDB`/`ARGUS_GEOIP_COUNTRY_MMDB`, opt-in) and rejects any unresolvable
  address to `rejects.log` instead of crashing, verified against a mocked `geoip2.database.Reader`
  (no real database file needed).
- **`sklearn.cluster.HDBSCAN.fit`'s Windows stack overflow at real-world wallet counts was
  reproduced exactly** — 150,000 purely synthetic random embeddings (no real data) crashed with
  exit code 127 and no Python traceback on this same machine, matching the report's observed
  symptom precisely. Fixed and confirmed: `argus.er.embed_cluster.run_with_big_stack()` runs
  HDBSCAN on a 128 MB thread stack, always on (no behavioral downside — it only changes where
  the recursion runs, not the result) — the exact same 150,000-point case that crashed with
  exit 127 now completes successfully (687.2s, exit code 0, 150,000 rows, 9,136 non-noise).
  Peeling false positives on real transaction shapes remain diagnosed but not yet fixed (the
  fix requires a new synthetic hard-negative pattern, not attempted this session).
