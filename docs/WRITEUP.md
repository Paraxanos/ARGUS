# ARGUS — Dev A Write-Up

This document covers **Dev A's owned sections only** (SIH26146 / NTRO Bitcoin transaction
monitoring, team Doomsbyte). See `CLAUDE.md` for the exact scope boundary and the list of
files this repo deliberately never implements. Dev B's sections are left as marked TODOs
below — not drafted here.

## Scale tested

All results in this document are from the scale configured in `configs/default.yaml`:

- 2,000 entities, 50,000 wallets
- 200,000 baseline transactions + several hundred planted illicit-pattern transactions
  (200,784 total in the run these numbers come from)
- Raw exports at this scale: ~46 MB CSV, ~120 MB JSON, ~159 MB XML

This is **not** the plan's optional 1M-transaction stretch goal, which was not attempted (per
instruction — see `docs/offline_install.md`'s sibling scale note, or the section below). A
rough scaling estimate for 1M transactions (5x this run):

- Raw export sizes scale roughly linearly with row count: ~230 MB CSV, ~600 MB JSON, ~800 MB
  XML — the XML format in particular becomes a meaningful disk-space and parse-time cost at
  that scale, more so than the other two.
- Graph size scales with transaction count too: this run produced ~343k nodes / ~824k edges;
  5x would land near ~1.7M nodes / ~4M edges, still well within what `python-igraph`'s C
  backend handles in memory on ordinary hardware, but `node_features.parquet` and
  `graph_edges.parquet` would grow proportionally (a few hundred MB each).
- The full pipeline (`data` through `fusion`) took 1.5-3.5 minutes wall-clock at 200k
  transactions on the development machine across several Dev-A-only runs; naive linear scaling
  would put 1M transactions at roughly 10-20 minutes for that portion, though several stages
  (ER pass 1's transaction scan, feature engineering's per-wallet grouping) are closer to O(n)
  than O(n²), so this is a rough upper bound rather than a hard prediction.
- Dev B's Phase 1 (ER pass 2: GraphSAGE training + HDBSCAN, in `er2`) and Phases 2-3 (shared
  GAT-v2 encoder + autoencoder training + pattern-similarity search, all in `detect` as of
  Phase 3's refactor — see "Pattern-similarity detection" for why that training moved out of
  `fusion`) each add roughly 1-1.5 minutes at this scale, measured directly (~60-100s for `er2`;
  `detect` measured at ~4m25s total, of which the shared-encoder training is the dominant cost —
  the classical detectors and pattern-similarity search itself are seconds, not minutes;
  `fusion` is back to sub-10-second pure score-reading, as it was pre-Phase-2). Both trained
  models run a small number of full-batch epochs over the full ~343k-node graph, not
  mini-batched, so this is expected to scale roughly linearly with graph size same as the rest
  of the pipeline. CPU-only throughout — no GPU used or required.
- This was not run. If asked to attempt 1M transactions, expect to hit the Windows long-path
  `pip install` issue documented in the Phase 1 fresh-clone verification again if working from
  a deeply-nested directory — unrelated to scale, but worth flagging alongside it.

## Data generation & ingestion

`src/argus/synth/` generates a fully-seeded synthetic dataset: entities (licit, exchange,
ransomware, darknet, mixer), wallets, and transactions, with an IP-subnet affinity and
broadcast-time profile per entity, three illicit structural patterns (peeling chains, CoinJoin
rounds, ransomware collect→layer→cash-out lifecycles), a known-illicit seed set, and
deliberate small-scale data-quality corruption (~0.5% of baseline rows: bad checksum, negative
amount, unparseable timestamp) — all exported identically to CSV, JSON, and XML.

`src/argus/ingest/` streams all three formats back in (`csv.DictReader`, `ijson`,
`lxml.iterparse` — never loading a whole file into memory), rejects the corrupted rows to
`data/artifacts/rejects.log` with a specific reason per row (never silently dropped), enriches
each row with `geo_country`/`asn` via an offline synthetic GeoIP lookup (see
`src/argus/ingest/geoip.py`'s docstring for why this is not a real MaxMind database — the
dataset's IPs are synthetic, so a real lookup would be meaningless), and writes the validated
rows to `canonical/transactions.parquet` with exactly the contracted columns. A dedicated test
(`test_cross_format_equality`) asserts all three input formats parse to row-for-row identical
canonical output.

## Entity resolution (pass 1)

`src/argus/er/union_find.py` implements the two classical heuristics — common-input-ownership
and change-address-return — over a Union-Find structure, with an explicit structural guard
against CoinJoin over-merging (transactions with many distinct FUNDS inputs and many
near-equal-value PAYS outputs are skipped for the common-input heuristic).

**Measured result: precision 1.0000, recall 0.0000.** This is not a bug being hidden — it's a
real, verified finding. The only transactions in this dataset with more than one genuinely
distinct input wallet are CoinJoin rounds, which the guard must skip; every other transaction
either has a single input or repeats the same wallet as "multiple inputs." The change-address
heuristic only ever fires when an output literally equals one of the transaction's own inputs
— in this dataset, that only happens when it's the same wallet (baseline's own change output),
never two different wallets. Net effect: heuristic 1 has zero real material to act on once the
CoinJoin guard is applied, and heuristic 2 never produces a new cross-wallet link. Every wallet
resolves to its own singleton cluster, `CO_SPEND` edges never materialize (0 in every run of
this generator), and precision is a vacuous 1.0 (zero false merges among zero merges) rather
than a real success. See the Phase 2 commit message and this repo's test suite
(`tests/test_er.py`) for the same finding, verified directly against the canonical data before
any code was written to explain it away.

This has a direct downstream consequence, updated by Dev B's Phase 1 below: with zero
`CO_SPEND` edges, `src/argus/detectors/risk_ppr.py`'s seeded Personalized PageRank has no
graph structure to propagate over from pass 1 alone — see the "Entity resolution pass 2"
section for what changes once `SAME_ENTITY` edges exist. `docs/contracts.md` documents the
current state precisely.

## Entity resolution pass 2 (Dev B, Phase 1)

`src/argus/models/sage.py` trains a heterogeneous GraphSAGE encoder (PyTorch Geometric
`HeteroConv` of `SAGEConv` per relation, `ToUndirected()`-symmetrized for message passing,
self-supervised via negative-sampling link prediction) over the full dual-layer graph, using
`node_features.parquet`'s `f_*` columns directly as input features. `src/argus/er/embed_cluster.py`
clusters the resulting wallet embeddings with HDBSCAN (`cluster_selection_method="leaf"`,
`max_cluster_size=500`) and reconciles the result against pass 1's `entities.parquet`: a
non-noise HDBSCAN cluster spanning more than one pass-1 entity triggers a **merge**; a pass-1
entity whose wallets land in more than one HDBSCAN bucket triggers a **split**. Every action is
logged to `artifacts/er_pass2_log.parquet` with its evidence, referenced from
`entities.parquet`'s `merge_split_log_ref` — see `docs/contracts.md`.

**Algorithm correctness — verified independently of data quality:** a hand-built fixture
(`tests/test_er_embed_cluster.py::test_merge_and_split_and_noise_all_correct`) with known-correct
merge, split, and noise cases passes exactly: three singleton pass-1 entities whose embeddings
cluster together get merged into one; a pre-merged pass-1 entity whose wallets embed into two
separate clusters gets split into two; an isolated wallet is correctly left as noise, untouched,
`source` still `"pass1"`. `tests/test_models_sage.py` separately verifies the GraphSAGE encoder
itself produces valid, non-degenerate embeddings on a toy graph (two wallets sharing a
transaction/IP neighborhood embed measurably closer to each other than to two wallets in an
unrelated neighborhood).

**Measured result on the full 200k-tx dataset, and the diagnosed root cause (same standard as
pass 1's 0.0-recall finding above — verified, not hidden):** pairwise recall against ground
truth stays near 0 (0.0000-0.0002 across every `min_cluster_size` / `cluster_selection_method` /
`max_cluster_size` combination tried) despite the algorithm itself being correct. Root cause,
diagnosed directly: `src/argus/graph/build.py` gives every `IP` node its literal full address,
and `src/argus/synth/transactions.py` randomizes an IP's last octet per transaction while
holding the entity's home `/24` fixed — so two transactions from the *same* entity almost never
broadcast to the *same* IP node (up to 254 distinct last-octet values), even though they share a
subnet. GraphSAGE's message passing therefore sees an essentially unique `IP` node per
transaction; the only place same-entity transactions actually meet in the graph is two hops out
at the shared `ASN` node, diluted by every other entity coincidentally sharing one of only 64
synthetic ASNs (~31 entities/ASN on average). The `/24`-level affinity the architecture doc
describes is real in the generator (`src/argus/synth/networks.py`) but not representable through
current graph topology at a granularity this encoder can exploit. **Not fixed in this phase** —
the concrete fix is a graph-schema change (an `IP` node keyed by subnet prefix, or a dedicated
`Subnet` node type, in `graph/build.py`, propagating through `features/build.py`'s IP-based
features), out of scope here; see `er/embed_cluster.py`'s module docstring for the full
diagnosis and the same note repeated at the point it matters.

Before landing on `max_cluster_size=500`, an uncapped run (`cluster_selection_method="eom"`,
sklearn's own default) was measured directly: HDBSCAN merged 49,962 of 49,983 wallets into six
clusters, corrupting `entities.parquet` far worse than pass 1's inert-but-harmless singleton
output (precision 0.0040 — i.e. actively wrong, not merely unhelpful). `max_cluster_size` (a
native `sklearn.cluster.HDBSCAN` parameter, not a hand-rolled post-hoc filter) rejects any
cluster above that size as noise instead of a real entity — a permanent safety net independent
of the diagnosis above, since no real entity in this dataset's own ground truth exceeds 308
wallets. With the guard: 2,135 merges, 9,981 wallets touched, precision 0.0068 — no longer
corrupting the majority of the population, but not a meaningful improvement over pass 1 either,
consistent with the root cause above.

**Downstream consequence:** `detectors/risk_ppr.py` now propagates over `SAME_ENTITY` edges
alongside `CO_SPEND` (see `docs/contracts.md`) — measured: seeds go from 928 rows (no
propagation, pass 1 alone) to 1,616 scored rows (propagation over 7,846 `SAME_ENTITY` edges).
The propagation machinery is correctly wired and exercised; its output quality on this dataset
inherits pass 2's diagnosed weakness above, not a separate issue in `risk_ppr.py`.

## Anomaly detection (Dev B, Phase 2)

`src/argus/models/encoder.py` implements the architecture doc's sec 4.3 shared detection core:
a **Temporal Heterogeneous GAT-v2 encoder** — `HeteroConv` of `GATv2Conv` per relation (same
per-relation-weights pattern as pass 2's GraphSAGE, but attention instead of mean/sum
aggregation), with `BROADCAST_VIA` edges additionally carrying a **Time2Vec**-encoded timestamp
feature (Kazemi & Poole 2019: one learnable linear term + learnable-frequency periodic terms —
the learnable-frequency property is what makes it Time2Vec rather than a fixed sinusoidal
positional encoding) that `GATv2Conv`'s attention conditions on via `edge_dim`. This encoder is
deliberately **shared** — sec 4.3's stated intent is that anomaly/pattern/risk signals reason
over the same learned representation; `models/anomaly.py` is its first real consumer.

`src/argus/models/anomaly.py` is a graph autoencoder: the shared encoder produces each node's
embedding, a per-node-type linear decoder reconstructs that node's own (already z-scored)
input features, and the whole thing trains end-to-end via MSE. Reconstruction error is
z-scored **within** each node type (a Wallet's error is only meaningful relative to other
Wallets — IPs/ASNs have different feature semantics entirely) and squashed to [0,1] via
sigmoid, exactly matching the architecture doc's "reconstruction error z-scored against
population" wording.

**Mechanism verified correct, independent of real-dataset quality:** a hand-built fixture — 20
near-identical "normal" wallets plus one wallet with deliberately wild-magnitude features, all
with otherwise-identical local graph structure (isolating the signal to features, not topology)
— is scored correctly: the planted outlier gets the single highest anomaly z-score among all
wallets, by a wide margin (z=4.47 vs. a normal-population range of roughly -0.24 to -0.21;
`tests/test_models_anomaly.py`). `tests/test_models_encoder.py` separately verifies the shared
encoder itself: valid non-NaN embeddings for every node type, the Time2Vec edge feature
correctly attached to both `BROADCAST_VIA` and its `ToUndirected()`-generated reverse relation
in matching per-edge order, and — the one failure mode that wouldn't show up as a shape or NaN
error — gradients actually reaching Time2Vec's learnable parameters (confirming the temporal
signal is genuinely wired into training, not a silently-disconnected branch).

**Measured result on the full 200k-tx dataset (verified, not assumed — same standard as every
other finding in this document):** AUC-ROC 0.510 / precision@50 0.30 against
`ground_truth/entities.parquet`'s illicit labels — essentially chance-level overall ranking,
with modest lift in the very top scores over the population base rate. This is a real,
diagnosable gap, not a bug: the mechanism is independently proven correct above, so the
shortfall is what the mechanism is measuring, not whether it works. Unsupervised
reconstruction-error anomaly detection answers "is this node statistically unusual" — a
different question from "is this node one of the specific labeled ransomware/darknet/mixer
entity types," and on this dataset's current `f_*` feature schema those only partially overlap:
illicit entities aren't necessarily feature-space outliers (several behave in
ordinary-looking ways by construction), while legitimate high-volume entities like exchanges
can be structural outliers without being illicit at all. **Not fixed in this phase** — a
supervised or semi-supervised variant, or features specifically discriminative of the illicit
campaign types (rather than the current general topological/temporal/cross-layer set), would be
the concrete next step; see `docs/contracts.md`'s `scores_anomaly.parquet` section for the same
diagnosis at the point it matters.

## Classical pattern detection

`src/argus/detectors/peeling.py` walks `FUNDS`/`PAYS` edges to find single-input,
single-dominant-output transaction chains with strictly decreasing value hop-over-hop.
`src/argus/detectors/coinjoin.py` flags transactions with many distinct `FUNDS` inputs and
many near-equal-value `PAYS` outputs (the same structural check ER pass 1 uses for its guard,
shared as a single source of truth).

**Measured result** (full dataset, overlap-based matching against
`ground_truth/patterns.parquet`): peeling recall 1.0000 / precision 0.9836 (60/61 detected
chains correct); CoinJoin recall 1.0000 / precision 1.0000.

The peeling detector went through a real debugging cycle worth recording here, not just the
final number: an early version's "value must decrease" check compared a hop's output against
that same transaction's own input, which is trivially true for any transaction with a positive
fee and provided no actual discrimination — precision was 0.008 (4,365 false positives from
purely coincidental baseline structure). Requiring genuine amount continuity between
consecutive hops (next hop's input must match the previous hop's output to within 2%) fixed
that, but surfaced two further bugs in how chain "start" wallets were selected (picking the
largest-amount candidate could lock onto an unrelated coincidental transaction instead of the
real chain; a wallet whose own baseline "change" output happened to be dominant was wrongly
excluded as a candidate start because it looked like someone else's target). Fixing both
brought recall to 1.0 with precision 0.28; the final fix — requiring at least 3 hops, matching
`argus.synth.patterns.PEELING_HOP_RANGE`'s actual minimum — removed all but one residual false
positive, a single coincidental 3-hop chain that is expected statistical noise at this scale.

CoinJoin detection reached perfect scores immediately because it reuses the exact structural
check ER pass 1 already uses, and CoinJoin rounds are provably the only transactions in this
dataset with ≥3 genuinely distinct input wallets.

## Pattern-similarity detection (Dev B, Phase 3)

`src/argus/detectors/pattern_sim.py` implements architecture sec 4.3's Pattern-head recall
extension: "Embedding-similarity search catches near-variants the hard-coded pattern missed."
It reuses `models/anomaly.py`'s already-trained shared encoder embeddings directly — no second
training pass — which required moving that training call from `fusion/cli.py` into
`detectors/cli.py` (this phase), so the ONE trained instance is available to both the anomaly
head and this detector in the same run; `fusion/cli.py` is now pure score-reading again, as it
was before Phase 2. For each of the two pattern types, every confirmed structural detection's
own wallets/transactions form a reference set; every other node of the same type is scored by
cosine similarity to its nearest reference embedding (not a centroid — a pattern's members can
play structurally different roles that an average would blur together).

**A single fixed similarity threshold does not work here — measured, not assumed.** An initial
flat 0.95 cosine cutoff was tested directly against the full dataset before being shipped:
142,351 matches out of ~343,360 graph nodes (41%). Root cause, diagnosed by measuring the actual
candidate-vs-reference similarity distribution per (pattern_type, node_type) pair rather than
guessing: Wallet-vs-peeling, Wallet-vs-CoinJoin, and Transaction-vs-peeling are all compressed
(median similarity already ~0.94-0.95 — the same structural-embedding-collapse root cause
diagnosed in `er/embed_cluster.py`'s Phase 1 finding: most wallets/transactions are
ordinary-looking single-hop activity, so their embeddings cluster tightly regardless of true
pattern membership), while Transaction-vs-CoinJoin is genuinely well-separated (median ~0.04,
only ~0.1% of candidates reach 0.95 — a CoinJoin transaction's many-in/many-out shape really is
a rare, distinctive local topology). One global cutoff cannot serve both regimes correctly.

**Fix:** an adaptive gate — a candidate must clear an absolute floor (0.95) **and** be a genuine
statistical outlier *within its own candidate population* (z-score > 3, computed per
(pattern_type, node_type) pair — exactly `models/anomaly.py`'s own z-scoring methodology, reused
here on similarity instead of reconstruction error), falling back to the absolute floor alone
below 30 candidates where population statistics aren't trustworthy. This is the same principle
as ER pass 2's `max_cluster_size` guard (`docs/WRITEUP.md`'s "Entity resolution pass 2"
section): don't trust one global constant to mean the same thing regardless of the population
it's being compared against.

**Measured result with the fix (full dataset):** 102 matches, **all** `PATTERN_SIM_COINJOIN` on
`Transaction`-type nodes — zero `PATTERN_SIM_PEELING` matches, zero `Wallet`-type matches. This
is the mechanism working correctly, not underperforming: it found signal exactly where the
measured similarity distribution said signal exists, and correctly found none where the
distribution said there wasn't any (the peeling/wallet embedding-collapse limitation already
diagnosed for ER pass 2 applies here too, for the same underlying reason, and is not
independently re-solved by this module). `tests/test_detectors_pattern_sim.py` verifies the
mechanism itself on hand-built cases: a genuine near-duplicate is flagged, an unrelated node is
not, and — the specific regression this measured finding requires guarding against — a large,
compressed population where most candidates clear the absolute floor does NOT flood the match
list.

## Ablation: dual-layer vs on-chain-only

`src/argus/eval/ablation.py` reruns ER/pattern/risk metrics with `BROADCAST_VIA`/`RESOLVES_TO`
edges dropped, swept across `ip_noise ∈ {0.01, 0.05, 0.20}` ("low"/"medium"/"high"). Results:
`docs/ablation_results.csv` (raw), `docs/ablation_plot.png` (plot).

**Every metric is byte-identical between the `dual_layer` and `on_chain_only` conditions, at
every noise level tested.** This was verified directly, not assumed: `grep` across
`er/union_find.py`, `detectors/peeling.py`, `detectors/coinjoin.py`, and
`detectors/risk_ppr.py` found zero references to `BROADCAST_VIA`, `RESOLVES_TO`, or
`node_features.parquet` anywhere. None of the classical detectors implemented in this repo
were ever wired to consume network-layer or cross-layer signal — they operate purely on
`FUNDS`/`PAYS`/`CO_SPEND` and raw canonical amounts. The dual-layer graph and cross-layer
features (`f_ip_reuse_count`, `f_entity_asn_entropy`, `f_geo_hop_count`/`f_geo_hop_rate`) are
correctly computed and NaN-free (Phase 2), but nothing downstream currently reads them for a
detection decision.

This is the honest, unadjusted result. Per instruction, the generator and detectors were not
modified to manufacture a difference. If dual-layer signal is to demonstrate measurable value,
it needs a detector that actually incorporates it — e.g. a peeling-chain confidence adjustment
based on IP reuse across hops, or an ER heuristic considering shared broadcast infrastructure
— which did not exist in this repo as of the Phase 1 write-up above.

**Scope note (Dev B Phases 1-3):** this ablation covers the CLASSICAL pipeline only (ER pass 1,
peeling/CoinJoin, risk propagation) — it was never extended to the ML stages, which is a
separate, deliberate scope boundary, not an oversight. Those stages *do* consume `BROADCAST_VIA`
(via Time2Vec, `models/encoder.py`) and the cross-layer `f_*` features (as GraphSAGE/GAT-v2
input) directly — see "Entity resolution pass 2", "Anomaly detection", and
"Pattern-similarity detection" above. Whether that consumption translates into *measurably
better* detections is answered by those sections' own measured results, not by this ablation.

## Score fusion

`src/argus/fusion/blend.py` combines `scores_pattern`, `scores_risk`, and — as of Dev B Phase
2 — `scores_anomaly` from a real model (`models/anomaly.py`, see above; no longer the constant
0.5 placeholder) via a logistic regression calibrated on `ground_truth/entities.parquet`'s
labels when enough labeled nodes exist (both paths — calibrated and fixed-weight fallback —
are implemented and tested). Anomaly reason codes now flow into `build_alerts`'s evidence and
rationale text too, gated by `ANOMALY_NOTABLE_Z` (z > 1.5) so an unremarkable anomaly score
doesn't produce a misleading "flagged as anomalous" sentence for a node that wasn't — every
alert's `components.anomaly` value is still always shown regardless.

Measured on the full dataset with the real anomaly model wired in: fusion universe 2,877,
1,997 alerts at the 0.6 threshold, fused scores ranging 0.257-0.9999 (median 0.661) — broadly
similar shape to the placeholder-era bimodal distribution (pattern-only detections cluster
lower, risk/seed detections cluster near-1.0), now with a small amount of continuous spread
between them contributed by the real (if weak per the section above) anomaly signal, rather
than every node sharing the exact same anomaly value. The alert threshold (0.6) is unchanged
from the placeholder era and still sits below the risk/seed cluster for the same reason
documented then — see `blend.py`'s inline comments for the full reasoning.

## Offline / install

See `docs/offline_install.md` for the offline-runtime verification procedure, what was
actually run and how, and the codebase audit for accidental network calls.

---

## Dev B's sections

- **DONE (Phase 1): ER pass 2 — GraphSAGE + HDBSCAN embedding-based clustering**
  (`er/embed_cluster.py`, `models/sage.py`) — see "Entity resolution pass 2" above.
- **DONE (Phase 2): Temporal hetero GAT-v2 encoder** (`models/encoder.py`) — a separate model
  from pass 2's GraphSAGE encoder (architecture doc sec 4.2 vs 4.3): this one feeds the
  anomaly/pattern/risk detection heads, not entity resolution. See "Anomaly detection" above.
- **DONE (Phase 2): Graph autoencoder anomaly detection** (`models/anomaly.py`) — replaces the
  former placeholder entirely (deleted); see "Anomaly detection" above and `docs/contracts.md`.
- **DONE (Phase 3): Embedding-similarity pattern detector** (`detectors/pattern_sim.py`) — reuses
  `models/anomaly.py`'s already-trained shared embeddings (moved that training into
  `detectors/cli.py` this phase so it's available here without a second training pass); see
  "Pattern-similarity detection" above and `docs/contracts.md`.
- **TODO (Dev B): Attention-based evidence extractor** (`fusion/evidence.py`)
- **TODO (Dev B): Full rationale-templating engine** (`fusion/rationale.py`) — this repo only
  ships a simple fixed-template rationale string; see `fusion/blend.py`'s docstring.
- **TODO (Dev B): Streamlit dashboard** (`dashboard/`)
