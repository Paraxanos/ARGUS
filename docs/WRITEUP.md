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
  `detect` measured at ~4m25s-4m51s total, of which the shared-encoder training is the dominant
  cost — the classical detectors and pattern-similarity search itself are seconds, not minutes).
  Both trained models run a small number of full-batch epochs over the full ~343k-node graph,
  not mini-batched, so this is expected to scale roughly linearly with graph size same as the
  rest of the pipeline. CPU-only throughout — no GPU used or required.
- Phase 4 (attention-based evidence extraction) adds `fusion`'s cost back: measured at ~1m32s on
  the full dataset — `encoder_checkpoint.pt` (~76 MB) load + one attention-capturing forward
  pass over the full graph + evidence lookups for 2,096 alerts. Still a single-digit-minutes
  pipeline overall; see "Explainable evidence extraction" for the real performance bug caught
  and fixed before this number was this small (an earlier version measured over 9 minutes on
  just the ~15k-wallet test dataset alone, extrapolating to far worse at full scale).
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
are implemented and tested). Anomaly reason codes flow into evidence and rationale text too,
gated by `ANOMALY_NOTABLE_Z` (z > 1.5, now living in `fusion/rationale.py` — see below) so an
unremarkable anomaly score doesn't produce a misleading "flagged as anomalous" sentence for a
node that wasn't — every alert's `components.anomaly` value is still always shown regardless.

Measured on the full dataset with the real anomaly model wired in: fusion universe 2,877,
1,997 alerts at the 0.6 threshold, fused scores ranging 0.257-0.9999 (median 0.661) — broadly
similar shape to the placeholder-era bimodal distribution (pattern-only detections cluster
lower, risk/seed detections cluster near-1.0), now with a small amount of continuous spread
between them contributed by the real (if weak per the section above) anomaly signal, rather
than every node sharing the exact same anomaly value. The alert threshold (0.6) is unchanged
from the placeholder era and still sits below the risk/seed cluster for the same reason
documented then — see `blend.py`'s inline comments for the full reasoning.

## Real-data benchmarking findings and Tier-1 fix (2026-09-30)

A teammate's benchmarking report ran the full pipeline (commit `cff9f19`) against a realistic
Bitcoin-shaped dataset (Track M, ~146k transactions, ~0.2% illicit prevalence, kept external to
this repo — see the fix plan below for why). Five genuine problems were found; the diagnosis
plan treats each as a **marker our synthetic generator lacks** (multi-input transactions, IP
reuse across unrelated owners, realistic scale, benign peeling-shaped chains, a sparse fixed
seed set) and fixes/verifies each locally by reproducing the marker in isolation — a hand-built
fixture or a synthetic-scale stress test — never the hidden dataset itself. The benchmarking
team re-measures against the real data once each fix lands.

**Tier 1 (done): alert threshold floods the queue at realistic prevalence.** Confirmed by
reading, not assuming: `ALERT_THRESHOLD = 0.6` (`fusion/blend.py`) was calibrated against
synthetic data's own comment ("keeps the entire fusion universe... already a curated,
reasonably-sized watchlist") — a prevalence artifact, not a severity cutoff. At the real
dataset's ~0.2% prevalence the report measured 31,059 alerts from 40,016 scored nodes (78%).
Fix: `build_alerts()` gained an optional `top_k` cap (`DEFAULT_TOP_K = 500`), applied **after**
the threshold floor and the existing descending sort — the threshold still excludes genuinely
low-confidence nodes, `top_k` bounds investigator-facing queue size regardless of how many nodes
clear it. `fusion/cli.py` exposes `--top-k` (default `DEFAULT_TOP_K`, `0` = unbounded, matching
the pre-fix behavior exactly). Verified with a hand-built low-prevalence fixture reproducing the
marker directly (998 benign nodes clustered just above threshold, 2 planted high-scorers, no
real data involved) — `tests/test_fusion.py::test_build_alerts_top_k_caps_a_low_prevalence_flood`
confirms the flood reproduces without the cap and that `top_k` bounds it while never dropping
the true high scorers. Direct callers of `build_alerts()` (existing tests) are unaffected —
`top_k` defaults to `None`. **Not yet re-measured against the real dataset** — that's the
benchmarking team's own next step; the report's target (queue ≤ 500, precision@100 ≥ 0.5) is the
number to watch for. Note this changes `make pipeline`'s own default output on synthetic data
too (previously up to 2,109 alerts unbounded, now capped at 500 by default) — the "Measured on
the full dataset" numbers just above this section reflect a run from before this change and are
not being force-regenerated for this fix alone; `--top-k=0` reproduces the old, unbounded number
exactly if needed.

**Tier 2 (done): ER pass-2 over-merging, and a scored universe too narrow to
hold most illicit nodes.**

*ER pass-2 over-merging.* The report measured pass 1 at precision 0.960 on
real data (unlike this repo's own vacuous 0.0-recall synthetic result), yet
pass 2 still dropped overall ER precision to 0.209 by splitting pass 1's
(mostly correct) multi-wallet clusters and merging unrelated wallets that
happened to share broadcast infrastructure (a light-wallet server relaying
many owners' transactions). Three fixes, all opt-in (default off, so every
existing test/behavior — including the deliberate split-capability test,
`test_merge_and_split_and_noise_all_correct` — is untouched):
- `reconcile_with_pass1(..., protect_multiwallet_pass1_entities=True)`: any
  pass-1 entity with more than one wallet can never be split (its wallets
  keep one shared provisional id regardless of individual hdbscan label) —
  it stays fully eligible to be *merged* into a larger cross-entity group,
  since that's a separate decision. Verified with a hand-built fixture
  reproducing the exact split mechanism the existing test exercises,
  confirming protection keeps the entity whole
  (`test_protect_multiwallet_pass1_entities_never_splits`).
- `reconcile_with_pass1(..., min_merge_probability=...)`: skips a merge
  entirely unless every wallet in the triggering hdbscan cluster clears this
  confidence floor — verified with an impossible-to-satisfy floor (1.1,
  since hdbscan_prob is always <= 1.0) to avoid depending on HDBSCAN's exact
  probability output (`test_min_merge_probability_blocks_low_confidence_merges`).
- `drop_high_fanin_broadcasters()` (new function, `er/embed_cluster.py`):
  drops BROADCAST_VIA edges into any IP node whose in-degree is a
  population-relative outlier (z-score, same "don't trust one global
  constant" principle as `detectors/pattern_sim.py`'s own adaptive gate) —
  before embedding training only, never the persisted graph. Verified with a
  hand-built graph fixture: one hub IP at fan-in 40 among 30 ordinary IPs at
  fan-in 1-2 loses its edges; a too-small IP population is correctly a no-op
  (`test_drop_high_fanin_broadcasters_removes_only_the_outlier_ip`,
  `test_drop_high_fanin_broadcasters_is_noop_below_min_population`). Wired
  into `er/cli.py embed` as `--drop-broadcaster-outliers`, alongside
  `--protect-cospend-clusters` and `--min-merge-probability`.

*Scored universe too narrow.* Only 14.7% of illicit targets ever entered the
alert list — `component_table()`'s universe was pattern-or-risk hits only,
by original design (anomaly scores every node, so admitting it unconditionally
would blow the universe up to ~340k trivial rows). Fix:
`component_table(..., anomaly_admission_threshold=...)` additionally admits
any node whose anomaly score alone exceeds the cutoff — `None` (default)
preserves the exact pre-existing universe. Verified with a hand-built
fixture: 3 "quiet illicit" nodes (elevated anomaly score, no pattern/risk
hit) among 500 all-normal nodes are missed by default and captured once
admitted, without the all-normal population flooding in
(`test_component_table_anomaly_admission_widens_universe_without_flooding`).
Separately, `detectors/risk_ppr.compute_risk_scores()` now exposes `damping`
(previously a hardcoded module constant) — higher damping = lower restart
probability = propagation reaches farther from a sparse real seed set (17
wallets). Verified directly: on a 7-node CO_SPEND chain, a farther node's
score relative to the seed's own score is measurably higher under a higher
damping value, the expected personalized-PageRank property
(`test_higher_damping_widens_propagation_reach`). Both wired in as CLI flags
(`fusion/cli.py --anomaly-admission-threshold`, `detectors/cli.py
--risk-damping`), both off/unchanged by default.

**Not yet re-measured against the real dataset** for either fix — that's the
benchmarking team's own next step. Targets to watch: pass-2 F1 >= pass-1 F1
(0.960 -> currently 0.209), and >= 50% of illicit targets in the scored
universe (currently 14.7%).

**Tier 3 (diagnosis only — two hypotheses tested and REJECTED, not
confirmed): the anomaly head's real-data flatness and the ER pass-2 loss
blowup are NOT pure scale or degree-skew artifacts.** Both `models/sage.py`
(now tracks per-epoch loss via `TrainingResult.losses`, already existed) and
`models/anomaly.py` (gained the same `AnomalyResult.losses` field this
phase) can now answer "did training converge" directly on ANY future run,
ours or the benchmarking team's — that instrumentation is the one concrete
code change from this diagnosis. Two hypotheses for the report's findings
were tested empirically and did not hold up, reported honestly rather than
forcing a fix that isn't verified to help:

- *Pure scale.* Generating our own synthetic data 4x larger (60k vs 15k
  wallets, same generator, same fixed 30-epoch budget) barely moved either
  model: GraphSAGE final loss 1.44 -> 1.48 (report's real-data finding was
  1.79 -> 21.6, an order of magnitude), and the anomaly score distribution
  was essentially identical (mean 0.485/0.486, std 0.107/0.108) at both
  scales. If under-convergence from a fixed epoch budget were the cause, a
  4x node-count jump should have shown meaningfully worse numbers — it did
  not.
- *Degree skew (the same shared-broadcaster-IP marker Tier 2's
  `drop_high_fanin_broadcasters` targets).* Rewiring 30% of one run's
  BROADCAST_VIA edges onto a single injected hub IP (in-degree 4,590) barely
  moved GraphSAGE's loss either (1.44 baseline vs 1.40 hub-skewed vs 1.42
  with the Tier-2 fix applied) — all three within noise of each other. A
  single injected hub does not reproduce the report's loss blowup.

Both models were mildly still-descending at epoch 30 in every condition
tested (5-17% of that point's loss in the last 5 epochs) — a real, if minor,
pre-existing inefficiency independent of either hypothesis, not something
newly caused by scale or skew.

**Conclusion:** the report's loss blowup and anomaly flatness are most
likely driven by real transaction *content* (feature-value complexity,
real-world amount/timing distributions our synthetic generator doesn't
reproduce) rather than graph size or topology shape — genuinely not
diagnosable further without the hidden dataset. **Handoff ask for the
benchmarking team's next re-run** (aggregate stats only, no raw data needed):
(1) the per-epoch GraphSAGE loss curve, now directly available via
`TrainingResult.losses` — still descending at epoch 30 means "train longer,"
a plateaued high value means "the features themselves are hard to
reconstruct, not a training-time problem"; (2) the raw per-node `z_score`
(not the sigmoid-squashed `score`) from `scores_anomaly.parquet`, split by
the real ground truth's illicit/licit label — tells us whether the anomaly
head has ANY separable signal on real data at all, independent of z-score's
[0,1] compression.

**Tier 4 (done): GeoIP real-address support, and a Windows HDBSCAN crash
guard.**

*GeoIP.* `ingest/geoip.py`'s synthetic-pool lookup (keyed on the generator's
own fixed `/16` pool) used to crash the entire ingest run the moment it saw
a real-shaped address outside that pool (e.g. `240.x.x.x`) or IPv6 — a real
bug, not a hypothetical: this repo's OWN generator never produces those
shapes, so the crash path was simply never exercised before real data hit
it. Fixed: `enrich()` now returns `None` on any unresolvable address instead
of raising, and `ingest/pipeline.py` rejects that row to `rejects.log`
(`reason: "unresolvable_geoip"`) like any other bad row — never a crash.
When `ARGUS_GEOIP_ASN_MMDB`/`ARGUS_GEOIP_COUNTRY_MMDB` point at real
MaxMind-format databases, `enrich()` uses them via the already-pinned (until
now genuinely unused) `geoip2` dependency instead of the synthetic pool,
handling IPv6 natively; absent those two env vars this path is never
reached, `make pipeline`'s own default behavior is unaffected. Verified
against a mocked `geoip2.database.Reader` (no real database file needed,
matching this repo's offline-testing standard) plus an ingest-level
integration test confirming one bad `src_ip` is rejected while every other
row in the same file still ingests normally
(`tests/test_ingest_geoip.py`, `tests/test_ingest.py`).

*HDBSCAN Windows stack overflow.* The report found `sklearn.cluster.HDBSCAN
.fit`'s recursive single-linkage-tree construction overflows Windows'
default 1 MB thread stack at real-world wallet counts (150k+) — a genuine
platform limitation (Linux's default 8 MB stack doesn't hit this), not a
bug in this repo's own code, crashing with no Python traceback, exit 127.
**Reproduced exactly, with no real data at all:** `run_hdbscan()` on 150,000
purely synthetic random embeddings (16-dim, no graph or GraphSAGE training —
the crash is inside sklearn's own C-level recursion, which only cares about
point count and dimensionality) on this same Windows machine terminated
with **exit code 127 and no Python traceback**, matching the report's
observed symptom precisely. It took far longer than the report's own
observed ~4 minutes before crashing — plausibly because unclustered random
Gaussian points are a harder case for the single-linkage tree than real,
naturally-clustering embeddings, but the failure mode itself matches exactly.

Fixed: `argus.er.embed_cluster.run_with_big_stack()` runs a given function
on a thread with a 128 MB stack (vs. Windows' 1 MB default), always on (not
an opt-in guard like Tier 2/3's — it changes nothing about the result, only
where the C-level recursion runs), wrapping `er/cli.py embed`'s
`run_hdbscan` call. Verified two ways: (1) unit-level — returns the wrapped
function's value unchanged, propagates exceptions correctly, and actually
requests the larger stack size before restoring the previous one afterward
(`tests/test_er_embed_cluster.py`); (2) **the exact same 150,000-point
reproduction that crashed with exit 127 above, rerun through
`run_with_big_stack`, now completes successfully — 687.2s, exit code 0,
150,000 rows returned, 9,136 non-noise** — confirming the fix resolves the
specific crash it was built for, not just a plausible-looking mechanism.

**Not ours to build:** exporting more hard negatives and seeds into the
ground truth (`tools/export_to_argus_prototype.py`) lives in the dataset
repo, not here.

## Explainable evidence extraction (Dev B, Phase 4)

Architecture doc sec 4.4: "extract an evidence subgraph: the k-hop neighborhood restricted to
the top attention-weighted edges." Two new modules, both reusing the shared encoder rather than
training anything new:

`src/argus/fusion/evidence.py` adds `forward_with_attention` to `models/encoder.py`'s shared
GAT-v2 encoder — a second, side call to each relation's underlying `GATv2Conv` with
`return_attention_weights=True` (PyG's `HeteroConv` wrapper does not expose this itself),
verified directly to reproduce `HeteroConv`'s own per-relation output bit-for-bit. Since the
encoder is 2 layers, layer 2's attention explains a node's 1-hop neighbors' contribution to its
final embedding, and layer 1's explains those neighbors' own 2-hop contributions — together, the
"k-hop neighborhood restricted to top attention-weighted edges" the architecture doc asks for.
`detectors/cli.py` persists the trained encoder + its input graph (`encoder_checkpoint.pt`,
~76 MB on the full dataset) right after anomaly training; `fusion/cli.py` reloads it, so
evidence extraction needs no third training pass and only runs over the final (~2,000-alert)
list, never the full ~343k-node population.

`src/argus/fusion/rationale.py` replaces `blend.py`'s former inline `_rationale`/
`_extract_evidence` if/elif chain with a registry: each reason_code prefix maps to an
(evidence_extractor, clause_renderer) pair, extended with a `PATTERN_SIM_*` entry (Phase 3's
detector had none before this phase) and an optional attention-neighbor clause naming the
single highest-weight evidence-subgraph neighbor when `fusion/evidence.py` supplies one.
`register_template()` is the extensibility point for a future detector's reason_code.

**A real performance bug caught before shipping, same standard as every other finding in this
document:** the first working version called `encoder.forward_with_attention` — a full pass
over the *entire* graph — once **per alert**. Measured directly: this turned `fusion`'s runtime
from ~10 seconds (pure score-reading, pre-Phase-4) into over 9 minutes for the ~15k-wallet test
dataset alone, and would have been far worse at the full 343k-node scale. Root cause: the
forward pass is identical for every target node, only the neighbor lookup varies, so computing
it fresh per node was pure waste. Fixed by splitting the module into
`build_attention_context` (the expensive part, called ONCE per run) and
`extract_attention_evidence` (a cheap dict lookup against that context, called once per alert).
Measured after the fix: `fusion/cli.py` on the full 200k-tx dataset took ~1m32s (encoder
checkpoint load + one attention-forward-pass + evidence for 2,096 alerts) — a real, honest cost
of this design (not free), but two orders of magnitude better than the bug it replaced, and
`tests/test_fusion_evidence.py::test_one_context_serves_multiple_targets_without_recomputation`
guards against this regressing silently again.

**Measured result (full 200k-tx dataset):** 2,095 of 2,096 alerts (99.95%) now carry a concrete
attention clause naming a specific neighbor node — e.g. `"the model's own attention weighted
104.188.132.98 most heavily among 24 evidence-subgraph neighbors (weight 1.00)"`. This is a
genuine capability increase, most valuable for `ANOMALY_ZSCORE`-only alerts specifically:
`fusion/rationale.py`'s anomaly evidence extractor deliberately returns no structural evidence
of its own (see its docstring — a reconstruction-error anomaly is a property of the node
itself), so before this phase those alerts had nothing beyond the flagged node and a z-score
number. Mechanism correctness (not exact attention *values*, which are learned and not
independently hand-verifiable) is tested directly: hop-1 neighbors are real graph neighbors of
the target and never leak into an unrelated node's neighborhood, hop-2 neighbors always connect
through a genuine hop-1 node, and no node occupies more than one ranking slot under two
different relation labels (`tests/test_fusion_evidence.py`, `tests/test_fusion_rationale.py`).

## Dashboard (Dev B, Phase 5)

Architecture doc sec 4.5: "ranked alert table... + pyvis/streamlit-agraph link-analysis graph
view that highlights the evidence subgraph on click." `src/argus/dashboard/app.py` is a single
Streamlit file, read-only over `data/artifacts/alerts.json` and `node_features.parquet` — it
never re-runs any pipeline stage. `make dashboard` runs it standalone; `make demo` runs the full
pipeline first, matching every other stage's offline/single-process design.

**Offline verification, not assumed — same standard as `docs/offline_install.md`:** pyvis's
documented `cdn_resources="local"` default was tested directly (generate HTML, search for
`https?://`) and still emits `cdnjs.cloudflare.com` (vis-network) and `cdn.jsdelivr.net`
(Bootstrap) URLs despite the name — confirmed by reading the installed pyvis version's own
`templates/template.html`: the `local` branch only localizes `tom-select`, not vis-network
itself. Switching to `cdn_resources="in_line"` correctly inlines vis-network, but the same
template still hardcodes the two Bootstrap CDN tags **unconditionally**, outside every
`{% if cdn_resources==... %}` branch — a real template limitation, not a misconfiguration here.
Bootstrap there is decorative page chrome around the graph container, not load-bearing for
vis-network, so `_strip_external_cdn_links` regex-strips those two tags post-generation and
asserts no `http(s)://` reference survives (`tests/test_dashboard.py`), rather than trusting
either parameter name at face value.

**Verified against the real pipeline output**, since no browser-automation harness is available
in this environment: `main()` invoked directly as a plain Python call against the full-scale
dataset (2,096 alerts, 343,360 node-type entries) raised no exception, and `curl` against the
running `streamlit run` server returned HTTP 200. Every non-Streamlit-specific helper
(`load_alerts`, `load_node_types`, `_alerts_dataframe`, `_strip_external_cdn_links`,
`render_evidence_graph`) has a direct unit test in `tests/test_dashboard.py`.

**A second real bug caught before shipping, found by reading Streamlit's own source rather than
assuming `streamlit run` is side-effect-free:** `streamlit.runtime.credentials.check_credentials`
(called on every `streamlit run`) calls `Credentials.get_current()._check_activated()` whenever
`server.headless` is `False` (its default on Windows, and on Linux with a `DISPLAY` set) —
which, on any machine with no pre-existing `~/.streamlit/credentials.toml`, drops into an
interactive `click.prompt(...)` asking for an email address and blocks on stdin. Separately,
`browser.gatherUsageStats` defaults to `True` and gates both a startup `GET
https://data.streamlit.io/metrics.json` call and ongoing per-session telemetry
(`runtime/metrics_util.py`). Both would silently break this repo's single-command/offline-runtime
guarantees on a genuinely fresh machine — this dev machine's own pre-existing
`~/.streamlit/credentials.toml` (unrelated prior local Streamlit use) masked the first issue
during initial testing, which is exactly why it needed tracing through source rather than trusting
one successful local run. Fixed by adding `--server.headless=true
--browser.gatherUsageStats=false` to the Makefile's `dashboard` target — both flags verified
directly: launched with a scratch, empty fake-`HOME` (no `.streamlit` directory at all,
simulating a fresh machine) and `timeout`, the server answered HTTP 200 immediately with no
stdin block, and no `.streamlit` directory was created at all afterward (proving neither the
activation prompt nor the telemetry file write fired).

Two Streamlit deprecation warnings were found and triaged: `st.dataframe(...,
use_container_width=True)` was renamed to `width="stretch"`. `st.components.v1.html(...)`'s
suggested replacement, `st.iframe(...)`, only accepts a `src: str | Path` (confirmed via
`inspect.signature`) — not raw HTML content — so it is not a valid drop-in replacement here;
`components.html()` was kept deliberately (still fully functional, only a warning), documented
inline in `app.py`.

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
- **DONE (Phase 4): Attention-based evidence extractor** (`fusion/evidence.py`) and **full
  rationale-templating engine** (`fusion/rationale.py`, replacing `blend.py`'s former inline
  version entirely) — see "Explainable evidence extraction" above and `docs/contracts.md`.
- **DONE (Phase 5): Streamlit dashboard** (`dashboard/app.py`) — ranked alert table + pyvis
  evidence-subgraph graph view; see "Dashboard" above and `docs/contracts.md`.

## Real-data generalisation (branch `improve/generalization`, 2026-09-30)

**Why.** On the ARGUS dataset, `fusion/blend.py`'s calibrated blend was fitted on
`ground_truth/entities.parquet` of the same run it scored — in-sample. On real, unlabelled data it falls back to
`FALLBACK_WEIGHTS` + `ALERT_THRESHOLD=0.6`, which raised 2–4 alerts, all wrong, on every dataset window tested.
The changes below make every stage work without labels, and every choice was made on held-out dev data.

**Protocol.** Two dev windows (ARGUS dataset, 23 Sep 2026 00–06 and 06–12 UTC) for all design choices; a sealed test
(24 Sep 00–06) run once at the end; then a fresh window (24 Sep 12–18) that nobody had looked at, whose injected
crime flows share nothing with any earlier set. Seeds are excluded from every hit count.

**Changes** (defaults in brackets; every new behaviour has a CLI flag):
- `fusion`: label-free percentile-rank blend [`--fusion-mode rank`, weights risk 0.55 / pattern 0.45 / anomaly 0];
  a blend fitted on a dev run can be saved (`--save-fusion-model`) and applied elsewhere (`--fusion-mode model`);
  one alert per resolved entity [`--dedupe-entities`]; `in-sample` keeps the old behaviour for comparison.
- `detectors/risk_ppr`: money-flow taint in time order [`--risk-mode both`] — ordinary transactions carry their
  owner's full taint (common-input ownership), CoinJoin-shaped ones split it by value (haircut), extreme-degree
  service wallets absorb taint without passing it on (cutoff = this run's 99.9th-percentile degree).
- `detectors/peeling` + `scores`: chains scored by behaviour [`--pattern-scoring behaviour`] — share of peels later
  swept into a consolidation (cash-out to an exchange) and whether the source is a service hot wallet; CoinJoin
  participants scored at half [`--coinjoin-participant-factor 0.5`] (mixing is a signal, not proof).
- `models/iforest`: label-free isolation-forest ensemble [`--anomaly-scorer iforest`, opt-in], stable across
  seeds (rank correlation 0.97–0.997). Both anomaly scorers ranked illicit nodes BELOW licit ones on both dev
  windows (AUC 0.21–0.37), so anomaly has weight 0 in the ranking and is shown as evidence only.
- `er`: general equal-output CoinJoin rule in pass 1; address-type change heuristic [`--type-change`, off:
  +0.02 F1 but −5 pts precision on dev]; pass 2 can be skipped [`er embed --skip`] — with every guard on it still
  lowered F1 on dev (0.202 -> 0.142), so the real-data runs below skip it. Ground truth is optional everywhere.

**Results** (`--skip` for er2, `--anomaly-scorer iforest`, all other defaults; precision = share of the top-k alerts
that are illicit wallets or transactions):

| Window | Code | P@10 | P@50 | P@100 | AP | ER precision / F1 |
|---|---|---|---|---|---|---|
| Sealed test | shipped, no labels | 0 | 0 | 0 | 0 | 0.21 / 0.076 |
| Sealed test | shipped, fitted on the test's own labels | 0.70 | 0.62 | 0.53 | 0.56 | 0.21 / 0.076 |
| Sealed test | this branch, no labels | **1.00** | **0.80** | **0.70** | **0.76** | **0.96 / 0.107** |
| Fresh window | this branch, no labels | **1.00** | **0.90** | 0.51 | **0.97** | **0.97 / 0.119** |

**Still open.** Only ~10% of illicit activity reaches the queue: risk propagation needs seeds (9–17 per window,
covering about a third of campaigns), and neither anomaly scorer carries signal on realistic data. Finding
unseeded crime needs a better unsupervised signal; that is the next piece of work.
