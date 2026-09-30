"""Classical peeling-chain detector.

Walks FUNDS/PAYS edges from the graph to find sequences of single-input,
single-dominant-output transactions whose dominant amount strictly decreases
hop over hop — the same structural signal argus.synth.patterns plants.

This is Dev A's classical/structural detector only. Dev B's
detectors/pattern_sim.py (embedding-similarity variant) would append MORE rows
to the same artifacts/scores_pattern.parquet later — not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import igraph as ig

# A transaction only counts as "single-dominant-output" if its largest output
# is at least this fraction of the total output value — comfortably below the
# ~80-95% dominance argus.synth.patterns actually plants (decay_rate range),
# leaving margin for heuristic_break_rate's noise output.
DOMINANCE_THRESHOLD = 0.6

# A 1- or 2-hop "chain" is statistically indistinguishable from two ordinary,
# unrelated two-output payments that happen to chain together by coincidence
# (empirically confirmed: on the full dataset, spurious detections cluster
# almost entirely at hop_count=2 with confidence <0.80, while every true
# planted chain has hop_count>=3 with confidence >=0.80 — argus.synth.patterns'
# PEELING_HOP_RANGE never plants anything shorter). Requiring >=3 hops costs
# no recall here and removes nearly all false positives.
MIN_HOP_COUNT = 3

# A real chain's next hop spends almost exactly the previous hop's dominant
# output (minus a tiny fee) — its own input amount should match the value
# carried forward to within this fraction. Without this check, two totally
# unrelated transactions with coincidentally-similar amounts get strung
# together: "dominant < this hop's own input" is trivially true for ANY
# transaction with a positive fee, so it provides no protection on its own.
CONTINUITY_TOLERANCE = 0.02


@dataclass
class PeelingChain:
    chain_id: str
    hops: list[tuple[str, str, str, float]]  # (txid, from_wallet, to_wallet, dominant_amount)
    confidence: float  # mean per-hop dominance fraction, naturally in (DOMINANCE_THRESHOLD, 1]

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def wallets(self) -> list[str]:
        return [self.hops[0][1], *(h[2] for h in self.hops)]

    @property
    def txids(self) -> list[str]:
        return [h[0] for h in self.hops]


def _build_funds_pays_index(g: ig.Graph):
    names = g.vs["name"]
    edgelist = g.get_edgelist()
    edge_types = g.es["type"]
    amounts = g.es["amount"]

    funds_in: dict[str, list[tuple[str, float]]] = {}
    pays_out: dict[str, list[tuple[str, float]]] = {}
    for (s, t), et, amt in zip(edgelist, edge_types, amounts):
        if et == "FUNDS":
            funds_in.setdefault(names[t], []).append((names[s], amt))
        elif et == "PAYS":
            pays_out.setdefault(names[s], []).append((names[t], amt))
    return funds_in, pays_out


def _qualifying_hops_by_source(
    funds_in: dict[str, list[tuple[str, float]]], pays_out: dict[str, list[tuple[str, float]]]
) -> dict[str, list[tuple[str, float, str, float, float]]]:
    """wallet -> [(txid, source_amount, dominant_wallet, dominant_amount, dominance_fraction)]
    for every transaction where that wallet is the sole (single) input.
    """
    by_source: dict[str, list[tuple[str, float, str, float, float]]] = {}
    for txid, inputs in funds_in.items():
        distinct_inputs = {w for w, _ in inputs}
        if len(distinct_inputs) != 1:
            continue  # peeling chains are single-input by construction; multi-input is CoinJoin's domain

        source_wallet, source_amount = inputs[0]
        outputs = pays_out.get(txid, [])
        if len(outputs) < 2:
            continue
        total = sum(a for _, a in outputs)
        if total <= 0:
            continue
        dominant_wallet, dominant_amount = max(outputs, key=lambda o: o[1])
        if dominant_wallet == source_wallet:
            continue  # a self-payment isn't a hop to anywhere — not a chain step
        dominance_fraction = dominant_amount / total
        if dominance_fraction < DOMINANCE_THRESHOLD:
            continue
        if dominant_amount >= source_amount:
            continue  # must actually decrease hop over hop

        by_source.setdefault(source_wallet, []).append(
            (txid, source_amount, dominant_wallet, dominant_amount, dominance_fraction)
        )
    return by_source


def _extend_from_first_hop(
    start_wallet: str,
    first: tuple[str, float, str, float, float],
    by_source: dict[str, list[tuple[str, float, str, float, float]]],
    globally_visited: set[str],
) -> tuple[list[tuple[str, str, str, float]], list[float], set[str]]:
    txid, _source_amount, dom_wallet, dom_amount, dom_fraction = first
    hops = [(txid, start_wallet, dom_wallet, dom_amount)]
    fractions = [dom_fraction]
    used = {txid}

    current_wallet, current_value = dom_wallet, dom_amount
    while True:
        candidates = [
            c for c in by_source.get(current_wallet, []) if c[0] not in globally_visited and c[0] not in used
        ]
        if not candidates:
            break
        # Amount-continuity: the next hop's own input should almost exactly
        # match the value just carried forward — "dominant < this hop's own
        # input" is trivially true for any transaction with a positive fee, so
        # continuity is the actual signal that this is the SAME pot of money,
        # not some unrelated transaction with a coincidentally similar amount.
        txid, source_amount, dom_wallet, dom_amount, dom_fraction = min(
            candidates, key=lambda c: abs(c[1] - current_value)
        )
        if abs(source_amount - current_value) > CONTINUITY_TOLERANCE * current_value:
            break
        used.add(txid)
        hops.append((txid, current_wallet, dom_wallet, dom_amount))
        fractions.append(dom_fraction)
        current_wallet, current_value = dom_wallet, dom_amount

    return hops, fractions, used


def _walk_chains(by_source: dict[str, list[tuple[str, float, str, float, float]]]) -> list[PeelingChain]:
    # Deliberately does NOT restrict to wallets that are nobody else's target:
    # a genuine chain's start can coincidentally ALSO be the target of some
    # unrelated coincidental-dominant baseline transaction elsewhere, which
    # would wrongly disqualify it. Instead: compute the best walk from EVERY
    # wallet with any qualifying outgoing tx, then resolve overlaps by
    # accepting the longest chains first — an overlapping shorter walk
    # starting partway through an already-claimed chain is discarded outright
    # rather than partially re-walked.
    candidates: list[tuple[list, list, set]] = []
    for wallet, hops_for_wallet in by_source.items():
        best: tuple[list, list, set] | None = None
        for first in hops_for_wallet:
            hops, fractions, used = _extend_from_first_hop(wallet, first, by_source, set())
            if best is None or len(hops) > len(best[0]):
                best = (hops, fractions, used)
        if best is not None and len(best[0]) >= MIN_HOP_COUNT:
            candidates.append(best)

    candidates.sort(key=lambda c: len(c[0]), reverse=True)

    globally_visited: set[str] = set()
    chains: list[PeelingChain] = []
    for hops, fractions, used in candidates:
        if used & globally_visited:
            continue
        globally_visited.update(used)
        chains.append(
            PeelingChain(
                chain_id=f"peel_det_{len(chains):04d}",
                hops=hops,
                confidence=sum(fractions) / len(fractions),
            )
        )
    return chains


# --- Behavioural chain scoring (real-data generalisation) ---------------------------------------------------
# The structural detector above finds peel-shaped chains, but on real traffic most of them are innocent (exchange
# hot-wallet withdrawals, wallets spending their own change): every chain it returns scores ~0.9, so the head cannot
# rank. Two label-free signals from the forensics literature separate laundering from service withdrawals:
#   * cash-out: criminals peel to exchange DEPOSIT addresses, which the exchange later sweeps into a multi-input
#     consolidation; an exchange's withdrawal peels go to its customers, whose addresses are not swept like that.
#     swept_fraction = share of a chain's peel outputs later spent in a non-CoinJoin transaction with at least
#     SWEEP_MIN_INPUTS inputs.
#   * source: a chain started by an extreme-activity wallet (degree above this run's SERVICE_DEGREE_QUANTILE) is
#     most likely a service's hot wallet.
# score = confidence * (0.5 + 0.5 * swept_fraction) * (SERVICE_SOURCE_FACTOR if service source else 1). The factors
# are fixed a priori (they halve or keep the structural confidence); nothing is fitted to labels.
SWEEP_MIN_INPUTS = 3
SERVICE_DEGREE_QUANTILE = 0.999
SERVICE_SOURCE_FACTOR = 0.5


def chain_behaviour_scores(g: ig.Graph, chains: list[PeelingChain]) -> dict[str, dict]:
    from argus.er.union_find import is_coinjoin_like

    funds_in, pays_out = _build_funds_pays_index(g)
    spent_in: dict[str, list[str]] = {}
    for txid, inputs in funds_in.items():
        for w, _ in inputs:
            spent_in.setdefault(w, []).append(txid)
    sweeps = {txid for txid, inputs in funds_in.items()
              if len({w for w, _ in inputs}) >= SWEEP_MIN_INPUTS
              and not is_coinjoin_like([w for w, _ in inputs], [a for _, a in pays_out.get(txid, [])])}
    degree: dict[str, int] = {}
    for txid, inputs in funds_in.items():
        for w, _ in inputs:
            degree[w] = degree.get(w, 0) + 1
    for txid, outputs in pays_out.items():
        for w, _ in outputs:
            degree[w] = degree.get(w, 0) + 1
    ordered = sorted(degree.values()) or [0]
    cutoff = ordered[min(len(ordered) - 1, int(SERVICE_DEGREE_QUANTILE * len(ordered)))]

    out: dict[str, dict] = {}
    for chain in chains:
        peels = []
        for txid, _from, dominant_to, _amt in chain.hops:
            peels += [w for w, _ in pays_out.get(txid, []) if w != dominant_to]
        swept = sum(1 for w in peels if any(t in sweeps for t in spent_in.get(w, [])))
        swept_fraction = swept / len(peels) if peels else 0.0
        service_source = degree.get(chain.hops[0][1], 0) > cutoff
        score = chain.confidence * (0.5 + 0.5 * swept_fraction) * (SERVICE_SOURCE_FACTOR if service_source else 1.0)
        out[chain.chain_id] = {"score": score, "swept_fraction": round(swept_fraction, 3), "service_source": service_source}
    return out


def detect_peeling_chains(g: ig.Graph) -> list[PeelingChain]:
    funds_in, pays_out = _build_funds_pays_index(g)
    by_source = _qualifying_hops_by_source(funds_in, pays_out)
    return _walk_chains(by_source)
