"""Entity resolution pass 1: Union-Find over wallets using two structural
heuristics — common-input-ownership and change-address-return. This is
deliberately a cheap, conservative pass, expected to have HIGH PRECISION (few
false merges) rather than high recall — see this module's evaluation results
and diagnosis in the Phase 2 report for why recall on this dataset is low.
Pass 2 (embedding/HDBSCAN-based fuzzy clustering) is Dev B's, out of scope here.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import igraph as ig
import pandas as pd

# A simple structural check — deliberately NOT importing anything from a future
# detectors/coinjoin.py module, per this pass's no-circular-dependency requirement.
MIN_COINJOIN_PARTICIPANTS = 3

COMMON_INPUT_CONFIDENCE = 0.90
CHANGE_ADDRESS_CONFIDENCE = 0.70
TYPE_CHANGE_CONFIDENCE = 0.60  # weaker than address reuse: an inference, not an observed self-payment
MERGED_CLUSTER_CONF = 0.85
SINGLETON_CLUSTER_CONF = 1.0


class UnionFind:
    def __init__(self, items) -> None:
        self._parent = {item: item for item in items}

    def __contains__(self, item) -> bool:
        return item in self._parent

    def find(self, item):
        path = []
        root = item
        while self._parent[root] != root:
            path.append(root)
            root = self._parent[root]
        for node in path:
            self._parent[node] = root
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict:
        members: dict = {}
        for item in self._parent:
            members.setdefault(self.find(item), []).append(item)
        return members


def is_coinjoin_like(input_addresses: list[str], output_amounts: list[float]) -> bool:
    """Many distinct FUNDS inputs AND many near-equal-value PAYS outputs. Allows
    at most one perturbed output (heuristic_break_rate may have broken exact
    equality on one), matching the tolerance used to plant/verify these rounds.

    Also flags the general equal-output CoinJoin shape used in the literature
    (e.g. Moser & Bohme's equal-output rule; Wasabi/WabiSabi rounds with several
    denominations and per-participant change): at least MIN_COINJOIN_PARTICIPANTS
    distinct inputs, one output value repeated at least that many times, and those
    equal outputs forming at least half of all outputs. Common-input ownership is
    invalid for such transactions (inputs belong to different participants).
    """
    if len(set(input_addresses)) < MIN_COINJOIN_PARTICIPANTS:
        return False
    if len(output_amounts) < MIN_COINJOIN_PARTICIPANTS:
        return False
    rounded = [round(a, 8) for a in output_amounts]
    most_common_count = Counter(rounded).most_common(1)[0][1]
    if most_common_count >= len(rounded) - 1:
        return True
    return (
        most_common_count >= MIN_COINJOIN_PARTICIPANTS
        and 2 * most_common_count >= len(rounded)
        and len(set(input_addresses)) >= most_common_count
    )


def script_class(address: str) -> str:
    """Output script class from the address encoding alone (mainnet)."""
    if address.startswith("bc1p"):
        return "p2tr"
    if address.startswith("bc1q"):
        return "p2wpkh" if len(address) == 42 else "p2wsh"
    if address.startswith("3"):
        return "p2sh"
    if address.startswith("1"):
        return "p2pkh"
    return "other"


def type_change_output(input_addresses: list[str], output_addresses: list[str], seen_outputs: set[str]) -> str | None:
    """Address-type change heuristic (Moser & Narayanan, FC 2022, and earlier
    wallet-fingerprinting work): wallets usually send change to an address of the
    SAME script type as the inputs they spend. With exactly two outputs, all inputs
    of one type, exactly one output of that type and the other of a different type,
    the matching output is taken as change, provided it has not appeared before as
    an output (change addresses are fresh). Deliberately conservative: it never
    fires when both outputs share the inputs' type, which is the common case."""
    if len(output_addresses) != 2 or len(set(output_addresses)) != 2:
        return None
    in_classes = {script_class(a) for a in input_addresses}
    if len(in_classes) != 1:
        return None
    in_class = next(iter(in_classes))
    matches = [a for a in output_addresses if script_class(a) == in_class]
    if len(matches) != 1:
        return None
    change = matches[0]
    if change in seen_outputs or change in input_addresses:
        return None
    return change


def resolve_entities(df: pd.DataFrame, type_change: bool = False) -> tuple[UnionFind, list[tuple[str, str, float]]]:
    """Runs both heuristics over every canonical transaction row and returns the
    fitted UnionFind plus the list of (wallet_a, wallet_b, confidence) pairwise
    links discovered — one entry per heuristic firing, used both to build
    CO_SPEND edges and (transitively, via the UnionFind) to derive entities.parquet.

    type_change=True additionally applies the address-type change heuristic
    (type_change_output); rows are then processed in timestamp order so "fresh
    address" means "not seen as an output earlier".
    """
    all_wallets: set[str] = set()
    for col in ("input_addresses", "output_addresses"):
        for arr in df[col]:
            all_wallets.update(arr)

    uf = UnionFind(all_wallets)
    links: list[tuple[str, str, float]] = []
    seen_outputs: set[str] = set()
    if type_change and "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="stable")

    for row in df.itertuples(index=False):
        input_addresses = list(row.input_addresses)
        output_addresses = list(row.output_addresses)
        output_amounts = list(row.output_amounts)

        if is_coinjoin_like(input_addresses, output_amounts):
            seen_outputs.update(output_addresses)
            continue

        # Heuristic 1: common-input-ownership — wallets co-spent as inputs on
        # the same transaction are probably the same entity.
        distinct_inputs = sorted(set(input_addresses))
        anchor = distinct_inputs[0]
        for other in distinct_inputs[1:]:
            uf.union(anchor, other)
            links.append((anchor, other, COMMON_INPUT_CONFIDENCE))

        # Heuristic 2: change-address — an output returning to one of this
        # transaction's own inputs is probably change, same entity.
        input_set = set(input_addresses)
        for addr in output_addresses:
            if addr in input_set and addr != anchor:
                uf.union(anchor, addr)
                links.append((anchor, addr, CHANGE_ADDRESS_CONFIDENCE))

        # Heuristic 3 (opt-in): address-type change — see type_change_output.
        if type_change:
            change = type_change_output(input_addresses, output_addresses, seen_outputs)
            if change is not None:
                uf.union(anchor, change)
                links.append((anchor, change, TYPE_CHANGE_CONFIDENCE))
        seen_outputs.update(output_addresses)

    return uf, links


def add_wallet_link_edges(g: ig.Graph, links: list[tuple[str, str, float]], edge_type: str) -> ig.Graph:
    """Adds one wallet<->wallet edge of the given type per distinct pair in
    `links` (deduplicated — a pair may be linked by more than one source
    fact, e.g. several co-spending transactions). Shared by ER pass 1
    (CO_SPEND, via add_co_spend_edges below) and ER pass 2
    (SAME_ENTITY, via argus.er.embed_cluster.add_same_entity_edges) — same
    edge shape, different type/provenance.
    """
    name_to_index = {name: i for i, name in enumerate(g.vs["name"])}

    seen: set[tuple[str, str]] = set()
    new_edges: list[tuple[int, int]] = []
    new_confidences: list[float] = []
    for a, b, confidence in links:
        pair = tuple(sorted((a, b)))
        if pair in seen:
            continue
        seen.add(pair)
        if a not in name_to_index or b not in name_to_index:
            continue
        new_edges.append((name_to_index[a], name_to_index[b]))
        new_confidences.append(confidence)

    if not new_edges:
        return g

    start_eid = g.ecount()
    g.add_edges(new_edges)
    g.es[start_eid:]["type"] = [edge_type] * len(new_edges)
    g.es[start_eid:]["confidence"] = new_confidences
    for attr in ("amount", "timestamp", "port"):
        if attr in g.es.attribute_names():
            g.es[start_eid:][attr] = [None] * len(new_edges)
    return g


def add_co_spend_edges(g: ig.Graph, links: list[tuple[str, str, float]]) -> ig.Graph:
    """Adds one CO_SPEND edge per distinct wallet pair discovered by
    resolve_entities (deduplicated — a pair can be linked by many transactions).
    """
    return add_wallet_link_edges(g, links, "CO_SPEND")


def write_entities_parquet(uf: UnionFind, path: Path) -> None:
    rows = []
    for root, members in uf.groups().items():
        entity_id = f"resolved_{root}"
        conf = SINGLETON_CLUSTER_CONF if len(members) == 1 else MERGED_CLUSTER_CONF
        for wallet_id in members:
            rows.append(
                {
                    "wallet_id": wallet_id,
                    "entity_id": entity_id,
                    "source": "pass1",
                    "conf": conf,
                    "merge_split_log_ref": None,
                }
            )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
