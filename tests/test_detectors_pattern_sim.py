"""argus.detectors.pattern_sim: embedding-similarity search for near-variant
peeling-chain/CoinJoin members the classical structural detectors missed.
Uses hand-built embeddings with a known-correct answer (a deliberately
near-duplicate wallet, a deliberately unrelated one) rather than trusting
similarity scores at face value — same standard as every other detector's
tests in this repo.
"""
from __future__ import annotations

import numpy as np

from argus.detectors.coinjoin import CoinjoinRound
from argus.detectors.pattern_sim import COINJOIN, PEELING, detect_pattern_similarity, pattern_sim_rows
from argus.detectors.peeling import PeelingChain


def _chain(chain_id: str, wallets: list[str], txids: list[str]) -> PeelingChain:
    hops = [(txids[i], wallets[i], wallets[i + 1], 100.0 - i) for i in range(len(txids))]
    return PeelingChain(chain_id=chain_id, hops=hops, confidence=0.9)


def _round(txid: str, inputs: list[str], outputs: list[str]) -> CoinjoinRound:
    return CoinjoinRound(txid=txid, input_wallets=inputs, output_wallets=outputs, denomination=1.0, equal_output_fraction=1.0)


def test_near_duplicate_of_a_peeling_wallet_is_flagged():
    chain = _chain("peel_det_0000", ["w_a", "w_b", "w_c", "w_d"], ["tx_a", "tx_b", "tx_c"])
    chains = [chain]
    rounds: list[CoinjoinRound] = []

    node_ids = {"Wallet": ["w_a", "w_b", "w_c", "w_d", "w_near", "w_far"], "Transaction": ["tx_a", "tx_b", "tx_c"]}
    embeddings = {
        "Wallet": np.array(
            [
                [1.0, 0.0],  # w_a (reference)
                [0.0, 1.0],  # w_b (reference)
                [1.0, 1.0],  # w_c (reference)
                [0.9, 0.1],  # w_d (reference)
                [0.999, 0.001],  # w_near: near-duplicate of w_a
                [-1.0, -1.0],  # w_far: unrelated direction
            ]
        ),
        "Transaction": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    }

    matches = detect_pattern_similarity(chains, rounds, embeddings, node_ids)
    by_id = {m.node_id: m for m in matches}

    assert "w_near" in by_id
    assert by_id["w_near"].pattern_type == PEELING
    assert by_id["w_near"].nearest_reference_id == "w_a"
    assert by_id["w_near"].similarity >= 0.95

    assert "w_far" not in by_id
    # Reference members themselves are never candidates.
    assert not {"w_a", "w_b", "w_c", "w_d"} & set(by_id)


def test_coinjoin_reference_set_is_separate_from_peeling():
    chains = [_chain("peel_det_0000", ["w_a", "w_b"], ["tx_a"])]
    rounds = [_round("tx_cj", ["w_cj1", "w_cj2", "w_cj3"], ["w_cj4", "w_cj5", "w_cj6"])]

    node_ids = {"Wallet": ["w_a", "w_b", "w_cj1", "w_cj2", "w_cj3", "w_cj4", "w_cj5", "w_cj6", "w_near_cj"], "Transaction": ["tx_a", "tx_cj"]}
    embeddings = {
        "Wallet": np.array(
            [
                [1.0, 0.0],  # w_a
                [0.0, 1.0],  # w_b
                [5.0, 5.0],  # w_cj1
                [5.1, 4.9],  # w_cj2
                [4.9, 5.1],  # w_cj3
                [5.0, 5.2],  # w_cj4
                [4.8, 5.0],  # w_cj5
                [5.2, 4.8],  # w_cj6
                [5.001, 4.999],  # w_near_cj: near-duplicate of a CoinJoin participant
            ]
        ),
        "Transaction": np.array([[1.0, 0.0], [5.0, 5.0]]),
    }

    matches = detect_pattern_similarity(chains, rounds, embeddings, node_ids)
    by_id = {m.node_id: m for m in matches}

    assert by_id["w_near_cj"].pattern_type == COINJOIN
    assert "w_a" not in by_id and "w_b" not in by_id  # too dissimilar to the CoinJoin cluster


def test_rows_are_contract_shaped_and_score_in_unit_interval():
    chains = [_chain("peel_det_0000", ["w_a", "w_b"], ["tx_a"])]
    node_ids = {"Wallet": ["w_a", "w_b", "w_near"], "Transaction": ["tx_a"]}
    embeddings = {
        "Wallet": np.array([[1.0, 0.0], [0.0, 1.0], [0.999, 0.001]]),
        "Transaction": np.array([[1.0, 0.0]]),
    }
    matches = detect_pattern_similarity(chains, [], embeddings, node_ids)
    rows = pattern_sim_rows(matches)

    assert rows
    for row in rows:
        assert set(row) == {"node_id", "score", "reason_code", "evidence_json"}
        assert 0.0 <= row["score"] <= 1.0
        assert row["reason_code"].startswith("PATTERN_SIM_PEELING=")


def test_no_matches_below_threshold_returns_empty():
    chains = [_chain("peel_det_0000", ["w_a", "w_b"], ["tx_a"])]
    node_ids = {"Wallet": ["w_a", "w_b", "w_unrelated"], "Transaction": ["tx_a"]}
    embeddings = {
        "Wallet": np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        "Transaction": np.array([[1.0, 0.0]]),
    }
    matches = detect_pattern_similarity(chains, [], embeddings, node_ids)
    assert matches == []


def test_adaptive_threshold_prevents_flooding_a_compressed_embedding_space():
    """A large candidate population where almost everyone ALREADY clears a
    flat 0.95 cosine bar (measured directly on the real dataset — see
    module docstring's THRESHOLDING section) must NOT flood the match list;
    only a genuine outlier within that population should qualify. Cosine
    similarities are constructed exactly (a unit vector (c, sqrt(1-c^2)) has
    cosine similarity exactly c to reference direction (1, 0)) rather than
    via random noise, so this test doesn't depend on a noise draw happening
    to land in the right range. Below MIN_CANDIDATES_FOR_ADAPTIVE_THRESHOLD
    this falls back to the absolute floor alone
    (test_near_duplicate_of_a_peeling_wallet_is_flagged covers that path);
    this test is deliberately large enough to exercise the z-score gate.
    """
    chains = [_chain("peel_det_0000", ["w_a", "w_b"], ["tx_a"])]

    def unit_vector_at_similarity(c: float) -> list[float]:
        return [c, (1.0 - c**2) ** 0.5]

    n = 500
    # Background: a narrow band [0.95, 0.96], all clearing the flat 0.95
    # floor but with little internal spread — mirrors the real dataset's
    # compressed distribution. One candidate is an exact duplicate of the
    # reference (similarity 1.0), far outside that narrow band.
    background_sims = np.linspace(0.95, 0.96, n)
    candidate_emb = np.array([unit_vector_at_similarity(c) for c in background_sims])
    candidate_emb[0] = [1.0, 0.0]  # exact duplicate of w_a

    wallet_ids = ["w_a", "w_b"] + [f"w_c{i}" for i in range(n)]
    embeddings = {
        "Wallet": np.vstack([[1.0, 0.0], [0.0, 1.0], candidate_emb]),
        "Transaction": np.array([[1.0, 0.0]]),
    }
    node_ids = {"Wallet": wallet_ids, "Transaction": ["tx_a"]}

    matches = detect_pattern_similarity(chains, [], embeddings, node_ids)
    matched_ids = {m.node_id for m in matches}

    assert len(matches) < n * 0.1  # nowhere near the ~41%-of-population flood this fix replaces
    assert "w_c0" in matched_ids  # the genuine near-duplicate must still survive the stricter gate
