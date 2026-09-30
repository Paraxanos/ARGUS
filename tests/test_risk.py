import random

import igraph as ig

from argus.detectors.risk_ppr import compute_risk_scores
from argus.graph.build import build_graph
from argus.ingest.pipeline import run_ingest
from argus.synth.config import SynthConfig
from argus.synth.corrupt import inject_corruption
from argus.synth.entities import generate_entities
from argus.synth.export import write_csv
from argus.synth.patterns import generate_illicit_patterns
from argus.synth.seeds import ILLICIT_TYPES, generate_seed_wallets
from argus.synth.transactions import generate_transactions
from argus.synth.wallets import generate_wallets


def _mock_co_spend_graph() -> ig.Graph:
    """w0-w1-w2-w3 chained by CO_SPEND, w4 isolated. Not derived from the real
    pipeline — this dataset's generator never produces a CO_SPEND edge at all
    (see risk_ppr.py's scope-limitation #2), so the ALGORITHM's distance-decay
    behavior can only be exercised against a hand-built fixture that actually
    has CO_SPEND connectivity.
    """
    g = ig.Graph(directed=True)
    names = ["w0", "w1", "w2", "w3", "w4"]
    g.add_vertices(len(names))
    g.vs["name"] = names
    g.vs["type"] = ["Wallet"] * len(names)
    edges = [(0, 1), (1, 2), (2, 3)]
    g.add_edges(edges)
    n = len(edges)
    g.es["type"] = ["CO_SPEND"] * n
    g.es["confidence"] = [0.9] * n
    g.es["amount"] = [None] * n
    g.es["timestamp"] = [None] * n
    g.es["port"] = [None] * n
    return g


def test_seed_wallet_scores_highest():
    g = _mock_co_spend_graph()
    scores = compute_risk_scores(g, ["w0"])
    by_id = {r.node_id: r for r in scores}

    assert by_id["w0"].seed_distance == 0
    assert by_id["w0"].score == max(r.score for r in scores)


def test_score_decays_with_distance():
    g = _mock_co_spend_graph()
    scores = compute_risk_scores(g, ["w0"])
    by_id = {r.node_id: r for r in scores}

    assert by_id["w1"].seed_distance == 1
    assert by_id["w2"].seed_distance == 2
    assert by_id["w3"].seed_distance == 3
    assert by_id["w0"].score > by_id["w1"].score > by_id["w2"].score > by_id["w3"].score
    assert "w4" not in by_id  # unreachable from any seed over CO_SPEND — no row emitted


def test_higher_damping_widens_propagation_reach():
    """Real-data lever (ARGUS dataset Track M benchmarking report,
    2026-09-30): a sparse real seed set (17 wallets) needs risk to reach
    farther than this dataset's own default DAMPING was tuned for — damping
    is now an exposed parameter for exactly this. Confirm the mechanism
    directly on a longer CO_SPEND chain: a farther node's score relative to
    the seed's own score should be higher under a higher damping value
    (standard personalized-PageRank property — higher damping = lower
    restart probability = score reaches farther before decaying).
    """
    g = ig.Graph(directed=True)
    names = [f"w{i}" for i in range(7)]
    g.add_vertices(len(names))
    g.vs["name"] = names
    g.vs["type"] = ["Wallet"] * len(names)
    edges = [(i, i + 1) for i in range(6)]
    g.add_edges(edges)
    n = len(edges)
    g.es["type"] = ["CO_SPEND"] * n
    g.es["confidence"] = [0.9] * n
    g.es["amount"] = [None] * n
    g.es["timestamp"] = [None] * n
    g.es["port"] = [None] * n

    default_scores = {r.node_id: r.score for r in compute_risk_scores(g, ["w0"])}
    wider_scores = {r.node_id: r.score for r in compute_risk_scores(g, ["w0"], damping=0.99)}

    default_ratio = default_scores["w6"] / default_scores["w0"]
    wider_ratio = wider_scores["w6"] / wider_scores["w0"]
    assert wider_ratio > default_ratio


def test_precision_at_50_on_full_dataset(tmp_path):
    cfg = SynthConfig(
        random_seed=31,
        num_entities=1000,
        num_wallets=15000,
        num_transactions=15000,
        ip_noise=0.05,
        heuristic_break_rate=0.1,
        mixer_fraction=0.05,
    )
    rng = random.Random(cfg.random_seed)
    entities = generate_entities(cfg, rng)
    wallets = generate_wallets(cfg, rng, entities)
    baseline = generate_transactions(cfg, rng, entities, wallets)
    pattern_rows, _ = generate_illicit_patterns(cfg, rng, entities, wallets)
    corrupted_baseline = inject_corruption(baseline, rng)
    all_rows = corrupted_baseline + pattern_rows
    seed_wallets = generate_seed_wallets(rng, entities, wallets)
    assert len(seed_wallets) >= 50

    csv_path = tmp_path / "transactions.csv"
    write_csv(all_rows, csv_path)
    df = run_ingest(csv_path, "csv", tmp_path / "rejects.log")
    g = build_graph(df)

    scores = compute_risk_scores(g, seed_wallets)
    top50 = sorted(scores, key=lambda r: r.score, reverse=True)[:50]

    entities_by_id = {e.entity_id: e for e in entities}
    wallet_to_entity = {w.wallet_id: w.entity_id for w in wallets}

    def is_illicit(wallet_id: str) -> bool:
        entity_id = wallet_to_entity.get(wallet_id)
        return entity_id is not None and entities_by_id[entity_id].entity_type in ILLICIT_TYPES

    hits = sum(1 for r in top50 if is_illicit(r.node_id))
    precision_at_50 = hits / len(top50) if top50 else 0.0
    assert precision_at_50 >= 0.6, f"precision@50 {precision_at_50} below target"
