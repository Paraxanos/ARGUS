import json
import random
from pathlib import Path

import pandas as pd

from argus.ingest.pipeline import run_ingest
from argus.synth.config import SynthConfig
from argus.synth.corrupt import CORRUPTION_RATE, CORRUPTION_TYPES, inject_corruption
from argus.synth.entities import generate_entities
from argus.synth.export import write_csv, write_json, write_xml
from argus.synth.transactions import generate_transactions
from argus.synth.wallets import generate_wallets


def _small_dataset(seed: int = 7) -> list[dict]:
    cfg = SynthConfig(
        random_seed=seed,
        num_entities=20,
        num_wallets=100,
        num_transactions=1000,
        ip_noise=0.1,
        heuristic_break_rate=0.1,
        mixer_fraction=0.02,
    )
    rng = random.Random(cfg.random_seed)
    entities = generate_entities(cfg, rng)
    wallets = generate_wallets(cfg, rng, entities)
    transactions = generate_transactions(cfg, rng, entities, wallets)
    return inject_corruption(transactions, rng)


def _write_all_formats(rows: list[dict], tmp_path: Path) -> dict[str, Path]:
    paths = {
        "csv": tmp_path / "transactions.csv",
        "json": tmp_path / "transactions.json",
        "xml": tmp_path / "transactions.xml",
    }
    write_csv(rows, paths["csv"])
    write_json(rows, paths["json"])
    write_xml(rows, paths["xml"])
    return paths


def test_cross_format_equality(tmp_path):
    rows = _small_dataset()
    paths = _write_all_formats(rows, tmp_path)

    dfs = {}
    for fmt, path in paths.items():
        rejects_path = tmp_path / f"rejects_{fmt}.log"
        dfs[fmt] = run_ingest(path, fmt, rejects_path).reset_index(drop=True)

    pd.testing.assert_frame_equal(dfs["csv"], dfs["json"])
    pd.testing.assert_frame_equal(dfs["csv"], dfs["xml"])


def test_rejects_match_injected_corruption(tmp_path):
    rows = _small_dataset()
    path = tmp_path / "transactions.csv"
    write_csv(rows, path)
    rejects_path = tmp_path / "rejects.log"

    df = run_ingest(path, "csv", rejects_path)

    rejected = [json.loads(line) for line in open(rejects_path, encoding="utf-8")]
    assert len(rejected) == round(len(rows) * CORRUPTION_RATE)
    assert {r["reason"] for r in rejected} == set(CORRUPTION_TYPES)

    valid_txids = set(df["txid"])
    rejected_txids = {r["txid"] for r in rejected}
    assert valid_txids.isdisjoint(rejected_txids)
    assert len(df) + len(rejected) == len(rows)


def test_real_shaped_ip_outside_synthetic_pool_is_rejected_not_crashed(tmp_path):
    """Real-data finding (ARGUS dataset Track M benchmarking report,
    2026-09-30): a real-shaped IPv4 address (this generator's synthetic pool
    never produces one) used to crash the entire ingest run with an
    uncaught ValueError. It must now be rejected to rejects.log like any
    other bad row, and every OTHER row must still ingest normally.
    """
    # generate_transactions directly, skipping inject_corruption: isolates
    # the geoip reject path from the unrelated bad-checksum/negative-amount/
    # bad-timestamp corruption paths, which could otherwise coincidentally
    # also touch row 0 and break this test's "exactly 1 reject" assumption.
    cfg = SynthConfig(
        random_seed=7, num_entities=20, num_wallets=100, num_transactions=50,
        ip_noise=0.1, heuristic_break_rate=0.1, mixer_fraction=0.02,
    )
    rng = random.Random(cfg.random_seed)
    entities = generate_entities(cfg, rng)
    wallets = generate_wallets(cfg, rng, entities)
    rows = generate_transactions(cfg, rng, entities, wallets)
    rows[0] = dict(rows[0], src_ip="240.1.2.3")  # real-shaped, outside NETWORK_POOL
    path = tmp_path / "transactions.csv"
    write_csv(rows, path)
    rejects_path = tmp_path / "rejects.log"

    df = run_ingest(path, "csv", rejects_path)

    rejected = [json.loads(line) for line in open(rejects_path, encoding="utf-8")]
    geoip_rejects = [r for r in rejected if r["reason"] == "unresolvable_geoip"]
    assert len(geoip_rejects) == 1
    assert geoip_rejects[0]["txid"] == rows[0]["txid"]
    assert rows[0]["txid"] not in set(df["txid"])
