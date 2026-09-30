"""argus.ingest.geoip: the synthetic-pool lookup (this repo's own data) and
the real-data fix (ARGUS dataset Track M benchmarking report, 2026-09-30) —
MaxMind-mmdb-backed lookup via ARGUS_GEOIP_ASN_MMDB/ARGUS_GEOIP_COUNTRY_MMDB,
IPv6 support, and reject-not-crash on any unresolvable address. The mmdb
path is tested against a mocked geoip2.database.Reader — no real database
file needed, matching this repo's offline-testing standard.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from argus.ingest import geoip
from argus.synth.networks import NETWORK_POOL


@pytest.fixture(autouse=True)
def _reset_mmdb_cache(monkeypatch):
    """Module-level cache (see geoip.py's own comment on why it's cached) —
    reset between tests so one test's mmdb configuration never leaks into
    the next."""
    monkeypatch.setattr(geoip, "_mmdb_readers", None)
    monkeypatch.setattr(geoip, "_mmdb_checked", False)


def test_enrich_synthetic_pool_success():
    net = NETWORK_POOL[0]
    ip = f"{net.octet1}.{net.octet2}.5.5"
    assert geoip.enrich(ip) == (net.asn, net.country)


def test_enrich_synthetic_pool_returns_none_for_real_shaped_ip_outside_pool():
    """The exact real-data failure mode the report found: a real-shaped
    IPv4 address (240.x.x.x) that was never in this generator's synthetic
    pool must be rejected, not crash the pipeline."""
    assert geoip.enrich("240.1.2.3") is None


def test_enrich_returns_none_for_malformed_address():
    assert geoip.enrich("not-an-ip") is None


class _FakeGeoIPReader:
    def __init__(self, table: dict[str, tuple[int, str]]) -> None:
        self._table = table

    def asn(self, ip: str):
        import geoip2.errors

        if ip not in self._table:
            raise geoip2.errors.AddressNotFoundError("not found")
        return SimpleNamespace(autonomous_system_number=self._table[ip][0])

    def country(self, ip: str):
        import geoip2.errors

        if ip not in self._table:
            raise geoip2.errors.AddressNotFoundError("not found")
        return SimpleNamespace(country=SimpleNamespace(iso_code=self._table[ip][1]))


def test_enrich_uses_mmdb_when_configured_including_ipv6(monkeypatch):
    table = {"8.8.8.8": (15169, "US"), "2001:4860:4860::8888": (15169, "US")}
    fake_reader = _FakeGeoIPReader(table)
    monkeypatch.setattr(geoip, "_mmdb_readers", (fake_reader, fake_reader))
    monkeypatch.setattr(geoip, "_mmdb_checked", True)

    assert geoip.enrich("8.8.8.8") == (15169, "US")
    assert geoip.enrich("2001:4860:4860::8888") == (15169, "US")


def test_enrich_mmdb_rejects_unresolvable_ip_instead_of_crashing(monkeypatch):
    fake_reader = _FakeGeoIPReader({})
    monkeypatch.setattr(geoip, "_mmdb_readers", (fake_reader, fake_reader))
    monkeypatch.setattr(geoip, "_mmdb_checked", True)

    assert geoip.enrich("240.1.2.3") is None
    assert geoip.enrich("not-an-ip") is None


def test_get_mmdb_readers_wires_env_vars_to_reader_paths(monkeypatch):
    opened_paths = []

    class _StubReader:
        def __init__(self, path):
            opened_paths.append(path)

    monkeypatch.setattr("geoip2.database.Reader", _StubReader)
    monkeypatch.setenv(geoip.ASN_MMDB_ENV, "asn.mmdb")
    monkeypatch.setenv(geoip.COUNTRY_MMDB_ENV, "country.mmdb")

    readers = geoip._get_mmdb_readers()

    assert opened_paths == ["asn.mmdb", "country.mmdb"]
    assert readers is not None


def test_get_mmdb_readers_disabled_when_env_vars_absent(monkeypatch):
    monkeypatch.delenv(geoip.ASN_MMDB_ENV, raising=False)
    monkeypatch.delenv(geoip.COUNTRY_MMDB_ENV, raising=False)

    assert geoip._get_mmdb_readers() is None
