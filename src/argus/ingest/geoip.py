"""Offline, fully-deterministic GeoIP/ASN enrichment.

This repo's OWN dataset is entirely synthetic: src_ip values are made-up
addresses that were never assigned by any real registry, so a real
MaxMind/GeoLite2 lookup would return meaningless results for them — and would
need a licensed database file this repo can neither ship nor download at
runtime (see CLAUDE.md's offline rule). Absent ARGUS_GEOIP_ASN_MMDB /
ARGUS_GEOIP_COUNTRY_MMDB (below), this module instead re-derives
geo_country/asn the same way argus.synth.networks originally assigned them:
by looking up the IP's first two octets against that same fixed pool of
synthetic "/16" networks — a genuine re-lookup keyed on src_ip, not a copy of
whatever synth wrote into the raw export.

REAL-DATA FIX (ARGUS dataset Track M benchmarking report, 2026-09-30): real
Bitcoin-shaped data has IPv4 addresses outside this generator's synthetic
pool (e.g. 240.x.x.x) and IPv6, both of which the synthetic-only lookup above
correctly cannot resolve — but it used to CRASH the whole ingest run on the
first such address instead of rejecting just that row. When
ARGUS_GEOIP_ASN_MMDB and ARGUS_GEOIP_COUNTRY_MMDB point at real MaxMind-format
databases, `enrich()` uses them instead (via the already-pinned-but-until-now-
unused `geoip2` dependency) and handles IPv6 natively. `enrich()` now returns
None on ANY resolution failure (malformed address, address not in either
database) rather than raising — argus.ingest.pipeline rejects that row to
rejects.log with a specific reason, matching the "never silently drop, never
crash" standard the rest of ingestion already holds itself to.

configs/default.yaml's geoip_mmdb_path field remains an unused placeholder —
the two ARGUS_GEOIP_*_MMDB env vars (matching the benchmarking team's own
local workaround) are what this module actually reads, since a single path
can't address MaxMind's separate ASN/Country database files.
"""
from __future__ import annotations

import ipaddress
import os

from argus.synth.networks import NETWORK_POOL

_LOOKUP: dict[tuple[int, int], tuple[int, str]] = {
    (net.octet1, net.octet2): (net.asn, net.country) for net in NETWORK_POOL
}

ASN_MMDB_ENV = "ARGUS_GEOIP_ASN_MMDB"
COUNTRY_MMDB_ENV = "ARGUS_GEOIP_COUNTRY_MMDB"

# Lazily opened once (not per-row — enrich() runs per transaction, and
# geoip2.database.Reader opens/mmaps a file), then cached for the rest of the
# process. None = not yet checked; a (asn_reader, country_reader) tuple once
# opened; stays None forever if the env vars aren't both set.
_mmdb_readers: tuple | None = None
_mmdb_checked = False


def _get_mmdb_readers() -> tuple | None:
    global _mmdb_readers, _mmdb_checked
    if not _mmdb_checked:
        _mmdb_checked = True
        asn_path = os.environ.get(ASN_MMDB_ENV)
        country_path = os.environ.get(COUNTRY_MMDB_ENV)
        if asn_path and country_path:
            import geoip2.database

            _mmdb_readers = (geoip2.database.Reader(asn_path), geoip2.database.Reader(country_path))
    return _mmdb_readers


def enrich(src_ip: str) -> tuple[int, str] | None:
    """Returns (asn, geo_country) for src_ip, or None if it cannot be
    resolved — callers must reject the row, never crash the pipeline over one
    bad address (see module docstring's REAL-DATA FIX).
    """
    readers = _get_mmdb_readers()
    if readers is not None:
        import geoip2.errors

        asn_reader, country_reader = readers
        try:
            ipaddress.ip_address(src_ip)  # validates IPv4 AND IPv6
            asn = asn_reader.asn(src_ip).autonomous_system_number
            country = country_reader.country(src_ip).country.iso_code
        except (ValueError, geoip2.errors.AddressNotFoundError):
            return None
        if asn is None or country is None:
            return None
        return int(asn), str(country)

    try:
        octet1, octet2, _, _ = src_ip.split(".")
        key = (int(octet1), int(octet2))
    except ValueError:
        return None
    return _LOOKUP.get(key)
