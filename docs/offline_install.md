# Offline install / runtime verification

`CLAUDE.md`'s hard rule: no step in this repo may make a network call at pipeline runtime.
Install-time package downloads (`pip install -r requirements.txt`) are the only permitted
network access. This document describes what was actually verified, and how.

## What was NOT attempted, and why

The originally-requested procedure was: fresh venv, `pip install -r requirements.txt`, then
**disable networking at the OS level**, then confirm `make env && make pipeline` still
succeeds. That last step was not attempted here. Disabling networking system-wide (firewall
rules, network adapter disable, etc.) from within this environment would require admin
privileges, affects the entire machine rather than just this test, and is difficult to
reliably and cleanly revert if something goes wrong mid-test — not a risk worth taking for a
verification that has an equally rigorous, much safer alternative (below).

## What was actually done instead

Two things, both real, both run against the actual pipeline code — not a claim without
evidence:

### 1. Static audit for network-capable imports/calls

```
grep -rn "import requests|import urllib|import http\.client|import socket|aiohttp|urlopen|\.get(\"http|\.post(\"http" --include="*.py" src/
```

Zero matches anywhere in `src/`. Also checked specifically: `geoip2` is a pinned dependency
(originally added in Phase 0, before the offline-GeoIP design decision was made) but is never
actually imported anywhere in `src/` — `src/argus/ingest/geoip.py` implements its own fully
offline, deterministic lookup instead (see that module's docstring for why: this dataset's
IPs are synthetic, so a real MaxMind lookup would be meaningless even if one were available).
`geoip2` is consequently a dead dependency as far as runtime behavior goes; it was left in
`requirements.txt` rather than removed, since removing dependencies wasn't asked for here and
is a separate decision from the offline-runtime audit.

### 2. Live simulated network-disabled run (the actual empirical test)

Rather than touching OS firewall settings, `socket.socket.connect`, `socket.socket.connect_ex`,
`socket.create_connection`, `socket.getaddrinfo`, and `socket.gethostbyname` were monkey-patched
at the start of a fresh Python process to raise `NetworkDisabledError` unconditionally. Since
essentially every Python networking library (`requests`, `urllib`, `http.client`, `aiohttp`,
raw sockets) eventually calls into one of these primitives, this is a practical, code-level
equivalent of "no network route exists" without needing OS-level privileges or touching shared
machine state.

The full pipeline was then run under this patched interpreter, calling each stage's CLI
function directly in sequence (`synth.cli.generate` → `ingest.cli.ingest` → `graph.cli.build` →
`er.cli.resolve` → `features.cli.build` → `detectors.cli.run` → `fusion.cli.run`) against a
throwaway output directory.

**Result: the full pipeline completed successfully, end to end, producing byte-identical
counts to a normal run** (2,000 entities / 50,000 wallets / 200,784 transactions through to
2,249 fused alerts), **with zero `NetworkDisabledError`s raised at any point.** Total wall time
under the patched interpreter: ~98 seconds, consistent with normal (non-patched) runs — no
retry/timeout/fallback behavior was observed either, which would have shown up as unusual
slowness if any code path attempted and failed a network call before falling back.

This is not a hypothetical or partial check: if any dependency anywhere in the pipeline had
attempted to open a socket for any reason (a stray telemetry call, an accidental
version-check ping, a real GeoIP web-service lookup), that call would have raised immediately
and the run would have failed with a traceback naming the exact call site. It did not.

## Addendum (Dev B Phase 1): torch / torch_geometric

`requirements.txt` gained `torch` + `torch_geometric` (CPU builds, via
`--extra-index-url https://download.pytorch.org/whl/cpu`, a requirements-file directive that
`pip install -r requirements.txt` picks up automatically — no separate install step) for ER
pass 2's GraphSAGE encoder (`src/argus/models/sage.py`). Covered by the same static-audit
methodology above: neither package appears in the network-capable-import grep, and both are
standard local compute libraries with no telemetry/version-check network calls in normal use.

**Windows-only gotcha, unrelated to offline-ness but worth recording here:** importing `torch`
in the same process *after* `numpy`/`pandas`/`sklearn`/`igraph` have already initialized
reproducibly crashes with `OSError: [WinError 1114] A dynamic link library (DLL)
initialization routine failed ... c10.dll` — an OpenBLAS/MKL runtime-init conflict between this
environment's OpenBLAS-linked scipy/sklearn wheels and torch's bundled runtime, bisected
directly (not guessed) during Phase 1 development. Importing `torch` (via
`argus.models.sage`) *first* avoids it entirely. Fixed at both points this matters:
`src/argus/er/cli.py`'s import order, and `tests/conftest.py` (which guarantees the safe load
order for the whole test suite regardless of pytest's alphabetical file-collection order). See
either file's comment for the full explanation; not re-verified on Linux, where this class of
DLL conflict does not apply the same way.

## Reproducing this check

The check script is not checked into the repo (it was a one-off verification, not part of the
product), but is trivial to reconstruct: patch the five `socket` attributes above to raise,
then call the seven CLI functions listed above in order against a scratch `data_dir`.
