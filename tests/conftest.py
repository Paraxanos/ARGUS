"""Imports torch before anything else in the test process.

On Windows, importing torch AFTER numpy/pandas/sklearn/igraph have already
initialized (which pytest's alphabetical test-file collection order would do
here — e.g. test_er_embed_cluster.py's sklearn import running before
test_models_sage.py's torch import) reproducibly crashes with
"OSError: [WinError 1114] A dynamic link library (DLL) initialization
routine failed ... c10.dll" — an OpenBLAS/MKL runtime-init conflict between
this environment's OpenBLAS-linked scipy/sklearn wheels and torch's bundled
runtime. Verified empirically (see the Dev B Phase 1 commit message for the
bisection), not a guess. conftest.py is pytest's own hook for exactly this:
it is imported before any test module, regardless of collection order, so
this one `import torch` here guarantees the safe load order for the whole
suite. See src/argus/er/cli.py's matching comment for the same fix in the
CLI entrypoint.
"""
import torch  # noqa: F401
