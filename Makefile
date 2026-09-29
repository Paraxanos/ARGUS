VENV := .venv

ifeq ($(OS),Windows_NT)
VENV_BIN := $(VENV)/Scripts
else
VENV_BIN := $(VENV)/bin
endif

PY_CREATE := $(shell command -v py >/dev/null 2>&1 && echo "py -3.12" || echo python3)
PY := $(shell if [ -x "$(VENV_BIN)/python" ] || [ -x "$(VENV_BIN)/python.exe" ]; then echo "$(VENV_BIN)/python"; else echo python; fi)

.PHONY: env test data ingest graph er features er2 detect fusion eval pipeline dashboard demo

env:
	$(PY_CREATE) -m venv $(VENV)
	"$(VENV_BIN)/python" -m pip install --upgrade pip
	"$(VENV_BIN)/python" -m pip install -r requirements.txt

test:
	@"$(PY)" -m pytest tests/ -v; ec=$$?; if [ $$ec -eq 5 ]; then exit 0; else exit $$ec; fi

data:
	PYTHONPATH=src "$(PY)" -m argus.synth.cli

ingest: data
	PYTHONPATH=src "$(PY)" -m argus.ingest.cli

graph: ingest
	PYTHONPATH=src "$(PY)" -m argus.graph.cli

# "resolve" named explicitly: argus.er.cli now has two commands (resolve =
# pass 1, embed = pass 2 below), so typer requires the subcommand name once
# there's more than one registered.
er: graph
	PYTHONPATH=src "$(PY)" -m argus.er.cli resolve

features: er
	PYTHONPATH=src "$(PY)" -m argus.features.cli

# ER pass 2 (GraphSAGE + HDBSCAN, Dev B): needs node_features.parquet (built
# from pass 1's entities.parquet) as its embedding input, and runs before
# detect so the risk head sees SAME_ENTITY edges alongside CO_SPEND.
er2: features
	PYTHONPATH=src "$(PY)" -m argus.er.cli embed

detect: er2
	PYTHONPATH=src "$(PY)" -m argus.detectors.cli

fusion: detect
	PYTHONPATH=src "$(PY)" -m argus.fusion.cli

eval: graph
	PYTHONPATH=src "$(PY)" -m argus.eval.cli

# data -> ingest -> graph -> er -> features -> er2 -> detect -> fusion, via
# the existing dependency chain above, ending at alerts.json.
pipeline: fusion
	@echo "pipeline complete: data/artifacts/alerts.json"

# Dev B Phase 5: the Streamlit dashboard (src/argus/dashboard/app.py) — a
# read-only viewer over the artifacts `pipeline` already produced; it never
# re-runs any pipeline stage itself. Run standalone once alerts.json exists.
#
# --server.headless + --browser.gatherUsageStats=false: verified directly (by
# reading streamlit.runtime.credentials.check_credentials, not assumed) that
# without these, `streamlit run` on a machine with no pre-existing
# ~/.streamlit/credentials.toml blocks on an interactive "enter your email"
# terminal prompt, and, once past that, periodically calls out to
# data.streamlit.io for anonymous usage telemetry — both break this repo's
# single-command/offline-runtime guarantees (docs/offline_install.md).
dashboard:
	PYTHONPATH=src "$(VENV_BIN)/python" -m streamlit run src/argus/dashboard/app.py --server.headless=true --browser.gatherUsageStats=false

# The complete end-user-facing deliverable: run the full pipeline, then open
# the dashboard on its output. `streamlit run` blocks (a live local server),
# so this is meant to be run directly, not chained into other targets.
demo: pipeline dashboard
