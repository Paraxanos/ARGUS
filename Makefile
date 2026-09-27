VENV := .venv

ifeq ($(OS),Windows_NT)
VENV_BIN := $(VENV)/Scripts
else
VENV_BIN := $(VENV)/bin
endif

PY_CREATE := $(shell command -v py >/dev/null 2>&1 && echo "py -3.12" || echo python3)
PY := $(shell if [ -x "$(VENV_BIN)/python" ] || [ -x "$(VENV_BIN)/python.exe" ]; then echo "$(VENV_BIN)/python"; else echo python; fi)

.PHONY: env test data ingest graph er features er2 detect fusion eval pipeline

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

# data -> ingest -> graph -> er -> features -> detect -> fusion, via the
# existing dependency chain above. Named "pipeline", not "demo": there is no
# dashboard here (Dev B's, out of scope), so this is not a complete
# end-user-facing deliverable, just the full Dev-A pipeline through alerts.json.
pipeline: fusion
	@echo "pipeline complete: data/artifacts/alerts.json"
