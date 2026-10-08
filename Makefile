.PHONY: help install test lint run-api run-cli bench-p1 bench-p2 bench-p3 quant-sweep clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
UVICORN ?= .venv/bin/uvicorn
PYTEST ?= .venv/bin/pytest

help:
	@echo "Available targets:"
	@echo "  install      Create venv and install dependencies"
	@echo "  test         Run all 23 unit and integration tests offline"
	@echo "  run-api      Start FastAPI assistant server on :8000"
	@echo "  run-cli      Launch interactive terminal assistant"
	@echo "  bench-p1     Run Phase 1 Latency & TTFT benchmark"
	@echo "  bench-p2     Run Phase 2 Determinism & Variance benchmark"
	@echo "  bench-p3     Run Phase 3 Standardized Multi-Model evaluation"
	@echo "  quant-sweep  Run automated quantization sweep against Q8 baseline"
	@echo "  clean        Remove temporary build artifacts and test caches"

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -v

run-api:
	$(UVICORN) app.main:app --host 127.0.0.1 --port 8000 --reload

run-cli:
	$(PYTHON) assistant_cli.py --model llama3.2:3b

bench-p1:
	$(PYTHON) -m bench.phase1_latency --model llama3.2:3b --iterations 10 --cold-start

bench-p2:
	$(PYTHON) -m bench.phase2_determinism --model llama3.2:3b --runs 10 --temperatures 0.0 0.7

bench-p3:
	$(PYTHON) -m bench.phase3_compare --models llama3.2:3b phi4-mini mistral:7b --repeats 3 --allow-code-exec

quant-sweep:
	./scripts/quant_sweep.sh llama3.2

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
