# Local AI Assistant & Benchmarking Suite for Small Language Models (SLMs)

[![CI Suite](https://github.com/prosuvinhere/Local-AI-Assistant-Benchmarking-using-SLMs/actions/workflows/ci.yml/badge.svg)](https://github.com/prosuvinhere/Local-AI-Assistant-Benchmarking-using-SLMs/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.6+-e92063.svg)](https://docs.pydantic.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-0.3.0+-black.svg)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


A production-grade, local, privacy-first AI assistant and scientific benchmarking suite for Small Language Models (SLMs) running on [Ollama](https://ollama.com).

Every token is evaluated and generated on-device, keeping sensitive data private with zero external cloud dependencies. Designed for repeatable AI performance engineering, dual-clock latency profiling, deterministic structured outputs with auto-repair, and quantization trade-off analysis.

---

## Architecture Overview

```
                                      ┌────────────────────────────────────────────────┐
                                      │              FastAPI Server (:8000)            │
                                      │  - POST /v1/chat & /v1/chat/stream (SSE)       │
                                      │  - POST /v1/structured (Pydantic + 1-Retry)    │
                                      │  - POST /v1/benchmark (In-service profiling)   │
                                      └───────────────────────┬────────────────────────┘
                                                              │
                                    httpx (Async REST Client with Wire Timing)
                                                              │
                                                              ▼
                                      ┌────────────────────────────────────────────────┐
                                      │             Local Ollama Daemon                │
                                      │             (http://127.0.0.1:11434)           │
                                      │  - Llama 3.2 3B / Phi-4 Mini / Mistral 7B      │
                                      │  - Grammars / GGUF quantization (Q4, Q5, Q8)   │
                                      │  - GPU Acceleration (Metal / CUDA / ROCm)      │
                                      └────────────────────────────────────────────────┘
                                                              ▲
                                                              │
 ┌────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┐
 │                                              Benchmarking CLI Suite                                               │
 │  Phase 1: bench.phase1_latency      │  Phase 2: bench.phase2_determinism  │  Phase 3: bench.phase3_compare        │
 │  - TTFT, ITL, Prefill vs Decode     │  - Strict JSON Validation           │  - 14-Prompt Standardized Suite       │
 │  - Wall clock vs Server counters    │  - 1-Shot Error-Feedback Repair     │  - Peak RSS & /api/ps VRAM Sampling   │
 │  - Student-t 95% Confidence Bounds  │  - Variance @ Temp 0.0 vs 0.7       │  - Pareto Front & Quant Sweeps        │
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Layout

```
├── app/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Settings, env parsing & loopback normalization
│   ├── metrics.py            # Latency math, percentile interpolation, Student-t CI95
│   ├── ollama_client.py      # Async HTTP client streaming wire metrics & model lifecycle
│   ├── output_schemas.py     # Pydantic schemas (extra="forbid" for strict JSON Schema)
│   ├── structured.py         # Validation layer with single repair retry and graceful fallback
│   ├── benchmarking.py       # Reusable latency profiling engine with cache-busting
│   ├── schemas.py            # API request/response models
│   └── main.py               # FastAPI application with SSE and structured endpoints
├── bench/
│   ├── __init__.py
│   ├── common.py             # System hardware capture (OS, CPU, RAM, battery) & tables
│   ├── memory.py             # Threaded 10 Hz RSS sampler + Ollama /api/ps accounting
│   ├── quality.py            # Objective test verifiers (regex, code execution, JSON fields)
│   ├── prompts.json          # 14 standardized test prompts across 6 domains
│   ├── phase1_latency.py     # Phase 1 CLI: TTFT, ITL, decode tok/s, prefill tok/s
│   ├── phase2_determinism.py # Phase 2 CLI: Temp 0.0 vs 0.7 variance & JSON retry tracking
│   └── phase3_compare.py     # Phase 3 CLI: Automated multi-model comparison & report gen
├── docs/
│   └── QUANTIZATION.md       # In-depth guide on GGUF quant levels (Q4_K_M vs Q5 vs Q8)
├── results/
│   └── example/              # Reference benchmark output artifacts
├── scripts/
│   └── quant_sweep.sh        # Automation to pull quants and run baseline comparisons
├── tests/
│   ├── test_api.py           # Integration tests for FastAPI endpoints
│   ├── test_structured.py    # Unit tests for retry mechanism and schema validation
│   └── test_metrics_quality.py # Unit tests for metric calculations and verifiers
├── .env.example              # Template environment variables
├── .github/workflows/ci.yml  # GitHub Actions CI matrix
├── Makefile                  # Developer task shortcuts
├── pyproject.toml            # Modern Python package configuration
├── requirements.txt          # Python dependencies
└── assistant_cli.py          # Interactive terminal chat with real-time token telemetry
```

---

## Quickstart

### 1. Prerequisites
Install and launch [Ollama](https://ollama.com) on your system:
```bash
ollama serve
# Pull a small model (e.g. Llama 3.2 3B)
ollama pull llama3.2:3b
```

### 2. Installation
```bash
git clone https://github.com/prosuvinhere/Local-AI-Assistant-Benchmarking-using-SLMs.git
cd Local-AI-Assistant-Benchmarking-using-SLMs


# Using Makefile
make install
# Or manually
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Offline Tests
All 23 unit and integration tests run offline without needing Ollama:
```bash
make test
# Or: pytest -v
```

### 4. Interactive Assistant CLI
Chat with your local model in your terminal with live per-token telemetry:
```bash
make run-cli
# Or: python assistant_cli.py --model llama3.2:3b
```
```
Local assistant on llama3.2:3b (100% on-device). /reset /stats /quit

you › In one sentence, why are SLMs valuable for edge devices?
ai  › Small Language Models enable low-latency, privacy-compliant intelligence directly on device without recurring cloud API costs.
      [ttft 298 ms · 12.3 tok/s · 23 tok · 2.17 s]
```

### 5. Launch FastAPI Service
```bash
make run-api
# Or: uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## Phase 1: Local Inference & Rigorous Latency Measurement

### Methodology
- **Dual-Clock Telemetry**:
  - **Client Wall Clock**: Captures HTTP roundtrip, serialization, OS scheduling, and TTFT.
  - **Server-Side Counters**: Ollama's `eval_duration` and `prompt_eval_duration` isolate pure hardware inference.
- **Prefix-Cache Busting**: Generates unique nonces to prevent Ollama KV-cache hits from artificially lowering prefill measurements.
- **Student-t Confidence Intervals**: Accurately computes 95% confidence intervals on small sample sizes ($df = n - 1$).

### Run Phase 1 Benchmark
```bash
make bench-p1
# Or: python -m bench.phase1_latency --model llama3.2:3b --iterations 10 --cold-start
```

#### Empirical Results on Apple M1 (8 GB RAM)
| Prompt Length | Input Tokens | Output Tokens | TTFT p50 | Decode Throughput | Stability (CV) | Prefill Throughput | Total Latency p50 |
|---|---|---|---|---|---|---|---|
| **Short** | 45 | 73 | **300 ms** | **12.2 ± 0.2 tok/s** | 1.6% | 157 tok/s | 6,244 ms |
| **Medium** | 210 | 128 | **1,636 ms** | **12.0 ± 1.1 tok/s** | 7.4% | 132 tok/s | 11,880 ms |
| **Long** | 1,389 | 128 | **13,507 ms** | **11.0 ± 0.8 tok/s** | 5.6% | 104 tok/s | 24,885 ms |

**Cold Start**: 3,495 ms model load | **Cold TTFT**: 3,852 ms vs. **Warm TTFT**: 300 ms.

---

## Phase 2: Structure, Determinism & Fault-Tolerant Auto-Repair

### Structure & Repair Pipeline
1. **Schema Constraints**: Pydantic models configured with `model_config = ConfigDict(extra="forbid")` output strict `"additionalProperties": false`.
2. **Strict Validation**: Evaluates model JSON via `model_validate_json(strict=True)`. String booleans (`"true"`) and string floats (`"0.9"`) are rejected.
3. **One-Shot Repair Re-prompt**: On validation failure, exact Pydantic diagnostics are packaged into a repair prompt.
4. **Graceful Fallback**: If repair fails, a structured 422 payload with raw output and error history is returned.

### Run Phase 2 Benchmark
```bash
make bench-p2
# Or: python -m bench.phase2_determinism --model llama3.2:3b --runs 10 --temperatures 0.0 0.7
```

#### Variance Analysis: Temperature 0.0 vs 0.7
| Prompt | Temp | Unique Outputs | Mean Similarity | Field Agreement (Enums / Booleans) | Field Agreement (Free Text) |
|---|---|---|---|---|---|
| `tomato` | 0.0 → 0.7 | 1 → 3 | 1.000 → 0.556 | **100% agreement** | 33% (wording drift) |
| `ticket` | 0.0 → 0.7 | 1 → 3 | 1.000 → 0.911 | **100% agreement** | 33% (summary drift) |

---

## Phase 3: Model Comparison & Quantization Trade-offs

### Standardized Evaluation Suite
Evaluates models across 14 test prompts in `bench/prompts.json` spanning 6 domains:
`factual`, `reasoning`, `coding`, `summarization`, `instruction_following`, and `structured_json`.
Every check is objectively verified using regex, AST parsers, or isolated subprocess execution.

### Run Phase 3 Comparison
```bash
make bench-p3
# Or: python -m bench.phase3_compare --models llama3.2:3b phi4-mini mistral:7b --repeats 3 --allow-code-exec
```

### Quantization Sweeps (GGUF Q4 vs Q5 vs Q8)
To evaluate the speed vs. quality trade-off of quantized variants against a Q8 baseline:
```bash
make quant-sweep
# Or: ./scripts/quant_sweep.sh llama3.2
```

See [docs/QUANTIZATION.md](docs/QUANTIZATION.md) for theory on memory bandwidth bounds and quantization levels.

---

## Contributing

Contributions are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for details on code style, offline test requirements, and the pull request process.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
