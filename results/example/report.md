# Phase 1 - Local Inference Latency Report

## Environment
| Property | Value |
|---|---|
| timestamp | 2026-10-08T23:18:03 |
| os | macOS-26.6.2-arm64-arm-64bit |
| cpu | Apple M1 |
| cpu_cores_physical | 8 |
| ram_total_gb | 8.59 |
| ram_available_gb_at_start | 1.7 |
| ollama_version | 0.33.1 |
| python | 3.9.6 |
| battery | {'percent': 22, 'power_plugged': True} |

## Methodology
- 1 warm-up iteration(s) discarded, then 5 measured iterations per (model, prompt).
- temperature=0.0, seed=42, num_predict=128, num_ctx=4096.
- Prefix-cache busting: ON (unique nonce per request).
- **TTFT** = client wall clock from request send to first streamed token (includes HTTP + prefill).
- **Decode tok/s** = Ollama `eval_count / eval_duration` (pure generation speed). Shown as mean ± 95% CI.
- **CV** = coefficient of variation of decode tok/s (run-to-run stability; <5% is a stable rig).
- **Prefill tok/s** = `prompt_eval_count / prompt_eval_duration`.
- **ITL** = median inter-token latency between streamed chunks.
- **E2E tok/s** = completion tokens / total wall latency (what a user perceives).

## Results
| Model | Prompt | In tok | Out tok | TTFT p50 (ms) | TTFT p95 (ms) | Decode tok/s | CV | Prefill tok/s | ITL p50 (ms) | Latency p50 (ms) | Latency p95 (ms) | E2E tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | short | 45 | 73 | 300 | 321 | 12.2 ± 0.2 | 1.6% | 157 | 83.6 | 6,244 | 6,606 | 11.5 |
| llama3.2:3b | medium | 210 | 128 | 1,636 | 1,643 | 12.0 ± 1.1 | 7.4% | 132 | 80.6 | 11,880 | 13,507 | 10.8 |
| llama3.2:3b | long | 1,389 | 128 | 13,507 | 14,386 | 11.0 ± 0.8 | 5.6% | 104 | 89.6 | 24,885 | 26,739 | 5.1 |

## Cold start (model unloaded first)
| Model | Prompt | Load (ms) | Cold TTFT (ms) | Warm TTFT p50 (ms) |
|---|---|---|---|---|
| llama3.2:3b | short | 3,495 | 3,852 | 300 |

## Warnings
- Less than 3 GB RAM free at start: expect swapping and inflated latencies.
