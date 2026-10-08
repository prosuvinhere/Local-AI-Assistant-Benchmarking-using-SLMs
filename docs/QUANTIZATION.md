# Testing Quantized Models: Speed vs. Quality Trade-off

This guide shows how to benchmark GGUF quantizations (Q4 / Q5 / Q8 / FP16) of the
same model with this suite and how to read the results.

---

## 1. Why quantization matters for local SLMs

Token generation (decode) is **memory-bandwidth bound**: every generated token
requires streaming (almost) all weights from RAM to the compute units. So:

$$
\text{decode tok/s} \;\lesssim\; \frac{\text{memory bandwidth (GB/s)}}{\text{model weights (GB)}}
$$

Halving bits-per-weight roughly halves the bytes moved per token, which is why
Q4 models are often ~1.5–2× faster than Q8 and use ~half the RAM. Prefill (and
hence TTFT) is more **compute bound**, so it benefits less.

Approximate weight memory:

$$
\text{weights (GB)} \approx \frac{\text{params} \times \text{bits per weight}}{8 \times 10^9}
$$

Then add KV cache (grows linearly with `num_ctx`) and compute buffers — this is
why the suite pins `num_ctx` across models.

| GGUF type | ≈ bits/weight | Typical use | Quality impact (rule of thumb) |
|---|---|---|---|
| `Q4_0` | 4.5 | legacy, fastest | noticeable degradation |
| `Q4_K_M` | ~4.8 | **Ollama default**, best size/quality sweet spot | small |
| `Q5_K_M` | ~5.7 | quality-sensitive work on limited RAM | very small |
| `Q6_K` | ~6.6 | near-lossless | negligible |
| `Q8_0` | 8.5 | reference-grade baseline | ~lossless |
| `F16` / `BF16` | 16 | true baseline | none |

`K` = k-quants (block-wise scales with super-blocks); `_S/_M/_L` = how many
tensors (e.g. attention/output) get bumped to higher precision.

Approximate download sizes (check the tags page – they change):

| Model | Q4_K_M | Q5_K_M | Q8_0 | FP16 |
|---|---|---|---|---|
| Llama 3.2 3B | ~2.0 GB | ~2.3 GB | ~3.4 GB | ~6.4 GB |
| Phi-4 Mini 3.8B | ~2.5 GB | ~2.8 GB* | ~4.1 GB | ~7.7 GB |
| Mistral 7B | ~4.4 GB | ~5.1 GB | ~7.7 GB | ~14.5 GB |

\* not every quant is published in the Ollama library for every model; use the
Hugging Face route below.

> [!WARNING]
> On an **8 GB** Mac, keep total loaded size under ~5–5.5 GB. Mistral 7B Q8_0 and
> any 7B FP16 will swap and produce meaningless latency numbers. Llama 3.2 3B
> FP16 (~6.4 GB) is borderline — close other apps.

---

## 2. Getting quantized variants

### A. Ollama library tags (easiest)

Every library model has explicit quant tags. Browse
`https://ollama.com/library/<model>/tags`, e.g.:

```bash
# Llama 3.2 3B
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q5_K_M
ollama pull llama3.2:3b-instruct-q8_0
ollama pull llama3.2:3b-instruct-fp16

# Phi-4 Mini
ollama pull phi4-mini:3.8b-q4_K_M
ollama pull phi4-mini:3.8b-q8_0

# Mistral 7B
ollama pull mistral:7b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q5_K_M
```

Verify what you actually got (the untagged name is usually Q4_K_M):

```bash
ollama show llama3.2:3b | grep -i quantization
```

### B. GGUF files from Hugging Face

Ollama can pull any GGUF repo directly; the tag selects the quant file:

```bash
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M
ollama pull hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q5_K_M
```

Or import a local `.gguf` with a Modelfile:

```bash
cat > Modelfile <<'EOF'
FROM ./Mistral-7B-Instruct-v0.3-Q5_K_M.gguf
PARAMETER temperature 0
EOF
ollama create mistral-7b-q5km -f Modelfile
```

> [!IMPORTANT]
> A raw GGUF imported with only `FROM` may lack the correct **chat template**,
> which silently tanks quality (and would wrongly be blamed on quantization).
> Copy the `TEMPLATE` / `PARAMETER stop` lines from
> `ollama show --modelfile <library-model>` into your Modelfile.

### C. Quantize yourself from FP16

```bash
ollama pull llama3.2:3b-instruct-fp16
ollama show --modelfile llama3.2:3b-instruct-fp16 > Modelfile.fp16
ollama create llama3.2-3b-q5km --quantize q5_K_M -f Modelfile.fp16
```

---

## 3. Running a quantization sweep

Phase 3 has a `--baseline` mode that adds a relative-comparison table
(speedup, memory ratio, quality retained, output drift):

```bash
python -m bench.phase3_compare \
  --baseline llama3.2:3b-instruct-q8_0 \
  --models llama3.2:3b-instruct-q4_K_M llama3.2:3b-instruct-q5_K_M \
  --repeats 3 --allow-code-exec
```

Or use the helper script, which pulls the tags first:

```bash
./scripts/quant_sweep.sh llama3.2   # or: phi4-mini | mistral
```

Complement it with Phase 1 (latency under different prompt lengths) and Phase 2
(does lower precision hurt JSON validity / determinism?):

```bash
python -m bench.phase1_latency --model llama3.2:3b-instruct-q4_K_M --model llama3.2:3b-instruct-q8_0 --unload-after
python -m bench.phase2_determinism --model llama3.2:3b-instruct-q4_K_M --enforcement prompt
python -m bench.phase2_determinism --model llama3.2:3b-instruct-q8_0  --enforcement prompt
```

> [!TIP]
> Using **Q8_0 as the baseline** instead of FP16 is pragmatic on 8–16 GB machines:
> Q8_0 is effectively lossless and fits in memory, so you measure the quant
> effect rather than swap thrashing.

---

## 4. Reading the results

The baseline table in `report.md` contains:

| Column | Meaning | What "good" looks like |
|---|---|---|
| Decode speedup | tok/s vs baseline | Q4 ≈ 1.4–2× vs Q8 on bandwidth-bound HW |
| TTFT speedup | prefill gains | smaller than decode speedup (compute-bound) |
| Memory vs base | loaded size ratio | Q4 ≈ 55–65 % of Q8 |
| Quality retained | suite score / baseline score | ≥ 95 % is usually "free" speed |
| Output similarity | greedy-output text overlap with baseline | < 0.6 means behaviour drifts even if checks pass |

Decision rule of thumb:

1. Discard any variant that does not fit comfortably in RAM (swap = invalid run).
2. Among the rest, plot quality (y) vs decode tok/s (x). The **Pareto front**
   (listed in the executive summary) are the only rational choices.
3. Pick the fastest point on the front whose quality retention meets your bar
   (e.g. ≥ 95 %). For structured-output workloads, also require JSON first-try
   validity not to drop (lower-bit models tend to fail format rules first).

Typical pattern: Q5_K_M ≈ Q8_0 quality at ~25 % less memory; Q4_K_M is the
best speed/quality knee; below Q4 (Q3/Q2) reasoning and format adherence fall
off a cliff, especially for ≤ 4B models, which have less redundancy than 7B+.

---

## 5. Rigor checklist

- **Same everything except the quant:** temperature 0, fixed seed, same
  `num_ctx`, same prompts, same Ollama version.
- **Isolation:** Phase 3 unloads all models before each one. Don't run other
  GPU-heavy apps; plug in the laptop (battery mode throttles).
- **Repeats:** use `--repeats 3+` and look at the CI95 / CV columns — a 5 %
  speed difference inside overlapping CIs is noise.
- **Thermals:** long sweeps on fanless machines (MacBook Air) throttle. Put the
  model you care most about first, or insert pauses (`--settle-s 30`).
- **Quality suite size:** 14 prompts detects large regressions, not 1–2 %
  differences. For finer resolution add prompts to `bench/prompts.json` or run
  a standard harness (e.g. `lm-evaluation-harness` against Ollama's
  OpenAI-compatible endpoint).
- **KV-cache quantization** is a separate knob: `OLLAMA_FLASH_ATTENTION=1
  OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve` cuts KV memory roughly in half with
  long contexts. Benchmark it as its own variant.
