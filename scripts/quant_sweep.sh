#!/usr/bin/env bash
# Quantization sweep: pull several quants of one model family and compare them
# against a Q8_0 baseline with Phase 3.
#   ./scripts/quant_sweep.sh llama3.2 [extra phase3 args...]
set -euo pipefail
cd "$(dirname "$0")/.."

family="${1:-llama3.2}"
shift || true

case "$family" in
  llama3.2)  baseline="llama3.2:3b-instruct-q8_0"
             variants=("llama3.2:3b-instruct-q4_K_M" "llama3.2:3b-instruct-q5_K_M") ;;
  phi4-mini) baseline="phi4-mini:3.8b-q8_0"
             variants=("phi4-mini:3.8b-q4_K_M") ;;
  mistral)   baseline="mistral:7b-instruct-q5_K_M"   # Q8_0 (7.7 GB) won't fit an 8 GB machine
             variants=("mistral:7b-instruct-q4_K_M") ;;
  *) echo "unknown family: $family (llama3.2 | phi4-mini | mistral)"; exit 1 ;;
esac

for tag in "$baseline" "${variants[@]}"; do
  echo ">> ollama pull $tag"
  ollama pull "$tag"
done

PY="${PYTHON:-.venv/bin/python}"
"$PY" -m bench.phase3_compare --baseline "$baseline" --models "${variants[@]}" --repeats 3 "$@"
