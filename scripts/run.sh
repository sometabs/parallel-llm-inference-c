#!/usr/bin/env bash
set -u

MODEL=${MODEL:-../model_zoo/Llama-3.2-1B}
RANKS=${RANKS:-4}
THREADS=${THREADS:-2}
STEPS=${STEPS:-100}
PROMPT=${1:-"Once upon a time there were three"}

extra=""
if [ "${GREEDY:-0}" != "0" ]; then
  extra="--temp 0"
fi

# --oversubscribe is required whenever RANKS exceeds the core count.
exec mpirun --oversubscribe -np "$RANKS" \
  --mca btl_base_warn_component_unused 0 \
  ./strasgpt -m "$MODEL" -p "$PROMPT" -n "$STEPS" -t "$THREADS" $extra
