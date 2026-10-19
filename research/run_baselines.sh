#!/usr/bin/env bash
# Frozen metric snapshot for R2 (NCERT-Ground) and R10 (systems numbers).
#
#   bash research/run_baselines.sh
#
# Prerequisites (all on Sol, in the storytutor env):
#   1. stack up:   bash scripts/sol.sh up        # Ollama serving for --report below
#   2. repo pull:  git pull
#
# Every number quoted in a paper or a deck comes from a dated folder produced
# by this script -- never from an ad-hoc rerun (research/README.md ground
# rule 1). Outputs land in research/reports/<YYYY-MM-DD>/:
#   retrieval_eval.json   recall@k / MRR by language  (R2 retrieval slice)
#   rag_eval.json         citation, groundedness, refusal rates (R2 answer slice)
#   chat_test_report.json 53 behavioural questions vs the live server
#
# Run time: ~10-15 min (rag_eval makes real LLM calls).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
cd "$PROJ"

STAMP="$(date +%Y-%m-%d)"
OUT="research/reports/$STAMP"
mkdir -p "$OUT"

echo "== StoryTutor baseline snapshot -> $OUT"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | sed 's/^/  GPU /' || echo "  (no GPU visible)"

echo
echo "[1/3] retrieval eval (golden set, verified subset, rerank as configured)"
python eval/run_retrieval_eval.py --report "$OUT/retrieval_eval.json"

echo
echo "[2/3] RAG eval: citations, groundedness, refusal, language match"
python eval/run_rag_eval.py --limit 60 --report "$OUT/rag_eval.json"

echo
echo "[3/3] 53 behavioural chat tests against the live server"
python eval/run_chat_tests.py --url "${STORYTUTOR_URL:-http://127.0.0.1:5050}" \
    --report "$OUT/chat_test_report.json"

echo
echo "== snapshot complete:"
ls -la "$OUT"
echo "   freeze this folder: it is the baseline every later experiment compares to."
