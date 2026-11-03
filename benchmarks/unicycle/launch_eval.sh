#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

MANIFEST="models/agents.json"
SETUPS="1,2,3,4"
EPISODES=100
OUT="results.csv"
SESSION="unicycle_eval"
RETRAIN_DIR="models/retrained"
FINETUNE_DIR="models/finetuned"

: "${VIRTUAL_ENV:?activate your venv before running}"  # tmux shells need it activated too

tmux new-session -d -s "$SESSION" \
  "source '$VIRTUAL_ENV/bin/activate' && python evaluate.py --manifest '$MANIFEST' --setups '$SETUPS' --episodes $EPISODES --out '$OUT' --retrain-dir '$RETRAIN_DIR' --finetune-dir '$FINETUNE_DIR'"
echo "eval -> tmux session: $SESSION"
