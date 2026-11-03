#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

CONFIG="best_hps.json"   # top-N HP configs from a prior HP sweep
OUT_DIR="models"
TOP=5
SESSION="unicycle_train"

: "${VIRTUAL_ENV:?activate your venv before running}"  # tmux shells need it activated too

tmux new-session -d -s "$SESSION" \
  "source '$VIRTUAL_ENV/bin/activate' && python train_agents.py --config '$CONFIG' --out-dir '$OUT_DIR' --top $TOP --wandb"
echo "train ($CONFIG) -> tmux session: $SESSION"
