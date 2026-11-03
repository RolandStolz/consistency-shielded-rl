#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

MANIFEST="models/agents.json"
OUT_DIR="models/finetuned"

: "${VIRTUAL_ENV:?activate your venv before running}"  # tmux shells need it activated too

# One tmux session per grid config; each fine-tunes all base agents for that config.
for tag in $(python -c "import common; print('\n'.join(map(common.config_tag, common.build_grid())))"); do
  session="finetune_${tag//./_}"  # tmux session names can't contain '.'
  tmux new-session -d -s "$session" \
    "source '$VIRTUAL_ENV/bin/activate' && python finetune.py --tag '$tag' --manifest '$MANIFEST' --out-dir '$OUT_DIR' --wandb"
  echo "$tag -> tmux session: $session"
done
