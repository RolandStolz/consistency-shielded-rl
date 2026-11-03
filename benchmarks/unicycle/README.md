# Unicycle consistency experiments

All commands are run from this directory with the virtual environment activated (`source ../../.venv/bin/activate`).

The experiment compares four setups on a grid of perturbed dynamics. In each grid config, one parameter is varied away from the baseline while the other two stay at baseline: acceleration force $F_a$, brake force $F_b$, or mass $m$. The baseline is $F_a = F_b = 2$, $m = 1$.

1. **Shield non-consistent**: the shield assumes the baseline dynamics; the environment is perturbed.
2. **Shield consistent**: the shield assumes the perturbed dynamics.
3. **Agent fine-tuned**: the agent is fine-tuned with the consistent shield on the perturbed dynamics.
4. **Agent retrained**: the agent is trained from scratch with the consistent shield on the perturbed dynamics.

## Only recreating the figure

To recreate the figure of the paper from the results we obtained, run:

```bash
python plot_evaluation.py
```

This reads `evaluation_results_paper/results.csv` and writes `evaluation_results_paper/results.pdf`.

## Running the full experiment

The launch scripts start one or more `tmux` sessions. Logging goes to tensorboard by default.

To log to [Weights & Biases](https://wandb.ai) instead:
- set the `WANDB_ENTITY` environment variable, and optionally `WANDB_PROJECT`;
- keep the `--wandb` flag in the launch scripts, or remove it there to log to tensorboard only.

### 1. Train the 5 baseline agents

`best_hps.json` contains the five best hyperparameter configurations from a prior hyperparameter sweep. Train one agent per configuration on the baseline dynamics:

```bash
bash launch_train.sh
```

The agents and the manifest `models/agents.json` are written to `models/`.

### 2. Fine-tune on the perturbed configurations (setup 3)

```bash
bash launch_finetune.sh
```

This starts one tmux session per grid config. It writes the fine-tuned agents and one manifest per config to `models/finetuned/`.

### 3. Retrain on the perturbed configurations (setup 4)

```bash
bash launch_retrain.sh
```

This starts one tmux session per grid config. It writes the retrained agents and one manifest per config to `models/retrained/`.

### 4. Run the evaluation

```bash
bash launch_eval.sh
```

This evaluates every agent for 100 episodes in each setup and grid config, and writes `results.csv`.

### 5. Plot the results

Copy `results.csv` to `evaluation_results_paper/results.csv`, or change `results_dir` in `plot_evaluation.py`. Then run:

```bash
python plot_evaluation.py
```
