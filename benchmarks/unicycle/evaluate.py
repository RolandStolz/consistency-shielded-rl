"""
Evaluate the V-SUM consistency experiments and write a results CSV.

For each grid config (one dynamics parameter perturbed off baseline at a time) we evaluate each
of the trained base agents over N episodes under four setups:
  1. shield NOT adapted  -- shield assumes baseline dynamics, env is perturbed (consistency violated)
  2. shield adapted       -- shield assumes the perturbed dynamics
  3. agent fine-tuned     -- agent fine-tuned with the adapted shield (200k steps), eval'd with it
  4. agent re-trained     -- agent trained from scratch with the adapted shield, eval'd with it

Metrics: goal-reaching rate, collision rate (fraction of the N episodes), intervention rate
(fraction of steps the shield intervened), and mean episode return.
wandb-free; writes a CSV.

Setup 3 reads fine-tuned models produced separately by finetune.py / launch_finetune.sh
(per-config manifests `<finetune-dir>/<tag>.json`); setup 4 reads re-trained models from retrain.py
(`<retrain-dir>/<tag>.json`). Neither setup trains here.

Usage:
    python evaluate.py --manifest models/agents.json --setups 1,2 --out results.csv  # eval-only
    python evaluate.py --manifest models/agents.json --setups 1,2,3 --out results.csv # needs fine-tunes
"""

import argparse
import csv
import json
import os
from pathlib import Path

from acorl.envs.seeker.unicycle import SeekerUnicycleConfig
from acorl.tianshou.logging.info_metrics_collect import InfoMetricsCollectStats
from acorl.tianshou.utils import load_collector_and_algorithm_from_experiment
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.data import Collector
from tianshou.highlevel.env import EnvMode, VectorEnvType

from common import (
    BASELINE,
    DYNAMICS_KEYS,
    build_env_factory,
    build_grid,
    config_tag,
    make_env_config,
)

RATE_KEYS = ('goal_reached', 'collision')
STEP_MEAN_KEYS = ('required_intervention',)  # fraction of steps the shield intervened


def evaluate_policy(
    algorithm: Algorithm,
    env_config: SeekerUnicycleConfig,
    shield_config: SeekerUnicycleConfig,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Roll out `algorithm` for `n_episodes` and return the CSV metrics."""
    InfoMetricsCollectStats.RATE_KEYS = RATE_KEYS
    InfoMetricsCollectStats.STEP_MEAN_KEYS = STEP_MEAN_KEYS
    venv = build_env_factory(env_config, shield_config, VectorEnvType.DUMMY).create_venv(
        1, EnvMode.TEST, seed=seed
    )
    collector = Collector(algorithm.policy, venv, collect_stats_class=InfoMetricsCollectStats)
    collector.reset()
    stats = collector.collect(n_episode=n_episodes)
    stats.refresh_all_sequence_stats()  # ensure info metrics are populated from the accumulators
    return {
        'goal_reached_rate': stats.info_rates.get('goal_reached', 0.0),
        'collision_rate': stats.info_rates.get('collision', 0.0),
        'intervention_rate': stats.info_step_means.get('required_intervention', 0.0),
        'return_mean': float(stats.returns.mean()),
    }


def _load_manifest(model_dir: str, tag: str, setup: int, script: str) -> dict[int, str]:
    """Read the per-config manifest `<model_dir>/<tag>.json` (idx -> dir)."""
    path = os.path.join(model_dir, f'{tag}.json')
    if not os.path.exists(path):
        raise SystemExit(
            f'missing manifest {path!r} for setup {setup}. '
            f'Run {script} --tag {tag} first, or drop setup {setup}.'
        )
    with open(path) as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--manifest', default='models/agents.json', help='agents.json from train_agents.py'
    )
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--setups', default='1,2,3,4', help='comma-separated subset of {1,2,3,4}')
    parser.add_argument(
        '--finetune-dir', default='models/finetuned', help='per-config manifests for setup 3'
    )
    parser.add_argument(
        '--retrain-dir', default='models/retrained', help='per-config manifests for setup 4'
    )
    parser.add_argument('--out', default='results.csv')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    setups = {int(s) for s in args.setups.split(',')}
    with open(args.manifest) as fh:
        agents = json.load(fh)

    # Load each base agent's policy once.
    base_algorithms: dict[int, Algorithm] = {}
    for a in agents:
        pth = Path(a['persist_directory'])
        if not pth.exists():
            raise ValueError(f'Path {pth} does not exist.')
        _, algorithm = load_collector_and_algorithm_from_experiment(pth)

        base_algorithms[a['idx']] = algorithm

    baseline_shield = make_env_config(**BASELINE)

    rows = []
    for cfg in build_grid():
        dyn = {k: cfg[k] for k in DYNAMICS_KEYS}
        env_config = make_env_config(**dyn)
        tag = config_tag(cfg)
        print(f'=== config {tag}: {dyn} ===')

        finetuned = _load_manifest(args.finetune_dir, tag, 3, 'finetune.py') if 3 in setups else {}
        retrained = _load_manifest(args.retrain_dir, tag, 4, 'retrain.py') if 4 in setups else {}

        for a in agents:
            idx = a['idx']
            # setup -> (agent, config the shield assumes)
            runs = {
                1: lambda: (base_algorithms[idx], baseline_shield),
                2: lambda: (base_algorithms[idx], env_config),
                3: lambda: (
                    load_collector_and_algorithm_from_experiment(Path(finetuned[idx]))[1],
                    env_config,
                ),
                4: lambda: (
                    load_collector_and_algorithm_from_experiment(Path(retrained[idx]))[1],
                    env_config,
                ),
            }
            for setup in sorted(setups):
                algo, shield_config = runs[setup]()
                metrics = evaluate_policy(algo, env_config, shield_config, args.episodes, args.seed)
                rows.append(
                    {
                        **dyn,
                        'param_varied': cfg['param_varied'],
                        'setup': setup,
                        'agent_idx': idx,
                        **metrics,
                    }
                )

    fieldnames = [
        'param_varied',
        'mass',
        'max_acc_force',
        'max_brake_force',
        'setup',
        'agent_idx',
        'goal_reached_rate',
        'collision_rate',
        'intervention_rate',
        'return_mean',
    ]
    with open(args.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
