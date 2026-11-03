"""
Train the base agents for the V-SUM consistency experiments.

Takes the top-N HP configs from best_hps.json (results of a prior HP sweep) and trains one
PPO agent per config on the *consistent* nominal dynamics (F_a = F_b = 2, m = 1), guarded by the
closest-point-projection shield. Each agent is persisted to its own directory and a manifest
(`agents.json`) records the HPs and persistence path for downstream finetune/eval.

Usage:
    python train_agents.py --config best_hps.json --out-dir models
    python train_agents.py --out-dir models          # no --config: default HPs, --top agents
"""

import argparse
import json
import os

import wandb
from common import (
    BASELINE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_env_factory,
    build_ppo_builder,
    make_env_config,
)
from tianshou.highlevel.logger import LoggerFactoryDefault

from acorl.tianshou.logging.info_metrics_collect import configure_experiment_metrics_logging

MAX_EPOCHS, EPOCH_NUM_STEPS = 10, 200_000


def _load_hp_configs(path: str, top: int) -> list[dict]:
    """Return up to `top` hyperparameter dicts from a best_hps.json-style file (or a bare list)."""
    with open(path) as fh:
        data = json.load(fh)
    configs = data['configs'] if isinstance(data, dict) else data
    hps = [
        c['hyperparameters'] if isinstance(c, dict) and 'hyperparameters' in c else c
        for c in configs
    ]
    return hps[:top]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', help='best-HP JSON, e.g. best_hps.json (default: default HPs)')
    parser.add_argument('--out-dir', default='models', help='persistence dir for the agents')
    parser.add_argument('--top', type=int, default=5, help='number of agents / HP configs to train')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--wandb', action='store_true', help="log each run to wandb (tag 'training')"
    )
    args = parser.parse_args()

    # No --config: train `--top` agents with default HPs (empty cfg -> builder defaults).
    hp_configs = _load_hp_configs(args.config, args.top) if args.config else [{}] * args.top
    os.makedirs(args.out_dir, exist_ok=True)

    # env == shield == baseline: the agent is trained under consistent dynamics.
    config = make_env_config(**BASELINE)

    manifest = []
    for idx, hp in enumerate(hp_configs):
        print(f'=== training agent {idx + 1}/{len(hp_configs)} ===')
        if args.wandb:
            # one wandb run per agent; sync_tensorboard so tianshou's tb logs reach wandb.
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config=hp,
                name=f'agent_{idx}',
                tags=['training'],
                sync_tensorboard=True,
            )
            logger_factory = LoggerFactoryDefault(
                'wandb', wandb_project=WANDB_PROJECT, wand_entity=WANDB_ENTITY
            )
        else:
            logger_factory = LoggerFactoryDefault('tensorboard')
        try:
            env_factory = build_env_factory(config, config)
            builder = build_ppo_builder(
                hp,
                env_factory,
                max_epochs=MAX_EPOCHS,
                epoch_num_steps=EPOCH_NUM_STEPS,
                logger_factory=logger_factory,
                persist_base_dir=args.out_dir,
                persistence_enabled=True,
                seed=args.seed + idx,
                num_train_envs=10,
            )
            builder = configure_experiment_metrics_logging(
                builder,
                rate_keys=('goal_reached', 'collision'),
                step_mean_keys=('required_intervention',),
            )
            result = builder.build().run(run_name=f'agent_{idx}')
            persist_dir = os.path.abspath(result.world.persist_directory)
            print(f'    persisted to {persist_dir}')
            manifest.append({'idx': idx, 'hyperparameters': hp, 'persist_directory': persist_dir})
        finally:
            if args.wandb:
                wandb.finish()

    manifest_path = os.path.join(args.out_dir, 'agents.json')
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'Wrote manifest with {len(manifest)} agents to {manifest_path}')


if __name__ == '__main__':
    main()
