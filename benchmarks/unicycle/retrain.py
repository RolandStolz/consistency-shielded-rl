"""
Re-train the agents from scratch under one perturbed env config with the *adapted* shield (setup 2).

Like finetune.py regarding the setup (one grid config per --tag, env == shield == perturbed dynamics),
but instead of restoring base weights and continuing for 200k steps, this trains a fresh PPO agent per
HP config from scratch with the full schedule. Writes a per-config manifest
`<out-dir>/<tag>.json` mapping agent idx -> persistence dir, which evaluate.py reads.

One config per process, so configs can be re-trained in parallel.

Usage:
    python retrain.py --tag mass_1.2 --manifest models/agents.json --out-dir models/retrained
"""

import argparse
import json
import os

import wandb
from common import (
    DYNAMICS_KEYS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_env_factory,
    build_grid,
    build_ppo_builder,
    config_tag,
    make_env_config,
)
from tianshou.highlevel.env import VectorEnvType
from tianshou.highlevel.logger import LoggerFactoryDefault

from acorl.envs.seeker.unicycle import SeekerUnicycleConfig
from acorl.tianshou.logging.info_metrics_collect import configure_experiment_metrics_logging

MAX_EPOCHS, EPOCH_NUM_STEPS = 10, 100_000


def retrain_agent(
    hp: dict,
    env_config: SeekerUnicycleConfig,
    out_base_dir: str,
    run_name: str,
    *,
    epochs: int = MAX_EPOCHS,
    epoch_steps: int = EPOCH_NUM_STEPS,
    seed: int = 0,
    use_wandb: bool = False,
) -> str:
    """Train a fresh policy on `env_config` with an adapted shield.

    Returns the persistence directory of the re-trained agent.
    """
    if use_wandb:
        # one wandb run per agent; sync_tensorboard so tianshou's tb logs reach wandb.
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=hp,
            name=run_name,
            tags=['retrain'],
            sync_tensorboard=True,
        )
        logger_factory = LoggerFactoryDefault(
            'wandb', wandb_project=WANDB_PROJECT, wand_entity=WANDB_ENTITY
        )
    else:
        logger_factory = LoggerFactoryDefault('tensorboard')

    # env == shield == perturbed dynamics: the shield is adapted to the new env.
    env_factory = build_env_factory(env_config, env_config, venv_type=VectorEnvType.DUMMY)
    try:
        builder = build_ppo_builder(
            hp,
            env_factory,
            max_epochs=epochs,
            epoch_num_steps=epoch_steps,
            logger_factory=logger_factory,
            persist_base_dir=out_base_dir,
            persistence_enabled=True,
            seed=seed,
            num_train_envs=1,
        )
        builder = configure_experiment_metrics_logging(
            builder,
            rate_keys=('goal_reached', 'collision'),
            step_mean_keys=('required_intervention',),
        )
        result = builder.build().run(run_name=run_name)
    finally:
        if use_wandb:
            wandb.finish()
    return os.path.abspath(result.world.persist_directory)


def retrain_config(
    agents: list[dict],
    env_config: SeekerUnicycleConfig,
    tag: str,
    out_dir: str,
    *,
    epochs: int = MAX_EPOCHS,
    epoch_steps: int = EPOCH_NUM_STEPS,
    base_seed: int = 0,
    use_wandb: bool = False,
) -> str:
    """Re-train every HP config for one config; write `<out-dir>/<tag>.json` (idx -> dir)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for a in agents:
        idx = a['idx']
        print(f'=== re-training {tag} / agent {idx} ===')
        manifest[idx] = retrain_agent(
            a['hyperparameters'],
            env_config,
            out_dir,
            f'{tag}_agent_{idx}',
            epochs=epochs,
            epoch_steps=epoch_steps,
            seed=base_seed + idx,
            use_wandb=use_wandb,
        )
    manifest_path = os.path.join(out_dir, f'{tag}.json')
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'wrote {manifest_path}')
    return manifest_path


def _config_for_tag(tag: str) -> dict:
    for cfg in build_grid():
        if config_tag(cfg) == tag:
            return cfg
    raise SystemExit(f'unknown config tag {tag!r}; see common.build_grid()')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tag', required=True, help="grid config tag, e.g. 'mass_1.2'")
    parser.add_argument(
        '--manifest', default='models/agents.json', help='agents.json from train_agents.py'
    )
    parser.add_argument('--out-dir', default='models/retrained')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--steps', type=int, default=EPOCH_NUM_STEPS, help='env steps per epoch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--wandb', action='store_true', help="log each run to wandb (tag 'retrain')"
    )
    args = parser.parse_args()

    with open(args.manifest) as fh:
        agents = json.load(fh)
    cfg = _config_for_tag(args.tag)
    env_config = make_env_config(**{k: cfg[k] for k in DYNAMICS_KEYS})
    retrain_config(
        agents,
        env_config,
        args.tag,
        args.out_dir,
        epochs=args.epochs,
        epoch_steps=args.steps,
        base_seed=args.seed,
        use_wandb=args.wandb,
    )


if __name__ == '__main__':
    main()
