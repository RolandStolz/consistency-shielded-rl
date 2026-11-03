"""
Fine-tune the base agents under one perturbed env config with the *adapted* shield (setup 3).

For a single grid config (identified by --tag, e.g. 'mass_1.2'), restore each base agent's weights
and continue PPO training on an env whose dynamics are perturbed accordingly, with the shield
adapted to match. Same architecture as the base agent (required for weight restore) and the same
HPs except the learning rate is halved. Writes a per-config manifest
`<out-dir>/<tag>.json` mapping agent idx -> persistence dir, which evaluate.py reads.

One config per process, so configs can be fine-tuned in parallel (see launch_finetune.sh).

Usage:
    python finetune.py --tag mass_1.2 --manifest models/agents.json --out-dir models/finetuned
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

FINETUNE_EPOCHS = 10  # 10 x 20k = 200k env steps total.
FINETUNE_EPOCH_STEPS = 20_000


def finetune_agent(
    base_dir: str,
    hp: dict,
    env_config: SeekerUnicycleConfig,
    out_base_dir: str,
    run_name: str,
    *,
    epochs: int = FINETUNE_EPOCHS,
    epoch_steps: int = FINETUNE_EPOCH_STEPS,
    seed: int = 0,
    use_wandb: bool = False,
) -> str:
    """Restore `base_dir`'s policy and fine-tune it on `env_config` with an adapted shield.

    Returns the persistence directory of the fine-tuned agent.
    """
    finetune_hp = {**hp}
    if finetune_hp.get('lr') is not None:
        finetune_hp['lr'] = finetune_hp['lr'] / 2  # gentler fine-tuning

    if use_wandb:
        # one wandb run per agent; sync_tensorboard so tianshou's tb logs reach wandb.
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=finetune_hp,
            name=run_name,
            tags=['finetune'],
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
            finetune_hp,
            env_factory,
            max_epochs=epochs,
            epoch_num_steps=epoch_steps,
            logger_factory=logger_factory,
            persist_base_dir=out_base_dir,
            persistence_enabled=True,
            restore_dir=base_dir,
            seed=seed,
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


def finetune_config(
    agents: list[dict],
    env_config: SeekerUnicycleConfig,
    tag: str,
    out_dir: str,
    *,
    epochs: int = FINETUNE_EPOCHS,
    epoch_steps: int = FINETUNE_EPOCH_STEPS,
    base_seed: int = 0,
    use_wandb: bool = False,
) -> str:
    """Fine-tune every base agent for one config; write `<out-dir>/<tag>.json` (idx -> dir)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for a in agents:
        idx = a['idx']
        print(f'=== fine-tuning {tag} / agent {idx} ===')
        manifest[idx] = finetune_agent(
            a['persist_directory'],
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
    parser.add_argument('--out-dir', default='models/finetuned')
    parser.add_argument('--epochs', type=int, default=FINETUNE_EPOCHS)
    parser.add_argument(
        '--steps', type=int, default=FINETUNE_EPOCH_STEPS, help='env steps per epoch'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--wandb', action='store_true', help="log each run to wandb (tag 'finetune')"
    )
    args = parser.parse_args()

    with open(args.manifest) as fh:
        agents = json.load(fh)
    cfg = _config_for_tag(args.tag)
    env_config = make_env_config(**{k: cfg[k] for k in DYNAMICS_KEYS})
    finetune_config(
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
