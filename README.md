This repository is a stripped-down version of a to-be-published Python package on action-constrained reinforcement learning and safety shields. It contains only what is needed for the unicycle experiments of the MODELS NIER submission.

The experiments are located in `benchmarks/unicycle`. Please read `benchmarks/unicycle/README.md` for instructions.

## Installation

Requires Python >= 3.12. No system dependencies are needed.

Create a Python environment and install all dependencies using [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras
```

This creates a `.venv`, which you can activate with `source .venv/bin/activate`.

## Structure

- `src/acorl/envs/seeker/`: the unicycle seeker environment (`SeekerUnicycleEnv`).
- `src/acorl/envs/constraints/seeker/unicycle.py`: the safety shield (`SeekerUnicycleClosestPointProjection`), a discrete control-barrier-function shield that projects the agent's action onto the closest safe action.
- `src/acorl/common/env_wrapper/adaption.py`: the gymnasium wrapper that applies the shield to every action.
- `src/acorl/tianshou/`: tianshou glue (env factory, metric logging, experiment loading).
- `benchmarks/unicycle/`: training, fine-tuning, re-training, evaluation, and plotting scripts.
