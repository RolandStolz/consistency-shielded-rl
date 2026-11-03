import functools
from pathlib import Path

import torch
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.data import BaseCollector
from tianshou.highlevel.experiment import Experiment


def load_collector_and_algorithm_from_experiment(
    log_path: Path,
) -> tuple[BaseCollector, Algorithm]:
    """
    Load a collector and policy from a Tianshou experiment directory.
    IMPORTANT: The function bypasses the safe torch 'weights_only' load.
    Therefore, only load experiments that you know are safe to load.
    """

    assert log_path.exists()

    loaded_experiment = Experiment.from_directory(log_path.__str__(), restore_policy=True)
    loaded_experiment.config.persistence_enabled = False

    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def unsafe_torch_load(*args, **kwargs):
        """Monkey-patched torch.load that forces weights_only=False."""
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    torch.load = unsafe_torch_load

    world = loaded_experiment.create_experiment_world()
    assert world.test_collector is not None
    assert world.algorithm is not None

    return world.test_collector, world.algorithm
