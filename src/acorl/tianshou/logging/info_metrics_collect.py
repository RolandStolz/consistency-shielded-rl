"""
Logging of custom env-info metrics in tianshou trainings.

Mechanism: a `CollectStats` subclass accumulates the metrics during collection and
exposes them as nested `dict[str, float]` fields, which the trainer logs via
`asdict(collect_stats)`. The subclass is injected into both the train and test
collectors through a `CollectorFactory` set on the experiment builder.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.data import Collector, CollectStats, ReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.highlevel.experiment import ExperimentBuilder
from tianshou.highlevel.params.collector import CollectorFactory


@dataclass(kw_only=True)
class InfoMetricsCollectStats(CollectStats):
    """Tracks env-info metrics during collection.

    `RATE_KEYS`: boolean info keys that are true only on the terminating step;
    logged as the fraction of episodes that ended with the flag set.
    `STEP_MEAN_KEYS`: numeric info keys present every step; logged as the mean
    over all collected steps.

    Keys are class variables (the collector instantiates the stats class with no
    args). They are set by :func:`configure_experiment_seeker_metrics_logging`.
    Class attributes (rather than a dynamically generated subclass) are used so the
    class stays importable, which the high-level experiment requires for pickling.
    """

    RATE_KEYS: ClassVar[tuple[str, ...]] = ()
    STEP_MEAN_KEYS: ClassVar[tuple[str, ...]] = ()

    info_rates: dict[str, float] = field(default_factory=dict)
    info_step_means: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # non-field accumulators (not serialized by asdict, so not logged raw)
        self._flag_lists: dict[str, list[bool]] = {k: [] for k in self.RATE_KEYS}
        self._step_sums: dict[str, float] = {k: 0.0 for k in self.STEP_MEAN_KEYS}
        self._step_counts: dict[str, int] = {k: 0 for k in self.STEP_MEAN_KEYS}

    @staticmethod
    def _extract(info: Any, key: str) -> np.ndarray:
        """Extract per-entry values for `key` from an info batch/array of dicts.

        Missing values become NaN. Returns an empty array if the key is absent.
        """
        if hasattr(info, key):
            return np.asarray(getattr(info, key), dtype=float).ravel()
        try:
            return np.array([d.get(key, np.nan) for d in info], dtype=float)
        except TypeError:
            return np.array([], dtype=float)

    def update_at_step_batch(self, step_batch: Any, refresh_sequence_stats: bool = False) -> None:
        super().update_at_step_batch(step_batch, refresh_sequence_stats=refresh_sequence_stats)
        for k in self.STEP_MEAN_KEYS:
            values = self._extract(step_batch.info, k)
            values = values[~np.isnan(values)]
            self._step_sums[k] += float(values.sum())
            self._step_counts[k] += int(values.size)
        if refresh_sequence_stats:
            self._refresh_info_metrics()

    def update_at_episode_done(
        self,
        episode_batch: Any,
        episode_return: float,
        refresh_sequence_stats: bool = False,
    ) -> None:
        super().update_at_episode_done(
            episode_batch, episode_return, refresh_sequence_stats=refresh_sequence_stats
        )
        for k in self.RATE_KEYS:
            values = self._extract(episode_batch.info, k)
            flag = bool(values[-1]) if values.size and not np.isnan(values[-1]) else False
            self._flag_lists[k].append(flag)
        if refresh_sequence_stats:
            self._refresh_info_metrics()

    def refresh_all_sequence_stats(self) -> None:
        super().refresh_all_sequence_stats()
        self._refresh_info_metrics()

    def _refresh_info_metrics(self) -> None:
        self.info_rates = {k: float(np.mean(v)) for k, v in self._flag_lists.items() if v}
        self.info_step_means = {
            k: self._step_sums[k] / self._step_counts[k]
            for k in self.STEP_MEAN_KEYS
            if self._step_counts[k] > 0
        }


class InfoMetricsCollectorFactory(CollectorFactory):
    def create_collector(
        self,
        algorithm: Algorithm,
        vector_env: BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
    ) -> Collector:
        return Collector(
            algorithm.policy,
            vector_env,
            buffer=buffer,
            exploration_noise=exploration_noise,
            collect_stats_class=InfoMetricsCollectStats,
        )


def configure_experiment_metrics_logging(
    builder: ExperimentBuilder,
    rate_keys: tuple[str, ...] = (),
    step_mean_keys: tuple[str, ...] = (),
) -> ExperimentBuilder:
    """
    Log env-info metrics in train and test collectors.
    Must be called before `builder.build()`.

    Logs `info_rates/<key>` (fraction of episodes ending with the flag) and
    `info_step_means/<key>` (mean over all steps) per epoch.

    Note: `rate_keys` / `step_mean_keys` are applied as class attributes on the
    shared `InfoMetricsCollectStats`, so all experiments in a process use the same
    metrics.
    """
    InfoMetricsCollectStats.RATE_KEYS = tuple(rate_keys)
    InfoMetricsCollectStats.STEP_MEAN_KEYS = tuple(step_mean_keys)
    builder.with_collector_factory(InfoMetricsCollectorFactory())
    return builder
