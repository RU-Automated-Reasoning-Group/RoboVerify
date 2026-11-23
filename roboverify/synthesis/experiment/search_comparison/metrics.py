"""Ordered, speed-insensitive spatial imitation score in metres."""

import numpy as np


def arc_resample(values, points=64):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or len(values) == 0
        or not np.isfinite(values).all()
        or points < 2
    ):
        raise ValueError("Need a finite nonempty trajectory and at least two samples")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))]
    keep = np.r_[True, np.diff(cumulative) > 1e-12]
    values, cumulative = values[keep], cumulative[keep]
    if len(values) == 1:
        return np.repeat(values, points, axis=0)
    grid = np.linspace(0.0, cumulative[-1], points)
    return np.column_stack(
        [np.interp(grid, cumulative, values[:, j]) for j in range(values.shape[1])]
    )


def role_features(observations, example):
    obs = np.asarray(observations)
    n, source = example["num_blocks"], example["bindings"]["b_prime"]
    other = [i for i in range(n) if i != source]
    parts = [obs[:, :3], obs[:, 10 + 12 * source : 13 + 12 * source]]
    parts += [obs[:, 10 + 12 * i : 13 + 12 * i] for i in other]
    weights = np.asarray([1.0, 2.0] + [2.0 / max(1, len(other))] * len(other))
    return np.concatenate(
        [p * np.sqrt(w / weights.sum()) for p, w in zip(parts, weights)], axis=1
    )


class ArcMetric:
    def __init__(self, examples):
        self.examples = examples
        self.references = [
            arc_resample(
                role_features(e["observations"][: e["execution_end_index"] + 1], e)
            )
            for e in examples
        ]

    def __call__(self, trajectories):
        if len(trajectories) != len(self.examples):
            raise ValueError("All fitting examples must be scored together")
        return float(
            np.mean(
                [
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (arc_resample(role_features(obs, e)) - reference) ** 2,
                                axis=1,
                            )
                        )
                    )
                    for obs, e, reference in zip(
                        trajectories, self.examples, self.references
                    )
                ]
            )
        )


def objective(distance, failures, scale=0.05):
    """Maximize; each fewer failed episode dominates the imitation tie break."""
    if failures < 0 or not np.isfinite(distance) or scale <= 0:
        raise ValueError(
            "Finite score, nonnegative failures and positive scale required"
        )
    return -float(failures) - (0.5 + np.arctan(float(distance) / scale) / np.pi)
