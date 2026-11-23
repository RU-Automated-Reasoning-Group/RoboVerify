"""Cached density distances and optional ordered spatial trajectory distance."""

from typing import Protocol

import numpy as np
from scipy.special import logsumexp


class TrajectoryDistance(Protocol):
    name: str

    def __call__(self, states, *, samples=256): ...


class KDEDistance:
    name = "kl_kde"

    def __init__(self, demos, *, bandwidth=0.02, seed=0):
        self.demos = self._states(demos)
        if bandwidth <= 0:
            raise ValueError("KDE bandwidth must be positive")
        self.bandwidth = bandwidth
        self.seed = seed
        # Cache demo-side scaling/density components across every CEM evaluation.
        self.scale = np.maximum(self.demos.std(axis=0), bandwidth)
        self.reference = self.demos / self.scale

    @staticmethod
    def _states(values):
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("Distance requires finite nonempty N x D states")
        return array

    def _log_density(self, points, centers):
        # Isotropic Gaussian KDE in demo-normalized coordinates, chunked by samples.
        result = []
        for chunk in np.array_split(points, max(1, (len(points) + 63) // 64)):
            square = ((chunk[:, None, :] - centers[None, :, :]) / self.bandwidth) ** 2
            result.extend(
                logsumexp(-0.5 * square.sum(axis=2), axis=1) - np.log(len(centers))
            )
        return np.asarray(result)

    def __call__(self, states, *, samples=256):
        policy = self._states(states) / self.scale
        if policy.shape[1] != self.reference.shape[1] or samples < 1:
            raise ValueError("Mismatched dimensions or sample budget")
        rng = np.random.default_rng(self.seed)
        points = policy[rng.integers(len(policy), size=samples)] + rng.normal(
            0, self.bandwidth, (samples, policy.shape[1])
        )
        # Monte Carlo can be slightly negative. Keep that estimator value; do not
        # silently rescale MMD or manufacture a KL threshold from it.
        return float(
            np.mean(
                self._log_density(points, policy)
                - self._log_density(points, self.reference)
            )
        )


class GMMDistance:
    name = "kl_gmm"

    def __init__(self, demos, *, components=3, seed=0):
        from sklearn.mixture import GaussianMixture

        self.seed, self.components = seed, components
        self.factory = GaussianMixture
        self.reference = self._fit(KDEDistance._states(demos))

    def _fit(self, states):
        if len(states) == 1:
            states = np.concatenate([states, states], axis=0)
        return self.factory(
            n_components=min(self.components, len(states)),
            covariance_type="diag",
            reg_covar=1e-5,
            random_state=self.seed,
        ).fit(states)

    def __call__(self, states, *, samples=256):
        policy = self._fit(KDEDistance._states(states))
        points, _ = policy.sample(samples)
        return float(
            np.mean(policy.score_samples(points) - self.reference.score_samples(points))
        )


class MMDDistance:
    name = "mmd"

    def __init__(self, demos):
        self.demos = KDEDistance._states(demos)

    def __call__(self, states, *, samples=256):
        from synthesis.mcmc.cost_func import maximum_mean_discrepancy_rbf

        return float(
            maximum_mean_discrepancy_rbf(KDEDistance._states(states), self.demos)
        )


def arc_resample(values, *, points=64):
    """Resample one ordered polyline by arc length, ignoring pauses and speed.

    Endpoints are retained. A constant trajectory repeats its only position;
    separate demonstrations must never be concatenated before this operation.
    """
    values = KDEDistance._states(values)
    if not isinstance(points, int) or points < 2:
        raise ValueError("Arc distance requires at least two resampling points")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))]
    keep = np.r_[True, np.diff(cumulative) > 1e-12]
    values, cumulative = values[keep], cumulative[keep]
    if len(values) == 1:
        return np.repeat(values, points, axis=0)
    grid = np.linspace(0.0, cumulative[-1], points)
    return np.column_stack(
        [np.interp(grid, cumulative, values[:, j]) for j in range(values.shape[1])]
    )


class ArcDistance:
    """One demonstration's speed-insensitive spatial RMS distance, in metres.

    Raw RoboVerify observations supply gripper XYZ and every block XYZ. An
    observed block excursion greater than ``motion_threshold`` (5 mm by default)
    gives that block weight 2; other blocks get 0.25 and the gripper gets 1.
    Weights depend only on demonstration motion, never symbolic role bindings.
    Their sum normalizes squared Euclidean object errors, preserving metre units.

    This compares ordered paths, not timing, finger aperture or velocities. It
    does not replace endpoint, PostScore, physical-safety or verification checks.
    The ``samples`` argument is accepted for the shared distance interface; this
    deterministic metric always uses its fixed ``points`` resolution.
    """

    name = "arc"

    def __init__(self, demos, *, num_blocks, points=64, motion_threshold=0.005):
        if not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("Arc distance needs a positive physical block count")
        if not np.isfinite(motion_threshold) or motion_threshold < 0:
            raise ValueError("Arc motion threshold must be finite and nonnegative")
        self.num_blocks = num_blocks
        self.points = points
        objects = self._objects(demos)
        excursion = np.max(
            np.linalg.norm(objects[:, 1:] - objects[0, 1:], axis=2), axis=0
        )
        self.moving_blocks = excursion > motion_threshold
        self.weights = np.r_[1.0, np.where(self.moving_blocks, 2.0, 0.25)]
        self.scale = np.sqrt(self.weights / self.weights.sum())[None, :, None]
        self.reference = arc_resample(self._features(objects), points=self.points)

    def _objects(self, observations):
        observations = KDEDistance._states(observations)
        if observations.shape[1] < 10 + 12 * self.num_blocks:
            raise ValueError("Arc distance requires raw gripper and block observations")
        return np.stack(
            [observations[:, :3]]
            + [
                observations[:, 10 + 12 * i : 13 + 12 * i]
                for i in range(self.num_blocks)
            ],
            axis=1,
        )

    def _features(self, objects):
        return (objects * self.scale).reshape(len(objects), -1)

    def __call__(self, states, *, samples=256):
        actual = arc_resample(self._features(self._objects(states)), points=self.points)
        return float(np.sqrt(np.mean(np.sum((actual - self.reference) ** 2, axis=1))))


def make_distance(name, demos, **kwargs):
    return {
        "kl_kde": KDEDistance,
        "kl_gmm": GMMDistance,
        "mmd": MMDDistance,
        "arc": ArcDistance,
    }[name](demos, **kwargs)
