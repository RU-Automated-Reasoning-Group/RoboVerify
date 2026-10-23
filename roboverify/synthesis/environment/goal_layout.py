"""Immutable goal-only traversal graphs and geometric occupancy for execution."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

GOAL_TOLERANCE = 0.02
GOAL_REFERENCE_MODE = "goal-center-xyz-v1"


def positive_size(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def pyramid_layers(bottom_row_blocks):
    return tuple(range(positive_size(bottom_row_blocks, "bottom_row_blocks"), 0, -1))


@dataclass(frozen=True)
class GoalRef:
    index: int


@dataclass(frozen=True)
class Goal:
    position: tuple[float, float, float]
    r: GoalRef | None
    d: GoalRef | None


@dataclass(frozen=True)
class GoalLayout:
    goals: tuple[Goal, ...]
    row_lengths: tuple[int, ...]

    def as_dict(self):
        return dict(
            row_lengths=list(self.row_lengths),
            positions=self.positions.tolist(),
            head=self.head.index,
            r=[None if g.r is None else g.r.index for g in self.goals],
            d=[None if g.d is None else g.d.index for g in self.goals],
        )

    @classmethod
    def from_dict(cls, value):
        layout = cls.from_positions(value["positions"], value["row_lengths"])
        if layout.as_dict() != value:
            raise ValueError("Goal topology does not match its row layout")
        return layout

    @classmethod
    def from_positions(cls, positions, row_lengths):
        rows = tuple(positive_size(n, "row length") for n in row_lengths)
        xyz = np.asarray(positions, dtype=float)
        if not rows or xyz.shape != (sum(rows), 3) or not np.isfinite(xyz).all():
            raise ValueError(
                "Goal coordinates must be finite and match the row lengths"
            )
        distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances <= 2 * GOAL_TOLERANCE):
            raise ValueError("Goal occupancy regions must be disjoint")
        goals, start = [], 0
        for length in rows:
            for column in range(length):
                index = start + column
                right = GoalRef(index + 1) if column + 1 < length else None
                down = (
                    GoalRef(start + length)
                    if column == 0 and start + length < len(xyz)
                    else None
                )
                goals.append(Goal(tuple(map(float, xyz[index])), right, down))
            start += length
        return cls(tuple(goals), rows)

    @property
    def head(self):
        return GoalRef(0)

    @property
    def positions(self):
        return np.asarray([g.position for g in self.goals])

    def get(self, reference):
        if (
            not isinstance(reference, GoalRef)
            or type(reference.index) is not int
            or not 0 <= reference.index < len(self.goals)
        ):
            raise ValueError(f"Unknown or null goal reference: {reference}")
        return self.goals[reference.index]


def occupancy(block_positions, goal_positions, tolerance=GOAL_TOLERANCE):
    """Return [..., block, goal] occupancy, without assigning block identities."""
    blocks = np.asarray(block_positions, dtype=float)
    goals = np.asarray(goal_positions, dtype=float)
    return (
        np.linalg.norm(blocks[..., :, None, :] - goals[..., None, :, :], axis=-1)
        <= tolerance
    )


def goal_bindings(env):
    mapping = getattr(env, "symbolic_name_to_goal", None)
    if not isinstance(mapping, dict):
        raise ValueError("Goal execution requires symbolic_name_to_goal")
    return mapping


def encode_goal_bindings(mapping):
    return {
        name: None if value is None else value.index for name, value in mapping.items()
    }


def decode_goal_bindings(mapping, layout):
    if (
        not isinstance(mapping, dict)
        or mapping.get("h0") != 0
        or "null" not in mapping
        or mapping["null"] is not None
    ):
        raise ValueError("Goal bindings require the fixed h0 and null constants")
    result = {}
    for name, index in mapping.items():
        if not isinstance(name, str):
            raise ValueError("Goal aliases must be names")
        reference = None if index is None else GoalRef(index)
        if reference is not None:
            layout.get(reference)
        result[name] = reference
    return result


def resolve_reference(env, name):
    blocks = getattr(env, "symbolic_name_to_box_id", {})
    goals = getattr(env, "symbolic_name_to_goal", {})
    if name in blocks and name in goals:
        raise ValueError(f"Ambiguous block/goal alias: {name}")
    if name in blocks:
        return blocks[name]
    if name in goals:
        reference = goals[name]
        env.goal_layout.get(reference)  # Null is not a physical coordinate.
        return reference
    raise KeyError(f"Unknown block or goal alias: {name}")
