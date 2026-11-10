"""Finite contract-guided proposals for straight-line manipulation programs.

The nominal frame and swept-box envelope are search heuristics, not statements
about controller dynamics. Every returned program still needs BMC, numeric
execution and the caller's symbolic/motion verification. No task name, supplied
program, teacher action label or fixed primitive sequence enters this module.

``None`` in SearchExample.initial_holding means *known empty hand*. Callers must
reject an unknown interior holding state or supply a separately justified
hypothesis; this module never interprets an unknown state as empty.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np


class ContractInputError(ValueError):
    """The finite nominal model cannot represent the supplied search input."""


class TransitionBudgetExceeded(RuntimeError):
    """The requested number of nominal transition computations was exhausted."""


class DeadlineExceeded(TimeoutError):
    """The shared absolute deadline was reached before an operation completed."""


def _deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise DeadlineExceeded("Contract search deadline reached")


def _array(value, shape, label):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ContractInputError(f"{label} must have finite shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SearchExample:
    positions: np.ndarray
    gripper: np.ndarray
    bindings: Mapping[str, int | str]
    gripper_path: np.ndarray
    positions_path: np.ndarray
    initial_holding: int | None = None
    table_reference: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        positions = np.asarray(self.positions)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
            raise ContractInputError("positions must be nonempty [blocks,3]")
        n = len(positions)
        path = np.asarray(self.gripper_path)
        if path.ndim != 2 or path.shape[1:] != (3,) or not len(path):
            raise ContractInputError("gripper_path must be nonempty [states,3]")
        values = {
            "positions": _array(positions, (n, 3), "positions"),
            "gripper": _array(self.gripper, (3,), "gripper"),
            "gripper_path": _array(path, (len(path), 3), "gripper_path"),
            "positions_path": _array(
                self.positions_path, (len(path), n, 3), "positions_path"
            ),
        }
        if not np.allclose(
            values["gripper_path"][0], values["gripper"], rtol=0, atol=1e-8
        ):
            raise ContractInputError("gripper_path must begin at the supplied entry")
        if not np.allclose(
            values["positions_path"][0], values["positions"], rtol=0, atol=1e-8
        ):
            raise ContractInputError("positions_path must begin at the supplied entry")
        if self.initial_holding is not None and (
            type(self.initial_holding) is not int or not 0 <= self.initial_holding < n
        ):
            raise ContractInputError(
                "initial_holding must be a physical ID or None (known empty)"
            )
        if (
            self.initial_holding is not None
            and np.linalg.norm(
                values["positions"][self.initial_holding] - values["gripper"]
            )
            > 0.03
        ):
            raise ContractInputError("Initially held object must be near the gripper")
        bindings = dict(self.bindings)
        if set(bindings) & {"gripper", "Y_min", "Z_m"}:
            raise ContractInputError(
                "Observation reference names cannot be object bindings"
            )
        for key, value in bindings.items():
            if not isinstance(key, str) or not key:
                raise ContractInputError("Binding names must be nonempty strings")
            if not (value == "tbl" or (type(value) is int and 0 <= value < n)):
                raise ContractInputError(f"Unresolved physical binding {key!r}")
        values["bindings"] = bindings
        if self.table_reference is not None:
            values["table_reference"] = _array(
                self.table_reference, (3,), "table_reference"
            )
        for key, value in values.items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class ContractConfig:
    observation_proposals: bool = True
    observation_references: bool = True
    support_clearance: bool = True
    gripper_envelope: bool = True
    require_supported_release: bool = True
    require_empty_hand: bool = True
    allow_empty_goal: bool = True
    block_half_extent: float = 0.025
    geometry_tolerance: float = 0.002
    max_height: float = 0.95
    minimum_motion: float = 0.006
    observation_samples: int = 25
    proposal_top_references: int = 2
    trace_samples: int = 48
    rank_samples: int = 24
    grid_heights: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2)
    grid_axis_offsets: tuple[float, ...] = (-0.15, -0.1, -0.05, 0.05, 0.1, 0.15)
    release_heights: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2)
    rounding_digits: int = 4

    def __post_init__(self):
        if not 0 < self.geometry_tolerance < self.block_half_extent:
            raise ContractInputError(
                "Require 0 < geometry tolerance < block half extent"
            )
        if (
            not math.isfinite(self.max_height)
            or not math.isfinite(self.minimum_motion)
            or self.minimum_motion < 0
        ):
            raise ContractInputError("Invalid nominal motion limits")
        for key in ("observation_samples", "trace_samples", "rank_samples"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 2:
                raise ContractInputError(f"{key} must be an integer at least two")
        if (
            type(self.proposal_top_references) is not int
            or self.proposal_top_references < 1
        ):
            raise ContractInputError("proposal_top_references must be positive")
        for offsets in (
            self.grid_heights,
            self.grid_axis_offsets,
            self.release_heights,
        ):
            if not offsets or not all(math.isfinite(x) for x in offsets):
                raise ContractInputError("Proposal grids must be finite and nonempty")
        if not 0 <= self.rounding_digits <= 12:
            raise ContractInputError("rounding_digits must be between zero and twelve")


@dataclass(frozen=True)
class Action:
    kind: str
    operands: tuple
    offsets: tuple = ()
    proposal_id: str = field(default="grid", compare=False)

    def __post_init__(self):
        arity = {"Pick": (1, 0), "Move": (3, 3), "Release": (1, 1)}
        if self.kind not in arity:
            raise ContractInputError(f"Unsupported primitive {self.kind}")
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "offsets", tuple(float(x) for x in self.offsets))
        if (len(self.operands), len(self.offsets)) != arity[self.kind]:
            raise ContractInputError(f"Wrong operands/offsets for {self.kind}")
        if not all(math.isfinite(x) for x in self.offsets):
            raise ContractInputError("Offsets must be finite")
        if any(
            not ((type(x) is int and x >= 0) or (isinstance(x, str) and x))
            for x in self.operands
        ):
            raise ContractInputError("Operands must be physical IDs or nonempty names")

    def row(self):
        return {
            "kind": self.kind,
            "operands": list(self.operands),
            "offsets": list(self.offsets),
        }


@dataclass(frozen=True)
class NominalState:
    positions: np.ndarray
    gripper: np.ndarray
    holding: tuple[int, ...]

    def __post_init__(self):
        # Internal arrays are owned or already read-only; avoid repeated copies.
        self.positions.setflags(write=False)
        self.gripper.setflags(write=False)
        object.__setattr__(self, "holding", tuple(int(x) for x in self.holding))

    @property
    def p(self):
        return self.positions

    @property
    def ee(self):
        return self.gripper


def state_key(state):
    """Exact nominal key, including all examples, coordinates and holding state."""
    return (
        state.positions.shape,
        state.positions.dtype.str,
        state.positions.tobytes(),
        state.gripper.shape,
        state.gripper.dtype.str,
        state.gripper.tobytes(),
        state.holding,
    )


@dataclass
class Node:
    state: NominalState
    actions: tuple[Action, ...]
    path: np.ndarray
    score: float = 0.0

    def to_program(self, named=True, prefix=()):
        from synthesis.api.instructions import (
            Move,
            MoveByName,
            Pick,
            PickByName,
            Release,
            ReleaseByName,
        )
        from synthesis.api.program import Program

        instructions = deepcopy(list(prefix))
        types = {
            "Pick": PickByName if named else Pick,
            "Move": MoveByName if named else Move,
            "Release": ReleaseByName if named else Release,
        }
        for action in self.actions:
            if any(
                (not isinstance(x, str)) if named else type(x) is not int
                for x in action.operands
            ):
                raise ContractInputError(
                    "Named and numeric program operands cannot be mixed"
                )
            options = {}
            if action.kind == "Move":
                options["target_offset"] = list(action.offsets)
            elif action.kind == "Release":
                options["target_z"] = action.offsets[0]
            instructions.append(types[action.kind](*action.operands, **options))
        return Program(len(instructions), instructions)


def resample_batch(paths, count=24):
    """Arc-resample [examples,states,features], handling dwell and zero paths."""
    paths = np.asarray(paths, dtype=float)
    e, k, _ = paths.shape
    if k == 1:
        return np.repeat(paths, count, axis=1)
    cumulative = np.concatenate(
        [
            np.zeros((e, 1)),
            np.cumsum(np.linalg.norm(np.diff(paths, axis=1), axis=2), axis=1),
        ],
        axis=1,
    )
    grid = cumulative[:, -1, None] * np.linspace(0.0, 1.0, count)[None, :]
    indices = np.clip(
        np.sum(cumulative[:, None, :] <= grid[:, :, None], axis=2) - 1, 0, k - 2
    )
    ix = np.arange(e)[:, None]
    lo, hi = cumulative[ix, indices], cumulative[ix, indices + 1]
    fraction = np.divide(
        grid - lo, hi - lo, out=np.zeros_like(grid), where=hi - lo > 1e-12
    )
    return paths[ix, indices] + fraction[:, :, None] * (
        paths[ix, indices + 1] - paths[ix, indices]
    )


def swept_hit(start, end, obstacles, half):
    """Strict segment / expanded-box intersection, per example and obstacle."""
    a = start[:, None, :]
    direction = (end - start)[:, None, :]
    half = np.asarray(half)
    zero = abs(direction) < 1e-10
    safe = np.where(zero, 1.0, direction)
    u, v = (obstacles - half - a) / safe, (obstacles + half - a) / safe
    lo = np.where(zero, -np.inf, np.minimum(u, v))
    hi = np.where(zero, np.inf, np.maximum(u, v))
    outside = np.any(zero & (abs(a - obstacles) >= half), axis=2)
    lower = np.maximum(np.max(lo, axis=2), 0)
    upper = np.minimum(np.min(hi, axis=2), 1)
    return (~outside) & (upper > np.maximum(lower, 1e-6)) & (lower < 1 - 1e-6)


class ContractProblem:
    """Shared proposals, nominal transitions and ranking for beam or local search.

    ``goal(state, example_index)`` evaluates only the caller's postcondition.
    ``batch_goal(state)`` is an optional vectorized alternative returning E bools.
    A goal requires an empty hand by default, independently of that predicate.
    Table reference coordinates are never invented: absent coordinates remove
    table operands from the grammar. The default table support height is the
    minimum initial block height; callers with elevated-only scenes must supply
    the physical table reference explicitly.
    """

    def __init__(
        self,
        examples,
        goal=None,
        *,
        batch_goal=None,
        allowed_operands=None,
        config=None,
        deadline=None,
        actions=None,
    ):
        self.started = time.perf_counter()
        self.deadline = deadline
        _deadline(deadline)
        self.config = config or ContractConfig()
        self.examples = tuple(examples)
        if not self.examples or not all(
            isinstance(e, SearchExample) for e in self.examples
        ):
            raise ContractInputError("Nonempty SearchExample inputs required")
        if goal is None and batch_goal is None:
            raise ContractInputError("A postcondition callback is required")
        self.goal_callback, self.batch_goal_callback = goal, batch_goal
        self.E, self.N = len(self.examples), len(self.examples[0].positions)
        if any(len(e.positions) != self.N for e in self.examples):
            raise ContractInputError(
                "One nominal search requires a common physical block count"
            )
        self.ix = np.arange(self.E)
        self.initial = NominalState(
            np.stack([e.positions for e in self.examples]),
            np.stack([e.gripper for e in self.examples]),
            tuple(
                -1 if e.initial_holding is None else e.initial_holding
                for e in self.examples
            ),
        )
        self.bindings = {}
        common = set.intersection(*(set(e.bindings) for e in self.examples))
        for name in sorted(common):
            values = [e.bindings[name] for e in self.examples]
            if all(type(v) is int for v in values):
                self.bindings[name] = np.asarray(values, dtype=int)
            elif all(v == "tbl" for v in values) and all(
                e.table_reference is not None for e in self.examples
            ):
                self.bindings[name] = "tbl"
        if allowed_operands is None:
            allowed_operands = tuple(self.bindings)
        self.operands = tuple(
            sorted(set(allowed_operands), key=lambda x: (isinstance(x, str), str(x)))
        )
        for operand in self.operands:
            if type(operand) is int:
                if not 0 <= operand < self.N:
                    raise ContractInputError("Numeric operand outside physical domain")
            elif operand not in self.bindings:
                raise ContractInputError(f"Unavailable operand {operand!r}")
        if not self.operands:
            raise ContractInputError(
                "At least one physical/table reference is required"
            )
        if any(type(o) is int for o in self.operands) and any(
            isinstance(o, str) for o in self.operands
        ):
            raise ContractInputError("Use either named or numeric primitive operands")
        self.numeric = all(type(o) is int for o in self.operands)
        if self.numeric and self.config.observation_references:
            raise ContractInputError(
                "Numeric programs require observation_references=False"
            )
        self.table_reference = (
            None
            if any(e.table_reference is None for e in self.examples)
            else np.stack([e.table_reference for e in self.examples])
        )
        self.table_centres = np.asarray(
            [
                (
                    e.table_reference[2] + 2 * self.config.block_half_extent
                    if e.table_reference is not None
                    else float(np.min(e.positions[:, 2]))
                )
                for e in self.examples
            ]
        )
        self.counts = Counter()
        self.transition_limit = None
        self._goals = {}
        self._prepare_traces()
        self.actions = tuple(actions) if actions is not None else self._make_actions()
        for action in self.actions:
            self._validate_action(action)
        self.setup_seconds = time.perf_counter() - self.started

    def check_deadline(self, deadline=None):
        limits = [x for x in (deadline, self.deadline) if x is not None]
        _deadline(min(limits) if limits else None)

    def _prepare_traces(self):
        self.demo = []
        movers = []
        for example in self.examples:
            self.check_deadline()
            indices = (
                np.linspace(0, len(example.gripper_path) - 1, self.config.trace_samples)
                .round()
                .astype(int)
            )
            self.demo.append(
                np.concatenate(
                    [
                        example.gripper_path[indices],
                        example.positions_path[indices].reshape(len(indices), -1),
                    ],
                    axis=1,
                )
            )
            movers.append(
                np.max(
                    np.linalg.norm(
                        example.positions_path - example.positions[None], axis=2
                    ),
                    axis=0,
                )
                > 0.03
            )
        self.movers = np.asarray(movers)
        # At least one changed/dominant object is useful for ranking/progress, but
        # this observation-derived choice never restricts which object is picked.
        self.dominant = np.asarray(
            [
                int(
                    np.argmax(
                        np.max(
                            np.linalg.norm(
                                e.positions_path - e.positions[None], axis=2
                            ),
                            axis=0,
                        )
                    )
                )
                for e in self.examples
            ]
        )
        self.weights = np.concatenate(
            [
                np.ones((self.E, 3)),
                np.repeat(np.where(self.movers, 2.0, 0.25), 3, axis=1),
            ],
            axis=1,
        )
        self.demo = np.asarray(self.demo)
        prefixes = []
        for i in range(1, self.config.trace_samples):
            self.check_deadline()
            prefixes.append(
                resample_batch(self.demo[:, : i + 1], self.config.rank_samples)
            )
        self.demo_prefix = np.stack(prefixes, axis=1)

    def _physical_ids(self, operand):
        if type(operand) is int and 0 <= operand < self.N:
            return np.full(self.E, operand, dtype=int)
        values = self.bindings.get(operand)
        if isinstance(values, np.ndarray):
            return values
        raise ContractInputError(f"Pick requires a physical operand: {operand!r}")

    def _validate_action(self, action):
        if not isinstance(action, Action):
            raise ContractInputError("Expected Action values")
        axes = (
            (0, 1, 2)
            if action.kind == "Move"
            else ((2,) if action.kind == "Release" else (None,))
        )
        for operand, axis in zip(action.operands, axes):
            if action.kind == "Pick":
                if operand not in self.operands:
                    raise ContractInputError(
                        f"Operand {operand!r} is outside this grammar"
                    )
                self._physical_ids(operand)
            elif operand in ("gripper", "Y_min", "Z_m"):
                if (
                    not self.config.observation_references
                    or (operand == "Y_min" and axis != 1)
                    or (operand == "Z_m" and axis != 2)
                ):
                    raise ContractInputError(
                        "Observation extrema are scalar references on their own axis"
                    )
            elif operand not in self.operands:
                raise ContractInputError(f"Operand {operand!r} is outside this grammar")

    def reference(self, state, operand):
        if operand == "gripper":
            return state.gripper
        if operand == "Y_min":
            values = np.zeros((self.E, 3))
            values[:, 1] = state.positions[:, :, 1].min(axis=1)
            return values
        if operand == "Z_m":
            values = np.zeros((self.E, 3))
            values[:, 2] = state.positions[:, :, 2].max(axis=1)
            return values
        value = self.bindings.get(operand) if isinstance(operand, str) else None
        if isinstance(value, str) and value == "tbl":
            return self.table_reference
        return state.positions[self.ix, self._physical_ids(operand)]

    def _make_actions(self):
        cfg = self.config
        axis_refs = [list(self.operands) for _ in range(3)]
        if cfg.observation_references:
            axis_refs[0] += ["gripper"]
            axis_refs[1] += ["Y_min", "gripper"]
            axis_refs[2] += ["Z_m", "gripper"]
        actions, seen = [], set()

        def add(kind, refs, offsets=(), origin="grid"):
            self.check_deadline()
            action = Action(
                kind,
                tuple(refs),
                tuple(round(float(x), cfg.rounding_digits) for x in offsets),
                origin,
            )
            if action not in seen:
                seen.add(action)
                actions.append(action)

        for operand in self.operands:
            try:
                self._physical_ids(operand)
            except ContractInputError:
                continue
            add("Pick", (operand,))
        for refs in itertools.product(*axis_refs):
            for dz in cfg.grid_heights:
                add("Move", refs, (0, 0, dz))
        if cfg.observation_references:
            for axis, refs in enumerate(axis_refs):
                for ref in refs:
                    for offset in cfg.grid_axis_offsets:
                        operands, offsets = ["gripper"] * 3, [0.0] * 3
                        operands[axis], offsets[axis] = ref, offset
                        add("Move", operands, offsets)
        for ref in axis_refs[2]:
            for height in cfg.release_heights:
                add("Release", (ref,), (height,))
        if cfg.observation_proposals:
            for qi, quantile in enumerate(np.linspace(0, 1, cfg.observation_samples)):
                points = np.asarray(
                    [
                        e.gripper_path[round(quantile * (len(e.gripper_path) - 1))]
                        for e in self.examples
                    ]
                )
                options = []
                for axis, refs in enumerate(axis_refs):
                    fits = []
                    for ref in refs:
                        self.check_deadline()
                        delta = (
                            points[:, axis] - self.reference(self.initial, ref)[:, axis]
                        )
                        offset = float(np.median(delta))
                        error = float(np.mean(np.square(delta - offset)))
                        fits.append(
                            (error + 1e-7 * abs(offset), repr(ref), ref, offset)
                        )
                    options.append(sorted(fits)[: cfg.proposal_top_references])
                for combination in itertools.product(*options):
                    add(
                        "Move",
                        [x[2] for x in combination],
                        [x[3] for x in combination],
                        f"observation-q{qi}",
                    )
            for ref in axis_refs[2]:
                delta = (
                    np.asarray([e.gripper_path[-1, 2] for e in self.examples])
                    - self.reference(self.initial, ref)[:, 2]
                )
                add(
                    "Release",
                    (ref,),
                    (float(np.median(delta)),),
                    "observation-final-height",
                )
        return actions

    def state_key(self, state):
        return state_key(state)

    def vector(self, state):
        return np.concatenate(
            [state.gripper, state.positions.reshape(self.E, -1)], axis=1
        )

    def goal_mask(self, state):
        key = state_key(state)
        if key not in self._goals:
            self.check_deadline()
            values = np.asarray(
                (
                    self.batch_goal_callback(state)
                    if self.batch_goal_callback
                    else [self.goal_callback(state, i) for i in range(self.E)]
                ),
            )
            if values.shape != (self.E,) or values.dtype.kind != "b":
                raise ContractInputError(
                    "Postcondition callback must return one boolean per example"
                )
            self._goals[key] = values
        return self._goals[key].copy()

    def goal(self, state):
        return bool(
            np.all(self.goal_mask(state))
            and (
                not self.config.require_empty_hand or all(h < 0 for h in state.holding)
            )
        )

    def lower_bound(self, state):
        # Only necessary conditions shared by arbitrary caller goals. More
        # task-specific counting would silently exclude valid programs.
        if self.goal(state):
            return 0
        return 1

    def transition(self, state, action):
        self.check_deadline()
        self._validate_action(action)
        if (
            self.transition_limit is not None
            and self.counts["transition_trials"] >= self.transition_limit
        ):
            raise TransitionBudgetExceeded("Nominal transition budget reached")
        self.counts["transition_trials"] += 1
        cfg = self.config
        p, ee = state.positions, state.gripper
        holding = np.asarray(state.holding)
        held = holding >= 0
        if action.kind == "Pick":
            if np.any(held):
                return None
            source = self._physical_ids(action.operands[0])
            target = p[self.ix, source]
            delta = p - target[:, None]
            above = (
                (abs(delta[:, :, 0]) < 0.04)
                & (abs(delta[:, :, 1]) < 0.04)
                & (delta[:, :, 2] > cfg.block_half_extent)
            )
            if np.any(above):
                return None
            approach = ee.copy()
            approach[:, :2] = target[:, :2]
            if cfg.gripper_envelope:
                exempt = np.arange(self.N)[None, :] == source[:, None]
                for start, end in ((ee, approach), (approach, target)):
                    if np.any(
                        swept_hit(
                            start + [0, 0, 0.055],
                            end + [0, 0, 0.055],
                            p,
                            [0.08, 0.08, 0.103],
                        )
                        & ~exempt
                    ):
                        return None
            return NominalState(p, target.copy(), tuple(source))
        if action.kind == "Move":
            # Every operand is read from the same instruction-entry state.
            target = np.column_stack(
                [
                    self.reference(state, ref)[:, axis]
                    for axis, ref in enumerate(action.operands)
                ]
            ) + np.asarray(action.offsets)
            if (
                np.all(np.linalg.norm(target - ee, axis=1) < cfg.minimum_motion)
                or np.any(target[:, 2] < self.table_centres - cfg.geometry_tolerance)
                or np.any(target[:, 2] > cfg.max_height)
            ):
                return None
            if cfg.support_clearance:
                relative = ee[:, None] - p
                below = (
                    (abs(relative[:, :, 0]) < 2 * cfg.block_half_extent)
                    & (abs(relative[:, :, 1]) < 2 * cfg.block_half_extent)
                    & (relative[:, :, 2] > cfg.block_half_extent)
                    & (relative[:, :, 2] < 3 * cfg.block_half_extent)
                )
                lateral = np.linalg.norm(target[:, :2] - ee[:, :2], axis=1) > 0.004
                if np.any(held & lateral & np.any(below, axis=1)):
                    return None
            exempt = (np.arange(self.N)[None, :] == holding[:, None]) & held[:, None]
            payload_half = 2 * cfg.block_half_extent - cfg.geometry_tolerance
            if np.any(swept_hit(ee, target, p, payload_half) & ~exempt & held[:, None]):
                return None
            if cfg.gripper_envelope and np.any(
                swept_hit(
                    ee + [0, 0, 0.055], target + [0, 0, 0.055], p, [0.08, 0.08, 0.103]
                )
                & ~exempt
            ):
                return None
            new = p.copy()
            new[self.ix[held], holding[held]] = target[held]
            return NominalState(new, target, tuple(holding))
        if not np.all(held):
            return None
        if cfg.require_supported_release:
            source = p[self.ix, holding]
            on_table = abs(source[:, 2] - self.table_centres) <= cfg.geometry_tolerance
            difference = source[:, None] - p
            on_block = (
                (abs(difference[:, :, 0]) < cfg.block_half_extent)
                & (abs(difference[:, :, 1]) < cfg.block_half_extent)
                & (
                    abs(difference[:, :, 2] - 2 * cfg.block_half_extent)
                    <= cfg.geometry_tolerance
                )
            )
            if not np.all(on_table | np.any(on_block, axis=1)):
                return None
        target = ee.copy()
        target[:, 2] = (
            self.reference(state, action.operands[0])[:, 2] + action.offsets[0]
        )
        if np.any(target[:, 2] < ee[:, 2] + 0.02) or np.any(
            target[:, 2] > cfg.max_height
        ):
            return None
        exempt = np.arange(self.N)[None, :] == holding[:, None]
        if cfg.gripper_envelope and np.any(
            swept_hit(
                ee + [0, 0, 0.055], target + [0, 0, 0.055], p, [0.08, 0.08, 0.103]
            )
            & ~exempt
        ):
            return None
        return NominalState(p, target, (-1,) * self.E)

    def successors(self, state, deadline=None):
        if self.goal(state):
            return ()
        self.counts["successor_calls"] += 1
        successors = []
        # Moves dominate the finite corpus. Evaluate their identical entry-state
        # contracts together; Pick/Release retain their scalar implementations.
        references = {}
        for action in self.actions:
            if action.kind == "Move":
                for operand in action.operands:
                    if operand not in references:
                        references[operand] = self.reference(state, operand)
        for offset in range(0, len(self.actions), 64):
            self.check_deadline(deadline)
            chunk = self.actions[offset : offset + 64]
            truncated = False
            if self.transition_limit is not None:
                remaining = self.transition_limit - self.counts["transition_trials"]
                if remaining <= 0:
                    raise TransitionBudgetExceeded("Nominal transition budget reached")
                truncated = remaining < len(chunk)
                chunk = chunk[:remaining]
            move_actions = [action for action in chunk if action.kind == "Move"]
            move_results = iter(self._move_batch(state, move_actions, references))
            # All vectorized moves have now been computed, even if a deadline
            # interrupts consumption of this bounded batch below.
            self.counts["transition_trials"] += len(move_actions)
            for action in chunk:
                self.check_deadline(deadline)
                self.counts["action_trials"] += 1
                if action.kind == "Move":
                    result = next(move_results)
                else:
                    result = self.transition(state, action)
                if result is None:
                    self.counts["nominal_rejected"] += 1
                else:
                    successors.append((action, result))
            if truncated:
                raise TransitionBudgetExceeded("Nominal transition budget reached")
        self.counts["successors"] += len(successors)
        return tuple(successors)

    def _move_batch(self, state, actions, references):
        """Vectorize precisely the scalar Move contract, retaining action order."""
        if not actions:
            return ()
        cfg = self.config
        p, ee = state.positions, state.gripper
        holding = np.asarray(state.holding)
        held = holding >= 0
        target = np.asarray(
            [
                np.column_stack(
                    [
                        references[ref][:, axis]
                        for axis, ref in enumerate(action.operands)
                    ]
                )
                + np.asarray(action.offsets)
                for action in actions
            ]
        )
        invalid = (
            np.all(np.linalg.norm(target - ee, axis=2) < cfg.minimum_motion, axis=1)
            | np.any(
                target[:, :, 2] < self.table_centres - cfg.geometry_tolerance, axis=1
            )
            | np.any(target[:, :, 2] > cfg.max_height, axis=1)
        )
        if cfg.support_clearance:
            relative = ee[:, None] - p
            below = (
                (abs(relative[:, :, 0]) < 2 * cfg.block_half_extent)
                & (abs(relative[:, :, 1]) < 2 * cfg.block_half_extent)
                & (relative[:, :, 2] > cfg.block_half_extent)
                & (relative[:, :, 2] < 3 * cfg.block_half_extent)
            )
            lateral = np.linalg.norm(target[:, :, :2] - ee[:, :2], axis=2) > 0.004
            invalid |= np.any(held & lateral & np.any(below, axis=1), axis=1)
        exempt = (np.arange(self.N)[None, :] == holding[:, None]) & held[:, None]

        def collision(destinations, start, half):
            a = start[None, :, None, :]
            direction = (destinations - start)[:, :, None, :]
            zero = abs(direction) < 1e-10
            safe = np.where(zero, 1.0, direction)
            u, v = (p[None] - half - a) / safe, (p[None] + half - a) / safe
            lo = np.where(zero, -np.inf, np.minimum(u, v))
            hi = np.where(zero, np.inf, np.maximum(u, v))
            outside = np.any(zero & (abs(a - p[None]) >= half), axis=3)
            lower = np.maximum(np.max(lo, axis=3), 0)
            upper = np.minimum(np.min(hi, axis=3), 1)
            return (~outside) & (upper > np.maximum(lower, 1e-6)) & (lower < 1 - 1e-6)

        viable = np.flatnonzero(~invalid)
        if len(viable):
            destinations = target[viable]
            payload_half = 2 * cfg.block_half_extent - cfg.geometry_tolerance
            if np.any(held):
                invalid[viable] |= np.any(
                    collision(destinations, ee, payload_half) & ~exempt & held[:, None],
                    axis=(1, 2),
                )
            if cfg.gripper_envelope:
                invalid[viable] |= np.any(
                    collision(
                        destinations + [0, 0, 0.055],
                        ee + [0, 0, 0.055],
                        np.asarray([0.08, 0.08, 0.103]),
                    )
                    & ~exempt,
                    axis=(1, 2),
                )
        results = []
        for i in range(len(actions)):
            if invalid[i]:
                results.append(None)
            else:
                new = p.copy()
                new[self.ix[held], holding[held]] = target[i, held]
                results.append(NominalState(new, target[i], tuple(holding)))
        return tuple(results)

    def action_path(self, before, action, after):
        if action.kind == "Pick":
            middle = self.vector(before).copy()
            middle[:, :2] = after.gripper[:, :2]
            return np.stack([middle, self.vector(after)], axis=1)
        return self.vector(after)[:, None]

    def rank(self, node, complete=None):
        self.check_deadline()
        complete = self.goal(node.state) if complete is None else complete
        weights = self.weights[:, None]
        endpoint = node.path[:, -1]
        closest = (
            np.full(self.E, self.config.trace_samples - 1)
            if complete
            else np.argmin(
                np.sum(((self.demo - endpoint[:, None]) * weights) ** 2, axis=2), axis=1
            )
        )
        prefix = self.demo_prefix[self.ix, np.maximum(1, closest) - 1]
        distance = np.mean(
            ((resample_batch(node.path, self.config.rank_samples) - prefix) * weights)
            ** 2,
            axis=(1, 2),
        )
        travel = np.linalg.norm(np.diff(node.path[:, :, :3], axis=1), axis=2).sum(
            axis=1
        )
        endpoint_error = np.mean(
            ((endpoint - self.demo[:, -1]) * self.weights) ** 2, axis=1
        )
        progress = (
            0 if complete else 0.0003 * (1 - closest / (self.config.trace_samples - 1))
        )
        return float(
            np.mean(
                distance
                + 0.03 * endpoint_error
                + 0.00015 * travel
                + 0.000025 * len(node.actions)
                + progress
            )
        )

    def rank_many(self, nodes, complete=None, *, chunk_size=64):
        """The scalar history score evaluated in bounded NumPy batches.

        Grouping by path length changes neither the path samples nor the score.
        There is no state-only rank cache: equal endpoints reached by different
        histories retain their own scores. Local search can use ``rank`` while
        beam expansion amortizes array setup across its available children.
        """
        if type(chunk_size) is not int or chunk_size < 1:
            raise ContractInputError("Rank chunk_size must be positive")
        nodes = tuple(nodes)
        complete = (
            tuple(self.goal(node.state) for node in nodes)
            if complete is None
            else tuple(complete)
        )
        if len(complete) != len(nodes):
            raise ContractInputError("One completion flag per rank node required")
        result = np.empty(len(nodes))
        groups = defaultdict(list)
        for i, node in enumerate(nodes):
            groups[node.path.shape[1]].append(i)
        for indices in groups.values():
            for offset in range(0, len(indices), chunk_size):
                self.check_deadline()
                selected = indices[offset : offset + chunk_size]
                paths = np.stack([nodes[i].path for i in selected])
                batch, _, steps, features = paths.shape
                terminal = np.asarray([complete[i] for i in selected])
                endpoint = paths[:, :, -1]
                closest = np.argmin(
                    np.sum(
                        (
                            (self.demo[None] - endpoint[:, :, None])
                            * self.weights[None, :, None]
                        )
                        ** 2,
                        axis=3,
                    ),
                    axis=2,
                )
                closest[terminal] = self.config.trace_samples - 1
                prefix = self.demo_prefix[self.ix[None], np.maximum(1, closest) - 1]
                sampled = resample_batch(
                    paths.reshape(batch * self.E, steps, features),
                    self.config.rank_samples,
                ).reshape(batch, self.E, self.config.rank_samples, features)
                distance = np.mean(
                    ((sampled - prefix) * self.weights[None, :, None]) ** 2,
                    axis=(2, 3),
                )
                travel = np.linalg.norm(
                    np.diff(paths[:, :, :, :3], axis=2), axis=3
                ).sum(axis=2)
                endpoint_error = np.mean(
                    ((endpoint - self.demo[None, :, -1]) * self.weights[None]) ** 2,
                    axis=2,
                )
                progress = 0.0003 * (1 - closest / (self.config.trace_samples - 1))
                progress[terminal] = 0
                lengths = np.asarray([len(nodes[i].actions) for i in selected])
                result[selected] = np.mean(
                    distance
                    + 0.03 * endpoint_error
                    + 0.00015 * travel
                    + 0.000025 * lengths[:, None]
                    + progress,
                    axis=1,
                )
        return result

    def replay(self, actions):
        state = self.initial
        path = self.vector(state)[:, None]
        completed = []
        for action in actions:
            self.check_deadline()
            successor = self.transition(state, action)
            if successor is None:
                return None
            path = np.concatenate(
                [path, self.action_path(state, action, successor)], axis=1
            )
            completed.append(action)
            state = successor
        node = Node(state, tuple(completed), path)
        node.score = self.rank(node)
        return node

    def diversity_key(self, state):
        # Coarse progress observations are heuristics, not a dominance theorem.
        source = state.positions[self.ix, self.dominant]
        initial = self.initial.positions[self.ix, self.dominant]
        separated = np.max(abs(state.positions[:, :, :2] - source[:, None, :2]), axis=2)
        separated[self.ix, self.dominant] = np.inf
        return (
            state.holding,
            bool(np.all(source[:, 2] > initial[:, 2] + 0.02)),
            bool(np.all(np.min(separated, axis=1) >= 0.1 - 1e-7)),
            bool(np.all(abs(source[:, 2] - self.table_centres) <= 0.001)),
            int(np.sum(self.goal_mask(state))),
        )


@dataclass(frozen=True)
class SearchConfig:
    max_length: int = 8
    beam_width: int = 24
    max_expansions: int = 300
    max_transition_trials: int = 100000
    seconds: float = 35.0
    memoize: bool = True
    diversity: bool = True
    pool_limit: int = 12
    deadline: float | None = None

    def __post_init__(self):
        if type(self.max_length) is not int or self.max_length < 0:
            raise ContractInputError("max_length must be a nonnegative integer")
        for field_name in (
            "beam_width",
            "max_expansions",
            "max_transition_trials",
            "pool_limit",
        ):
            if (
                type(getattr(self, field_name)) is not int
                or getattr(self, field_name) <= 0
            ):
                raise ContractInputError(f"{field_name} must be a positive integer")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ContractInputError("seconds must be positive and finite")


@dataclass
class SearchResult:
    candidates: tuple[Node, ...]
    stats: dict
    status: str


def beam_search(
    problem, config=None, *, on_candidate: Callable[[Node], object] | None = None
):
    """Bounded history-sensitive beam; exact successor caching only.

    The optional callback observes each nominal goal candidate. Returning False
    stops search. Callback work counts against the same absolute deadline. A
    timeout returns only fully constructed candidates, never a partial prefix
    represented as a goal. Candidate existence is not physical acceptance.
    """
    cfg = config or SearchConfig()
    began = time.perf_counter()
    limits = [began + cfg.seconds] + [
        x for x in (cfg.deadline, problem.deadline) if x is not None
    ]
    deadline = min(limits)
    stats = Counter()
    before_counts = problem.counts.copy()
    previous_limit = problem.transition_limit
    requested_limit = problem.counts["transition_trials"] + cfg.max_transition_trials
    problem.transition_limit = (
        min(requested_limit, previous_limit)
        if previous_limit is not None
        else requested_limit
    )
    initial = Node(problem.initial, (), problem.vector(problem.initial)[:, None])
    beam = [initial]
    completed, best_by_length, memo = [], {}, {}
    status = "length_exhausted"
    first_goal = None
    stopped = False
    try:
        problem.check_deadline(deadline)
        if problem.config.allow_empty_goal and problem.goal(problem.initial):
            initial.score = problem.rank(initial, complete=True)
            completed.append(initial)
            best_by_length[0] = initial
            first_goal = time.perf_counter() - problem.started
            if on_candidate is not None and on_candidate(initial) is False:
                status = "callback_stopped"
                stopped = True
        for depth in range(cfg.max_length):
            if stopped:
                break
            children = []
            for node in beam:
                problem.check_deadline(deadline)
                if stats["expanded"] >= cfg.max_expansions:
                    status = "expansion_budget_exhausted"
                    stopped = True
                    break
                stats["expanded"] += 1
                key = state_key(node.state)
                if cfg.memoize and key in memo:
                    successors = memo[key]
                    stats["memo_hits"] += 1
                else:
                    successors = problem.successors(node.state, deadline)
                    if cfg.memoize:
                        memo[key] = successors
                pending, terminal = [], []
                for action, state in successors:
                    problem.check_deadline(deadline)
                    stats["generated"] += 1
                    goal = problem.goal(state)
                    if not goal and depth + 1 == cfg.max_length:
                        stats["backward_rejected"] += 1
                        continue
                    child = Node(
                        state,
                        node.actions + (action,),
                        np.concatenate(
                            [node.path, problem.action_path(node.state, action, state)],
                            axis=1,
                        ),
                    )
                    pending.append(child)
                    terminal.append(goal)
                scores = problem.rank_many(pending, terminal)
                for child, goal, score in zip(pending, terminal, scores):
                    problem.check_deadline(deadline)
                    child.score = float(score)
                    if goal:
                        completed.append(child)
                        stats["goal_candidates"] += 1
                        if first_goal is None:
                            first_goal = time.perf_counter() - problem.started
                        length = len(child.actions)
                        if (
                            length not in best_by_length
                            or child.score < best_by_length[length].score
                        ):
                            best_by_length[length] = child
                        if on_candidate is not None and on_candidate(child) is False:
                            status = "callback_stopped"
                            stopped = True
                            break
                    else:
                        children.append(child)
                if stopped:
                    break
            children.sort(key=lambda n: (n.score, repr(n.actions)))
            groups, seen = defaultdict(list), set()
            for node in children:
                problem.check_deadline(deadline)
                key = node.state.holding
                cap = max(1, cfg.beam_width // 2)
                if cfg.diversity:
                    exact = state_key(node.state)
                    if exact in seen:
                        stats["diversity_duplicate_pruned"] += 1
                        continue
                    seen.add(exact)
                    key = problem.diversity_key(node.state)
                    cap = max(1, cfg.beam_width // 8)
                if len(groups[key]) < cap:
                    groups[key].append(node)
            beam = sorted(
                [node for nodes in groups.values() for node in nodes],
                key=lambda n: (n.score, repr(n.actions)),
            )[: cfg.beam_width]
            stats["beam_discarded"] += len(children) - len(beam)
            stats["depth"] = depth + 1
            if not beam and not stopped:
                status = "frontier_exhausted"
                break
    except DeadlineExceeded:
        status = "deadline_exhausted"
    except TransitionBudgetExceeded:
        status = "transition_budget_exhausted"
    finally:
        problem.transition_limit = previous_limit
    pool = select_candidates(completed, cfg.pool_limit)
    stats.update(problem.counts - before_counts)
    stats = dict(stats)
    stats.update(
        seconds=time.perf_counter() - began,
        setup_seconds=problem.setup_seconds,
        total_seconds=time.perf_counter() - problem.started,
        first_goal_seconds=first_goal,
        candidate_lengths=sorted(best_by_length),
        shortlist_size=len(pool),
        status=status,
    )
    return SearchResult(tuple(pool), dict(stats), status)


def select_candidates(nodes, limit=12):
    """Common finite shortlist for beam and matched local-search comparisons.

    Retain the best representative of each observed length first, then fill by
    score with distinct action/operand structures. Numeric offsets do not define
    a new structure; the lowest-scoring offsets for that structure are retained.
    """
    if type(limit) is not int or limit <= 0:
        raise ContractInputError("Candidate limit must be a positive integer")
    completed = list(nodes)
    best_by_length = {}
    for node in completed:
        length = len(node.actions)
        if length not in best_by_length or (node.score, repr(node.actions)) < (
            best_by_length[length].score,
            repr(best_by_length[length].actions),
        ):
            best_by_length[length] = node
    completed.sort(key=lambda node: (node.score, len(node.actions), repr(node.actions)))
    pool, signatures = [], set()
    for node in (
        sorted(
            best_by_length.values(), key=lambda node: (node.score, len(node.actions))
        )
        + completed
    ):
        signature = tuple((action.kind, action.operands) for action in node.actions)
        if signature not in signatures:
            signatures.add(signature)
            pool.append(node)
        if len(pool) >= limit:
            break
    return tuple(pool)
