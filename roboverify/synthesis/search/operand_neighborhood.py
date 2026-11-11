"""Bounded reference alternatives after nominal search, before physical checks.

Equal nominal states can conceal different controller dependencies. Keep close
goal programs using different coordinate references available to BMC, endpoint
audits and full-program validation. These proposals make no physical claim.
"""

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from synthesis.search.contract import (
    Action,
    ContractInputError,
    DeadlineExceeded,
    SearchResult,
    TransitionBudgetExceeded,
)


@dataclass(frozen=True)
class OperandNeighborhoodConfig:
    max_candidates: int = 4096
    max_transition_trials: int = 100000
    seconds: float = 5.0
    pool_limit: int = 8
    relative_slack: float = 0.01

    def __post_init__(self):
        for name in ("max_candidates", "max_transition_trials", "pool_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractInputError(f"{name} must be a positive integer")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ContractInputError("Neighborhood seconds must be positive and finite")
        if not math.isfinite(self.relative_slack) or self.relative_slack < 0:
            raise ContractInputError(
                "Nominal relative slack must be finite and nonnegative"
            )


def _signature(node):
    return tuple((action.kind, action.operands) for action in node.actions)


def _order(node):
    return node.score, len(node.actions), repr(node.actions)


def _coherent_xy(node):
    # Index within the same action-kind skeleton identifies a Move site. The
    # reference itself can be any permitted name/ID; no task role is preferred.
    skeleton = tuple(action.kind for action in node.actions)
    return {
        (skeleton, index, action.operands[0])
        for index, action in enumerate(node.actions)
        if action.kind == "Move" and action.operands[0] == action.operands[1]
    }


def select_reference_diverse(nodes, *, limit=8, relative_slack=0.01):
    """Cover close coherent XY frames at each Move site, then fill by rank.

    Every selected program lies within ``best + abs(best)*relative_slack`` of
    the best complete nominal imitation score. This dimensionless proposal
    filter is separate from the physical distance's theta/epsilon thresholds.
    Numeric variants of one operand structure retain their lowest score.
    """
    OperandNeighborhoodConfig(pool_limit=limit, relative_slack=relative_slack)
    ranked = sorted((node for node in nodes if math.isfinite(node.score)), key=_order)
    if not ranked:
        return ()
    cutoff = ranked[0].score + abs(ranked[0].score) * relative_slack
    unique = {}
    for node in ranked:
        if node.score <= cutoff:
            unique.setdefault(_signature(node), node)
    remaining = list(unique.values())
    selected = [remaining.pop(0)]
    covered = _coherent_xy(selected[0])
    sites = sorted(
        {
            (skeleton, index)
            for node in remaining
            for skeleton, index, _ in _coherent_xy(node)
        }
    )
    while len(selected) < limit:
        changed = False
        for skeleton, index in sites:
            choices = [
                node
                for node in remaining
                if any(
                    token[:2] == (skeleton, index) and token not in covered
                    for token in _coherent_xy(node)
                )
            ]
            if not choices:
                continue
            node = min(choices, key=_order)
            selected.append(node)
            remaining = [other for other in remaining if other is not node]
            covered.update(_coherent_xy(node))
            changed = True
            if len(selected) == limit:
                break
        if not changed:
            break
    selected.extend(remaining[: limit - len(selected)])
    return tuple(selected)


def expand_operand_neighborhood(problem, goal_nodes, config=None):
    """Recheck seeds and one-action operand products under the unchanged goal.

    All coordinates of one primitive may change reference together; action
    kinds, lengths and numeric offsets remain fixed. Shared absolute deadlines
    and transition limits are preserved. Exhaustion returns only complete,
    fully scored nominal goal candidates, including qualifying original seeds.
    If the original deadline is already exhausted, no unchecked seed is
    promoted. Callers owning already validated beam goals can retain those
    originals when this optional expansion returns an empty shortlist.
    """
    cfg = config or OperandNeighborhoodConfig()
    began = perf_counter()
    original_deadline, original_limit = problem.deadline, problem.transition_limit
    deadline = problem.deadline = min(
        value for value in (original_deadline, began + cfg.seconds) if value is not None
    )
    requested_limit = problem.counts["transition_trials"] + cfg.max_transition_trials
    problem.transition_limit = min(
        value for value in (original_limit, requested_limit) if value is not None
    )
    before_counts = problem.counts.copy()
    stats, seen, completed = Counter(), set(), []
    status = "neighborhood_exhausted"

    def replay(actions, *, seed=False):
        nonlocal status
        problem.check_deadline()
        if actions in seen:
            return None
        if stats["candidates_checked"] >= cfg.max_candidates:
            status = "candidate_budget_exhausted"
            return False
        seen.add(actions)
        stats["candidates_checked"] += 1
        node = problem.replay(actions)
        if node is None:
            stats["nominal_rejected"] += 1
        elif not problem.goal(node.state) or (
            not node.actions and not problem.config.allow_empty_goal
        ):
            stats["goal_rejected"] += 1
        else:
            # replay() uses the existing complete-path imitation rank whenever
            # goal() holds. Never substitute a local/prefix score here.
            completed.append(node)
            stats["goal_candidates"] += 1
            if seed:
                stats["seed_goals"] += 1
            return node
        return None

    try:
        seeds = []
        for original in goal_nodes:
            node = replay(tuple(original.actions), seed=True)
            if node is False:
                break
            if node is not None:
                seeds.append(node)
        if status != "candidate_budget_exhausted":
            for seed in seeds:
                for index, action in enumerate(seed.actions):
                    for operands in itertools.product(
                        *problem.operand_references(action.kind)
                    ):
                        actions = list(seed.actions)
                        actions[index] = Action(
                            action.kind,
                            operands,
                            action.offsets,
                            "operand-neighborhood",
                        )
                        if replay(tuple(actions)) is False:
                            break
                    if status == "candidate_budget_exhausted":
                        break
                if status == "candidate_budget_exhausted":
                    break
    except DeadlineExceeded:
        status = "deadline_exhausted"
    except TransitionBudgetExceeded:
        status = "transition_budget_exhausted"
    finally:
        problem.deadline, problem.transition_limit = original_deadline, original_limit
    pool = select_reference_diverse(
        completed, limit=cfg.pool_limit, relative_slack=cfg.relative_slack
    )
    if status == "neighborhood_exhausted" and perf_counter() >= deadline:
        status = "deadline_exhausted"
    stats.update(problem.counts - before_counts)
    stats = dict(stats)
    stats.update(
        shortlist_size=len(pool), seconds=perf_counter() - began, status=status
    )
    return SearchResult(pool, dict(stats), status)
