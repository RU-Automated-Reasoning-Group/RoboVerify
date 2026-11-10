"""Matched Metropolis structural search over the beam backend's exact grammar.

This is a controlled search-algorithm baseline, not the unchanged legacy MCMC
pipeline. Partial nominal programs receive the same prefix rank as beam nodes;
only terminal goal candidates enter the shared physical/CEM evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np

from synthesis.mcmc.search_core import acceptance_probability, temperature_at


@dataclass(frozen=True)
class MatchedConfig:
    max_length: int = 8
    max_iterations: int = 100000
    max_transition_trials: int = 100000
    seconds: float = 20.0
    seed: int = 0
    temperature: float = 0.01
    decay: float = 0.999
    floor: float = 0.001
    shortlist: int = 12

    def __post_init__(self):
        if (
            self.max_length < 1
            or self.max_iterations < 1
            or self.max_transition_trials < 1
            or self.seconds <= 0
            or self.shortlist < 1
        ):
            raise ValueError("Positive matched-search budgets required")
        temperature_at(0, self.temperature, self.decay, self.floor)


@lru_cache(maxsize=8)
def _action_families(alphabet):
    families = {}
    for item in alphabet:
        families.setdefault(getattr(item, "kind", "action"), []).append(item)
    return tuple(tuple(items) for items in families.values())


def mutate_actions(actions, alphabet, rng, max_length):
    """One insert/delete/replace/order edit; every alphabet member is reachable."""
    if not alphabet or max_length < 1 or len(actions) > max_length:
        raise ValueError("Nonempty alphabet and legal length required")
    result = list(actions)
    kinds = ["replace", "insert", "delete", "swap"]
    weights = np.asarray([0.4, 0.3, 0.15, 0.15])
    allowed = [
        len(result) > 0,
        len(result) < max_length,
        len(result) > 0,
        len(result) > 1,
    ]
    probabilities = weights * allowed
    probabilities /= probabilities.sum()
    operation = str(rng.choice(kinds, p=probabilities))
    # The full grammar has hundreds of coordinate variants but few Pick
    # variants. Balance primitive families as production mutations do, so an
    # action's physical role does not vanish because its offset grid is smaller.
    choices = _action_families(tuple(alphabet))

    def action():
        family = choices[int(rng.integers(len(choices)))]
        return family[int(rng.integers(len(family)))]

    if operation == "replace":
        result[int(rng.integers(len(result)))] = action()
    elif operation == "insert":
        result.insert(int(rng.integers(len(result) + 1)), action())
    elif operation == "delete":
        result.pop(int(rng.integers(len(result))))
    else:
        first, second = rng.choice(len(result), 2, replace=False)
        result[first], result[second] = result[second], result[first]
    return tuple(result), operation


def shortlist(nodes, limit=12):
    """Same endpoint selector for both arms; no physical outcome is consulted."""
    from synthesis.search.contract import select_candidates

    return select_candidates(nodes, limit=limit)


def search(problem, config=None, *, deadline=None):
    from synthesis.search.contract import TransitionBudgetExceeded

    config = config or MatchedConfig()
    start = perf_counter()
    stop = min(
        start + config.seconds, deadline if deadline is not None else float("inf")
    )
    rng = np.random.default_rng(config.seed)
    previous_limit = problem.transition_limit
    requested_limit = problem.counts["transition_trials"] + config.max_transition_trials
    problem.transition_limit = (
        min(previous_limit, requested_limit)
        if previous_limit is not None
        else requested_limit
    )
    cache, goals = {}, {}
    stats = dict(
        iterations=0,
        proposed=0,
        nominal_rejected=0,
        cache_hits=0,
        accepted=0,
        goal_candidates=0,
        transition_steps_requested=0,
        empty_initial_program=True,
        progress=[],
        current_length_visits={},
        valid_proposal_lengths={},
        accepted_lengths={},
        action_sampling="uniform primitive family, then uniform family member",
    )
    best_rank, first_goal = None, None
    status = "iteration_limit"
    try:
        current = problem.replay(())
        if current is None:
            raise ValueError("The declared initial nominal state must be admissible")
        initial_goal = problem.goal(current.state)
        current.score = problem.rank(current, complete=initial_goal)
        cache[()] = current
        best_rank = float(current.score)
        if initial_goal and getattr(
            getattr(problem, "config", None), "allow_empty_goal", True
        ):
            goals[()] = current
            first_goal = perf_counter() - getattr(problem, "started", start)
        for iteration in range(config.max_iterations):
            if perf_counter() >= stop:
                status = "deadline"
                break
            length = str(len(current.actions))
            stats["current_length_visits"][length] = (
                stats["current_length_visits"].get(length, 0) + 1
            )
            actions, operation = mutate_actions(
                current.actions, problem.actions, rng, config.max_length
            )
            stats["proposed"] += 1
            stats[operation + "_proposals"] = stats.get(operation + "_proposals", 0) + 1
            stats["iterations"] = iteration + 1
            if actions in cache:
                node = cache[actions]
                stats["cache_hits"] += 1
            else:
                stats["transition_steps_requested"] += len(actions)
                node = problem.replay(actions)
                cache[actions] = node
            if node is None:
                stats["nominal_rejected"] += 1
                continue
            terminal = problem.goal(node.state)
            length = str(len(node.actions))
            stats["valid_proposal_lengths"][length] = (
                stats["valid_proposal_lengths"].get(length, 0) + 1
            )
            node.score = problem.rank(node, complete=terminal)
            if terminal and (
                node.actions
                or getattr(getattr(problem, "config", None), "allow_empty_goal", True)
            ):
                goals[actions] = node
                stats["goal_candidates"] += 1
                if first_goal is None:
                    first_goal = perf_counter() - getattr(problem, "started", start)
            temperature = temperature_at(
                iteration, config.temperature, config.decay, config.floor
            )
            probability = acceptance_probability(
                current.score - node.score, temperature
            )
            if rng.random() < probability:
                current = node
                stats["accepted"] += 1
                stats["accepted_lengths"][length] = (
                    stats["accepted_lengths"].get(length, 0) + 1
                )
            if node.score < best_rank:
                best_rank = float(node.score)
                stats["progress"].append(
                    dict(
                        iteration=iteration,
                        seconds=perf_counter() - start,
                        rank=best_rank,
                        length=len(node.actions),
                        terminal=bool(terminal),
                    )
                )
    except TimeoutError:
        status = "deadline"
    except TransitionBudgetExceeded:
        status = "transition_limit"
    finally:
        problem.transition_limit = previous_limit
    candidates = shortlist(goals.values(), config.shortlist)
    stats.update(
        seconds=perf_counter() - start,
        cached_programs=len(cache),
        best_nominal_rank=best_rank,
        candidate_count=len(candidates),
        first_goal_seconds=first_goal,
        unique_goal_programs=len(goals),
    )
    stats["transition_trials"] = problem.counts["transition_trials"]
    return candidates, stats, status
