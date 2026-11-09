"""Mutation or enumeration with CEM, epsilon pool, then PostScore ranking."""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from synthesis.cfg.execute import execute_current
from synthesis.cfg.refine import scene_at
from synthesis.cfg.reset import reset_segment
from synthesis.entry.search_options import validate_search_strategy
from synthesis.mcmc.distance import make_distance
from synthesis.mcmc.search_core import (
    CandidatePool,
    acceptance_probability,
    imitation_objective,
    temperature_at,
)
from synthesis.predicates.scene import Scene, evaluate, scene_from_obs
from synthesis.util import on


@dataclass(frozen=True)
class SearchBudget:
    iterations: int = 20
    cem_iterations: int = 2
    cem_samples: int = 8
    cem_elites: int = 2
    theta: float = 1.0
    epsilon: float = 0.1
    pool_limit: int = 10
    distance: str = "kl_kde"
    inner_samples: int = 128
    final_samples: int = 2048
    seed: int = 0
    temperature: float = 1.0
    decay: float = 0.95
    search_strategy: str = "mcmc"

    def __post_init__(self):
        validate_search_strategy(self.search_strategy)


@dataclass
class StraightLineResult:
    program: object
    ok: bool
    status: str
    distance: float
    post_score: float = 0.0
    elapsed: float = 0.0
    postscore_seconds: float = 0.0
    iterations: int = 0
    candidates_checked: int = 0
    bmc_rejected: int = 0


class SegmentRollout(list):
    """Predicate scenes, including binding boundaries, plus recorded observations.

    Boundary callbacks do not advance the simulation. Keep them for predicates,
    but score only the observation sequence returned by program execution.
    """

    def __init__(self):
        super().__init__()
        self.observations = []


def _features(states, num_blocks):
    if isinstance(states, SegmentRollout):
        states = states.observations
    if isinstance(states[0], Scene):
        if states[0].observation is not None:
            if any(s.observation is None for s in states):
                raise ValueError("Cannot mix recorded and observation-free scenes")
            return np.asarray([s.observation for s in states])[
                :, on.state_comparison_indices(num_blocks)
            ]
        return np.asarray(
            [
                [
                    value
                    for key in sorted(
                        s.positions, key=lambda key: (isinstance(key, str), key)
                    )
                    if key != "tbl"
                    for value in s.positions[key]
                ]
                for s in states
            ],
            dtype=float,
        )
    return np.asarray(states)[:, on.state_comparison_indices(num_blocks)]


def segment_rollout(program, segment, env_factory, reset_mode="replay"):
    from synthesis.mcmc.synthesis import preserved_global_rng, set_np_seed

    with preserved_global_rng():
        set_np_seed(segment.trace.seed)
        env = env_factory(segment.trace)
        try:
            first = reset_segment(env, segment, mode=reset_mode)
            scenes = SegmentRollout()
            entry = scene_at(segment, segment.entry_index).entry_positions

            def record(obs, bindings):
                scene = scene_from_obs(
                    obs,
                    segment.trace.num_blocks,
                    bindings,
                    include_table="tbl" in bindings,
                )
                scenes.append(
                    Scene(scene.positions, scene.bindings, entry, scene.observation)
                )

            scenes.observations = list(
                execute_current(program, env, first, on_state=record)
            )
            return scenes
        finally:
            env.close()


def postcondition_reached(states, segment, post):
    """Evaluate rollout states with this invocation's recorded frozen geometry."""
    entry_positions = scene_at(segment, segment.entry_index).entry_positions
    for state in states:
        scene = (
            state
            if isinstance(state, Scene)
            else scene_from_obs(
                state,
                segment.trace.num_blocks,
                segment.bindings,
                include_table="tbl" in segment.bindings,
            )
        )
        # Copy rather than mutate caller-owned Scene snapshots. Rollout positions
        # and bindings are current; ON_star_zero always uses the recorded entry.
        scene = Scene(scene.positions, scene.bindings, entry_positions)
        if evaluate(post, scene):
            return True
    return False


def straight_line_synthesize(
    segments,
    post,
    initial_program,
    propose,
    *,
    rollout,
    budget=None,
    logger=None,
    block_id="block",
    penalty=None,
    penalty_weight=1.0,
    candidates=None,
    feasible=None,
):
    budget = budget or SearchBudget()
    enumeration = budget.search_strategy == "enumeration"
    if enumeration and (candidates is None or feasible is None):
        raise ValueError("Enumeration requires a candidate iterator and BMC screening")
    candidate_iterator = iter(candidates) if enumeration else None
    if (
        not segments
        or budget.iterations < 0
        or not 1 <= budget.cem_elites <= budget.cem_samples
    ):
        raise ValueError("Nonempty segments and valid search budgets required")
    start = perf_counter()
    rng = np.random.default_rng(budget.seed)
    demos = np.concatenate([_features(s.states, s.trace.num_blocks) for s in segments])
    distance = make_distance(budget.distance, demos)
    pool = CandidatePool(budget.epsilon, budget.pool_limit)

    def trajectories(candidate):
        return [rollout(candidate, s) for s in segments]

    def measure(candidate, samples):
        values = trajectories(candidate)
        return distance(
            np.concatenate(
                [_features(t, s.trace.num_blocks) for t, s in zip(values, segments)]
            ),
            samples=samples,
        )

    def objective(candidate):
        failures = 0 if penalty is None else penalty(candidate)
        return imitation_objective(
            measure(candidate, budget.inner_samples), failures, penalty_weight
        )

    def optimize(candidate):
        mean = np.asarray(candidate.register_trainable_parameter(), dtype=float)
        std = np.full(len(mean), 0.05)
        best, best_score = deepcopy(candidate), objective(candidate)
        for _ in range(budget.cem_iterations if len(mean) else 0):
            scored = []
            for params in rng.normal(mean, std, (budget.cem_samples, len(mean))):
                trial = deepcopy(candidate)
                trial.update_trainable_parameter(list(params))
                score = objective(trial)
                scored.append((score, params))
                if score > best_score:
                    best, best_score = trial, score
            elite = np.asarray(
                [
                    p
                    for _, p in sorted(scored, key=lambda pair: pair[0], reverse=True)[
                        : budget.cem_elites
                    ]
                ]
            )
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 1e-6)
        return best, best_score

    def post_score(candidate):
        success = []
        for states, segment in zip(trajectories(candidate), segments):
            success.append(postcondition_reached(states, segment, post))
        return float(np.mean(success))

    current = deepcopy(initial_program)
    current_score = -float("inf") if enumeration else objective(current)
    last_distance = float("inf")
    checked, rejected = 0, 0
    status, iteration = "budget_exhausted", 0
    for iteration in range(budget.iterations + 1):
        if enumeration:
            try:
                candidate = (
                    deepcopy(initial_program)
                    if iteration == 0
                    else next(candidate_iterator)
                )
            except StopIteration:
                status = "space_exhausted"
                break
            check = feasible(candidate)
            checked += 1
            if not check:
                rejected += 1
                if logger:
                    logger.log_metrics(
                        iteration,
                        phase="straightline",
                        block_id=block_id,
                        search_strategy="enumeration",
                        bmc_feasible=False,
                        bmc_reason=getattr(check, "reason", "BMC rejected"),
                    )
                    logger.set_progress(
                        iteration,
                        budget.iterations + 1,
                        phase="straightline",
                        block_id=block_id,
                    )
                continue
            current, current_score = optimize(candidate)
        elif iteration:
            candidate, score = optimize(propose(deepcopy(current), rng))
            temperature = temperature_at(
                iteration - 1, budget.temperature, budget.decay
            )
            if rng.random() < acceptance_probability(
                score - current_score, temperature
            ):
                current, current_score = candidate, score
        last_distance = measure(current, budget.final_samples)
        pool.add(current, last_distance)
        if logger:
            logger.log_metrics(
                iteration,
                phase="straightline",
                block_id=block_id,
                distance=last_distance,
                pool_size=len(pool.entries),
                cost=current_score,
                search_strategy=budget.search_strategy,
                bmc_feasible=True if enumeration else None,
            )
            logger.set_progress(
                iteration,
                budget.iterations + 1,
                phase="straightline",
                block_id=block_id,
            )
        if last_distance < budget.theta:
            ranking_start = perf_counter()
            winner, success = pool.select(post_score)
            ranking_seconds = perf_counter() - ranking_start
            # Imitation convergence alone is not task success. Fail closed when
            # every close candidate misses a demonstrated postcondition.
            if success == 1.0:
                return StraightLineResult(
                    winner,
                    True,
                    "converged",
                    last_distance,
                    success,
                    perf_counter() - start,
                    ranking_seconds,
                    iteration,
                    checked,
                    rejected,
                )
    if enumeration and pool.entries:
        current = deepcopy(pool.entries[0].program)
        last_distance = pool.entries[0].distance
    elif enumeration:
        status = "no_feasible_candidate"
    return StraightLineResult(
        current,
        False,
        status,
        last_distance,
        elapsed=perf_counter() - start,
        iterations=max(0, checked - 1) if enumeration else iteration,
        candidates_checked=checked,
        bmc_rejected=rejected,
    )


StraightLineSynthesize = straight_line_synthesize
