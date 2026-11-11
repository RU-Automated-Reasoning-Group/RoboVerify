"""Structural proposals with CEM, epsilon pool, then PostScore ranking."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
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
    contract_beam_width: int = 24
    contract_max_expansions: int = 300
    contract_seconds: float = 35.0
    contract_observation_proposals: bool = True
    contract_memoize: bool = True
    contract_diversity: bool = True
    contract_pool_limit: int = 12
    cem_std: float = 0.05
    contract_endpoint_checks: bool = False
    contract_solve_offsets: bool = False
    contract_solver_timeout_ms: int = 250
    contract_solver_attempts: int = 24
    contract_offset_radius: float = 0.02

    def __post_init__(self):
        validate_search_strategy(self.search_strategy)
        if (
            type(self.contract_solver_timeout_ms) is not int
            or self.contract_solver_timeout_ms < 1
            or type(self.contract_solver_attempts) is not int
            or self.contract_solver_attempts < 1
            or not np.isfinite(self.contract_offset_radius)
            or self.contract_offset_radius <= 0
        ):
            raise ValueError(
                "Contract solver budgets and offset radius must be positive"
            )
        if not np.isfinite(self.cem_std) or self.cem_std <= 0:
            raise ValueError("CEM standard deviation must be positive and finite")
        if self.search_strategy == "contract" and (
            self.contract_beam_width < 1
            or self.contract_max_expansions < 1
            or not np.isfinite(self.contract_seconds)
            or self.contract_seconds <= 0
            or self.contract_pool_limit < 1
        ):
            raise ValueError("Contract proposal budgets must be positive and finite")


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
    search_stats: dict = field(default_factory=dict)
    search_reason: str = ""


class SegmentRollout(list):
    """Predicate scenes, including binding boundaries, plus recorded observations.

    Boundary callbacks do not advance the simulation. Keep them for predicates,
    but score only the observation sequence returned by program execution.
    """

    def __init__(self):
        super().__init__()
        self.observations = []
        self.contract_audit = None


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


def _observations(states):
    """Keep raw per-episode observations for an ordered spatial metric."""
    if isinstance(states, SegmentRollout):
        states = states.observations
    if not len(states):
        raise ValueError("Arc distance requires nonempty recorded observations")
    if isinstance(states[0], Scene):
        if any(s.observation is None for s in states):
            raise ValueError("Arc distance requires recorded gripper observations")
        return np.asarray([s.observation for s in states])
    return np.asarray(states)


def segment_rollout(
    program,
    segment,
    env_factory,
    reset_mode="replay",
    *,
    audit_contract=False,
    frame_tolerance_m=0.005,
):
    from synthesis.mcmc.synthesis import preserved_global_rng, set_np_seed

    with preserved_global_rng():
        set_np_seed(segment.trace.seed)
        env = env_factory(segment.trace)
        try:
            first = reset_segment(env, segment, mode=reset_mode)
            scenes = SegmentRollout()
            recorder = None
            if audit_contract:
                from synthesis.cfg.endpoint_contract import EndpointRecorder

                recorder = EndpointRecorder(
                    env, segment, first, frame_tolerance_m=frame_tolerance_m
                )
            entry = scene_at(segment, segment.entry_index).entry_positions

            def record(obs, bindings):
                if recorder is not None:
                    recorder.observe(obs)
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
                execute_current(
                    program,
                    env,
                    first,
                    on_state=record,
                    on_instruction=None if recorder is None else recorder.instruction,
                )
            )
            if recorder is not None:
                scenes.contract_audit = recorder.finish()
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
    cache_rollouts=False,
    endpoint_acceptance=None,
):
    """Search a block; optional caching requires deterministic ``rollout``.

    The integrated CFG runner restores and seeds every segment execution. It
    can reuse those physical trajectories while drawing fresh KL estimates.
    Custom callers retain uncached behavior unless they explicitly opt in.
    """
    budget = budget or SearchBudget()
    candidate_stream = budget.search_strategy in ("enumeration", "contract")
    if candidate_stream and (candidates is None or feasible is None):
        raise ValueError(
            f"{budget.search_strategy.capitalize()} requires a candidate iterator and BMC screening"
        )
    candidate_iterator = iter(candidates) if candidate_stream else None
    rollout_cache = OrderedDict()
    cache_counts = {"rollout_cache_hits": 0, "rollout_cache_misses": 0}
    endpoint_counts = {}

    def proposal_stats():
        return {
            **dict(getattr(candidates, "stats", {})),
            **(cache_counts if cache_rollouts else {}),
            **(
                {"endpoint_rejections": dict(endpoint_counts)}
                if endpoint_counts
                else {}
            ),
        }

    def proposal_reason():
        return str(getattr(candidates, "reason", ""))

    if (
        not segments
        or budget.iterations < 0
        or not 1 <= budget.cem_elites <= budget.cem_samples
    ):
        raise ValueError("Nonempty segments and valid search budgets required")
    start = perf_counter()
    rng = np.random.default_rng(budget.seed)
    if budget.distance == "arc":
        # Different scenes and segment boundaries are separate trajectories.
        # Concatenating them would invent spatial jumps between demonstrations.
        distances = [
            make_distance("arc", _observations(s.states), num_blocks=s.trace.num_blocks)
            for s in segments
        ]
    else:
        demos = np.concatenate(
            [_features(s.states, s.trace.num_blocks) for s in segments]
        )
        distance = make_distance(budget.distance, demos)
    pool_key = str
    if budget.search_strategy == "contract":
        from synthesis.cfg.program_source import program_fingerprint

        pool_key = program_fingerprint
    pool = CandidatePool(budget.epsilon, budget.pool_limit, key=pool_key)

    def trajectories(candidate):
        if not cache_rollouts:
            return [rollout(candidate, s) for s in segments]
        from synthesis.cfg.program_source import program_fingerprint

        key = program_fingerprint(candidate)
        if key in rollout_cache:
            cache_counts["rollout_cache_hits"] += 1
            rollout_cache.move_to_end(key)
            return rollout_cache[key]
        cache_counts["rollout_cache_misses"] += 1
        values = [rollout(candidate, s) for s in segments]
        # Only complete batches enter the cache. Eviction changes work, not
        # scores; parameters, operands and controller settings enter the key.
        rollout_cache[key] = values
        while len(rollout_cache) > max(1, budget.pool_limit):
            rollout_cache.popitem(last=False)
        return values

    def measure(candidate, samples):
        values = trajectories(candidate)
        if budget.distance == "arc":
            return float(
                np.mean(
                    [
                        metric(_observations(states), samples=samples)
                        for metric, states in zip(distances, values)
                    ]
                )
            )
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
        std = np.full(len(mean), budget.cem_std)
        best, best_score = deepcopy(candidate), objective(candidate)
        retain_samples = (
            budget.search_strategy == "contract"
            and budget.cem_iterations > 0
            and len(mean) > 0
        )
        if retain_samples:
            pool.add(candidate, measure(candidate, budget.final_samples))
        for _ in range(budget.cem_iterations if len(mean) else 0):
            scored = []
            for params in rng.normal(mean, std, (budget.cem_samples, len(mean))):
                trial = deepcopy(candidate)
                trial.update_trainable_parameter(list(params))
                score = objective(trial)
                if retain_samples:
                    # CEM still fits imitation alone. Preserve its already
                    # simulated parameter trials in the same bounded epsilon
                    # pool, so later PostScore can select a close trial that
                    # actually crosses the demonstrated relational boundary.
                    pool.add(trial, measure(trial, budget.final_samples))
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
            if budget.contract_endpoint_checks:
                from synthesis.cfg.endpoint_contract import check_endpoint_contract

                result = check_endpoint_contract(candidate, segment, states, post)
                if result and endpoint_acceptance is not None:
                    result = endpoint_acceptance(candidate, segment, states, post)
                for reason in getattr(result, "reasons", ()):
                    endpoint_counts[reason] = endpoint_counts.get(reason, 0) + 1
                success.append(bool(result))
            elif endpoint_acceptance is not None:
                success.append(
                    bool(endpoint_acceptance(candidate, segment, states, post))
                )
            else:
                success.append(postcondition_reached(states, segment, post))
        return float(np.mean(success))

    current = deepcopy(initial_program)
    current_score = -float("inf") if candidate_stream else objective(current)
    last_distance = float("inf")
    checked, rejected = 0, 0
    status, iteration = "budget_exhausted", 0
    for iteration in range(budget.iterations + 1):
        if candidate_stream:
            try:
                candidate = (
                    deepcopy(initial_program)
                    if iteration == 0 and budget.search_strategy == "enumeration"
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
                        search_strategy=budget.search_strategy,
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
                bmc_feasible=True if candidate_stream else None,
                search_stats=proposal_stats(),
            )
            logger.set_progress(
                iteration,
                budget.iterations + 1,
                phase="straightline",
                block_id=block_id,
            )
        if last_distance < budget.theta and not getattr(
            current, "_contract_refinement_only", False
        ):
            ranking_start = perf_counter()
            winner, success = pool.select(post_score)
            ranking_seconds = perf_counter() - ranking_start
            # Imitation convergence alone is not task success. Fail closed when
            # every close candidate misses a demonstrated postcondition.
            if success == 1.0:
                selected_distance = last_distance
                if budget.search_strategy == "contract":
                    selected_distance = next(
                        entry.distance
                        for entry in pool.entries
                        if pool_key(entry.program) == pool_key(winner)
                    )
                return StraightLineResult(
                    winner,
                    True,
                    "converged",
                    selected_distance,
                    success,
                    perf_counter() - start,
                    ranking_seconds,
                    checked if budget.search_strategy == "contract" else iteration,
                    checked,
                    rejected,
                    proposal_stats(),
                    proposal_reason(),
                )
    if candidate_stream and pool.entries:
        current = deepcopy(pool.entries[0].program)
        last_distance = pool.entries[0].distance
    elif candidate_stream:
        status = "no_feasible_candidate"
        if budget.search_strategy == "contract":
            from synthesis.api.program import Program

            current = deepcopy(getattr(candidates, "fallback_program", Program(0, [])))
    if budget.search_strategy == "contract":
        proposal_status = getattr(candidates, "status", "")
        if proposal_status == "unsupported":
            status = "unsupported_contract_input"
        elif proposal_status == "deadline_exhausted":
            status = "contract_deadline_exhausted"
    return StraightLineResult(
        current,
        False,
        status,
        last_distance,
        elapsed=perf_counter() - start,
        iterations=(
            checked
            if budget.search_strategy == "contract"
            else max(0, checked - 1) if candidate_stream else iteration
        ),
        candidates_checked=checked,
        bmc_rejected=rejected,
        search_stats=proposal_stats(),
        search_reason=proposal_reason(),
    )


StraightLineSynthesize = straight_line_synthesize
