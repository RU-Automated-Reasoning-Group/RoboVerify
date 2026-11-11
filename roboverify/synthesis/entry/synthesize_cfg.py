"""Instrumented CFG synthesis; synthesized candidates still require verification."""

import argparse
import json
import math
import random
import signal
from dataclasses import asdict, replace
from functools import partial
from time import perf_counter

from synthesis.api.instructions import LoopBudgetExceeded, Skip, While
from synthesis.api.program import Program, generate_random_program
from synthesis.cfg.artifacts import describe_cfg
from synthesis.cfg.bindings import (
    close_objects,
    enumerate_scoped,
    mutate_ids,
    mutate_scoped,
    require_closed,
    require_ids,
    seed_from_scope,
)
from synthesis.cfg.candidate_traces import prepare_candidate
from synthesis.cfg.collection import execution_deadline
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.execute import execute_cfg
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.lower import lower, lower_region
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_io import (
    describe_program_artifact,
    load_program_artifact,
    save_program_artifact,
)
from synthesis.cfg.program_source import ProgramDefinition, load_program
from synthesis.cfg.realization_cache import SuccessfulRealizationCache
from synthesis.cfg.recordings import load_traces
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.straightline import (
    SearchBudget,
    postcondition_reached,
    segment_rollout,
    straight_line_synthesize,
)
from synthesis.cfg.synthesize import synthesize_cfg
from synthesis.cfg.tasks import task_identity, task_spec
from synthesis.cfg.verified_synthesis import verified_synthesis
from synthesis.entry.inference_options import add_inference_options
from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.entry.search_options import (
    add_contract_search_options,
    add_search_options,
)
from synthesis.entry.verification_options import (
    add_verification_options,
    wp_options_from_args,
)
from synthesis.experiment.run_logger import RunLogger
from synthesis.inference_lib.minimization import using_invariant_minimizer
from synthesis.mcmc.synthesis import (
    make_roboverify_env,
    preserved_global_rng,
    set_np_seed,
)
from synthesis.predicates.language import Language
from synthesis.util.on import DEFAULT_HIGHER_TOLERANCE, using_higher_tolerance
from synthesis.verification_lib.bmc_lib import NoiseSpec
from synthesis.verification_lib.height_predicate import using_height_predicate
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.solver_options import using_unsat_core_minimization
from synthesis.verification_lib.wp_options import using_wp_system


def run(args, logger):
    with using_height_predicate(
        getattr(args, "height_predicate", "Higher")
    ), using_invariant_minimizer(
        getattr(args, "invariant_minimizer", None)
    ), using_higher_tolerance(
        getattr(args, "higher_tolerance", DEFAULT_HIGHER_TOLERANCE)
    ), using_unsat_core_minimization(
        getattr(args, "minimize_unsat_core", None)
    ), using_wp_system(
        **wp_options_from_args(args).kwargs()
    ):
        return _run(args, logger)


def _synthesize_only(cfg, realize, execute, context, args, logger, quotient):
    """Run the existing CFG algorithm without proof or invariant bootstrap.

    A reloadable result is a candidate; fresh simulation acceptance is a separate
    evaluation and no successful synthesis status is promoted to a proof.
    """

    def checkpoint(candidate, round_id):
        directory = f"checkpoints/pre-quotient-round-{round_id:04d}"
        definition = _candidate_definition(candidate, context)
        payload = describe_program_artifact(
            definition,
            metadata=dict(
                task=args.task,
                synthesis_num_blocks=args.num_blocks,
                synthesis_status="search_complete_unrolled",
                accepted_by_block_search=True,
                formal_verification="not_run",
                simulation_evaluation="not_run",
                round=round_id,
                config=vars(args),
            ),
        )
        logger.write_artifact(
            f"{directory}/program.json",
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
        )
        logger.write_artifact(
            f"{directory}/cfg.json", json.dumps(describe_cfg(candidate), indent=2)
        )
        logger.write_artifact(f"{directory}/program.txt", str(definition.program))
        logger.log_event(
            "complete_candidate_saved",
            "Saved the complete unrolled candidate before loop generalization",
            step=round_id,
            force=True,
            artifact=f"{directory}/program.json",
            formal_verification="not_run",
        )

    result = synthesize_cfg(
        cfg,
        realize,
        execute,
        quotient=quotient,
        max_refinements=args.refinements,
        language=Language(
            relations=("ON", "ON_star", context.height_predicate, "Scattered", "eq"),
            timeout_seconds=args.predicate_seconds,
        ),
        logger=logger,
        checkpoint=checkpoint,
        boundary_budget=_boundary_budget(args),
    )
    status = "synthesized_unverified" if result else result.status
    validation = None
    if result and getattr(args, "contract_refinement", False):
        validation = _validate_synthesis_candidate(cfg, context, args, logger)
        if not validation["accepted"]:
            status = "simulation_rejected"
    logger.write_artifact("cfg.json", json.dumps(describe_cfg(cfg), indent=2))
    saved = None
    try:
        definition = _candidate_definition(cfg, context)
    except ValueError:
        # A failed attempt may still contain unresolved blocks. Keep its CFG;
        # do not invent executable code for the missing regions.
        if result:
            raise
    else:
        path = logger.artifact_dir() / "program.json"
        save_program_artifact(
            path,
            definition,
            metadata=dict(
                task=args.task,
                synthesis_num_blocks=args.num_blocks,
                synthesis_status=status,
                accepted_by_cfg_search=bool(result),
                training_simulation_passed=(
                    None if validation is None else validation["accepted"]
                ),
                formal_verification="not_run",
                config=vars(args),
            ),
        )
        logger.write_artifact("program.txt", str(definition.program))
        saved = str(path)
    logger.finish(
        status,
        reason=result.reason,
        formal_verification="not_run",
        simulation_evaluation=(
            "not_run"
            if validation is None
            else "training_passed" if validation["accepted"] else "training_rejected"
        ),
        training_validation=validation,
        search_status=result.status,
        rounds=result.rounds,
        failed_block=result.failed_block,
        task=args.task,
        program_artifact=saved,
    )
    return 0 if result and (validation is None or validation["accepted"]) else 2


def _validate_synthesis_candidate(cfg, context, args, logger, revision=0):
    from synthesis.cfg.candidate_selection import select_validated_candidate

    deadline = perf_counter() + getattr(args, "candidate_validation_seconds", 120.0)

    def validate(candidate, trial):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return {
                "accepted": False,
                "passed": 0,
                "total": 0,
                "selection_budget_exhausted": True,
                "seconds": 0.0,
            }
        return _validate_one_synthesis_candidate(
            candidate, context, args, logger, revision, trial, remaining
        )

    result = select_validated_candidate(
        cfg, validate, limit=getattr(args, "candidate_validation_limit", 8)
    )
    logger.write_artifact(
        f"training-selection-{revision}.json",
        json.dumps(result["candidate_selection"], indent=2),
    )
    return result


def _validate_one_synthesis_candidate(
    cfg, context, args, logger, revision, trial, seconds
):
    from synthesis.cfg.lower import lower_with_locations
    from synthesis.cfg.robust_validation import (
        ValidationBudget,
        validate_training_program,
    )

    definition = _candidate_definition(cfg, context)
    _, locations = lower_with_locations(cfg, context, physical=True)
    contracts = {
        path: {"paths": paths, "post": graph.outgoing(name)[0].label}
        for path, (graph, name, paths) in locations["blocks"].items()
    }
    traces = list(
        {
            id(trace): trace
            for trace in (
                *(row.trace for row in cfg._task_demos),
                *getattr(cfg, "_validation_traces", ()),
            )
        }.values()
    )
    result = validate_training_program(
        definition,
        traces,
        lambda trace: make_roboverify_env(trace.task, num_blocks=trace.num_blocks),
        budget=ValidationBudget(
            settling_steps=getattr(args, "candidate_settle_steps", 50),
            max_actions=getattr(args, "candidate_max_actions", 5000),
            max_loop_iterations=getattr(args, "max_loop_iterations", 100),
            seconds=seconds,
            frame_tolerance_m=getattr(args, "contract_frame_tolerance", 0.005),
        ),
        post=cfg.postcondition,
        artifact_dir=logger.artifact_dir()
        / f"training-validation-{revision}-candidate-{trial}",
        block_contracts=contracts,
    )
    logger.log_event(
        "training_validation",
        f"{result['passed']}/{result['total']} complete executions accepted",
        force=True,
        revision=revision,
        candidate=trial,
        accepted=result["accepted"],
        seconds=result["seconds"],
    )
    return result


def _boundary_budget(args):
    if not getattr(args, "contract_refinement", False):
        return None
    from synthesis.cfg.boundaries import BoundaryBudget

    return BoundaryBudget(
        max_shift_steps=getattr(args, "boundary_max_shift", 80),
        max_first_actions=getattr(args, "boundary_first_actions", 32),
        max_proposals=getattr(args, "boundary_proposals", 3),
        seconds=getattr(args, "boundary_seconds", 15.0),
    )


def _handoff_acceptance(node):
    successors = getattr(node, "_contract_successors", None)
    if successors is None:
        return None

    def accept(candidate, segment, states, post):
        from synthesis.cfg.endpoint_contract import EndpointResult, check_next_action

        key = (id(segment.trace), segment.demo_idx, segment.t_start, segment.t_end)
        if key not in successors:
            return EndpointResult(False, ("missing_handoff_bindings",))
        bindings = successors[key]
        return (
            EndpointResult(True)
            if bindings is None
            else check_next_action(candidate, segment, states, bindings)
        )

    return accept


def _candidate_definition(cfg, context):
    program = lower(cfg, context, physical=True)
    initial = dict(cfg._task_demos[0].bindings)
    initial.update(getattr(cfg, "_initial_bindings", {}))
    initial.update(getattr(cfg, "_fixed_id_bindings", {}))
    names = set(cfg.initial_scope) | set(getattr(cfg, "_fixed_id_bindings", {}))
    bindings = {name: initial[name] for name in names}
    return ProgramDefinition(program, bindings, "synthesized-candidate")


def _save_block_candidate(logger, result, node, demos, args, attempt, post):
    """Keep every attempted body, including unsuccessful refinement evidence."""
    scope = sorted(node.available_scope)
    entries = [
        dict(
            demo=segment.demo_idx,
            seed=segment.trace.seed,
            start=segment.t_start,
            end=segment.t_end,
            entry_index=segment.entry_index,
            milestone_index=segment.milestone_index,
            bindings={
                name: segment.bindings[name]
                for name in scope
                if name in segment.bindings
            },
        )
        for segment in demos
    ]
    fixed = {
        name: entries[0]["bindings"][name]
        for name in scope
        if entries
        and name in entries[0]["bindings"]
        and all(
            row["bindings"].get(name) == entries[0]["bindings"][name] for row in entries
        )
    }
    payload = describe_program_artifact(
        ProgramDefinition(result.program, fixed, "block-search-candidate"),
        metadata=dict(
            task=args.task,
            num_blocks=args.num_blocks,
            block=node.name,
            attempt=attempt,
            status=result.status,
            accepted_by_block_search=result.ok,
            formal_verification="not_run",
            search_strategy=args.search_strategy,
            synthesis_approach=node.synthesis_approach,
            distance=result.distance if math.isfinite(result.distance) else None,
            post_score=result.post_score,
            candidates_checked=result.candidates_checked,
            bmc_rejected=result.bmc_rejected,
            entry_bindings_required=sorted(set(scope) - set(fixed)),
            demonstration_archive=str(args.demos),
            target_postcondition=str(post),
            target_postcondition_ast=asdict(post),
            entries=entries,
        ),
    )
    logger.write_artifact(
        f"blocks/attempt-{attempt:04d}-{node.name.replace('/', '_')}/program.json",
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
    )
    for rank, alternative in enumerate(getattr(result, "alternatives", ())):
        alternate = describe_program_artifact(
            ProgramDefinition(alternative.program, fixed, "block-search-alternative"),
            metadata={
                **payload["metadata"],
                "alternative_rank": rank,
                "distance": alternative.distance,
                "post_score": alternative.post_score,
            },
        )
        logger.write_artifact(
            f"blocks/attempt-{attempt:04d}-{node.name.replace('/', '_')}/alternatives/candidate-{rank:04d}.json",
            json.dumps(alternate, indent=2, allow_nan=False) + "\n",
        )


def _run(args, logger):
    invariant_factory = None
    if getattr(args, "loop_invariant", None):
        import importlib

        module, separator, name = args.loop_invariant.rpartition(":")
        if not separator or not name.isidentifier():
            raise ValueError("Use --loop-invariant module:factory")
        invariant_factory = getattr(importlib.import_module(module), name)
    context = HighLevelContext(
        use_tbl=args.task in ("unstack", "reverse"),
        exists_top=args.task in ("unstack", "reverse"),
    )
    traces = load_traces(args.demos, require_valid=True)
    if any(t.metadata.get("task_spec") != task_identity(args.task) for t in traces):
        raise ValueError(
            "Demonstration specification differs from the requested task; recollect demonstrations"
        )
    if any(t.task != args.task or t.num_blocks != args.num_blocks for t in traces):
        raise ValueError(
            "Demonstration task/block count differs from requested synthesis"
        )
    segments = [
        DemoSegment(
            i,
            0,
            len(t.states) - 1,
            t,
            {
                **t.metadata["initial_bindings"],
                **{f"o{j}": j for j in range(t.num_blocks)},
                **({"tbl": "tbl"} if context.use_tbl else {}),
            },
        )
        for i, t in enumerate(traces)
    ]
    pre, post = task_spec(args.task, height_predicate=context.height_predicate)
    from synthesis.cfg.demo_validation import validate_demonstrations

    validation = validate_demonstrations(segments, pre, post)
    demo_checks = validation.as_dict()
    logger.log_event(
        "demo_validation", f"Task predicate checks: {demo_checks}", force=True
    )
    if not validation:
        logger.finish(
            "invalid_demonstrations",
            demonstration_checks=demo_checks,
            formal_verification="not_run",
            reason="Provide recordings satisfying the initial and final task conditions",
        )
        return 2
    cfg = RelationalCFG.initial(
        segments, pre, post, ("b0", "tbl") if context.use_tbl else ("b0",)
    )
    if args.mode == "verify":
        if args.program.endswith(".json"):
            definition = load_program_artifact(args.program, context, args.num_blocks)
        else:
            definition = load_program(
                args.program,
                context,
                args.num_blocks,
                allow_observation_references=True,
            )
        if not getattr(args, "candidate_program", False) and any(
            t.metadata.get("fingerprint") != definition.metadata["fingerprint"]
            for t in traces
        ):
            raise ValueError(
                "Provided program differs from the demonstration source; recollect demonstrations"
            )
        if getattr(args, "candidate_program", False):
            logger.log_event(
                "candidate_program",
                "Verify the loaded candidate on the original demonstration starts; "
                "candidate execution supplies invariant-learning traces",
                force=True,
                fingerprint=definition.metadata["fingerprint"],
            )
        cfg = program_to_cfg(definition, segments, pre, post)
        cfg._candidate_program = getattr(args, "candidate_program", False)
    cfg.synthesis_approach = args.synthesis_approach
    cfg._task_demos = list(segments)
    cfg._validation_traces = []
    for path in getattr(args, "validation_starts", ()):
        extra = load_traces(path)
        if any(trace.task != args.task or trace.num_blocks < 1 for trace in extra):
            raise ValueError("Additional validation starts must use the requested task")
        cfg._validation_traces.extend(extra)
    budget = SearchBudget(
        search_strategy=getattr(args, "search_strategy", "mcmc"),
        iterations=args.iterations,
        cem_iterations=args.cem_iterations,
        cem_samples=args.cem_samples,
        cem_elites=args.cem_elites,
        cem_std=getattr(args, "cem_std", 0.05),
        theta=args.theta,
        epsilon=getattr(args, "epsilon", 0.1),
        distance=args.distance,
        seed=args.seed,
        inner_samples=args.inner_samples,
        final_samples=args.final_samples,
        pool_limit=args.pool_limit,
        contract_beam_width=getattr(args, "contract_beam_width", 24),
        contract_max_expansions=getattr(args, "contract_max_expansions", 300),
        contract_seconds=getattr(args, "contract_seconds", 35.0),
        contract_observation_proposals=not getattr(args, "contract_grid_only", False),
        contract_memoize=not getattr(args, "contract_no_memo", False),
        contract_diversity=not getattr(args, "contract_no_diversity", False),
        contract_pool_limit=getattr(args, "contract_pool_limit", 12),
        contract_endpoint_checks=getattr(args, "contract_refinement", False),
        contract_solve_offsets=getattr(args, "contract_refinement", False),
        contract_operand_neighborhood=(
            getattr(args, "contract_refinement", False)
            and not getattr(args, "contract_no_operand_neighborhood", False)
        ),
        contract_solver_timeout_ms=getattr(args, "contract_solver_timeout_ms", 250),
        contract_solver_attempts=getattr(args, "contract_solver_attempts", 24),
        contract_offset_radius=getattr(args, "contract_offset_radius", 0.02),
    )

    def env_factory(trace):
        return make_roboverify_env(trace.task, num_blocks=trace.num_blocks)

    rollout = partial(
        segment_rollout,
        env_factory=env_factory,
        reset_mode=args.reset_mode,
        audit_contract=budget.contract_endpoint_checks,
        frame_tolerance_m=getattr(args, "contract_frame_tolerance", 0.005),
    )
    block_attempt = 0
    accepted_blocks = SuccessfulRealizationCache()
    base_budget = budget

    def realize(node, demos, post, penalty=None):
        nonlocal block_attempt
        budget = base_budget
        if getattr(node, "_inferred_loop_body", False) and getattr(
            args, "loop_body_distance", None
        ):
            budget = replace(
                budget,
                distance=args.loop_body_distance,
                theta=args.loop_body_theta,
                epsilon=args.loop_body_epsilon,
            )
        if budget.search_strategy == "contract":
            cached = accepted_blocks.get(node, demos, post, penalty=penalty)
            if cached is not None:
                logger.log_event(
                    "block_reused",
                    f"{node.name}: reuse the unchanged accepted physical body",
                    force=True,
                    block_id=node.name,
                )
                return cached, True
        # A new search may use changed demonstrations, handoffs or penalties.
        # Alternatives accepted by an earlier search are no longer certified.
        node._contract_alternatives = ()
        if isinstance(node.region, LoopRegion):
            physical = Program(
                len(lower_region(node.region, context, physical=True)),
                lower_region(node.region, context, physical=True),
            )

            def bound_loops(instructions):
                for instruction in instructions:
                    if isinstance(instruction, While):
                        instruction.max_iters = min(
                            args.max_loop_iterations,
                            (
                                instruction.max_iters
                                if instruction.max_iters is not None
                                else args.max_loop_iterations
                            ),
                        )
                        bound_loops(instruction.body)

            # Bound only the lowered execution copy, not the returned program.
            bound_loops(physical.instructions)
            ok = True
            try:
                for segment in demos:
                    with execution_deadline(args.trajectory_timeout_seconds):
                        states = rollout(physical, segment)
                        if budget.contract_endpoint_checks:
                            from synthesis.cfg.endpoint_contract import (
                                check_endpoint_contract,
                            )

                            ok = bool(
                                check_endpoint_contract(physical, segment, states, post)
                            )
                        else:
                            ok = postcondition_reached(states, segment, post)
                    if not ok:
                        break
            except LoopBudgetExceeded:
                ok = False
            return node.region, ok
        id_first = node.synthesis_approach == "id-first"
        if isinstance(node.region, BlockRegion) and node.region.physical:
            initial = Program(len(node.region.physical), list(node.region.physical))
        elif not id_first and isinstance(node.region, BlockRegion):
            initial = seed_from_scope(
                args.slots, node.available_scope, random.Random(args.seed)
            )
        else:
            initial = generate_random_program(
                args.slots, range(args.num_blocks), random.Random(args.seed)
            )

        def propose(candidate, rng):
            with preserved_global_rng():
                set_np_seed(int(rng.integers(2**31)))
                return (
                    mutate_ids(candidate, args.num_blocks)
                    if id_first
                    else mutate_scoped(candidate, node.available_scope)
                )

        if id_first:
            require_ids(initial.instructions, args.num_blocks)
        else:
            initial = close_objects(
                initial,
                demos,
                node.available_scope,
                context,
                prefix="object_" + node.name.replace(".", "_"),
            )

        search_options = {}
        if budget.search_strategy == "enumeration":
            from synthesis.api.instructions import Move, Pick, Release
            from synthesis.cfg.search_feasibility import check_candidate_feasibility
            from synthesis.mcmc.enumeration import enumerate_programs

            search_options = {
                "candidates": (
                    enumerate_programs(
                        {"Box": list(range(args.num_blocks))},
                        [Pick, Move, Release],
                        args.slots,
                    )
                    if id_first
                    else enumerate_scoped(initial, node.available_scope, args.slots)
                ),
                "feasible": partial(check_candidate_feasibility, segments=demos),
            }
        elif budget.search_strategy == "contract":
            from synthesis.cfg.contract_search import (
                check_contract_feasibility,
                make_contract_candidates,
            )

            search_options = {
                "candidates": make_contract_candidates(
                    demos,
                    post,
                    initial,
                    node.available_scope,
                    max_length=args.slots,
                    budget=budget,
                    id_first=id_first,
                    refinement_fallback=True,
                ),
                "feasible": partial(check_contract_feasibility, segments=demos),
            }
        result = straight_line_synthesize(
            demos,
            post,
            initial,
            propose,
            rollout=rollout,
            budget=budget,
            logger=logger,
            block_id=node.name,
            penalty=penalty,
            cache_rollouts=budget.search_strategy == "contract",
            endpoint_acceptance=(
                _handoff_acceptance(node) if budget.contract_endpoint_checks else None
            ),
            **search_options,
        )
        _save_block_candidate(logger, result, node, demos, args, block_attempt, post)
        block_attempt += 1
        logger.log_event(
            "block_result",
            f"{node.name}: {result.status}",
            force=True,
            block_id=node.name,
            search_approach=node.synthesis_approach,
            search_strategy=budget.search_strategy,
            candidates_checked=result.candidates_checked,
            bmc_rejected=result.bmc_rejected,
            distance=result.distance,
            post_score=result.post_score,
            elapsed=result.elapsed,
            postscore_seconds=result.postscore_seconds,
            search_stats=result.search_stats,
            search_reason=result.search_reason,
        )
        exports = (
            frozenset()
            if id_first
            else (
                require_closed(result.program.instructions, node.available_scope)
                - node.available_scope
            )
        )
        physical = tuple(result.program.instructions)
        if result.ok and not physical:
            # Empty physical blocks otherwise denote unresolved search. Keep a
            # search-accepted identity executable without weakening lowering.
            physical = (Skip(0),)
            logger.log_event(
                "accepted_empty_block_identity",
                "Represent the accepted zero-action program as Skip(0)",
                force=True,
                block_id=node.name,
            )
        # No unsupported relational summary is invented for arbitrary physical code.
        region = BlockRegion(
            node.region.symbolic if isinstance(node.region, BlockRegion) else None,
            physical,
            exports,
        )
        if result.ok and budget.contract_endpoint_checks:
            node._contract_alternatives = tuple(
                BlockRegion(
                    region.symbolic,
                    tuple(alternative.program.instructions) or (Skip(0),),
                    (
                        frozenset()
                        if id_first
                        else (
                            require_closed(
                                alternative.program.instructions, node.available_scope
                            )
                            - node.available_scope
                        )
                    ),
                )
                for alternative in getattr(result, "alternatives", ())
            )
        if budget.search_strategy == "contract":
            accepted_blocks.remember(
                node, demos, post, region, accepted=result.ok, penalty=penalty
            )
        return region, result.ok

    quotient_fn = None
    if args.quotient:
        from synthesis.cfg.quotient import quotient

        quotient_fn = partial(
            quotient,
            boundary_budget=_boundary_budget(args),
            stable_guards=getattr(args, "contract_refinement", False),
            language=Language(
                relations=(
                    "ON",
                    "ON_star",
                    context.height_predicate,
                    "Scattered",
                    "eq",
                ),
                timeout_seconds=args.predicate_seconds,
            ),
        )
    extra_paths = iter(args.additional_demos)

    def provide_demos(request):
        path = next(extra_paths, None)
        if path is None:
            return None
        extra = load_traces(path, require_valid=True)
        if any(
            t.task != args.task
            or t.num_blocks != args.num_blocks
            or t.metadata.get("task_spec") != task_identity(args.task)
            for t in extra
        ):
            raise ValueError(
                "Additional demonstrations must use the same task, specification, and block count"
            )
        return [
            DemoSegment(
                i,
                0,
                len(t.states) - 1,
                t,
                {
                    **t.metadata["initial_bindings"],
                    **getattr(cfg, "_initial_bindings", {}),
                    **getattr(cfg, "_fixed_id_bindings", {}),
                    **{f"o{j}": j for j in range(t.num_blocks)},
                    **({"tbl": "tbl"} if context.use_tbl else {}),
                },
            )
            for i, t in enumerate(extra, start=len(cfg._task_demos))
        ]

    def repair(node, demos, post, penalty):
        region, ok = realize(node, demos, post, penalty=penalty)
        return region if ok else None

    if args.mode == "synthesize":
        return _synthesize_only(
            cfg,
            realize,
            partial(
                execute_cfg,
                context=context,
                env_factory=env_factory,
                reset_mode=args.reset_mode,
            ),
            context,
            args,
            logger,
            quotient_fn,
        )

    result = verified_synthesis(
        cfg,
        realize,
        partial(
            execute_cfg,
            context=context,
            env_factory=env_factory,
            reset_mode=args.reset_mode,
        ),
        context,
        quotient=quotient_fn,
        boundary_budget=_boundary_budget(args),
        initial_candidate=args.mode == "verify",
        validate_candidate=(
            (
                lambda candidate, revision: _validate_synthesis_candidate(
                    candidate, context, args, logger, revision
                )
            )
            if getattr(args, "contract_refinement", False)
            else None
        ),
        verification_level=getattr(args, "verification_level", "both"),
        fixed_invariants=invariant_factory is not None,
        prepare=lambda candidate, revision: prepare_candidate(
            candidate,
            context,
            relations=args.invariant_relations,
            variables=args.invariant_variables,
            max_loop_iterations=args.max_loop_iterations,
            timeout_seconds=args.trajectory_timeout_seconds,
            logger=logger,
            revision=revision,
            invariant_factory=invariant_factory,
        ),
        demo_provider=provide_demos,
        repair_motion=repair,
        # Concrete IDs assume the demonstrated universe exists. Checking a
        # smaller universe can make fixed-alias premises inconsistent. The
        # existing verifier still requests its unbounded proof afterward.
        min_blocks=args.num_blocks if args.synthesis_approach == "id-first" else 2,
        max_blocks=(
            max(4, args.num_blocks) if args.synthesis_approach == "id-first" else 4
        ),
        symbolic_iterations=args.symbolic_iterations,
        motion_iterations=args.motion_iterations,
        timeout_ms=args.verification_timeout_ms,
        relations=args.invariant_relations,
        variables=args.invariant_variables,
        motion_options={
            "initial_arm": args.initial_arm,
            "noise": (
                NoiseSpec(*args.motion_noise) if args.motion_noise is not None else None
            ),
            "timeout_ms": args.motion_timeout_ms,
            "table_surface_height": args.table_surface_height,
            "supported_tower_model": args.supported_towers,
            # Unstack permits offset towers under the ordinary ON* geometry.
            # Remove the auxiliary exact-column premise together with its checks.
            "exact_column_alignment": args.task != "unstack",
        },
        max_refinements=args.refinements,
        language=Language(
            relations=("ON", "ON_star", context.height_predicate, "Scattered", "eq"),
            timeout_seconds=args.predicate_seconds,
        ),
        logger=logger,
    )
    logger.write_artifact("cfg.json", json.dumps(describe_cfg(cfg), indent=2))
    if result or result.status == "verified_symbolic":
        program = lower(cfg, context, physical=True)
        logger.write_artifact("program.txt", str(program))
    logger.finish(
        result.status,
        stages=result.history,
        reason=result.reason,
        formal_verification=result.status,
        symbolic=str(result.symbolic),
        motion=str(result.motion),
        demonstration_checks=demo_checks,
        task=args.task,
        wp_system=context.wp_system,
        support_complete=context.wp_options.support_complete,
    )
    return 0 if result or result.status == "verified_symbolic" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("stack", "unstack", "reverse"), default="stack"
    )
    parser.add_argument(
        "--verification-level", choices=("symbolic", "both"), default="both"
    )
    parser.add_argument(
        "--loop-invariant",
        help="Fixed invariant factory(context) for verification-only controls",
    )
    parser.add_argument(
        "--wall-timeout-seconds",
        type=float,
        help="Total run budget; Unstack defaults to 60 seconds",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "verify", "synthesize"),
        default="full",
        help="synthesize: CFG search and loop recovery without final verification",
    )
    parser.add_argument(
        "--program", help="DSL factory or saved program.json required by --mode verify"
    )
    parser.add_argument(
        "--candidate-program",
        action="store_true",
        help="Verify a candidate differing from the demonstration source; reexecute it for inference",
    )
    parser.add_argument(
        "--synthesis-approach",
        choices=("relational", "id-first"),
        default="relational",
        help="relational: bind names before search; id-first: search/refine IDs, then quotient and return ByName",
    )
    parser.add_argument("--max-loop-iterations", type=int, default=100)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    parser.add_argument(
        "--demos",
        required=True,
        help="Validated archive produced by synthesis.entry.collect_demos",
    )
    parser.add_argument(
        "--output-dir", default="runs", help="Directory for experiment results"
    )
    parser.add_argument(
        "--run-name", default=None, help="Optional readable label for this experiment"
    )
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slots", type=int, default=4)
    add_search_options(parser)
    add_contract_search_options(parser)
    group = parser.add_argument_group("contract-guided CFG refinement")
    group.add_argument(
        "--contract-refinement",
        action="store_true",
        help="Refine milestone cuts, solve shared offsets, require physical endpoint contracts, and prefer stable guard references",
    )
    group.add_argument("--boundary-max-shift", type=int, default=80)
    group.add_argument("--boundary-first-actions", type=int, default=32)
    group.add_argument("--boundary-proposals", type=int, default=3)
    group.add_argument("--boundary-seconds", type=float, default=15.0)
    group.add_argument("--contract-solver-timeout-ms", type=int, default=250)
    group.add_argument("--contract-solver-attempts", type=int, default=24)
    group.add_argument("--contract-offset-radius", type=float, default=0.02)
    group.add_argument(
        "--contract-no-operand-neighborhood",
        action="store_true",
        help="Disable bounded operand-reference alternatives in contract refinement",
    )
    group.add_argument("--contract-frame-tolerance", type=float, default=0.005)
    group.add_argument("--candidate-settle-steps", type=int, default=50)
    group.add_argument("--candidate-validation-seconds", type=float, default=120.0)
    group.add_argument("--candidate-max-actions", type=int, default=5000)
    group.add_argument("--candidate-validation-limit", type=int, default=8)
    group.add_argument(
        "--validation-starts",
        nargs="*",
        default=[],
        help="Additional exact-start archives for complete-candidate selection; may use other block counts; never used as imitation labels",
    )
    group.add_argument(
        "--loop-body-distance", choices=("kl_kde", "kl_gmm", "mmd", "arc")
    )
    group.add_argument("--loop-body-theta", type=float)
    group.add_argument("--loop-body-epsilon", type=float)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--refinements", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-samples", type=int, default=8)
    parser.add_argument("--cem-elites", type=int, default=2)
    parser.add_argument(
        "--cem-std",
        type=float,
        default=0.05,
        help="Initial CEM parameter standard deviation in metres",
    )
    parser.add_argument("--inner-samples", type=int, default=128)
    parser.add_argument("--final-samples", type=int, default=2048)
    parser.add_argument("--pool-limit", type=int, default=10)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Near-best pool width in the selected distance metric's units",
    )
    parser.add_argument(
        "--distance",
        choices=("kl_kde", "kl_gmm", "mmd", "arc"),
        default="kl_kde",
        help="Imitation metric; arc is ordered, speed-insensitive spatial RMS in metres",
    )
    parser.add_argument("--reset-mode", choices=("reset", "replay"), default="replay")
    parser.add_argument("--predicate-seconds", type=float, default=5.0)
    parser.add_argument(
        "--quotient", action="store_true", help="Enable conservative flat-loop folding"
    )
    parser.add_argument(
        "--invariant-relations",
        nargs="+",
        choices=(
            "ON_star",
            "ON_star_zero",
            "Higher",
            "EqualHeight",
            "Scattered",
            "equality",
        ),
    )
    parser.add_argument(
        "--height-predicate",
        choices=("Higher", "EqualHeight"),
        default="Higher",
        help="Height algebra for supplied-program verification (default Higher)",
    )
    parser.add_argument("--invariant-variables", type=int, default=2)
    add_inference_options(parser)
    add_verification_options(parser)
    add_motion_options(parser)
    add_predicate_options(parser)
    parser.add_argument(
        "--table-surface-height",
        type=float,
        help="Physical table height required for table-placement contracts",
    )
    parser.add_argument(
        "--supported-towers",
        action="store_true",
        help="Use complete uniform towers on a common table; check height, arm-clearance and column invariants",
    )
    parser.add_argument(
        "--additional-demos",
        nargs="*",
        default=[],
        help="Recordings supplied for verification-requested resynthesis",
    )
    parser.add_argument("--symbolic-iterations", type=int, default=10)
    parser.add_argument("--motion-iterations", type=int, default=10)
    parser.add_argument("--verification-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--initial-arm",
        type=float,
        nargs=3,
        help="Explicit initial arm position for the geometric verification model",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.validation_starts and not args.contract_refinement:
        parser.error("--validation-starts requires --contract-refinement")
    if args.candidate_validation_limit < 1:
        parser.error("Candidate validation limit must be positive")
    body_options = (
        args.loop_body_distance,
        args.loop_body_theta,
        args.loop_body_epsilon,
    )
    if any(value is not None for value in body_options):
        if not all(value is not None for value in body_options):
            parser.error(
                "Loop-body distance requires explicit theta and epsilon in that metric's units"
            )
        if not all(math.isfinite(value) and value >= 0 for value in body_options[1:]):
            parser.error("Loop-body thresholds must be nonnegative and finite")
    if args.contract_refinement:
        if args.search_strategy != "contract" or args.mode == "verify":
            parser.error(
                "--contract-refinement requires contract search in synthesize/full mode"
            )
        try:
            _boundary_budget(args)
            from synthesis.cfg.robust_validation import ValidationBudget

            ValidationBudget(
                settling_steps=args.candidate_settle_steps,
                seconds=args.candidate_validation_seconds,
                max_actions=args.candidate_max_actions,
                frame_tolerance_m=args.contract_frame_tolerance,
            )
            SearchBudget(
                contract_solver_timeout_ms=args.contract_solver_timeout_ms,
                contract_solver_attempts=args.contract_solver_attempts,
                contract_offset_radius=args.contract_offset_radius,
            )
            if (
                not math.isfinite(args.contract_frame_tolerance)
                or args.contract_frame_tolerance < 0
            ):
                raise ValueError("Frame tolerance must be nonnegative and finite")
        except ValueError as exc:
            parser.error(str(exc))
    try:
        wp_options_from_args(args)
        if args.wp_system != "old" and args.height_predicate != "Higher":
            raise ValueError("New WP backends require --height-predicate Higher")
    except ValueError as exc:
        parser.error(str(exc))
    if args.task == "reverse" and args.mode != "verify":
        parser.error("Reverse currently supports supplied-program verification only")
    if args.loop_invariant and args.mode != "verify":
        parser.error("--loop-invariant requires --mode verify")
    if args.candidate_program and args.mode != "verify":
        parser.error("--candidate-program requires --mode verify")
    if args.height_predicate == "EqualHeight" and args.mode != "verify":
        parser.error(
            "EqualHeight currently supports supplied-program verification only"
        )
    other_height = "Higher" if args.height_predicate == "EqualHeight" else "EqualHeight"
    if other_height in (args.invariant_relations or ()):
        parser.error("Invariant height relation must match --height-predicate")
    if args.task == "reverse" and args.invariant_relations is None:
        args.invariant_relations = ["ON_star", "ON_star_zero", "equality"]
    if args.invariant_relations is None:
        args.invariant_relations = [
            "ON_star",
            args.height_predicate,
            "Scattered",
            "equality",
        ]
    if args.wall_timeout_seconds is None and args.task == "unstack":
        args.wall_timeout_seconds = 60
    if (
        args.wall_timeout_seconds is not None
        and not 0 < args.wall_timeout_seconds < float("inf")
    ):
        parser.error("Wall-clock budget must be positive and finite")
    motion_noise_from_args(parser, args)
    if args.synthesis_approach == "id-first":
        args.quotient = True
    if args.mode == "verify" and not args.program:
        parser.error("--mode verify requires --program module:factory")
    if args.max_loop_iterations < 1 or not 0 < args.trajectory_timeout_seconds < float(
        "inf"
    ):
        parser.error("Execution budgets must be positive and finite")
    if args.smoke:
        args.iterations, args.cem_iterations, args.cem_samples, args.cem_elites = (
            2,
            1,
            4,
            2,
        )
        args.inner_samples, args.final_samples, args.refinements = 32, 128, 1
    if (
        args.num_blocks < 2
        or args.slots < 1
        or args.iterations < 0
        or args.refinements < 0
        or args.symbolic_iterations < 0
        or args.motion_iterations < 0
        or args.verification_timeout_ms <= 0
        or args.invariant_variables < 1
        or not math.isfinite(args.cem_std)
        or args.cem_std <= 0
        or not math.isfinite(args.epsilon)
        or args.epsilon < 0
    ):
        parser.error("Invalid block, slot, or iteration budget")

    def deadline(signum, frame):
        raise TimeoutError(
            f"End-to-end {args.wall_timeout_seconds:g}-second budget exhausted"
        )

    previous = signal.signal(signal.SIGALRM, deadline)
    with RunLogger(
        args.output_dir,
        "cfg",
        vars(args),
        slug=args.run_name or f"{args.task}-{args.mode}",
    ) as logger:
        logger.progress_line(f"CFG run: {logger.run_dir}")
        if args.wall_timeout_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, args.wall_timeout_seconds)
        try:
            code = run(args, logger)
        except TimeoutError as exc:
            logger.finish("budget_exhausted", reason=str(exc))
            code = 2
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", reason=str(exc))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    print(f"Read: uv run python -m synthesis.experiment.report --run {logger.run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
