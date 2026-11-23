"""Repair existing float parameters without changing a learned CFG or its gates."""

import math
from copy import deepcopy
from dataclasses import replace
from time import perf_counter

from synthesis.api.program import Program
from synthesis.cfg.candidate_selection import _blocks, _sync_loops
from synthesis.cfg.contract_search import check_contract_feasibility
from synthesis.cfg.lower import lower, lower_with_locations
from synthesis.cfg.program_source import program_fingerprint
from synthesis.cfg.robust_validation import ValidationBudgetExceeded
from synthesis.cfg.straightline import segment_rollout, straight_line_synthesize
from synthesis.search.parameter_refinement import refine_parameters


def repair_cfg_parameters(
    cfg,
    context,
    validate,
    *,
    budget,
    env_factory,
    action_budget=None,
    on_trial=None,
    endpoint_acceptance=None,
    frame_tolerance_m=0.005,
):
    """Screen fixed floats, replay complete tasks, then recheck original local gates.

    ``validate(cfg, trial)`` shares the caller's deadline, requested starts and
    action budget. Only a candidate passing every stage is installed. Local
    checks use original segments/metrics and the accepted body's epsilon radius.
    """
    nodes = tuple(_blocks(cfg))
    original_regions = tuple(node.region for node in nodes)
    original = lower(cfg, context, physical=True)
    fingerprints = {
        id(node): program_fingerprint(
            Program(len(region.physical), list(region.physical))
        )
        for node, region in zip(nodes, original_regions)
    }

    def install(program):
        parameters = list(program.register_trainable_parameter())
        cursor, replacements = 0, []
        for node, original_region in zip(nodes, original_regions):
            region = replace(
                original_region, physical=tuple(deepcopy(original_region.physical))
            )
            block = Program(len(region.physical), list(region.physical))
            count = len(block.register_trainable_parameter())
            if cursor + count > len(parameters):
                raise ValueError("Parameter vector is shorter than the CFG")
            block.update_trainable_parameter(parameters[cursor : cursor + count])
            replacements.append(replace(region, physical=tuple(block.instructions)))
            cursor += count
        if cursor != len(parameters):
            raise ValueError(
                "Parameter vector does not match the CFG's physical blocks"
            )
        previous = tuple(node.region for node in nodes)
        try:
            for node, region in zip(nodes, replacements):
                node.region = region
            _sync_loops(cfg)
            if program_fingerprint(
                lower(cfg, context, physical=True)
            ) != program_fingerprint(program):
                raise ValueError(
                    "Parameter repair changed the lowered executable structure"
                )
        except BaseException:
            for node, region in zip(nodes, previous):
                node.region = region
            _sync_loops(cfg)
            raise

    def check_budget(deadline):
        if perf_counter() >= deadline:
            raise ValidationBudgetExceeded("Parameter repair wall budget exhausted")
        if action_budget is not None and action_budget.used >= action_budget.limit:
            raise ValidationBudgetExceeded("Shared candidate action budget exhausted")

    def evaluate(program, trial, deadline):
        install(program)
        local_checks = []
        try:
            check_budget(deadline)
            _, locations = lower_with_locations(cfg, context, physical=True)
            changed = []
            for path, (graph, name, _) in locations["blocks"].items():
                node = graph.nodes[name]
                body = Program(len(node.region.physical), list(node.region.physical))
                if program_fingerprint(body) == fingerprints[id(node)]:
                    continue
                local_budget = getattr(node, "_contract_fit_budget", None)
                reference_distance = getattr(node, "_contract_fit_distance", None)
                if local_budget is None or reference_distance is None:
                    return {
                        "accepted": False,
                        "complete": False,
                        "stage": "missing_local_gate",
                        "block": path,
                    }
                segments = graph.demos.for_node(name)
                check_budget(deadline)
                for segment in segments:
                    check_budget(deadline)
                    screen = check_contract_feasibility(
                        body,
                        [segment],
                        fixed_offsets=True,
                        timeout_ms=max(
                            1, min(1000, int(1000 * (deadline - perf_counter())))
                        ),
                    )
                    check_budget(deadline)
                    if not screen:
                        return {
                            "accepted": False,
                            "complete": False,
                            "stage": "fixed_bmc",
                            "block": path,
                            "reason": screen.reason,
                        }
                changed.append(
                    (
                        path,
                        graph,
                        node,
                        body,
                        segments,
                        local_budget,
                        reference_distance,
                    )
                )

            report = validate(cfg, f"parameter-{trial:04d}")
            if not report["accepted"]:
                return {**report, "complete": False, "stage": "complete_execution"}

            def limited_environment(trace):
                check_budget(deadline)
                env = env_factory(trace)
                original_step = env.step

                def step(action):
                    check_budget(deadline)
                    if action_budget is not None:
                        action_budget.consume()
                    return original_step(action)

                env.step = step
                return env

            for (
                path,
                graph,
                node,
                body,
                segments,
                local_budget,
                reference_distance,
            ) in changed:
                check_budget(deadline)
                fit_budget = replace(
                    local_budget,
                    iterations=0,
                    cem_iterations=0,
                    search_strategy="contract",
                )

                def rollout(candidate, segment):
                    return segment_rollout(
                        candidate,
                        segment,
                        limited_environment,
                        reset_mode="reset",
                        audit_contract=True,
                        frame_tolerance_m=frame_tolerance_m,
                    )

                result = straight_line_synthesize(
                    segments,
                    graph.outgoing(node.name)[0].label,
                    body,
                    None,
                    rollout=rollout,
                    budget=fit_budget,
                    candidates=(body,),
                    # This exact body/segment batch already passed fixed BMC above.
                    feasible=lambda candidate: True,
                    cache_rollouts=True,
                    endpoint_acceptance=(
                        None
                        if endpoint_acceptance is None
                        else endpoint_acceptance(node)
                    ),
                )
                accepted = (
                    result.ok
                    and result.distance <= reference_distance + local_budget.epsilon
                )
                local_checks.append(
                    {
                        "block": path,
                        "accepted": accepted,
                        "distance": (
                            result.distance if math.isfinite(result.distance) else None
                        ),
                        "theta": local_budget.theta,
                        "epsilon": local_budget.epsilon,
                        "reference_distance": reference_distance,
                        "post_score": result.post_score,
                        "search_stats": result.search_stats,
                    }
                )
                if not accepted:
                    return {
                        **report,
                        "accepted": False,
                        "complete": False,
                        "stage": "local_gate",
                        "local_parameter_checks": local_checks,
                    }
            if perf_counter() >= deadline:
                raise ValidationBudgetExceeded("Parameter repair wall budget exhausted")
            return {
                **report,
                "complete": True,
                "stage": "accepted",
                "local_parameter_checks": local_checks,
            }
        except ValidationBudgetExceeded as exc:
            return {
                "accepted": False,
                "complete": False,
                "budget_exhausted": True,
                "reason": str(exc),
                "local_parameter_checks": local_checks,
            }

    try:
        result = refine_parameters(original, evaluate, budget=budget, on_trial=on_trial)
    finally:
        for node, region in zip(nodes, original_regions):
            node.region = region
        _sync_loops(cfg)
    if result:
        install(result.program)
    return result
