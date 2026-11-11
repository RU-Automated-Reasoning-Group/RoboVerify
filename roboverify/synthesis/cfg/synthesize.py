"""Algorithm 2 over structured single-entry CFGs, with bounded refinement."""

from copy import copy
from dataclasses import dataclass

from synthesis.cfg.refine import refine_cfg
from synthesis.cfg.region import LoopRegion
from synthesis.cfg.scope import scope


@dataclass
class SynthesisResult:
    cfg: object
    status: str
    rounds: int
    failed_block: str = None
    reason: str = ""

    def __bool__(self):
        return self.status == "synthesized"


def synthesize_cfg(
    cfg,
    realize,
    execute,
    *,
    quotient=None,
    max_refinements=10,
    language=None,
    logger=None,
    checkpoint=None,
    boundary_budget=None,
):
    """realize(node,segments,post) -> (region, ok); execute(cfg) returns negatives.

    Failure retains the best attempted physical block so Exec can run the
    realized prefix and stop at the first unresolved block. A successful search
    is a candidate synthesis result, not a formal verification result. An optional
    checkpoint(cfg, round_id) observes the complete numeric candidate immediately
    before quotienting can replace it with an as-yet unrealized loop body.
    """
    if cfg.synthesis_approach not in ("relational", "id-first"):
        raise ValueError(f"Unknown synthesis approach: {cfg.synthesis_approach}")
    id_first = cfg.synthesis_approach == "id-first"
    for round_id in range(max_refinements + 1):
        if quotient is not None and (not id_first or boundary_budget is not None):
            changed = quotient(cfg)
            if changed and id_first and boundary_budget is not None:
                # Contract-guided body search can complete an unrealized
                # repetition. Do not require a mixed-phase unrolled controller
                # before trying the recovered shared body.
                from synthesis.cfg.id_first import close_id_candidate

                close_id_candidate(cfg, allow_unresolved=True)
                cfg.synthesis_approach = "relational"
                id_first = False
            if changed and logger:
                logger.log_event(
                    "quotient_loop_found",
                    "Collapsed a repeated fragment",
                    step=round_id,
                    force=True,
                )
        failed = None
        retry_boundaries = False
        for name in cfg.order:
            node = cfg.nodes[name]
            if isinstance(node.region, LoopRegion) and node.region.body_cfg is not None:
                nested = synthesize_cfg(
                    node.region.body_cfg,
                    realize,
                    execute,
                    max_refinements=max_refinements,
                    language=language,
                    logger=logger,
                    boundary_budget=boundary_budget,
                )
                if not nested:
                    if boundary_budget is not None and round_id < max_refinements:
                        from synthesis.cfg.quotient import retry_boundary_fold

                        if retry_boundary_fold(cfg, name):
                            retry_boundaries = True
                            break
                    return SynthesisResult(
                        cfg,
                        nested.status,
                        round_id,
                        f"{name}/{nested.failed_block}",
                        nested.reason,
                    )
                body_cfg = node.region.body_cfg
                node.region.body = tuple(
                    body_cfg.nodes[n].region for n in body_cfg.order
                )
                node.region.postconditions = tuple(
                    body_cfg.outgoing(n)[0].label for n in body_cfg.order
                )
                node.region.body_demos = tuple(
                    tuple(body_cfg.demos.for_node(n)) for n in body_cfg.order
                )
            node.available_scope = scope(cfg)[name]
            node.synthesis_approach = cfg.synthesis_approach
            region, ok = realize(
                node, cfg.demos.for_node(name), cfg.outgoing(name)[0].label
            )
            if region is not None:
                node.region = region
            if not ok:
                failed = name
                break
        if retry_boundaries:
            continue
        if failed is None:
            if id_first:
                # Numeric search/refinement finishes before any loop variables
                # are introduced. The public synthesis boundary is named only.
                from synthesis.cfg.id_first import close_id_candidate

                if quotient is not None and checkpoint is not None:
                    checkpoint(cfg, round_id)
                changed = quotient is not None and quotient(cfg)
                if changed and logger:
                    logger.log_event(
                        "quotient_loop_found",
                        "Generalized concrete repetitions",
                        step=round_id,
                        force=True,
                    )
                close_id_candidate(cfg)
                if changed:
                    # Quotient establishes structure, not an executable body.
                    # Search every extracted body with its new named bindings,
                    # and recheck the loop and any repartitioned continuation.
                    # A shallow working graph selects this phase's policy while
                    # preserving ID-first for a later whole-task resynthesis.
                    named = copy(cfg)
                    named.synthesis_approach = "relational"
                    if logger:
                        logger.log_event(
                            "post_quotient_synthesis",
                            "Search named loop bodies and check the folded program",
                            step=round_id,
                            force=True,
                        )
                    continuation = synthesize_cfg(
                        named,
                        realize,
                        execute,
                        max_refinements=max_refinements,
                        language=language,
                        logger=logger,
                        boundary_budget=boundary_budget,
                    )
                    cfg.__dict__.update(named.__dict__)
                    cfg.synthesis_approach = "id-first"
                    return SynthesisResult(
                        cfg,
                        continuation.status,
                        round_id + continuation.rounds,
                        continuation.failed_block,
                        continuation.reason
                        or (
                            ""
                            if continuation
                            else "Post-quotient named synthesis failed at "
                            + str(continuation.failed_block)
                        ),
                    )
            return SynthesisResult(cfg, "synthesized", round_id)
        if round_id == max_refinements:
            return SynthesisResult(cfg, "budget_exhausted", round_id, failed)
        if boundary_budget is not None:
            from synthesis.cfg.boundaries import refine_failed_handoff
            from synthesis.cfg.quotient import retry_boundary_fold

            if retry_boundary_fold(cfg, failed):
                continue

            # Retry an outgoing cut too: the failed block may need to include
            # release/retreat before its following block can be realized.
            targets = [failed]
            position = cfg.order.index(failed)
            if position + 1 < len(cfg.order):
                targets.append(cfg.order[position + 1])
            if any(
                refine_failed_handoff(cfg, name, budget=boundary_budget)
                for name in targets
            ):
                if logger:
                    logger.log_event(
                        "boundary_refined",
                        f"{failed}: retry persistent milestone handoff",
                        step=round_id,
                        force=True,
                    )
                continue
        if isinstance(cfg.nodes[failed].region, LoopRegion):
            return SynthesisResult(
                cfg,
                "loop_execution_failed",
                round_id,
                failed,
                "Loop guard/body needs repair; retain the discovered loop",
            )
        negatives = execute(cfg)
        result = refine_cfg(
            cfg, failed, negatives, scope(cfg)[failed], language=language
        )
        if logger:
            logger.log_event(
                "refine" if result else result.status,
                f"{failed}: {result.status}",
                step=round_id,
                force=True,
            )
        if not result:
            return SynthesisResult(cfg, result.status, round_id, failed)
    raise AssertionError("Unreachable refinement exit")


Synthesize = synthesize_cfg
