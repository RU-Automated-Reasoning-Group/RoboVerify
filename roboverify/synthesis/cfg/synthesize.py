"""Algorithm 2 over structured single-entry CFGs, with bounded refinement."""

from dataclasses import dataclass

from synthesis.cfg.refine import refine_cfg
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
):
    """realize(node,segments,post) -> (region, ok); execute(cfg) returns negatives.

    Failure retains the best attempted physical block so Exec can run the
    realized prefix and stop at the first unresolved block. A successful search
    is a candidate synthesis result, not a formal verification result.
    """
    for round_id in range(max_refinements + 1):
        if quotient is not None:
            changed = quotient(cfg)
            if changed and logger:
                logger.log_event(
                    "quotient_loop_found",
                    "Collapsed a repeated fragment",
                    step=round_id,
                    force=True,
                )
        failed = None
        for name in cfg.order:
            node = cfg.nodes[name]
            region, ok = realize(
                node, cfg.demos.for_node(name), cfg.outgoing(name)[0].label
            )
            if region is not None:
                node.region = region
            if not ok:
                failed = name
                break
        if failed is None:
            return SynthesisResult(cfg, "synthesized", round_id)
        if round_id == max_refinements:
            return SynthesisResult(cfg, "budget_exhausted", round_id, failed)
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
