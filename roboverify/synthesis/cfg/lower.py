"""Fold stored regions into the existing execution/verification Program IR."""

from copy import deepcopy

import z3

from synthesis.api.instructions import Assign, While
from synthesis.api.program import Program
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.predicates.term import to_z3


def lower_region(region, context, *, physical=False):
    if isinstance(region, BlockRegion):
        if physical and not region.physical:
            raise ValueError("Region has no physical implementation")
        return deepcopy(list(region.physical if physical else region.symbolic))
    if not isinstance(region, LoopRegion):
        raise ValueError("Cannot lower an unresolved region")
    if region.invariant is None:
        raise ValueError("Loop lowering requires an explicit inferred invariant")
    body = [instruction for child in region.body for instruction in lower_region(child, context, physical=physical)]
    body.extend(Assign(a, b) for a, b in region.update)
    guard = to_z3(region.guard, context)
    loop = While(guard, [context.get_consts(v) for v in region.exists_vars], body,
                 region.invariant, max_iters=region.max_iters)
    loop.guard_term = region.guard
    return [*(Assign(a, b) for a, b in region.init), loop]


def lower(cfg, context, *, physical=False):
    cfg.validate_structure()
    instructions = [inst for key in cfg.order for inst in lower_region(cfg.nodes[key].region, context, physical=physical)]
    return Program(len(instructions), instructions)
