"""Close physical object references and keep mutation inside proven runtime scope."""

from copy import deepcopy

from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    ReleaseByName,
    Skip,
)
from synthesis.api.program import Program
from synthesis.cfg.physical import name_operands
from synthesis.predicates.term import atom, boolean, free_names, negate, ref, to_z3
from z3.z3util import get_vars


def require_closed(instructions, available):
    """Check uses in execution order; return definitely defined exit names."""
    names = set(available)
    for instruction in instructions:
        if isinstance(instruction, Get):
            bound = {str(v) for v in instruction.guard_exists_vars}
            condition = instruction.guard_term
            used = (
                set(free_names(condition))
                if condition is not None
                else {str(v) for v in get_vars(instruction.instantiated_cond)}
            ) - bound
            defined = bound
        elif isinstance(instruction, Assign):
            used, defined = {instruction.right}, {instruction.left}
        else:
            operands = instruction.get_operand()
            if any(o["type"] == "Box" for o in operands):
                raise ValueError("Numeric object IDs must be closed before export")
            used = {o["val"] for o in operands if o["type"] == "BoxName"}
            defined = set()
        missing = used - names
        if missing:
            raise ValueError(f"Out-of-scope operands: {sorted(missing)}")
        names.update(defined)
    return frozenset(names)


def close_objects(program, demos, available, context, *, prefix="object"):
    """Reuse only consistent in-scope aliases; close other IDs with typed Get.

    Get witnesses are arbitrary objects, not the recorded IDs. The caller must
    rescore this closed program and verify all choices before trusting it.
    """
    if not demos:
        raise ValueError("Object closure requires demonstrations")
    occupied = set(available) | set().union(*(set(d.bindings) for d in demos))
    aliases = {}
    for name in sorted(available):
        values = {d.bindings.get(name) for d in demos}
        if len(values) == 1 and None not in values:
            aliases.setdefault(next(iter(values)), name)
    numeric = sorted(
        {
            o["val"]
            for i in program.instructions
            for o in i.get_operand()
            if o["type"] == "Box"
        }
    )
    gets = []
    for object_id in numeric:
        if object_id in aliases:
            continue
        name = prefix
        index = 0
        while name in occupied:
            index += 1
            name = f"{prefix}{index}"
        occupied.add(name)
        aliases[object_id] = name
        condition = (
            negate(atom("eq", ref(name), ref("tbl")))
            if context.use_tbl
            else boolean(True)
        )
        gets.append(
            Get(
                name,
                to_z3(condition, context),
                [context.get_consts(name)],
                guard_term=condition,
            )
        )
    instructions = gets + [name_operands(i, aliases) for i in program.instructions]
    require_closed(instructions, available)
    return Program(len(instructions), instructions)


def mutate_scoped(program, available):
    """Freeze Get prefixes; structural search may mutate only the physical body."""
    from synthesis.mcmc.synthesis import mutate_program

    prefix, body = [], list(program.instructions)
    while body and isinstance(body[0], Get):
        prefix.append(body.pop(0))
    names = sorted(require_closed(prefix, available) - {"tbl"})
    require_closed(body, names)
    if not body or not names:
        return deepcopy(program)
    constructors = [
        lambda: PickByName(names[0]),
        lambda: MoveByName(names[0], names[0], names[0]),
        lambda: ReleaseByName(names[0]),
        Skip,
    ]
    candidate, _, _ = mutate_program(
        Program(len(body), body), {"BoxName": names}, constructors
    )
    instructions = deepcopy(prefix) + candidate.instructions
    require_closed(instructions, available)
    return Program(len(instructions), instructions)
