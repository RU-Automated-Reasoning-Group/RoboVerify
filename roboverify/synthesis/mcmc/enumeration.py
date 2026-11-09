"""Lazy, deterministic enumeration of primitive program structures.

Lengths increase first, then instruction kinds and operand tuples. Float
parameters start at zero and belong to CEM, not to the enumerated alphabet.
Skip padding is omitted: the empty program represents a zero-step no-op.
"""

import itertools
from copy import deepcopy

from synthesis.api.instructions import (
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Skip,
)
from synthesis.api.program import Program


def enumerate_programs(
    available_operands, available_instructions, max_length, *, prefix=()
):
    if type(max_length) is not int or max_length < 0:
        raise ValueError("Enumeration length must be a nonnegative integer")
    supported = {
        Pick: ("Box", 1),
        Move: ("Box", 3),
        Release: ("Box", 1),
        PickByName: ("BoxName", 1),
        MoveByName: ("BoxName", 3),
        ReleaseByName: ("BoxName", 1),
    }
    alphabet, seen = [], set()
    for kind in available_instructions:
        if kind is Skip or kind in seen:
            continue
        if kind not in supported:
            raise ValueError(
                "Enumeration supports only Pick, Move and Release primitives"
            )
        seen.add(kind)
        operand_type, arity = supported[kind]
        operands = list(available_operands.get(operand_type, ()))
        if not operands or any(
            (
                (type(v) is not int or v < 0)
                if operand_type == "Box"
                else (not isinstance(v, str) or not v)
            )
            for v in operands
        ):
            raise ValueError(
                f"Enumeration requires a nonempty {operand_type} operand pool"
            )
        operands = sorted(set(operands))
        alphabet.extend(
            (kind, values) for values in itertools.product(operands, repeat=arity)
        )
    if len({supported[kind][0] for kind in seen}) > 1:
        raise ValueError("Enumeration cannot mix numeric and named primitives")

    for length in range(max_length + 1):
        for structure in itertools.product(alphabet, repeat=length):
            body = deepcopy(list(prefix))
            for kind, operands in structure:
                if kind in (Move, MoveByName):
                    instruction = kind(*operands, target_offset=[0.0, 0.0, 0.0])
                elif kind in (Release, ReleaseByName):
                    instruction = kind(*operands, target_z=0.0)
                else:
                    instruction = kind(*operands)
                body.append(instruction)
            yield Program(len(body), body)
