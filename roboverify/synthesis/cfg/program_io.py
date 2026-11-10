"""Portable, exact executable artifacts for synthesized tower programs.

The JSON contains structured guards rather than executable Python or pickled Z3
objects. Loading constructs formulas in the caller's current verification context.
Acceptance and formal proof status are metadata, never inferred from reloadability.
"""

import json
import math
from pathlib import Path

from synthesis.api.control import ControlConfig
from synthesis.api.instructions import (
    Assign,
    Get,
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Skip,
    While,
)
from synthesis.api.program import Program
from synthesis.cfg.program_source import (
    ProgramDefinition,
    describe_program,
    guard_term,
    program_fingerprint,
)
from synthesis.predicates.term import Term, from_z3, to_z3

FORMAT = "roboverify-executable-program"
VERSION = 1


def _term_data(term):
    return dict(op=term.op, args=[_term_data(a) for a in term.args], value=term.value)


def _read_term(row):
    op = row["op"]
    args = tuple(_read_term(a) for a in row["args"])
    value = row["value"]
    if op in ("exists", "forall"):
        if not isinstance(value, list) or not value or len(args) != 1:
            raise ValueError("Invalid quantified guard")
        value = tuple(value)
    elif op in ("ref", "id", "bool"):
        if args:
            raise ValueError("Literal/reference guard nodes cannot have arguments")
    elif op in (
        "ON",
        "ON_star",
        "ON_star_zero",
        "Higher",
        "EqualHeight",
        "Scattered",
        "eq",
        "implies",
    ):
        if len(args) != 2:
            raise ValueError("Binary guard node needs two arguments")
    elif op == "not":
        if len(args) != 1:
            raise ValueError("Negated guard needs one argument")
    elif op not in ("and", "or"):
        raise ValueError(f"Unsupported guard operation: {op}")
    return Term(op, args, value)


def _rows(program):
    def enrich(instructions, rows):
        for instruction, row in zip(instructions, rows):
            if isinstance(instruction, (While, Get)):
                row["guard_ast"] = _term_data(guard_term(instruction))
            if isinstance(instruction, Get):
                row["variable"] = str(instruction.var)
            if isinstance(instruction, While):
                invariant = instruction.invariant
                row["invariant_ast"] = (
                    None if invariant is None else _term_data(from_z3(invariant))
                )
                enrich(instruction.body, row["body"])
        return rows

    return enrich(program.instructions, describe_program(program))


def save_program_artifact(path, definition, *, metadata=None):
    """Save exact runtime floats, bindings, control settings and structured guards."""
    if not isinstance(definition, ProgramDefinition):
        raise TypeError(
            "A ProgramDefinition with explicit initial bindings is required"
        )
    payload = dict(
        format=FORMAT,
        version=VERSION,
        initial_bindings=dict(definition.initial_bindings),
        instructions=_rows(definition.program),
        fingerprint=program_fingerprint(definition.program),
        metadata=dict(metadata or {}),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def load_program_artifact(path, context, num_blocks=None):
    """Reload an artifact without code execution, validating its executable hash."""
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("Unsupported executable program artifact")
    primitives = {
        c.__name__: c
        for c in (Pick, PickByName, Move, MoveByName, Release, ReleaseByName)
    }

    def finite(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Executable offsets must be finite")
        return value

    def load(row):
        kind = row["instruction"]
        if kind in primitives:
            operands = [o["val"] for o in row["operands"]]
            options = dict(limit=row["limit"], control=ControlConfig(**row["control"]))
            if kind in ("Move", "MoveByName"):
                options["target_offset"] = [finite(v) for v in row["offsets"]]
            elif kind in ("Release", "ReleaseByName"):
                options["target_z"] = finite(row["release_offset"])
            return primitives[kind](*operands, **options)
        if kind == "Assign":
            return Assign(row["left"], row["right"])
        if kind == "Skip":
            return Skip(row["steps"])
        if kind in ("Get", "While"):
            term = _read_term(row["guard_ast"])
            guard = to_z3(term, context)
            witnesses = [context.get_consts(n) for n in row["witnesses"]]
            if kind == "Get":
                return Get(row["variable"], guard, witnesses, guard_term=term)
            invariant = row["invariant_ast"]
            instruction = While(
                guard,
                witnesses,
                [load(child) for child in row["body"]],
                None if invariant is None else to_z3(_read_term(invariant), context),
                max_iters=row["max_iters"],
            )
            instruction.guard_term = term
            return instruction
        raise ValueError(f"Unsupported executable instruction: {kind}")

    rows = [load(row) for row in payload["instructions"]]
    program = Program(len(rows), rows)
    if program_fingerprint(program) != payload["fingerprint"]:
        raise ValueError("Executable artifact fingerprint mismatch")
    bindings = payload["initial_bindings"]
    if not isinstance(bindings, dict) or any(not isinstance(k, str) for k in bindings):
        raise ValueError("Initial bindings must map names to physical IDs or tbl")
    if any(v != "tbl" and (type(v) is not int or v < 0) for v in bindings.values()):
        raise ValueError("Invalid initial physical binding")
    if num_blocks is not None:
        if any(v != "tbl" and v >= num_blocks for v in bindings.values()):
            raise ValueError(
                "Saved physical binding is outside the requested block count"
            )
    if "tbl" in bindings.values() and not context.use_tbl:
        raise ValueError("Saved table binding requires a table-enabled context")
    return ProgramDefinition(program, bindings, str(path.resolve()))
