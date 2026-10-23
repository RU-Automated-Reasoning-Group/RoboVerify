"""Load an explicit primitive DSL factory and identify its executable contents."""

import hashlib
import importlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from z3.z3util import get_vars

from synthesis.api.control import TABLE_REFERENCE_MODE
from synthesis.api.instructions import (
    Assign,
    Get,
    GoalAssign,
    Move,
    MoveByName,
    MoveDown,
    MoveRight,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Skip,
    While,
)
from synthesis.api.program import Program
from synthesis.cfg.physical import name_operands
from synthesis.environment.goal_layout import GOAL_REFERENCE_MODE, GOAL_TOLERANCE
from synthesis.predicates.term import free_names, from_z3
from synthesis.util.actions import CARTESIAN_ACTION_MODE
from synthesis.util.symbols import fresh_name


def guard_term(instruction):
    term = getattr(instruction, "guard_term", None)
    return term if term is not None else from_z3(instruction.instantiated_cond)


def uses_goals(program):
    def walk(rows):
        return any(
            isinstance(i, (GoalAssign, MoveRight, MoveDown))
            or (isinstance(i, While) and walk(i.body))
            for i in rows
        )

    return walk(program.instructions)


def describe_program(program):
    goal_program = uses_goals(program)

    def guard(i):
        return i.instantiated_cond.sexpr() if goal_program else str(guard_term(i))

    def has_table_reference(rows):
        return any(
            (isinstance(i, Assign) and i.right == "tbl")
            or (isinstance(i, While) and has_table_reference(i.body))
            or any(o["val"] == "tbl" for o in i.get_operand())
            for i in rows
        )

    table_reference = has_table_reference(program.instructions)

    def describe(i):
        result = {"instruction": type(i).__name__}
        if isinstance(i, While):
            result.update(
                guard=guard(i),
                witnesses=list(map(str, i.guard_exists_vars)),
                body=[describe(child) for child in i.body],
                max_iters=i.max_iters,
            )
        elif isinstance(i, Get):
            result.update(guard=guard(i), witnesses=list(map(str, i.guard_exists_vars)))
        elif isinstance(i, (Assign, GoalAssign)):
            result.update(left=i.left, right=i.right)
        elif isinstance(i, (MoveRight, MoveDown)):
            result["var_name"] = i.var_name
        elif isinstance(i, Skip):
            result["steps"] = i.skip_steps
        elif type(i) in (Pick, PickByName, Move, MoveByName, Release, ReleaseByName):
            result.update(
                operands=i.get_operand(),
                limit=i.limit,
                control=asdict(i.control),
                action_scaling=CARTESIAN_ACTION_MODE,
            )
            if hasattr(i, "target_offset"):
                result["offsets"] = [p.numeric_val() for p in i.target_offset]
            if hasattr(i, "target_z_offset"):
                result["release_offset"] = i.target_z_offset.numeric_val()
            if table_reference and isinstance(i, (MoveByName, ReleaseByName)):
                result["table_reference"] = TABLE_REFERENCE_MODE
        else:
            raise ValueError(
                f"Unsupported instruction {type(i).__name__}; use explicit Pick/Move/Release primitives"
            )
        if goal_program:
            result.update(
                goal_reference=GOAL_REFERENCE_MODE, goal_tolerance=GOAL_TOLERANCE
            )
            if isinstance(i, (While, Get)):
                result["witness_sorts"] = [str(v.sort()) for v in i.guard_exists_vars]
        return result

    return [describe(i) for i in program.instructions]


def program_fingerprint(program):
    return hashlib.sha256(
        json.dumps(describe_program(program), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class ProgramDefinition:
    program: Program
    initial_bindings: dict
    source: str

    @property
    def metadata(self):
        return dict(
            source=self.source,
            program=describe_program(self.program),
            fingerprint=program_fingerprint(self.program),
            initial_bindings=self.initial_bindings,
        )


def prepare_program(program, num_blocks, source="", *, allow_goal_program=False):
    if not isinstance(program, Program):
        raise TypeError("Program factory must return synthesis.api.program.Program")
    goal_program = uses_goals(program)
    if goal_program and not allow_goal_program:
        raise ValueError(
            "Goal programs support execution/collection only; synthesis and verification are unsupported"
        )
    describe_program(program)
    occupied, ids = {"b0", "tbl"}, set()

    def inspect(rows, depth=0):
        for i in rows:
            if isinstance(i, (While, Get)):
                occupied.update(
                    map(str, get_vars(i.instantiated_cond))
                    if goal_program
                    else free_names(guard_term(i))
                )
                occupied.update(map(str, i.guard_exists_vars))
            if isinstance(i, While):
                if depth and not goal_program:
                    raise ValueError("Nested loops are not supported")
                inspect(i.body, depth + 1)
            elif isinstance(i, (Assign, GoalAssign)):
                occupied.update((i.left, i.right))
            elif isinstance(i, (MoveRight, MoveDown)):
                occupied.add(i.var_name)
            elif not isinstance(i, (Get, Skip)):
                for op in i.get_operand():
                    if op["type"] == "Box":
                        ids.add(op["val"])
                    else:
                        occupied.add(op["val"])

    inspect(program.instructions)
    if any(
        not isinstance(i, (int, np.integer)) or not 0 <= i < num_blocks for i in ids
    ):
        raise ValueError("Numeric object operand is outside the configured block count")
    aliases = {i: fresh_name(f"object_{i}", occupied) for i in sorted(ids)}

    def normalize(rows):
        output = []
        for i in rows:
            new = name_operands(i, aliases)
            if isinstance(new, While):
                new.body = normalize(i.body)
            if isinstance(new, (While, Get)) and not goal_program:
                new.guard_term = guard_term(new)
            output.append(new)
        return output

    return ProgramDefinition(
        Program(len(program.instructions), normalize(program.instructions)),
        {
            **({} if goal_program else {"b0": 0}),
            **{name: int(i) for i, name in aliases.items()},
        },
        source,
    )


def load_program(source, context, num_blocks, *, allow_goal_program=False):
    module_name, separator, factory_name = source.rpartition(":")
    if not separator or not module_name or not factory_name.isidentifier():
        raise ValueError("Use --program module:factory or path/to/program.py:factory")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        spec = importlib.util.spec_from_file_location("roboverify_user_program", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load program file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = str(path) + ":" + factory_name
    else:
        module = importlib.import_module(module_name)
    definition = prepare_program(
        getattr(module, factory_name)(context, num_blocks=num_blocks),
        num_blocks,
        source,
        allow_goal_program=allow_goal_program,
    )
    if context.use_tbl:
        definition.initial_bindings["tbl"] = "tbl"
    return definition
