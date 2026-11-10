"""BMC screening of an enumerated motion skeleton before simulation/CEM.

This checks existential trace feasibility in the existing BMC abstraction at
each demonstrated entry. It deliberately leaves the goal to the CFG's shared
PostScore/verification stages. Unrecorded initial holding state is unconstrained
for interior segments; a full trajectory starts with an empty hand. This is a
search filter, not a collision, controller, or postcondition certificate.
"""

import itertools
import math
from dataclasses import dataclass
from time import perf_counter

import z3

from synthesis.api.guard_eval import evaluate_z3
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
)
from synthesis.cfg.refine import scene_at
from synthesis.predicates.scene import evaluate
from synthesis.verification_lib.bmc_lib import bmc_feasible


@dataclass(frozen=True)
class CandidateFeasibility:
    feasible: bool
    reason: str = ""
    status: str = "unknown"
    seconds: float = 0.0

    def __bool__(self):
        return self.feasible


def resolve_entry_bindings(instructions, scene):
    """Evaluate a fixed logical prefix at one recorded entry.

    The chosen witness follows the same scene iteration order as the existing
    CFG screen. This grounds this demonstration only; it is not a proof that
    every legal Get witness is correct.
    """
    bindings = dict(scene.bindings)
    for instruction in instructions:
        if isinstance(instruction, Skip):
            continue
        if isinstance(instruction, Assign):
            bindings[instruction.left] = bindings[instruction.right]
            continue
        if not isinstance(instruction, Get):
            raise ValueError("Expected a logical entry prefix")
        names = [str(v) for v in instruction.guard_exists_vars]
        for values in itertools.product(scene.positions, repeat=len(names)):
            trial = dict(bindings, **dict(zip(names, values)))
            holds = (
                evaluate(instruction.guard_term, scene, trial)
                if instruction.guard_term is not None
                else evaluate_z3(instruction.instantiated_cond, scene, trial)
            )
            if holds:
                bindings = trial
                break
        else:
            raise ValueError("Get has no witness at this segment entry")
    return bindings


def check_scene_feasibility(
    candidate,
    scene,
    *,
    gripper=None,
    initial_holding="unknown",
    table_reference=None,
    fixed_offsets=False,
    timeout_ms=1000,
    goal=None,
):
    """Screen a finite scene with instruction-entry coordinate references.

    Reuses the production noiseless Pick/Move/Release transition kernels. For
    recorded held entries, Move preserves the object's observed gripper offset.
    All physical scene objects participate, including ones absent from operands.
    ``None`` means a known empty hand; ``"unknown"`` leaves it unconstrained.
    ``table_reference`` is an explicit runtime coordinate frame (its Z is the
    table surface minus half a block), not the relational table sentinel.
    The default goal is True. Existential offsets are independent per call;
    feasibility establishes neither shared parameters nor controller safety.
    """
    from synthesis.verification_lib.bmc_lib import (
        _encode_move,
        _encode_pick,
        _encode_release,
        build_trace_symbols,
    )

    started = perf_counter()

    def real(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Coordinates and fixed offsets must be finite")
        return z3.RealVal(str(number))

    try:
        if timeout_ms <= 0:
            raise ValueError("BMC timeout must be positive")
        ids = sorted(key for key in scene.positions if type(key) is int)
        if not ids:
            raise ValueError("BMC requires at least one physical scene object")
        instructions = list(candidate.instructions)
        physical = [i for i in instructions if not isinstance(i, (Assign, Get, Skip))]
        sym = build_trace_symbols(tuple(map(str, ids)), len(physical))
        solver = z3.Solver()
        solver.set(timeout=max(1, int(timeout_ms)))
        coords = (sym.bx, sym.by, sym.bz)
        for object_id in ids:
            position = scene.positions[object_id]
            if len(position) != 3:
                raise ValueError("Physical positions must contain XYZ")
            for axis in range(3):
                solver.add(coords[axis][str(object_id)][0] == real(position[axis]))
        if gripper is not None:
            if len(gripper) != 3:
                raise ValueError("Gripper position must contain XYZ")
            for coordinate, value in zip((sym.ee_x, sym.ee_y, sym.ee_z), gripper):
                solver.add(coordinate[0] == real(value))
        if initial_holding is None:
            solver.add(sym.holding[0] == sym.NONE)
        elif initial_holding != "unknown":
            if type(initial_holding) is not int or initial_holding not in ids:
                raise ValueError(
                    "Initial holding must be empty, unknown or a physical ID"
                )
            solver.add(sym.holding[0] == sym.block_consts[str(initial_holding)])
        bindings = dict(scene.bindings)
        first = sym.block_names[0]

        def coordinate(reference, axis, time):
            if reference == "gripper":
                return (sym.ee_x, sym.ee_y, sym.ee_z)[axis][time]
            if reference in ("Y_min", "Z_m"):
                key_axis = 1 if reference == "Y_min" else 2
                best_key = coords[key_axis][first][time]
                best_value = coords[axis][first][time]
                for name in sym.block_names[1:]:
                    key = coords[key_axis][name][time]
                    choose = key < best_key if reference == "Y_min" else key > best_key
                    best_value = z3.If(choose, coords[axis][name][time], best_value)
                    best_key = z3.If(choose, key, best_key)
                return best_value
            bound = bindings.get(reference, reference)
            if bound == "tbl":
                if table_reference is None or len(table_reference) != 3:
                    raise ValueError(
                        "Table coordinates require an explicit runtime frame"
                    )
                return real(table_reference[axis])
            if type(bound) is not int or bound not in ids:
                raise ValueError(
                    f"Unresolved physical coordinate operand: {reference!r}"
                )
            return coords[axis][str(bound)][time]

        time = 0
        for instruction in instructions:
            if isinstance(instruction, Skip):
                continue
            if isinstance(instruction, Assign):
                bindings[instruction.left] = bindings[instruction.right]
                continue
            if isinstance(instruction, Get):
                if time:
                    raise ValueError("Get is supported only before physical actions")
                from synthesis.predicates.scene import Scene

                entry = Scene(scene.positions, bindings, scene.entry_positions)
                bindings = resolve_entry_bindings([instruction], entry)
                continue
            operands = [o["val"] for o in instruction.get_operand()]

            def offset(parameter, axis):
                return (
                    real(parameter.concrete_float("BMC fixed-offset search screen"))
                    if fixed_offsets
                    else z3.FreshReal(f"search_offset_{time}_{axis}")
                )

            if type(instruction) in (Pick, PickByName):
                bound = bindings.get(operands[0], operands[0])
                if type(bound) is not int or bound not in ids:
                    raise ValueError("Pick requires a physical block operand")
                solver.add(_encode_pick(sym, time, ids.index(bound)))
            elif type(instruction) in (Move, MoveByName):
                targets = [
                    coordinate(operands[axis], axis, time)
                    + offset(instruction.target_offset[axis], axis)
                    for axis in range(3)
                ]
                # Rebase expression-valued targets through a physical object.
                # Interior held observations may have a nonzero grasp offset;
                # retain it instead of snapping the payload to the gripper.
                shifted = [targets[a] - coords[a][first][time] for a in range(3)]
                solver.add(
                    _encode_move(
                        sym,
                        time,
                        0,
                        0,
                        0,
                        *shifted,
                        preserve_grasp_offset=type(initial_holding) is int,
                    )
                )
            elif type(instruction) in (Release, ReleaseByName):
                target = coordinate(operands[0], 2, time) + offset(
                    instruction.target_z_offset, 0
                )
                solver.add(_encode_release(sym, time, 0, target - sym.bz[first][time]))
            else:
                raise ValueError(
                    f"Unsupported BMC instruction: {type(instruction).__name__}"
                )
            time += 1
        solver.add(z3.BoolVal(True) if goal is None else goal(sym, bindings))
        status = solver.check()
        reason = solver.reason_unknown() if status == z3.unknown else str(status)
        return CandidateFeasibility(
            status == z3.sat, reason, str(status), perf_counter() - started
        )
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return CandidateFeasibility(
            False, str(exc), "unsupported", perf_counter() - started
        )


def _ground(candidate, scene):
    bindings, physical = dict(scene.bindings), []
    for instruction in candidate.instructions:
        if isinstance(instruction, Skip):
            continue
        if isinstance(instruction, Assign):
            bindings[instruction.left] = bindings[instruction.right]
            continue
        if isinstance(instruction, Get):
            if physical:
                raise ValueError(
                    "BMC search screening supports Get only before physical actions"
                )
            names = [str(v) for v in instruction.guard_exists_vars]
            for values in itertools.product(scene.positions, repeat=len(names)):
                trial = dict(bindings, **dict(zip(names, values)))
                holds = (
                    evaluate(instruction.guard_term, scene, trial)
                    if instruction.guard_term is not None
                    else evaluate_z3(instruction.instantiated_cond, scene, trial)
                )
                if holds:
                    bindings = trial
                    break
            else:
                raise ValueError("Get has no witness at this segment entry")
            continue
        ids = [
            bindings[o["val"]] if o["type"] == "BoxName" else o["val"]
            for o in instruction.get_operand()
        ]
        if any(type(i) is not int or i not in scene.positions for i in ids):
            raise ValueError("BMC search screening requires physical block operands")
        if type(instruction) in (Pick, PickByName):
            physical.append(Pick(*ids))
        elif type(instruction) in (Move, MoveByName):
            physical.append(Move(*ids))
        elif type(instruction) in (Release, ReleaseByName):
            physical.append(Release(*ids))
        else:
            raise ValueError(
                f"Unsupported BMC search instruction: {type(instruction).__name__}"
            )
    return physical


def check_candidate_feasibility(candidate, segments, *, timeout_ms=10000):
    if not segments or timeout_ms <= 0:
        raise ValueError("BMC screening requires segments and a positive timeout")
    for index, segment in enumerate(segments):
        try:
            scene = scene_at(segment, segment.t_start)
            physical = _ground(candidate, scene)

            def initial(sym):
                constraints = (
                    [sym.holding[0] == sym.NONE] if segment.t_start == 0 else []
                )
                for name in sym.block_names:
                    position = scene.positions[int(name)]
                    constraints.extend(
                        coords[name][0] == float(value)
                        for coords, value in zip((sym.bx, sym.by, sym.bz), position)
                    )
                return constraints

            solver = z3.Solver()
            solver.set(timeout=timeout_ms)
            passed = bmc_feasible(
                physical,
                lambda sym: z3.BoolVal(True),
                initial_constraints=initial,
                assume_goal_false_at_start=False,
                solver=solver,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return CandidateFeasibility(False, f"segment {index}: {exc}")
        if not passed:
            return CandidateFeasibility(
                False, f"segment {index}: BMC did not establish feasibility"
            )
    return CandidateFeasibility(True, "BMC feasible on every segment")
