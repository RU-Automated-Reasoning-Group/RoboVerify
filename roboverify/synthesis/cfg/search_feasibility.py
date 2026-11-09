"""BMC screening of an enumerated motion skeleton before simulation/CEM.

This checks existential trace feasibility in the existing BMC abstraction at
each demonstrated entry. It deliberately leaves the goal to the CFG's shared
PostScore/verification stages. Unrecorded initial holding state is unconstrained
for interior segments; a full trajectory starts with an empty hand. This is a
search filter, not a collision, controller, or postcondition certificate.
"""

import itertools
from dataclasses import dataclass

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

    def __bool__(self):
        return self.feasible


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
