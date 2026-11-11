"""Empirical endpoint contracts for straight-line candidate acceptance.

These checks reject observed failures. They are finite observation checks, not a
motion proof: frame observations are once per controller action, contact checks
are at instruction boundaries, and no later settling is silently added.
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import product

import numpy as np
import z3

from synthesis.predicates.scene import Scene, evaluate, scene_from_obs
from synthesis.util import on


@dataclass(frozen=True)
class EndpointResult:
    ok: bool
    reasons: tuple = ()
    details: dict = field(default_factory=dict)

    def __bool__(self):
        return self.ok


def endpoint_scene(states, segment, *, bindings=None):
    """Use the invocation's frozen entry geometry and the actual last state."""
    from synthesis.cfg.refine import scene_at

    if not states:
        return None
    state = states[-1]
    scene = (
        state
        if isinstance(state, Scene)
        else scene_from_obs(
            state,
            segment.trace.num_blocks,
            segment.bindings,
            include_table="tbl" in segment.bindings,
        )
    )
    aliases = dict(scene.bindings)
    aliases.update(bindings or {})
    return Scene(
        scene.positions,
        aliases,
        scene_at(segment, segment.entry_index).entry_positions,
        scene.observation,
    )


def postcondition_at_endpoint(states, segment, post, *, bindings=None):
    scene = endpoint_scene(states, segment, bindings=bindings)
    return scene is not None and evaluate(post, scene)


def _on_placements(post, bindings):
    """Only positive explicit conjuncts justify a particular Put contract."""
    if post.op == "and":
        return [pair for term in post.args for pair in _on_placements(term, bindings)]
    if post.op != "ON":
        return []
    return [tuple(a.value if a.op == "id" else bindings[a.value] for a in post.args)]


def _ground(formula, domain):
    if z3.is_quantifier(formula):
        children = [
            _ground(z3.substitute_vars(formula.body(), *reversed(values)), domain)
            for values in product(domain, repeat=formula.num_vars())
        ]
        return (z3.And if formula.is_forall() else z3.Or)(children)
    if z3.is_app(formula) and formula.num_args():
        return formula.decl()(*(_ground(a, domain) for a in formula.children()))
    return formula


class _AxiomCollector:
    def __init__(self):
        self.formulas = {}

    def assert_and_track(self, formula, name):
        self.formulas[str(name)] = formula


@lru_cache(maxsize=None)
def _finite_context(count):
    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    # A fresh sort avoids collision with finite verifiers or repeated test runs.
    context = HighLevelContext(
        mode="enum",
        num_blocks=count,
        sort_name=str(z3.FreshConst(z3.IntSort(), "EndpointDomain")),
        wp_system="old",
        height_predicate="Higher",
    )
    collector = _AxiomCollector()
    context.add_axiom(collector)
    context.add_axiom_scattered(collector)
    return context, {
        name: _ground(formula, context.enum_blocks)
        for name, formula in collector.formulas.items()
    }


@lru_cache(maxsize=256)
def _finite_put(count, source, target):
    from synthesis.api.instructions import Put
    from synthesis.api.program import put_effect, put_precondition

    context, _ = _finite_context(count)
    instruction = Put("endpoint_source", "endpoint_target")
    substitutions = (
        (context.get_consts("endpoint_source"), context.enum_blocks[source]),
        (context.get_consts("endpoint_target"), context.enum_blocks[target]),
    )
    effects = {
        name: [
            [
                z3.simplify(
                    z3.substitute(
                        put_effect(instruction, getattr(context, name)(a, b), context),
                        *substitutions,
                    )
                )
                for b in context.enum_blocks
            ]
            for a in context.enum_blocks
        ]
        for name in ("ON_star", "Scattered")
    }
    enabled = z3.simplify(
        z3.substitute(
            _ground(put_precondition(instruction, context), context.enum_blocks),
            *substitutions,
        )
    )
    return effects, enabled


def _tables(positions, ids):
    return {
        name: [[bool(function(positions[a], positions[b])) for b in ids] for a in ids]
        for name, function in (
            ("ON_star", on.on_star_implementation),
            ("Scattered", on.scattered_implementation),
        )
    }


def _values(context, tables):
    return tuple(
        (getattr(context, name)(a, b), z3.BoolVal(table[i][j]))
        for name, table in tables.items()
        for i, a in enumerate(context.enum_blocks)
        for j, b in enumerate(context.enum_blocks)
    )


def _truth(formula, values):
    result = z3.simplify(z3.substitute(formula, *values))
    if not (z3.is_true(result) or z3.is_false(result)):
        raise ValueError("Endpoint audit contains an ungrounded formula")
    return z3.is_true(result)


def audit_put_effects(before, after, source, target):
    """Ground production Put effects and axioms; assume none while checking them.

    Scope is ON*/Scattered on physical objects, for a single physical-block Put.
    Higher, table placement, collision sweeps and arbitrary summaries are not
    certified by this function. The caller must separately check direct ON.
    """
    ids = sorted(
        k for k, position in before.positions.items() if not on.is_table(position)
    )
    if (
        set(before.positions) != set(after.positions)
        or source not in ids
        or target not in ids
    ):
        return EndpointResult(False, ("unsupported_put_universe",))
    context, axioms = _finite_context(len(ids))
    effects, enabled = _finite_put(len(ids), ids.index(source), ids.index(target))
    old, new = _tables(before.positions, ids), _tables(after.positions, ids)
    values = _values(context, old)
    reasons = []
    if not _truth(enabled, values):
        reasons.append("put_not_enabled")
    for label, tables in (("entry", old), ("endpoint", new)):
        failures = [
            name
            for name, formula in axioms.items()
            if not _truth(formula, _values(context, tables))
        ]
        reasons.extend(f"{label}_domain:{name}" for name in failures)
    mismatches = {
        name: [
            [ids[i], ids[j]]
            for i, row in enumerate(formulas)
            for j, formula in enumerate(row)
            if _truth(formula, values) != new[name][i][j]
        ]
        for name, formulas in effects.items()
    }
    reasons.extend(f"put_effect:{name}" for name, pairs in mismatches.items() if pairs)
    return EndpointResult(
        not reasons,
        tuple(reasons),
        {
            "mismatches": mismatches,
            "scope": "finite physical ON_star/Scattered; old Put; no Higher or motion proof",
        },
    )


def contact_state(env, observation, count):
    """Read physical hand occupancy and oriented support contacts, without stepping."""
    from synthesis.cfg.reset import inner_env

    sim = inner_env(env).sim
    model, data = sim.model, sim.data
    objects = {model.geom_name2id(f"object{i}"): i for i in range(count)}
    fingers = {
        i
        for i in range(model.ngeom)
        if (model.geom_id2name(i) or "").endswith("gripper_finger_link")
    }
    table = model.geom_name2id("table0")
    finger_objects = {}
    supports = set()
    positions = {i: on.get_block_pos(observation, i) for i in range(count)}
    for contact in data.contact[: data.ncon]:
        if contact.dist > 1e-6:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        for finger, obj in ((a, b), (b, a)):
            if finger in fingers and obj in objects:
                finger_objects.setdefault(objects[obj], set()).add(finger)
        if abs(contact.frame[2]) < 0.9:
            continue
        if a in objects and b in objects:
            i, j = objects[a], objects[b]
            upper, lower = (i, j) if positions[i][2] > positions[j][2] else (j, i)
            if (
                abs(positions[upper][2] - positions[lower][2] - on.BLOCK_LENGTH)
                <= 0.004
            ):
                supports.add((upper, lower))
        elif (a == table and b in objects) or (b == table and a in objects):
            supports.add((objects[b] if a == table else objects[a], "tbl"))
    aperture = float(np.sum(np.asarray(observation)[3:5]))
    nearest_center = min(
        (
            float(np.linalg.norm(np.asarray(position) - np.asarray(observation)[:3]))
            for position in positions.values()
        ),
        default=float("inf"),
    )
    # An initially closed hand can be empty. No contacts alone do not establish
    # that fact near an object: require an empty neighbourhood as well. The
    # radius is deliberately larger than the nominal grasp's 3 cm offset bound.
    empty_evidence = None
    if not finger_objects and aperture > 0.053:
        known, holding, empty_evidence = True, None, "open_without_object_contacts"
    elif not finger_objects and nearest_center > on.BLOCK_LENGTH:
        known, holding, empty_evidence = (
            True,
            None,
            "no_object_contacts_or_nearby_block",
        )
    elif len(finger_objects) == 1 and len(next(iter(finger_objects.values()))) >= 2:
        known, holding = True, next(iter(finger_objects))
    else:
        known, holding = False, None
    return {
        "known": known,
        "holding": holding,
        "supports": list(supports),
        "aperture": aperture,
        "finger_objects": sorted(finger_objects),
        "nearest_block_center_m": nearest_center,
        "empty_evidence": empty_evidence,
    }


class EndpointRecorder:
    """Cheap read-only instruction and controller-step observations."""

    def __init__(self, env, segment, first, *, frame_tolerance_m=0.005):
        self.env, self.segment = env, segment
        self.tolerance = float(frame_tolerance_m)
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Frame tolerance must be finite and nonnegative")
        self.reasons, self.records = [], []
        self.initial_observation = np.array(first, copy=True)
        self.contract_bindings = None
        self.active = None
        self.transfer_start = None
        self.transfer_mover = None
        self.maximum_drift = 0.0
        self.last_observation = np.array(first, copy=True)

    def _contact(self, obs):
        try:
            return contact_state(self.env, obs, self.segment.trace.num_blocks)
        except (AttributeError, KeyError, ValueError, TypeError) as exc:
            self.reasons.append("contact_observation_unavailable")
            return {"known": False, "holding": None, "supports": [], "error": str(exc)}

    def observe(self, obs):
        self.last_observation = np.array(obs, copy=True)
        if self.active is None:
            return
        # The same mover's permission spans its whole transfer, so per-action
        # errors cannot accumulate unnoticed on a support or other nonmover.
        baseline = (
            self.transfer_start
            if self.transfer_start is not None
            else self.active["before"]
        )
        mover = (
            self.transfer_mover
            if self.transfer_start is not None
            else self.active["mover"]
        )
        for block in range(self.segment.trace.num_blocks):
            if block == mover:
                continue
            drift = float(
                np.linalg.norm(
                    on.get_block_pos(obs, block) - on.get_block_pos(baseline, block)
                )
            )
            self.maximum_drift = max(self.maximum_drift, drift)
            if drift > self.tolerance:
                self.reasons.append("sampled_nonmover_frame")

    def instruction(self, instruction, phase, observation):
        from synthesis.api.instructions import (
            Move,
            MoveByName,
            Pick,
            PickByName,
            Release,
            ReleaseByName,
        )

        physical = isinstance(
            instruction, (Move, MoveByName, Pick, PickByName, Release, ReleaseByName)
        )
        if not physical:
            # Identity/binding operations are permitted. Unsupported physical
            # macros cannot silently receive an audit certificate.
            from synthesis.api.instructions import Assign, Get, Skip, While

            if isinstance(instruction, Skip):
                if phase == "start":
                    self.active = {
                        "kind": "Skip",
                        "before": np.array(observation, copy=True),
                        "mover": None,
                    }
                else:
                    self.observe(observation)
                    self.active = None
            if not isinstance(instruction, (Assign, Get, Skip, While)):
                self.reasons.append("unsupported_instruction_contract")
            return
        if phase == "start":
            if self.contract_bindings is None:
                self.contract_bindings = dict(self.env.symbolic_name_to_box_id)
            contact = self._contact(observation)
            mover = contact["holding"] if contact["known"] else None
            if isinstance(instruction, (Pick, PickByName)):
                mover = (
                    instruction.grab_box_id
                    if isinstance(instruction, Pick)
                    else self.env.symbolic_name_to_box_id[instruction.grab_box_name]
                )
                self.transfer_start, self.transfer_mover = (
                    np.array(observation, copy=True),
                    mover,
                )
            elif not contact["known"]:
                self.reasons.append("unknown_instruction_entry_holding")
            elif mover is not None and self.transfer_start is None:
                self.transfer_start, self.transfer_mover = (
                    np.array(observation, copy=True),
                    mover,
                )
            self.active = {
                "kind": type(instruction).__name__,
                "before": np.array(observation, copy=True),
                "mover": mover,
            }
            return
        self.observe(observation)
        contact = self._contact(observation)
        result = getattr(instruction, "last_control_result", None)
        if result is None or not result.converged:
            self.reasons.append("primitive_not_converged")
        mover = self.active["mover"]
        if isinstance(instruction, (Pick, PickByName)) and (
            not contact["known"] or contact["holding"] != mover
        ):
            self.reasons.append("pick_not_observed_held")
        if isinstance(instruction, (Release, ReleaseByName)):
            if not contact["known"] or contact["holding"] is not None:
                self.reasons.append("release_not_observed_empty")
            # Occupancy and opening are distinct obligations: a closed empty
            # hand at reset is valid, but Release must actually open it.
            if contact.get("aperture", 0.0) <= 0.053:
                self.reasons.append("release_not_observed_open")
            if mover is None or not any(
                pair[0] == mover for pair in contact["supports"]
            ):
                self.reasons.append("release_not_observed_supported")
            self.transfer_start = self.transfer_mover = None
        self.records.append(
            {
                "kind": self.active["kind"],
                "mover": mover,
                "converged": bool(result and result.converged),
                "endpoint": contact,
            }
        )
        self.active = None

    def finish(self):
        endpoint = self._contact(self.last_observation)
        return {
            "reasons": sorted(set(self.reasons)),
            "instructions": self.records,
            "endpoint": endpoint,
            "contract_bindings": self.contract_bindings or dict(self.segment.bindings),
            "max_nonmover_drift_m": self.maximum_drift,
            "frame_tolerance_m": self.tolerance,
            "scope": "empirical: controller-action sampled frame, instruction-end contacts/convergence; no settling or swept collision proof",
        }


def check_endpoint_contract(candidate, segment, states, post):
    """Require the demonstrated terminal phase and the actual logical endpoint."""
    from synthesis.cfg.contract_search import entry_holding
    from synthesis.cfg.refine import scene_at

    audit = getattr(states, "contract_audit", None)
    if audit is None:
        return EndpointResult(False, ("missing_physical_endpoint_audit",))
    bindings = audit["contract_bindings"]

    # Freeze existing operand roles at physical entry, before trailing updates.
    def contains_on(term):
        return term.op == "ON" or (
            term.op == "and" and any(contains_on(a) for a in term.args)
        )

    final = endpoint_scene(
        states, segment, bindings=bindings if contains_on(post) else None
    )
    reasons = list(audit["reasons"])
    if final is None or not evaluate(post, final):
        reasons.append("postcondition_not_at_endpoint")
    contact = audit["endpoint"]
    try:
        expected_holding = entry_holding(replace(segment, t_start=segment.t_end))
    except (ValueError, KeyError, TypeError):
        reasons.append("unknown_demonstrated_endpoint_holding")
    else:
        if not contact["known"] or contact["holding"] != expected_holding:
            reasons.append("endpoint_holding_mismatch")
    effects = []
    placements = _on_placements(post, final.bindings) if final is not None else []
    if placements and (not contact["known"] or contact["holding"] is not None):
        reasons.append("endpoint_placement_not_released")
    for source, target in placements:
        if (source, target) not in map(tuple, contact["supports"]):
            reasons.append("endpoint_designated_support_missing")
        # Only one completed transfer is represented by this Put abstraction.
        # Multiple releases / different movers need their own block summaries.
        movers = {
            row["mover"] for row in audit["instructions"] if row["mover"] is not None
        }
        releases = [
            row
            for row in audit["instructions"]
            if row["kind"] in ("Release", "ReleaseByName")
        ]
        if movers - {source} or len(releases) > 1:
            reasons.append("unsupported_multi_transfer_put_audit")
            continue
        if not releases and not movers:
            # An executable identity can preserve an already true milestone;
            # it does not claim to perform an enabled Put.
            continue
        before = scene_at(segment, segment.t_start)
        effect = audit_put_effects(before, final, source, target)
        effects.append(effect.details)
        reasons.extend(effect.reasons)
    return EndpointResult(
        not reasons,
        tuple(sorted(set(reasons))),
        {"physical": audit, "put_effects": effects},
    )


def check_next_action(candidate, segment, states, next_bindings):
    """Nominal readiness at the *actual* endpoint, under the next bindings.

    This checks the candidate's first physical action after resolving its logical
    prefix. It does not physically execute that next action or prove all future
    iterations. A terminal iteration has no continuation obligation.
    """
    if next_bindings is None:
        return EndpointResult(True, details={"continuation": "terminal"})
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
    from synthesis.cfg.contract_search import table_reference_at
    from synthesis.cfg.search_feasibility import resolve_entry_bindings
    from synthesis.search.contract import (
        Action,
        ContractConfig,
        ContractProblem,
        SearchExample,
    )

    audit = getattr(states, "contract_audit", None)
    if audit is None or not audit["endpoint"]["known"]:
        return EndpointResult(False, ("unknown_handoff_holding",))
    scene = endpoint_scene(states, segment, bindings=next_bindings)
    if scene is None or scene.observation is None:
        return EndpointResult(False, ("missing_handoff_observation",))
    prefix, physical = [], None
    for instruction in candidate.instructions:
        if isinstance(instruction, (Assign, Get, Skip)):
            prefix.append(instruction)
        else:
            physical = instruction
            break
    kinds = (
        (Pick, PickByName, "Pick"),
        (Move, MoveByName, "Move"),
        (Release, ReleaseByName, "Release"),
    )
    kind = next(
        (
            name
            for numeric, named, name in kinds
            if isinstance(physical, (numeric, named))
        ),
        None,
    )
    if kind is None:
        return EndpointResult(False, ("unsupported_handoff_first_action",))
    try:
        bindings = resolve_entry_bindings(prefix, scene)
        operands = tuple(row["val"] for row in physical.get_operand())
        offsets = []
        physical.register_trainable_parameter(offsets)
        action = Action(kind, operands, tuple(offsets))
        positions = np.stack(
            [scene.positions[i] for i in range(segment.trace.num_blocks)]
        )
        gripper = np.asarray(scene.observation[:3])
        example = SearchExample(
            positions,
            gripper,
            bindings,
            gripper[None],
            positions[None],
            initial_holding=audit["endpoint"]["holding"],
            table_reference=table_reference_at(segment),
        )
        named = any(isinstance(value, str) for value in operands)
        problem = ContractProblem(
            [example],
            lambda state, index: False,
            allowed_operands=(
                tuple(sorted(bindings))
                if named
                else tuple(range(segment.trace.num_blocks))
            ),
            config=ContractConfig(observation_references=named),
            actions=[action],
        )
        admitted = problem.transition(problem.initial, action) is not None
        return EndpointResult(
            admitted,
            () if admitted else ("next_action_not_ready",),
            {
                "action": action.row(),
                "bindings": bindings,
                "scope": "existing nominal first-action contract at observed endpoint; no physical next-action execution",
            },
        )
    except (ValueError, KeyError, TypeError) as exc:
        return EndpointResult(False, ("unsupported_handoff",), {"error": str(exc)})
