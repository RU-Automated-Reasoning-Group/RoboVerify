"""Thin shared interface over existing controllers and predicate semantics."""

import inspect
import itertools
import math
import uuid
from dataclasses import asdict

from .contracts import (
    ACTIONS,
    API_NAMES,
    GOAL_POSITION_QUERIES,
    GOAL_PREDICATES,
    GOAL_TASKS,
    POSITION_QUERIES,
    PREDICATES,
    QUERIES,
    TRAVERSALS,
    CapError,
    GoalHandle,
    ObjectHandle,
    Position3,
    PrimitiveFailure,
    task_api_names,
    task_input_types,
    task_inputs,
)


class RoboVerifyAdapter:
    def __init__(self, env, logger, budget, *, task="stack"):
        from synthesis.cfg.reset import inner_env

        self.env, self.inner, self.logger, self.budget = (
            env,
            inner_env(env),
            logger,
            budget,
        )
        trial = uuid.uuid4().hex
        self.goal_layout = None
        if task in GOAL_TASKS:
            # Goal tasks have no designated base block (Revision 5, Grid/Pyramid).
            self.goal_layout = getattr(self.inner, "goal_layout", None)
            if self.goal_layout is None:
                raise CapError(f"{task} requires an environment with a goal layout")
            base = None
        else:
            base = self.inner.roboverify_base_block_id
            if self.inner.symbolic_name_to_box_id.get("b0") != base:
                raise CapError(
                    "Snapshot b0 binding does not match the environment's designated base"
                )
        self.handles = tuple(
            ObjectHandle("b0" if i == base else f"object_{uuid.uuid4().hex}", trial)
            for i in range(self.inner.num_blocks)
        )
        self.ids = {handle: i for i, handle in enumerate(self.handles)}
        self.b0 = self.handles[base] if base is not None else None
        # tbl is a Move/Release reference only: never picked, read or queried.
        self.tbl = ObjectHandle("tbl", trial) if "tbl" in task_inputs(task) else None
        if self.goal_layout is not None:
            # Goal 0 is the layout head h0; null is the empty goal.
            self.goals = tuple(
                GoalHandle("h0" if i == 0 else f"goal_{uuid.uuid4().hex}", trial)
                for i in range(len(self.goal_layout.goals))
            )
            self.goal_ids = {handle: i for i, handle in enumerate(self.goals)}
            self.null = GoalHandle("null", trial)
            self.inputs = {"h0": self.goals[0], "null": self.null}
        else:
            self.goals, self.goal_ids, self.null = (), {}, None
            self.inputs = {"b0": self.b0} | ({"tbl": self.tbl} if self.tbl else {})
        self.input_types = task_input_types(task)
        # Physical names of the inputs given to the policy, for task scoring.
        self.input_bindings = (
            {}
            if self.goal_layout is not None
            else {"b0": base} | ({"tbl": "tbl"} if self.tbl else {})
        )
        self.entry_observation = self.observe()
        self.trajectory = [self.entry_observation.copy()]
        self.query_depth = 0
        self.api_names = task_api_names(task)
        # Counted independently: a generated policy may catch PrimitiveFailure.
        self.unconverged_primitives = 0

    def observe(self):
        return self.inner.flatten_observation(self.inner._get_obs()).copy()

    def object_id(self, handle):
        if isinstance(handle, GoalHandle):
            raise CapError("A goal is not a block; use it as a Move/Release reference")
        if self.tbl is not None and handle == self.tbl:
            raise CapError(
                "tbl is not a block; use it only as a Move/Release reference"
            )
        if not isinstance(handle, ObjectHandle) or handle not in self.ids:
            raise CapError("Unknown or cross-trial object handle")
        return self.ids[handle]

    def reference(self, handle):
        """Physical block ID, or the table for a Move/Release reference."""
        if self.tbl is not None and handle == self.tbl:
            return "tbl"
        return self.object_id(handle)

    def goal_index(self, handle):
        """Goal index for a goal handle; None for null."""
        if self.null is not None and handle == self.null:
            return None
        if not isinstance(handle, GoalHandle) or handle not in self.goal_ids:
            raise CapError("Unknown or cross-trial goal handle")
        return self.goal_ids[handle]

    def goal_equal(self, left, right):
        self.goal_index(left), self.goal_index(right)
        return left == right

    def located_goal(self, handle, name):
        index = self.goal_index(handle)
        if index is None:
            raise CapError(f"{name} requires a goal location; null has none")
        return index

    def occupied(self):
        """[block, goal] occupancy on the current observation (20 mm, 3D)."""
        import numpy as np

        from synthesis.environment.goal_layout import occupancy

        obs = self.observe()
        blocks = np.array(
            [obs[10 + 12 * i : 13 + 12 * i] for i in range(self.inner.num_blocks)]
        )
        return occupancy(blocks, self.goal_layout.positions)

    def namespace(self):
        return self.inputs | {name: self._wrapper(name) for name in self.api_names}

    def _wrapper(self, name):
        def call(*args):
            return self.call(name, list(args))

        return call

    def call(self, name, args):
        self.budget.tick()
        if name not in self.api_names:
            raise CapError(f"API unavailable for this task: {name}")
        if name in GOAL_POSITION_QUERIES:
            if len(args) != 1:
                raise CapError(f"{name} requires 1 argument")
            index = self.located_goal(args[0], name)
            result = Position3(self.goal_layout.goals[index].position)
            self.logger.emit(
                "coordinate_query", query=name, arguments=args, result=result
            )
            return result
        if name in GOAL_PREDICATES:
            if len(args) != 1:
                raise CapError(f"{name} requires 1 argument")
            occupied = self.occupied()
            if name == "AtGoal":
                result = bool(occupied[self.object_id(args[0])].any())
            else:
                index = self.goal_index(args[0])
                result = False if index is None else bool(occupied[:, index].any())
            self.logger.emit("predicate", predicate=name, arguments=args, result=result)
            return result
        if name in TRAVERSALS:
            if self.query_depth:
                raise CapError("Queries cannot update goal cursors")
            if len(args) != 1:
                raise CapError(f"{name} requires 1 argument")
            index = self.goal_index(args[0])
            if index is None:
                raise CapError(f"{name} cannot advance null")
            goal = self.goal_layout.goals[index]
            following = goal.r if name == "MoveRight" else goal.d
            result = self.null if following is None else self.goals[following.index]
            self.logger.emit(
                "goal_traversal", statement=name, argument=args[0], result=result
            )
            return result
        if name in POSITION_QUERIES:
            expected = 0 if name == "get_gripper_position" else 1
            if len(args) != expected:
                raise CapError(f"{name} requires {expected} arguments")
            start = 0 if not args else 10 + 12 * self.object_id(args[0])
            result = Position3(float(v) for v in self.observe()[start : start + 3])
            self.logger.emit(
                "coordinate_query", query=name, arguments=args, result=result
            )
            return result
        if name in QUERIES:
            if len(args) != 1 or not callable(args[0]):
                raise CapError("A query requires one callable condition")
            query = args[0]
            arity = getattr(query, "arity", None)
            if arity is None:
                arity = len(inspect.signature(query).parameters)
            if not 1 <= arity <= self.budget.max_depth or (
                name == "get" and arity != 1
            ):
                raise CapError("Invalid query binder count")
            self.query_depth += 1
            candidate_results = []
            evaluated_count = 0
            try:
                for objects in itertools.product(self.handles, repeat=arity):
                    self.budget.tick()
                    value = query(*objects)
                    evaluated_count += 1
                    if len(candidate_results) < 16:
                        candidate_results.append({"objects": objects, "result": value})
                    if type(value) is not bool:
                        raise CapError("Query condition must return a Boolean")
                    if (name in {"exists", "get"} and value) or (
                        name == "forall" and not value
                    ):
                        result = objects[0] if name == "get" else value
                        self.logger.emit(
                            "query_result", query=name, result=result, witnesses=objects
                        )
                        return result
                if name == "get":
                    self.logger.emit(
                        "query_failure",
                        query=name,
                        reason="no_witness",
                        candidates=candidate_results,
                        evaluated_count=evaluated_count,
                        omitted_candidates=evaluated_count - len(candidate_results),
                    )
                    raise CapError("No object satisfies get condition")
                result = name == "forall"
                self.logger.emit("query_result", query=name, result=result)
                return result
            finally:
                self.query_depth -= 1
        if name in PREDICATES:
            from synthesis.predicates.scene import evaluate, scene_from_obs
            from synthesis.predicates.term import atom, block_id, ref

            if len(args) != 2:
                raise CapError(f"{name} requires two objects")
            ids = [self.reference(arg) for arg in args]
            scene = scene_from_obs(
                self.observe(),
                self.inner.num_blocks,
                include_table=self.tbl is not None,
                entry_obs=self.entry_observation,
            )
            terms = (ref(i) if i == "tbl" else block_id(i) for i in ids)
            result = evaluate(atom(name, *terms), scene)
            self.logger.emit("predicate", predicate=name, arguments=args, result=result)
            return result
        if name not in ACTIONS:
            raise CapError(f"Unknown DSL API: {name}")
        if self.query_depth:
            raise CapError("Queries cannot execute physical actions")
        from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName

        count = {"Pick": 1, "Move": 4, "Release": 2}[name]
        if len(args) != count:
            raise CapError(f"{name} requires {count} arguments")
        references = args[:3] if name == "Move" else args[:1]
        aliases = []
        for handle in references:
            if name == "Pick" and self.tbl is not None and handle == self.tbl:
                raise CapError("Pick requires a block; tbl cannot be picked")
            if isinstance(handle, GoalHandle):
                if name == "Pick":
                    raise CapError("Pick requires a block; goals cannot be picked")
                goal = self.located_goal(handle, name)
                from synthesis.environment.goal_layout import GoalRef

                alias = f"cap_goal_{goal}"
                self.inner.symbolic_name_to_goal[alias] = GoalRef(goal)
                aliases.append(alias)
                continue
            physical_id = self.reference(handle)
            alias = "cap_table" if physical_id == "tbl" else f"cap_object_{physical_id}"
            self.inner.symbolic_name_to_box_id[alias] = physical_id
            aliases.append(alias)
        if name == "Pick":
            instruction = PickByName(*aliases)
        elif name == "Move":
            offsets = args[3]
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 3:
                raise CapError("Move requires three finite offsets")
            instruction = MoveByName(
                *aliases, target_offset=[finite(v) for v in offsets]
            )
        else:
            instruction = ReleaseByName(*aliases, target_z=finite(args[1]))
        contract = {}
        if getattr(self.logger, "primitive_phase_feedback_enabled", False):
            contract = {
                "controller_config": asdict(instruction.control),
                "step_limit": instruction.limit,
            }
        event = self.logger.emit(
            "primitive_start", primitive=name, arguments=args, **contract
        )
        previous_observer = getattr(self.env, "_cap_control_event", None)

        def observe_control(**fields):
            self.logger.emit(
                "controller_phase", primitive=name, primitive_event_id=event, **fields
            )

        self.env._cap_control_event = observe_control
        try:
            with self.logger.scoped(primitive=name, primitive_event_id=event):
                instruction.eval(self.env, self.trajectory)
        finally:
            if previous_observer is None:
                del self.env._cap_control_event
            else:
                self.env._cap_control_event = previous_observer
        result = instruction.last_control_result
        self.logger.emit(
            "primitive_end",
            primitive=name,
            parent_event_id=event,
            converged=bool(result and result.converged),
            steps=result.steps if result else 0,
            position_error=result.position_error if result else None,
            controller_phase=result.phase if result else None,
        )
        if not result or not result.converged:
            self.unconverged_primitives += 1
            raise PrimitiveFailure(
                f"{name} did not converge within its shared primitive budget"
            )
        return None


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CapError("Action parameters must be finite numbers")
    return float(value)


class GenerationOnlyAdapter:
    """Expose signatures/identity for a generation diagnostic, never a rollout."""

    def __init__(self, task="stack"):
        self.input_types = task_input_types(task)
        self.inputs = {
            name: (GoalHandle if kind == "goal" else ObjectHandle)(
                name, "generation-only"
            )
            for name, kind in self.input_types
        }
        self.b0 = self.inputs.get("b0")
        self.api_names = task_api_names(task)

    def call(self, name, args):
        raise CapError("Environment calls are unavailable in generation-only mode")

    def namespace(self):
        return self.inputs | {name: self._wrapper(name) for name in self.api_names}

    def _wrapper(self, name):
        return lambda *args: self.call(name, args)
