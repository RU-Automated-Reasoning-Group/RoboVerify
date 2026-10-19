"""Thin shared interface over existing controllers and predicate semantics."""

import inspect
import itertools
import math
import uuid

from .contracts import (
    ACTIONS,
    API_NAMES,
    POSITION_QUERIES,
    PREDICATES,
    QUERIES,
    CapError,
    ObjectHandle,
    Position3,
    PrimitiveFailure,
    task_api_names,
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
        self.b0 = self.handles[base]
        self.entry_observation = self.observe()
        self.trajectory = [self.entry_observation.copy()]
        self.query_depth = 0
        self.api_names = task_api_names(task)

    def observe(self):
        return self.inner.flatten_observation(self.inner._get_obs()).copy()

    def object_id(self, handle):
        if not isinstance(handle, ObjectHandle) or handle not in self.ids:
            raise CapError("Unknown or cross-trial object handle")
        return self.ids[handle]

    def namespace(self):
        return {"b0": self.b0} | {name: self._wrapper(name) for name in self.api_names}

    def _wrapper(self, name):
        def call(*args):
            return self.call(name, list(args))

        return call

    def call(self, name, args):
        self.budget.tick()
        if name not in self.api_names:
            raise CapError(f"API unavailable for this task: {name}")
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
            from synthesis.predicates.term import atom, block_id

            if len(args) != 2:
                raise CapError(f"{name} requires two objects")
            ids = [self.object_id(arg) for arg in args]
            scene = scene_from_obs(
                self.observe(), self.inner.num_blocks, entry_obs=self.entry_observation
            )
            result = evaluate(atom(name, *(block_id(i) for i in ids)), scene)
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
            physical_id = self.object_id(handle)
            alias = f"cap_object_{physical_id}"
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
        event = self.logger.emit("primitive_start", primitive=name, arguments=args)
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
        self.b0 = ObjectHandle("b0", "generation-only")
        self.api_names = task_api_names(task)

    def call(self, name, args):
        raise CapError("Environment calls are unavailable in generation-only mode")

    def namespace(self):
        return {"b0": self.b0} | {name: self._wrapper(name) for name in self.api_names}

    def _wrapper(self, name):
        return lambda *args: self.call(name, args)
