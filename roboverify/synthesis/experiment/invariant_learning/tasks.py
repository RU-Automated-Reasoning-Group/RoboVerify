"""Environment adapter boundary and the initial Stack implementation."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import z3

from synthesis.cfg.collection import (
    STACK_SETTLING_STEPS,
    execution_deadline,
    record_execution,
    validate_trace,
)
from synthesis.cfg.demos import DemoTrace
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import loop_store
from synthesis.cfg.reset import capture, inner_env
from synthesis.cfg.tasks import task_identity, task_spec
from synthesis.cfg.verification import (
    propose_summaries,
    symbolic_problem,
    verify_cfg_motion,
)
from synthesis.cfg.verified_synthesis import loop_regions
from synthesis.experiment.invariant_learning.witness import (
    WitnessQuery,
    missing_head_query,
)
from synthesis.inference_lib.demo_store import (
    InferenceVocabulary,
    LoopHeadState,
    observation_positions,
)
from synthesis.util import on
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.counterexamples import state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.symbolic_verify import (
    SymbolicVerificationResult,
    discharge_vc,
)


class ExperimentTask(Protocol):
    """Adapters own domain semantics; the runner owns inference and iteration.

    An adapter must preserve the fixed executable across sizes, validate complete
    simulator executions, and expose all physical heads/exits with frozen entry
    geometry. search_query returns a bounded *initial-state* coverage query.
    """

    context: object
    vocabulary: InferenceVocabulary
    loop_id: str

    def describe(self) -> dict: ...
    def set_invariant(self, invariant): ...
    def verify_symbolic(self, timeout_ms): ...
    def search_query(self, size, timeout_ms): ...
    def execute(self, search, *, seed, timeout_seconds, video_path): ...
    def validate(self, trace) -> bool: ...
    def learning_states(self, trace): ...
    def verify_motion(self, options): ...


class ExperimentFailure(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


@dataclass
class InitialScene:
    positions: list
    base_xy: list
    gripper: list
    height: float


class StackExperiment:
    """Stack scenes within the existing reset domain; no training demos required."""

    def __init__(
        self, source, *, relations=None, variables=2, max_loop_iterations=None
    ):
        self.source = source
        self.context = HighLevelContext()
        self.definition = load_program(source, self.context, 1)
        if self.definition.initial_bindings != {"b0": 0}:
            raise ValueError(
                "Stack experiment requires only the fixed entry alias b0=0"
            )
        self.cfg = propose_summaries(
            program_to_cfg(self.definition, [], *task_spec("stack")), self.context
        )
        loops = list(loop_regions(self.cfg))
        if len(loops) != 1:
            raise ValueError("The experiment supports one non-nested loop")
        self.loop = loops[0]
        from synthesis.api.instructions import Assign, Get, Skip, While

        index, _ = next(
            (i, row)
            for i, row in enumerate(self.definition.program.instructions)
            if isinstance(row, While)
        )
        if any(
            not isinstance(row, (Assign, Get, Skip))
            for row in self.definition.program.instructions[:index]
        ):
            raise ValueError("Stack prefix must preserve initial block geometry")
        self.loop_id = str(index)
        names = tuple(
            sorted(self.loop.body_cfg.initial_scope - set(self.loop.exists_vars))
        )
        self.vocabulary = InferenceVocabulary(
            variables,
            tuple(relations or ("ON_star", "Higher", "Scattered", "equality")),
            names,
        )
        self.max_loop_iterations = max_loop_iterations
        self._geometry = None
        self.set_invariant(z3.BoolVal(False))

    def describe(self):
        return dict(
            task=task_identity("stack"),
            program=self.definition.metadata,
            vocabulary=dict(
                k=self.vocabulary.k,
                relations=self.vocabulary.relations,
                constants=self.vocabulary.constants,
            ),
            initial_domain="Stack reset workspace; equal tabletop heights",
            settling_steps=STACK_SETTLING_STEPS,
            execution_bound=(
                self.max_loop_iterations
                if self.max_loop_iterations is not None
                else "num_blocks - 1"
            ),
        )

    def set_invariant(self, invariant):
        self.loop.invariant = invariant

    def verify_symbolic(self, timeout_ms):
        program, pre, post, context = symbolic_problem(self.cfg, self.context)
        return SymbolicVerificationResult(
            [
                discharge_vc(vc, context, timeout_ms)
                for vc in program.VC_gen(pre, post, context)
            ],
            context=context,
            scope="unbounded",
        )

    def _definition(self, size):
        definition = load_program(self.source, self.context, size)
        if (
            definition.metadata["fingerprint"]
            != self.definition.metadata["fingerprint"]
            or definition.initial_bindings != self.definition.initial_bindings
        ):
            raise ExperimentFailure(
                "program_changed", "Program factory changed with block count"
            )
        return definition

    def _read_geometry(self):
        if self._geometry is None:
            from synthesis.mcmc.synthesis import (
                make_roboverify_stack_env,
                preserved_global_rng,
                set_np_seed,
            )

            with preserved_global_rng():
                set_np_seed(0)
                env = make_roboverify_stack_env(num_blocks=1)
                try:
                    inner = inner_env(env)
                    self._geometry = (
                        inner.robot_base_xy.copy(),
                        inner.initial_gripper_xpos.copy(),
                        float(inner.height_offset),
                    )
                finally:
                    env.close()
        return self._geometry

    def search_query(self, size, timeout_ms):
        from synthesis.environment.stack_reset import (
            STACK_GRIPPER_CLEARANCE,
            STACK_MAX_BASE_DISTANCE,
            STACK_X_RANGE,
            STACK_Y_RANGE,
        )

        self._definition(size)
        program, pre, _, context = symbolic_problem(self.cfg, self.context, size)
        iterations = (
            size - 1 if self.max_loop_iterations is None else self.max_loop_iterations
        )
        coverage = missing_head_query(program, context, iterations)
        base, gripper, height = self._read_geometry()
        solver = context.new_solver(timeout_ms)
        solver.add(pre, context.get_consts("b0") == context.enum_blocks[0])
        positions = [
            tuple(fresh_const(z3.RealSort(), f"initial_{axis}") for axis in "xyz")
            for _ in range(size)
        ]
        for x, y, z in positions:
            dx, dy = x - float(base[0]), y - float(base[1])
            solver.add(
                dx >= STACK_X_RANGE[0],
                dx <= STACK_X_RANGE[1],
                dy >= STACK_Y_RANGE[0],
                dy <= STACK_Y_RANGE[1],
                dx * dx + dy * dy <= STACK_MAX_BASE_DISTANCE**2,
                (x - float(gripper[0])) ** 2 + (y - float(gripper[1])) ** 2
                >= STACK_GRIPPER_CLEARANCE**2,
                z == height,
            )
        length = z3.RealVal(str(on.BLOCK_LENGTH))
        for i, a in enumerate(context.enum_blocks):
            for j, b in enumerate(context.enum_blocks):
                p, q = positions[i], positions[j]
                above = z3.And(
                    z3.Abs(p[0] - q[0]) < length / 2,
                    z3.Abs(p[1] - q[1]) < length / 2,
                    p[2] >= q[2],
                )
                solver.add(
                    context.ON_star(a, b) == above,
                    context.ON_star_zero(a, b) == above,
                    context.Higher(a, b) == on.higher_z3(p[2], q[2]),
                    context.Scattered(a, b)
                    == z3.Or(
                        z3.Abs(p[0] - q[0]) >= 2 * length,
                        z3.Abs(p[1] - q[1]) >= 2 * length,
                    ),
                )

        def decode(model):
            def number(v):
                value = model.eval(v)
                return float(
                    (
                        value if z3.is_rational_value(value) else value.approx(30)
                    ).as_fraction()
                )

            xyz = [[number(v) for v in pos] for pos in positions]
            return InitialScene(xyz, base.tolist(), gripper.tolist(), height)

        return WitnessQuery(solver, coverage, decode, iterations, context)

    def execute(self, search, *, seed, timeout_seconds, video_path):
        from synthesis.api.control import get_move_action
        from synthesis.mcmc.synthesis import (
            make_roboverify_stack_env,
            preserved_global_rng,
            set_np_seed,
        )

        definition = self._definition(search.size)
        scene = search.witness
        with preserved_global_rng(), execution_deadline(timeout_seconds):
            set_np_seed(seed)
            env = make_roboverify_stack_env(num_blocks=search.size)
            try:
                inner = inner_env(env)
                # Initialize all robot/control state normally, then replace only
                # block free joints using the solver scene, before preparation.
                env.reset()
                inner.sim.set_state(inner.initial_state)
                for name, xyz in zip(inner.object_names, scene.positions):
                    qpos = inner.sim.data.get_joint_qpos(f"{name}:joint").copy()
                    qpos[:3] = xyz
                    inner.sim.data.set_joint_qpos(f"{name}:joint", qpos)
                    inner.sim.data.set_joint_qvel(f"{name}:joint", np.zeros(6))
                inner.sim.forward()
                first = inner.flatten_observation(inner._get_obs())
                target = first[:3].copy()
                for _ in range(STACK_SETTLING_STEPS):
                    env.step(get_move_action(first, target, close_gripper=False))
                    first = inner.flatten_observation(inner._get_obs())
                inner.symbolic_name_to_box_id = dict(definition.initial_bindings)
                snapshot = capture(env)
                positions = observation_positions(first, search.size)
                positions.pop("tbl")
                bindings = {"b0": "x1"}
                bindings.update(
                    {
                        str(obj): f"x{i+1}"
                        for i, obj in enumerate(search.query.context.enum_blocks)
                    }
                )
                settled = LoopHeadState("entry", positions, positions, bindings)
                _, pre, _, _ = symbolic_problem(self.cfg, self.context, search.size)
                # Both expressions use names/relations, evaluated over the actual
                # settled geometry, not the solver's pre-settling coordinates.
                if not state_holds(pre, settled) or not state_holds(
                    search.query.coverage, settled
                ):
                    return DemoTrace(
                        (first,),
                        (snapshot,),
                        ((), (0,)),
                        seed,
                        "stack",
                        search.size,
                        (),
                        dict(
                            definition.metadata,
                            status="realization_mismatch",
                            failure_kind="realization_mismatch",
                            reason="Settled scene does not satisfy the initial-state coverage query",
                            task_spec=task_identity("stack"),
                            higher_tolerance=on.get_higher_tolerance(),
                            counterexample_initialization=dict(
                                source="solver",
                                settling_steps=STACK_SETTLING_STEPS,
                                positions=scene.positions,
                            ),
                        ),
                    )
            finally:
                env.close()
            trace = record_execution(
                definition,
                seed=seed,
                num_blocks=search.size,
                task="stack",
                initial_snapshot=snapshot,
                max_loop_iterations=search.query.iterations,
                timeout_seconds=timeout_seconds,
                video_path=video_path,
            )
        trace.metadata["task_spec"] = task_identity("stack")
        trace.metadata["counterexample_initialization"] = dict(
            source="solver",
            settling_steps=STACK_SETTLING_STEPS,
            positions=scene.positions,
        )
        return trace

    def validate(self, trace):
        return validate_trace(trace)

    def learning_states(self, trace):
        return loop_store(
            [trace], loop_id=self.loop_id, names=self.vocabulary.constants
        ).for_loop(self.loop_id)

    def verify_motion(self, options):
        return verify_cfg_motion(self.cfg, self.context, **options)


TASKS = {"stack": StackExperiment}
