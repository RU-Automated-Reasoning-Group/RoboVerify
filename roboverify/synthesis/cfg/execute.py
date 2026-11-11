from synthesis.cfg.lower import lower
from synthesis.cfg.refine import scene_at
from synthesis.cfg.reset import reset_segment
from synthesis.predicates.scene import scene_from_obs


def execute_current(
    program, env, initial_observation, on_state=None, *, on_instruction=None
):
    class Trajectory(list):
        def append(self, state):
            super().append(state)
            if on_state is not None:
                on_state(state, dict(env.symbolic_name_to_box_id))

    trajectory = Trajectory()
    trajectory.append(initial_observation)
    paths = {}
    if on_instruction is not None:
        from synthesis.api.instructions import While
        from synthesis.api.runtime import execute_instruction

        def collect(instructions, prefix=""):
            for index, instruction in enumerate(instructions):
                path = f"{prefix}{index}"
                paths[path] = instruction
                if isinstance(instruction, While):
                    collect(instruction.body, f"{path}.")

        collect(program.instructions)

        def event(row):
            if row["kind"] in ("instruction_start", "instruction_end"):
                on_instruction(
                    paths[row["path"]],
                    "start" if row["kind"] == "instruction_start" else "end",
                    trajectory[-1],
                )

    for index, instruction in enumerate(program.instructions):
        if on_instruction is None:
            instruction.eval(env, trajectory)
        else:
            execute_instruction(
                instruction, env, trajectory, path=str(index), on_event=event
            )
        if on_state is not None:
            on_state(trajectory[-1], dict(env.symbolic_name_to_box_id))
    return trajectory


def execute_cfg(cfg, context, env_factory, *, reset_mode="replay"):
    """Generate negatives from the current graph, never a random unrelated program."""
    from synthesis.api.guard_eval import NoGuardWitness
    from synthesis.api.instructions import Get, LoopBudgetExceeded
    from synthesis.api.program import Program
    from synthesis.cfg.lower import lower_region
    from synthesis.cfg.region import BlockRegion
    from synthesis.predicates.term import to_z3

    instructions = []
    for name in cfg.order:
        region = cfg.nodes[name].region
        if region is None or (isinstance(region, BlockRegion) and not region.physical):
            # A failed proposal can retain an empty BlockRegion. Treat that as
            # the first unresolved physical block, just like region=None; its
            # reached prefix states still supply refinement negatives.
            break
        edge = cfg.incoming(name)[0]
        if edge.binds:
            names = sorted(edge.binds)
            instructions.append(
                Get(
                    names[0],
                    to_z3(edge.binding_condition, context),
                    [context.get_consts(n) for n in names],
                    guard_term=edge.binding_condition,
                )
            )
        instructions.extend(lower_region(region, context, physical=True))
    program = Program(len(instructions), instructions)
    output = []
    segments = cfg.demos.for_node(cfg.order[0])
    for segment in segments:
        env = env_factory(segment.trace)
        try:
            first = reset_segment(env, segment, mode=reset_mode)

            def record(obs, bindings):
                output.append(
                    scene_from_obs(
                        obs,
                        segment.trace.num_blocks,
                        bindings,
                        include_table="tbl" in bindings,
                        entry_obs=segment.trace.states[segment.entry_index],
                    )
                )

            try:
                execute_current(program, env, first, on_state=record)
            except (NoGuardWitness, LoopBudgetExceeded):
                # A partial candidate can stop at a failed Get; its actual
                # prefix states still provide classifier negatives.
                pass

        finally:
            env.close()
    return output
