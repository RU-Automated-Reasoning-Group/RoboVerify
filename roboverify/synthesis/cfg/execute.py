from synthesis.cfg.lower import lower
from synthesis.cfg.refine import scene_at
from synthesis.cfg.reset import reset_segment
from synthesis.predicates.scene import scene_from_obs


def execute_current(program, env, initial_observation):
    trajectory = [initial_observation]
    for instruction in program.instructions:
        instruction.eval(env, trajectory)
    return trajectory


def execute_cfg(cfg, context, env_factory, *, reset_mode="replay"):
    """Generate negatives from the current graph, never a random unrelated program."""
    from synthesis.api.instructions import Get
    from synthesis.api.program import Program
    from synthesis.cfg.lower import lower_region
    from synthesis.predicates.term import to_z3

    instructions = []
    for name in cfg.order:
        region = cfg.nodes[name].region
        if region is None:
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
            states = execute_current(program, env, first)
            output.extend(
                scene_from_obs(
                    obs,
                    segment.trace.num_blocks,
                    segment.bindings,
                    include_table="tbl" in segment.bindings,
                    entry_obs=first,
                )
                for obs in states
            )
        finally:
            env.close()
    return output
