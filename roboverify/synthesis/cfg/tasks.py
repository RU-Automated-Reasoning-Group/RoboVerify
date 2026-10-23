"""Task conditions shared by collection, search, and verification."""

from synthesis.predicates.term import (
    atom,
    conjunction,
    disjunction,
    forall,
    implies,
    negate,
    ref,
)


def task_spec(task):
    x, y, b0 = ref("x"), ref("y"), ref("b0")
    if task == "stack":
        different = negate(atom("eq", x, y))
        return (
            conjunction(
                forall(
                    ["x", "y"],
                    implies(
                        different,
                        conjunction(
                            negate(atom("ON_star", x, y)), atom("Scattered", x, y)
                        ),
                    ),
                ),
                # Both ordered pairs enforce one initial height level; concrete
                # observations use the configured Higher tolerance.
                forall(["x", "y"], atom("Higher", x, y)),
            ),
            forall(["x"], atom("ON_star", x, b0)),
        )
    if task == "unstack":
        return (
            forall(
                ["x"], disjunction(atom("eq", x, ref("tbl")), atom("ON_star", x, b0))
            ),
            forall(["x", "y"], implies(atom("ON_star", x, y), atom("eq", x, y))),
        )
    if task == "reverse":
        physical_x = negate(atom("eq", x, ref("tbl")))
        physical_y = negate(atom("eq", y, ref("tbl")))
        frozen = atom("ON_star_zero", x, y)
        current = atom("ON_star", x, y)
        reversed_order = atom("ON_star", y, x)
        return (
            conjunction(
                forall(["x"], implies(physical_x, atom("ON_star", x, b0))),
                forall(
                    ["x", "y"],
                    conjunction(implies(frozen, current), implies(current, frozen)),
                ),
            ),
            forall(
                ["x", "y"],
                implies(
                    conjunction(physical_x, physical_y),
                    conjunction(
                        implies(frozen, reversed_order), implies(reversed_order, frozen)
                    ),
                ),
            ),
        )
    raise ValueError(f"Unsupported task: {task}")


def task_identity(task):
    if task in ("grid", "pyramid"):
        from synthesis.environment.goal_layout import GOAL_TOLERANCE

        return dict(
            task=task,
            semantics="goal-occupancy-v1",
            goal_tolerance=GOAL_TOLERANCE,
            precondition="All goals empty; physical blocks pairwise Scattered; one block per goal available",
            postcondition="Every goal occupied by a physical block",
        )
    pre, post = task_spec(task)
    return {"task": task, "precondition": str(pre), "postcondition": str(post)}
