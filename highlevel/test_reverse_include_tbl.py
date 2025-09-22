from z3 import *

# set_option("smt.core.minimize", "true")

solver = Solver()
solver.set(unsat_core=True)

Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13'])
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
x, y, c, n0, t, next_box, x1, tbl = Consts("x y c n0 t next_box x1 tbl", Box)


def define_ON_func(func_name: str, solver: Solver):
    ON_star = Function(func_name, Box, Box, BoolSort())
    solver.assert_and_track(
        ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))),
        f"{func_name}_1",
    )
    solver.assert_and_track(ForAll([x], ON_star(x, x)), f"{func_name}_2")
    solver.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))
            ),
        ),
        f"{func_name}_3",
    )
    solver.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))
            ),
        ),
        f"{func_name}_4",
    )
    solver.assert_and_track(
        ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))),
        f"{func_name}_5",
    )
    solver.assert_and_track(
        ForAll(
            [x],
            Or(
                Exists(
                    [y],
                    And(
                        (y != x),
                        ForAll(
                            [x1], Implies(And(x1 != x, ON_star(x, x1)), ON_star(y, x1))
                        ),
                    ),
                ),
                on_table(x, ON_star),
            ),
        ),
        f"{func_name}_6",
    )
    solver.assert_and_track(
        ForAll([x], Implies(Or(ON_star(x, tbl), ON_star(tbl, x)), x == tbl)),
        f"{func_name}_tbl"
    )
    return ON_star


def func_equiv(func1, func2):
    return ForAll([x, y], func1(x, y) == func2(x, y))


def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # print(f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}")
        print("----------------")
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(
                    f"ON_star({name1}, {name2}): {x.model().evaluate(ON_star(box1, box2))}"
                )
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(
                    f"ON_star_zero({name1}, {name2}): {x.model().evaluate(ON_star_zero(box1, box2))}"
                )
        import pdb

        pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())


def precondition():
    return ForAll([x], Implies(x != tbl, ON_star(x, n0)))


def postcondition():
    return And(
        ForAll([x], ON_star(n0, x)),
        ForAll([x, y], ON_star_zero(x, y) == ON_star(y, x))
    )


def postcondition_substituted(b_prime, b):
    return And(
        Not(ON_star(b, b_prime)),
        ForAll([x], ON_func_substituted(n0, x, b_prime, b, ON_star)),
        ForAll([x, y], ON_star_zero(x, y) == ON_func_substituted(y, x, b_prime, b, ON_star))
    )


# def postcondition_top():
# return ForAll([x], Implies(ON_star_zero(x, n0), top(x, ON_star)))


# def postcondition_on_table():
# return ForAll([x], Implies(ON_star_zero(x, n0), on_table(x, ON_star)))


def top(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))


def on_table(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))


def while_cond():
    return Exists([x], And(top(x, ON_star), ON_star(x, n0), x != n0))


def while_cond_instance(x):
    return And(top(x, ON_star), ON_star(x, n0), x != n0)

# def loop_invariant_substituted_for_precondition(next_box):
    # return ForAll(
        # [x, y],
        # Or(
            # And(ON_func_substituted(x, n0, next_box, ON_star), ON_func_substituted(x, y, next_box, ON_star) == ON_func_substituted(x, y, next_box, ON_star_zero)),
            # And(Not(ON_func_substituted(x, n0, next_box, ON_star)), ON_func_substituted(x, y, next_box, ON_star) == ON_func_substituted(y, x, next_box, ON_star_zero))
        # )
    # )

def loop_invariant(t):
    return ForAll(
        [x, y],
        Implies(
            And(x != tbl, y != tbl),
            Or(
                And(ON_star(x, n0), ON_star(x, y) == ON_star_zero(x, y)),
                And(Not(ON_star(x, n0)), Implies(t != tbl, ON_star(t, x)), ON_star(x, y) == ON_star_zero(y, x))
            )
        )
    )

def loop_invariant_substituted(next_box):
    return And(
        ForAll(
            [t],
            Implies(
                ON_star_zero(t, n0),
                Or(
                    And(
                        top_substituted(t, next_box, ON_star),
                        on_table_substituted(t, next_box, ON_star),
                    ),
                    ON_func_substituted(t, n0, next_box, ON_star),
                ),
            ),
        ),
        Exists(
            [t],
            And(
                top_substituted(t, next_box, ON_star),
                ON_func_substituted(t, n0, next_box, ON_star),
            ),
        ),
    )


def not_loop_invar_instance(next_box):
    return And(
        Not(top_substituted(b11, next_box)), Not(ON_func_substituted(b11, n0, next_box))
    )


def ON_func_substituted(alpha, beta, b_prime, b, ON_func):
    return Or(
        ON_func(alpha, beta),
        And(ON_func(alpha, b_prime), ON_func(b, beta))
    )


# def top_substituted(x, next_box, ON_func):
#     return Not(
#         Exists(
#             [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
#         )
#     )


# def on_table_substituted(x, next_box, ON_func):
#     return Not(
#         Exists(
#             [y], And(Not(y == x), ON_func_substituted(x, y, next_box, ON_func=ON_func))
#         )
#     )


# add ON_star and ON_star_zero
ON_star = define_ON_func("ON_star", solver)
ON_star_zero = define_ON_func("ON_star_zero", solver)

print("verifying precondition")
solver.push()
solver.assert_and_track(func_equiv(ON_star, ON_star_zero), "two_on_equiv")
solver.assert_and_track(precondition(), "pre_cond")
solver.assert_and_track(Not(loop_invariant(tbl)), "not_loop_invar")
check_solver(solver)
solver.pop()

# print("verifying loop invariant")
# solver.push()
# solver.assert_and_track(while_cond_instance(x), "while_cond")
# solver.assert_and_track(loop_invariant(), "loop_invar")
# solver.assert_and_track(Not(loop_invariant_substituted(x)), "substituted_loop_invar")
# check_solver(solver)
# solver.pop()

# print("verifying post condition")
# solver.push()
# solver.assert_and_track(top(n0, ON_star), "top_n0")
# solver.assert_and_track(t != n0, "t_neq_n0")
# solver.assert_and_track(Exists([x], And(top(x, ON_star), ON_star(x, n0))), "top_n0")
# solver.assert_and_track(Not(while_cond()), "not_while_cond")
# solver.assert_and_track(loop_invariant(t), "loop_invar")
# solver.assert_and_track(
    # Not(ForAll([x], Implies(x != n0, ON_star(t, x)))),
    # "simple_postcond"
# )
# solver.assert_and_track(Not(postcondition()), "not_post_cond")
# solver.assert_and_track(Not(postcondition_substituted(n0, t)), "not_post_cond")
# check_solver(solver)
# solver.pop()
