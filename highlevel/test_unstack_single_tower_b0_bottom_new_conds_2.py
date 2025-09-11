from z3 import *
set_option("smt.core.minimize", "true")
solver = Solver()
# solver.set(unsat_core=True)
# solver.set(unsat_core=True)

Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13, b14, b15) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15'])
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
x, y, c, h, tmp, t_2, b0 = Consts("x y c h tmp t_2 b0", Box)

# define ON_star
ON_star = Function("ON_star", Box, Box, BoolSort())
solver.assert_and_track(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))), "on1")
solver.assert_and_track(ForAll([x], ON_star(x, x)), "on2")
solver.assert_and_track(
    ForAll(
        [x, y, c],
        Implies(And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))),
    ),
    "on3"
)
solver.assert_and_track(
    ForAll(
        [x, y, c],
        Implies(And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))),
    ),
    "on4"
)
solver.assert_and_track(ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))), "on5")

# define ON_star_zero
ON_star_zero = Function("ON_star_zero", Box, Box, BoolSort())
solver.assert_and_track(ForAll([x, y, c], Implies(And(ON_star_zero(x, y), ON_star_zero(y, c)), ON_star_zero(x, c))), "on_zero1")
solver.assert_and_track(ForAll([x], ON_star_zero(x, x)), "on_zero2")
solver.assert_and_track(
    ForAll(
        [x, y, c],
        Implies(And(ON_star_zero(x, y), ON_star_zero(x, c)), Or(ON_star_zero(y, c), ON_star_zero(c, y))),
    ),
    "on_zero3"
)
solver.assert_and_track(
    ForAll(
        [x, y, c],
        Implies(And(ON_star_zero(x, c), ON_star_zero(y, c)), Or(ON_star_zero(x, y), ON_star_zero(y, x))),
    ),
    "on_zero4"
)
solver.assert_and_track(ForAll([x, y], Implies(ON_star_zero(x, y), Implies(ON_star_zero(y, x), x == y))), "on_zero5")


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
                print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        import pdb

        pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())


# def ON_func_substituted(alpha, beta, next_box, ON_func=ON_star):
#     return And(
#         ON_func(alpha, beta), Or(Not(ON_func(alpha, next_box)), ON_func(beta, next_box))
#     )


# def top(x, ON_func=ON_star):
#     return Not(Exists([y], And(Not(y == x), ON_func(y, x))))


# def top_substituted(x, next_box, ON_func=ON_star):
#     return Not(
#         Exists(
#             [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
#         )
#     )


def on_table(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))


# def on_table_substituted(x, next_box, ON_func=ON_star):
#     return Not(
#         Exists(
#             [y], And(Not(y == x), ON_func_substituted(x, y, next_box, ON_func=ON_func))
#         )
#     )


def precondition():
    return And(
        ForAll(
            [x, y],
            And(
                Implies(ON_star_zero(x, y), ON_star(x, y)),
                Implies(ON_star(x, y), ON_star_zero(x, y))
            )
        ),
        on_table(b0)
    )


def postcondition():
    return ForAll(
        [x],
        And(
            Implies(ON_star_zero(h, x), on_table(x)),
            Implies(on_table(x), ON_star_zero(h, x))
        )
    )


def while_cond():
    return Exists(
        [t_2],
        And(
            t_2 != tmp,
            ForAll(
                [x],
                Implies(
                    And(ON_star(tmp, x), tmp != x),
                    ON_star(t_2, x)
                )
            )
        )
    )


# def while_cond_instantized(x):
#     ForAll(
#         [x],
#         Implies(
#             And(ON_star(t, x), t != x),
#             ON_star(t_2, x)
#         )
#     )


def loop_invariant(tmp):
    return And(
        ForAll(
            [x, y],
            Implies(
                ON_star(tmp, x),
                And(
                    Implies(ON_star(x, y), ON_star_zero(x, y)),
                    Implies(ON_star_zero(x, y), ON_star(x, y))
                )
            )
        ),
        ForAll(
            [x],
            Implies(
                And(ON_star_zero(h, x), Not(ON_star(tmp, x))),
                on_table(x)
            )
        ),
        ON_star_zero(h, tmp),
        on_table(b0)
    )

# def loop_invariant_substituted(t, next_box):
#     return And(
#         ForAll(
#             [a],
#             Or(on_table_substituted(a, next_box), ON_func_substituted(a, b0, next_box)),
#         ),
#         on_table_substituted(b0, next_box),
#     )


print("verifying precondition")
solver.push()
solver.assert_and_track(precondition(), "pre_cond")
solver.assert_and_track(Not(loop_invariant(h)), "not_loop_invar")
check_solver(solver)
solver.pop()

# print("verifying loop invariant")
# solver.push()
# solver.add(while_cond_instantized(x))
# solver.add(loop_invariant())
# solver.add(Not(loop_invariant_substituted(x)))
# check_solver(solver)
# solver.pop()

# solver2 = Solver()
# solver2.set(unsat_core=True)
# solver2.assert_and_track(while_cond(), "not_while")
# check_solver(solver2)

print("verifying postcondition")
solver.push()
solver.assert_and_track(loop_invariant(tmp), "loop_invar")
solver.assert_and_track(Not(while_cond()), "not_while")
# solver.assert_and_track(b0 == tmp, "gg")
solver.assert_and_track(Not(postcondition()), "not_post_cond")
check_solver(solver)
solver.pop()
