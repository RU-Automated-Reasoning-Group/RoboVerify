from z3 import *

# set_option("smt.mbqi", False)
# set_option("sat.euf", True)

solver = Solver()

# Box = DeclareSort("Box")
Box, (b9, b10, b11, b12, b13, b14, b15, b16) = EnumSort(
    "Box", ["b9", "b10", "b11", "b12", "b13", "b14", "b15", "b16"]
)
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)

# define ON_star
ON_star = Function("ON_star", Box, Box, BoolSort())
solver.add(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))))
solver.add(ForAll([x], ON_star(x, x)))
solver.add(
    ForAll(
        [x, y, c],
        Implies(And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))),
    )
)
solver.add(
    ForAll(
        [x, y, c],
        Implies(And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))),
    )
)
solver.add(ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))))

# define ON_star_zero
ON_star_zero = Function("ON_star_zero", Box, Box, BoolSort())
solver.add(
    ForAll(
        [x, y, c],
        Implies(And(ON_star_zero(x, y), ON_star_zero(y, c)), ON_star_zero(x, c)),
    )
)
solver.add(ForAll([x], ON_star_zero(x, x)))
solver.add(
    ForAll(
        [x, y, c],
        Implies(
            And(ON_star_zero(x, y), ON_star_zero(x, c)),
            Or(ON_star_zero(y, c), ON_star_zero(c, y)),
        ),
    )
)
solver.add(
    ForAll(
        [x, y, c],
        Implies(
            And(ON_star_zero(x, c), ON_star_zero(y, c)),
            Or(ON_star_zero(x, y), ON_star_zero(y, x)),
        ),
    )
)
solver.add(
    ForAll([x, y], Implies(ON_star_zero(x, y), Implies(ON_star_zero(y, x), x == y)))
)
# solver.add(ON_star_zero(b10, b9))
# solver.add(b0 == b9)


def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(
                    f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}"
                )
        print("----------------")
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        import pdb

        pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())


def ON_func_substituted(alpha, beta, next_box, ON_func=ON_star):
    return And(
        ON_func(alpha, beta), Or(Not(ON_func(alpha, next_box)), ON_func(beta, next_box))
    )


def top(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))


def top_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
        )
    )


def on_table(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))


def on_table_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(x, y, next_box, ON_func=ON_func))
        )
    )


def while_cond():
    return Exists([x], And(top(x, ON_star), ON_star(x, b0), x != b0))


def while_cond_instantized(x):
    return And(top(x, ON_star), ON_star(x, b0), x != b0)


def precondition():
    return ForAll(
        [a], Exists([x], Or(top(a, ON_star), And(top(x, ON_star), ON_star(x, a))))
    )


def postcondition():
    return ForAll(
        [x], Implies(ON_star_zero(x, b0), And(on_table(x, ON_star), top(x, ON_star)))
    )


def postcondition_before_on_table_n0():
    return ForAll(
        [x],
        Implies(
            ON_star_zero(x, b0),
            And(
                on_table_substituted(x, b0, ON_func=ON_star),
                top_substituted(x, b0, ON_func=ON_star),
            ),
        ),
    )


# def postcondition_before_on_table_n0():
#     return ForAll(
#         [a, x],
#         Implies(
#             ON_star_zero(a, b0),
#             Or(
#                 x == a, Not(And(ON_star(a, x), Or(Not(ON_star(a, b0)), ON_star(x, b0))))
#             ),
#         ),
#     )


def loop_invariant():
    return ForAll(
        [x],
        Implies(
            ON_star_zero(x, b0),
            Or(ON_star(x, b0), And(on_table(x, ON_star), top(x, ON_star))),
        ),
    )
    # return ForAll([x, y], And(Implies(ON_star_zero(b0, x), ON_star_zero(x, y) == ON_star(x, y)), Implies(ON_star_zero(x, b0), Or(ON_star(x, b0), on_table(x, ON_star)))))


def loop_invariant_substituted(next_box):
    return ForAll(
        [a],
        Implies(
            ON_star_zero(a, b0),
            Or(
                ON_func_substituted(a, b0, next_box, ON_func=ON_star),
                And(
                    on_table_substituted(a, next_box, ON_func=ON_star),
                    top_substituted(a, next_box, ON_star),
                ),
            ),
        ),
    )


print("verifying precondition")
solver.push()
solver.add(ForAll([x, y], ON_star_zero(x, y) == ON_star(x, y)))
solver.add(precondition())
solver.add(Not(loop_invariant()))
check_solver(solver)
solver.pop()

print("verifying loop invariant")
solver.push()
solver.add(while_cond_instantized(x))
solver.add(loop_invariant())
solver.add(Not(loop_invariant_substituted(x)))
check_solver(solver)
solver.pop()

print("verifying postcondition")
solver.push()
solver.add(loop_invariant())
solver.add(Not(while_cond()))
solver.add(Not(postcondition_before_on_table_n0()))
check_solver(solver)
solver.pop()


def next_loop_invariant(b):
    return And(
        ForAll(
            [a],
            Exists(
                [x],
                Or(
                    ON_star(a, b0), top(a, ON_star), And(top(x, ON_star), ON_star(x, a))
                ),
            ),
        ),
        ON_star(b, b0),
        top(b, ON_star),
        on_table(b0, ON_star),
    )


print("verifying postcondition implies the loop invariant of next loop")
solver.push()
solver.add(postcondition())
solver.add(Not(next_loop_invariant(b0)))
check_solver(solver)
solver.pop()
