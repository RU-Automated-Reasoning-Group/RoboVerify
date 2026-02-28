from z3 import *

solver = Solver()

Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13, b14, b15) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15'])
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


def ON_func_substituted(alpha, beta, next_box, ON_func=ON_star):
    return And(
        ON_func(alpha, beta), Or(Not(ON_func(alpha, next_box)), ON_func(beta, next_box))
    )


def top(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))


def top_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
        )
    )


def on_table(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))


def on_table_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(x, y, next_box, ON_func=ON_func))
        )
    )


def precondition():
    return And(ForAll([x], ON_star(x, b0)), on_table(b0))


def postcondition():
    return ForAll([x], top(x))


def while_cond():
    # return Exists([x], And(top(x), ON_star(x, b0), x != b0))
    return Exists([x], Implies(x != b0, And(top(x), ON_star(x, b0))))


def while_cond_instantized(x):
    return And(top(x), ON_star(x, b0), x != b0)


def loop_invariant():
    return And(ForAll([x], Or(on_table(x), ON_star(x, b0))), on_table(b0))


def loop_invariant_substituted(next_box):
    return And(
        ForAll(
            [a],
            Or(on_table_substituted(a, next_box), ON_func_substituted(a, b0, next_box)),
        ),
        on_table_substituted(b0, next_box),
    )


print("verifying precondition")
solver.push()
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
# solver.add(Not(top(b0)))
solver.add(Not(postcondition()))
check_solver(solver)
solver.pop()
