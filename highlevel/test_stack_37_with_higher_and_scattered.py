import pdb

from z3 import (
    And,
    BoolSort,
    Consts,
    DeclareSort,
    EnumSort,
    Exists,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Solver,
    sat,
    unsat,
)

pdb.set_trace()
# set_option("smt.mbqi", False)

solver = Solver()

Box = DeclareSort("Box")
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)
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

higher = Function("higher", Box, Box, BoolSort())
solver.add(ForAll([x, y, c], Implies(And(higher(x, y), higher(y, c)), higher(x, c))))
solver.add(ForAll([x], higher(x, x)))
solver.add(
    ForAll(
        [x, y, c],
        Implies(And(higher(x, y), higher(x, c)), Or(higher(y, c), higher(c, y))),
    )
)

# solver.add(
#     ForAll([x, y], Implies(And(ON_star(x, y), Not(ON_star(y, x))), And(higher(x, y), Not(higher(y, x)))))
# )


def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"ON({name1}, {name2}): {x.model().evaluate(ON_star(box1, box2))}")

        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"higher({name1}, {name2}): {x.model().evaluate(higher(box1, box2))}")
        import pdb

        pdb.set_trace()
    else:
        # print("NOT satisfiable")
        print(x.check())


def on_table(x):
    return ForAll([y], higher(y, x))


def top(x):
    return Not(Exists([y], And(Not(y == x), ON_star(y, x))))


def substituted_top(x, b_prime, b):
    return Not(
        Exists(
            [y],
            And(
                Not(y == x), Or(ON_star(y, x), And(ON_star(y, b_prime), ON_star(b, x)))
            ),
        )
    )


def substituted_on_table(x, b_prime, b):
    return ForAll(
        [y],
        Or(higher(y, x), And(higher(y, b_prime), higher(b, x)))
    )


def while_cond(b_prime):
    return Exists([b_prime], And(top(b_prime), b_prime != b))


def while_cond_instantized(b_prime):
    return And(top(b_prime), b_prime != b)


def postcondition():
    return ForAll([a], ON_star(a, b0))


def loop_invariant(b):
    return And(
        ForAll([a], Or(And(top(a), on_table(a)), ON_star(a, b0))),
        ON_star(b, b0),
        top(b),
    )


def substituted_loop_invariant(x, b_prime, b):
    return And(
        ForAll(
            [a],
            Or(
                And(substituted_top(a, b_prime, b), substituted_on_table(a, b_prime, b)),
                Or(ON_star(a, b0), And(ON_star(a, b_prime), ON_star(b, b0))),
            ),
        ),
        Or(ON_star(x, b0), And(ON_star(x, b_prime), ON_star(b, b0))),
        substituted_top(x, b_prime, b),
    )


# while loop correctness: verify while condition (true) + loop invariant implies another loop invariant
solver.push()
print("verifying inductive loop invariant")
solver.add(on_table(b0))
solver.add(while_cond_instantized(b_prime))
solver.add(loop_invariant(b))
wp = And(Not(ON_star(b, b_prime)), substituted_loop_invariant(b_prime, b_prime, b))
solver.add(Not(wp))
check_solver(solver)
solver.pop()

print("verifying postcondition")
solver.push()
solver.add(loop_invariant(b))
solver.add(Not(while_cond(b_prime)))
solver.add(Not(postcondition()))
check_solver(solver)
solver.pop()
