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
    Reals,
    RealSort,
    unsat,
)

pdb.set_trace()
# set_option("smt.mbqi", False)

solver = Solver()

Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
# Box, (b9, b10, b11, b12, b13) = EnumSort("Box", ["b9", "b10", "b11", "b12", "b13"])
x = Function("x", Box, RealSort())
y = Function("y", Box, RealSort())
z = Function("z", Box, RealSort())

# the loop invariant is (forall a. (top(a) AND on_table(a)) OR ON_star(a, b0)) AND ON_star(b, b0) AND top(b)
# now we want to directly translate them to the lowlevel encoding to prove that the 3 tubes are considered are clear.
b0, b, b_prime, a, alpha, tmp = Consts("b0 b b_prime a alpha tmp", Box)
L, = Reals("L")
solver.add(L == 1)


def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #         print(f"ON({name1}, {name2}): {x.model().evaluate(ON_star(box1, box2))}")

        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #         print(f"higher({name1}, {name2}): {x.model().evaluate(higher(box1, box2))}")
        import pdb

        pdb.set_trace()
    else:
        # print("NOT satisfiable")
        print(x.check())


def translate_on_star(a, b):
    return And(x(a) == x(b), y(a) == y(b), z(a) >= z(b))


def translate_top(b):
    return Or(box_equal(a, b), Not(translate_on_star(a, b)))


def translate_on_table(t):
    # return Implies(z(t) >= z(a), z(t) == z(a))
    return And(z(a) <= z(b), z(a) <= z(b_prime), z(a) <= z(b0))


def box_equal(a, b):
    return And(x(a) == x(b), y(a) == y(b), z(a) == z(b))


def translate_while_cond():
    # the while condition is top(b') AND b' != b
    return And(b_prime != b, translate_top(b_prime))


def translate_loop_invariant():
    return And(ForAll([a], Or()), translate_on_star(b, b0), translate_top(b))


def a_constraint():
    return Or(And(translate_top(a), translate_on_table(a)), translate_on_star(a, b0))


solver.assert_and_track(translate_top(b), "top_b")
solver.assert_and_track(translate_top(b_prime), "top_b_prime")
solver.assert_and_track(translate_on_star(b, b0), "b_on_b0")
# solver.assert_and_track(b_prime != b, "b_prime_neq_b")
solver.assert_and_track(Not(box_equal(b_prime, b)), "b_prime_neq_b")
solver.assert_and_track(a_constraint(), "a_constraints")

solver.push()
print("checking consistency")
check_solver(solver)
solver.pop()

solver.push()
import pdb; pdb.set_trace()
print("checking first tube")
# using the current formalism without any use of L, we cannot use L in the representation of the tube yet
solver.assert_and_track(
    And(x(tmp) > x(b_prime) - L/2,
        x(tmp) < (x(b_prime) + L/2),
        y(tmp) > (y(b_prime) - L/2),
        y(tmp) < (y(b_prime) + L/2),
        z(tmp) > z(b_prime),
    ),
    "tube1",
)
# solver.assert_and_track(translate_on_star(tmp, b_prime), "tube1")
# solver.assert_and_track(Not(box_equal(b_prime, tmp)), "tmp_neq_b_prime")
solver.assert_and_track(tmp == a, "tmp_eq_a")
check_solver(solver)
solver.pop()

solver.push()
print("checking third tube")
solver.assert_and_track(translate_on_star(tmp, b), "tube3")
solver.assert_and_track(Not(box_equal(b, tmp)), "tmp_neq_b")
solver.assert_and_track(tmp == a, "tmp_eq_a")
check_solver(solver)
solver.pop()

solver.push()
print("checking second tube")
solver.assert_and_track(And(z(tmp) > z(b_prime), z(tmp) > z(b)), "high_of_tube2_not_clear")
solver.assert_and_track(tmp == a, "tmp_eq_a")
check_solver(solver)
solver.pop()
