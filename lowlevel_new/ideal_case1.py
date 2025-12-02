"""in this file, the on* relation is modeled perfectly (exactly x,y position and differnt z position)
For collision check, we don't consider the length of the block
"""

from z3 import *
from matplotlib import pyplot as plt

set_option("smt.core.minimize", "true")
# --- Sorts and functions ---
Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13) = EnumSort("Box", ["b9", "b10", "b11", "b12", "b13"])
X = Function("X", Box, RealSort())
Y = Function("Y", Box, RealSort())
Z = Function("Z", Box, RealSort())


L, c, eps = Reals("L c eps")
b, b0, b_prime, sym = Consts("b b0 b_prime s", Box)

abs_diff = lambda u, v: If(u - v >= 0, u - v, v - u)

s = Solver()
s.set(unsat_core=True)
s.assert_and_track(L == 1.0, "L_val")


def box_equal(b1, b2):
    return And(X(b1) == X(b2), Y(b1) == Y(b2), Z(b1) == Z(b2))


def ideal_on(b1, b2):
    return And(X(b1) == X(b2), Y(b1) == Y(b2), Z(b1) >= Z(b2))


def top_one(b1, b2):
    # b1 is top
    return Or(box_equal(b1, b2), Not(ideal_on(b2, b1)))


def top(b1):
    return And(top_one(b1, b), top_one(b1, b0), top_one(b1, b_prime), top_one(b1, sym))


def higher(b1, b2):
    # b1 is higher than b2
    return Z(b1) >= Z(b2)


def on_table(b1):
    return And(higher(b, b1), higher(b0, b1), higher(b_prime, b1), higher(sym, b1))


def loop_inv(b1):
    return Or(ideal_on(b1, b0), And(top(b1), on_table(b1)))


def check_solver(s):
    print(s.check())
    if s.check() == unsat:
        print("Unsat Core:", s.unsat_core())
    elif s.check() == sat:
        print(s.model())


def encode_no_collision(a, p0, p1, L):
    """
    Returns a Z3 formula encoding that block `a` does NOT collide with
    the swept volume of a block moving from p0 to p1.

    p0 = (x0,y0,z0)
    p1 = (x1,y1,z1)
    L  = cube side length
    """

    (x0, y0, z0) = p0
    (x1, y1, z1) = p1

    t = Real("t")

    # moving-block center at time t
    cx = (1 - t) * x0 + t * x1
    cy = (1 - t) * y0 + t * y1
    cz = (1 - t) * z0 + t * z1

    # collision means the cubes overlap on ALL axes
    collision = And(
        t >= 0, t <= 1, Abs(X(a) - cx) < L, Abs(Y(a) - cy) < L, Abs(Z(a) - cz) < L
    )

    # non-collision is the negation of collision
    no_collision = Not(collision)

    return no_collision


s.assert_and_track(ideal_on(b, b0), "on(b, b0)")

s.assert_and_track(top(b), "top(b)")

s.assert_and_track(top(b_prime), "top(b')")

s.assert_and_track(
    And(loop_inv(b), loop_inv(b0), loop_inv(b_prime), loop_inv(sym)), "loop_inv"
)

s.assert_and_track(Not(box_equal(b, b_prime)), "b_neq_b'")


print("checking tube1")
s.push()
s.assert_and_track(
    And(X(sym) == X(b_prime), Y(sym) == Y(b_prime), Z(sym) > Z(b_prime)), "tube1"
)
# s.assert_and_track(encode_no_collision(sym, (X(b_prime), ), (), L), "tube1")
check_solver(s)
s.pop()

print("checking tube2")
s.push()
t = Real("t")
s.assert_and_track(And(t > 0, t < 1), "t_range")
s.assert_and_track(
    And(
        X(sym) == (X(b) + t * (X(b_prime) - X(b))),
        Y(sym) == (Y(b) + t * (Y(b_prime) - Y(b))),
        Z(sym) == (Z(b) + L / 2 + t * (Z(b_prime) - Z(b))),
    ),
    "tube2",
)
check_solver(s)
s.pop()

print("checking tube3")
s.push()
s.assert_and_track(And(X(sym) == X(b), Y(sym) == Y(b), Z(sym) > Z(b)), "tube3")
check_solver(s)
s.pop()
