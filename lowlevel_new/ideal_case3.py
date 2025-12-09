"""in this file, the on* relation is modeled using small offset in the x/y dimension (both x,y position  are within some offset and differnt z position)
For collision, we consider the length of the block.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_cube(ax, cx, cy, cz, L, color='blue'):
    """Draw a cube centered at (cx,cy,cz) with side length L."""
    r = L / 2.0
    # 8 cube vertices
    v = [
        (cx - r, cy - r, cz - r),
        (cx + r, cy - r, cz - r),
        (cx + r, cy + r, cz - r),
        (cx - r, cy + r, cz - r),
        (cx - r, cy - r, cz + r),
        (cx + r, cy - r, cz + r),
        (cx + r, cy + r, cz + r),
        (cx - r, cy + r, cz + r),
    ]

    # 6 cube faces
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]],
    ]

    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.3))


def visualize_scene(model, blocks, X, Y, Z, L=1.0):
    """
    Visualizes blocks b9, b10, b11 (or any others)
    using the coordinates from a Z3 model.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def f(z3val):
        s = str(model.eval(z3val).as_decimal(20))
        return float(s) if s[-1] != '?' else float(s[:-1])

    for b, color in zip(blocks, ["red", "blue", "yellow", "green", "gray"]):
        cx = f(X(b))
        cy = f(Y(b))
        cz = f(Z(b))

        draw_cube(ax, cx, cy, cz, L, color=color)
        print(str(b), cx, cy, cz)

        ax.text(cx, cy, cz, str(b), fontsize=12)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    # plt.show()
    return ax


def draw_vertical_swept_volume_from_block(
    ax, model, X, Y, Z, b_prime, L, color="red", alpha=0.2
):
    """
    Draws the swept volume of block b_prime moving vertically from:
        (X(b'), Y(b'), Z(b'))
    to:
        (X(b'), Y(b'), Z(b') + 2*L)
    using side length L.
    """
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    L = float(str(model.eval(L)))
    r = L / 2.0

    # Convert Z3 values to float
    def f(z3val):
        return float(str(model.eval(z3val).as_decimal(20))[:-1])

    import pdb


    x = f(X(b_prime))
    y = f(Y(b_prime))
    z0 = f(Z(b_prime))
    z1 = z0 + 2 * L  # upward end point

    # vertices of cube at start
    v0 = np.array(
        [
            [x - r, y - r, z0 - r],
            [x + r, y - r, z0 - r],
            [x + r, y + r, z0 - r],
            [x - r, y + r, z0 - r],
            [x - r, y - r, z0 + r],
            [x + r, y - r, z0 + r],
            [x + r, y + r, z0 + r],
            [x - r, y + r, z0 + r],
        ]
    )

    # vertices of cube at end
    v1 = np.array(
        [
            [x - r, y - r, z1 - r],
            [x + r, y - r, z1 - r],
            [x + r, y + r, z1 - r],
            [x - r, y + r, z1 - r],
            [x - r, y - r, z1 + r],
            [x + r, y - r, z1 + r],
            [x + r, y + r, z1 + r],
            [x - r, y + r, z1 + r],
        ]
    )

    # faces of the whole swept volume
    faces = [
        # start cube
        [v0[0], v0[1], v0[2], v0[3]],
        [v0[4], v0[5], v0[6], v0[7]],
        [v0[0], v0[1], v0[5], v0[4]],
        [v0[2], v0[3], v0[7], v0[6]],
        [v0[1], v0[2], v0[6], v0[5]],
        [v0[4], v0[7], v0[3], v0[0]],
        # end cube
        [v1[0], v1[1], v1[2], v1[3]],
        [v1[4], v1[5], v1[6], v1[7]],
        [v1[0], v1[1], v1[5], v1[4]],
        [v1[2], v1[3], v1[7], v1[6]],
        [v1[1], v1[2], v1[6], v1[5]],
        [v1[4], v1[7], v1[3], v1[0]],
        # connecting side faces
        [v0[0], v0[1], v1[1], v1[0]],
        [v0[1], v0[2], v1[2], v1[1]],
        [v0[2], v0[3], v1[3], v1[2]],
        [v0[3], v0[0], v1[0], v1[3]],
        [v0[4], v0[5], v1[5], v1[4]],
        [v0[5], v0[6], v1[6], v1[5]],
        [v0[6], v0[7], v1[7], v1[6]],
        [v0[7], v0[4], v1[4], v1[7]],
    ]

    poly = Poly3DCollection(faces, color=color, alpha=alpha)
    ax.add_collection3d(poly)


from z3 import *
from matplotlib import pyplot as plt

set_option("smt.core.minimize", "true")
# --- Sorts and functions ---
Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13) = EnumSort("Box", ["b9", "b10", "b11", "b12", "b13"])
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
X = Function("X", Box, RealSort())
Y = Function("Y", Box, RealSort())
Z = Function("Z", Box, RealSort())


L, c, eps = Reals("L c eps")
b, b0, b_prime, sym = Consts("b b0 b_prime sym", Box)

abs_diff = lambda u, v: If(u - v >= 0, u - v, v - u)

s = Solver()
s.set(unsat_core=True)
s.assert_and_track(L == 1.0, "L_val")


def box_equal(b1, b2):
    return And(X(b1) == X(b2), Y(b1) == Y(b2), Z(b1) == Z(b2))


def ideal_on(b1, b2):
    return Or(
        # box_equal(b1, b2),
        # And(X(b1) == X(b2), Y(b1) == Y(b2), Z(b1) >= Z(b2))
        And(Abs(X(b1) - X(b2)) < L / 2, Abs(Y(b1) - Y(b2)) < L / 2, Z(b1) >= Z(b2)),
    )


def top_one(b1, b2):
    # b1 is top
    return Or(box_equal(b1, b2), Not(ideal_on(b2, b1)))

def top_higher(b1, b2_bottom, b3_s):
    return Implies(And(ideal_on(b1, b2_bottom), ideal_on(b3_s, b2_bottom)), higher(b1, b3_s))


def top(b1):
    top_one_list = [top_one(b1, b), top_one(b1, b0), top_one(b1, b_prime), top_one(b1, sym)]
    top_higher_list = []
    for b2_bottom in [b, b0, b_prime, sym]:
        for b3_s in [b, b0, b_prime, sym]:
            top_higher_list.append(top_higher(b1, b2_bottom, b3_s))
    top_all_list = top_one_list + top_higher_list
    return And(*top_all_list)


def higher(b1, b2):
    # b1 is higher than b2
    return Z(b1) >= Z(b2)


def on_table(b1):
    return And(higher(b, b1), higher(b0, b1), higher(b_prime, b1), higher(sym, b1))


def loop_inv(b1):
    return Or(ideal_on(b1, b0), And(top(b1), on_table(b1)))


def scattered(t1, t2):
    return Or(Abs(X(t1) - X(t2)) > 2 * L, Abs(Y(t1) - Y(t2)) > 2 * L)


def imply_scattered(t1, t2):
    return Implies(And(t1 != t2, on_table(t1), on_table(t2)), scattered(t1, t2))


def scattered_all():
    conjuction = []
    for x in [b0, b, b_prime, sym]:
        for y in [b0, b, b_prime, sym]:
            conjuction.append(imply_scattered(x, y))
    return And(*conjuction)

def left_related(b1, b2, b3):
    return Implies(And(ideal_on(b1, b3), ideal_on(b2, b3)), Or(ideal_on(b1, b2), ideal_on(b2, b1)))

def left_related_all():
    conjunction = []
    for x in [b0, b, b_prime, sym]:
        for y in [b0, b, b_prime, sym]:
            for z in [b0, b, b_prime, sym]:
                conjunction.append(left_related(x, y, z))
    return And(*conjunction)


def check_solver(s):
    print(s.check())
    if s.check() == unsat:
        print("Unsat Core:", s.unsat_core())
    elif s.check() == sat:
        print(s.model())
        m = s.model()
        ax = visualize_scene(m, [b9, b10, b11, b12, b13], X, Y, Z, L=1.0)
        # draw_vertical_swept_volume_from_block(
        #     ax,
        #     model=s.model(),
        #     X=X, Y=Y, Z=Z,
        #     b_prime=s.model().eval(b),
        #     L=L,
        #     color='red',
        #     alpha=0.25
        # )

        plt.show()


def encode_collision(a, p0, p1, L):
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
        t >= 0,
        t <= 1,
        Abs(X(a) - cx) < L,
        Abs(Y(a) - cy) < L,
        Abs(Z(a) - cz) < L,
    )

    return collision


s.assert_and_track(ideal_on(b, b0), "on(b, b0)")

s.assert_and_track(top(b), "top(b)")

s.assert_and_track(top(b_prime), "top(b')")

s.assert_and_track(
    And(loop_inv(b), loop_inv(b0), loop_inv(b_prime), loop_inv(sym)), "loop_inv"
)

s.assert_and_track(Not(box_equal(b, b_prime)), "b_neq_b'")

s.assert_and_track(scattered_all(), "scattered_all")

# s.assert_and_track(left_related_all(), "left_related_all")

s.assert_and_track(on_table(b0), "on_table_b0")
s.assert_and_track(on_table(b_prime), "on_table_b_prime")

print("checking if all premises is consistent")
s.push()
print(s.check())
if s.check() != sat:
    print("INCONSISTENT premises!!!!!!")
else:
    print("valid premises")
s.pop()

print("checking tube1")
s.push()
s.assert_and_track(
    encode_collision(
        sym,
        (X(b_prime), Y(b_prime), Z(b_prime)),
        (X(b_prime), Y(b_prime), Z(b) + 2 * L),
        L,
    ),
    "tube1",
)
s.assert_and_track(sym != b_prime, "sym_neq_b_prime")
check_solver(s)
# import pdb; pdb.set_trace()
s.pop()


print("checking tube2")
s.push()
s.assert_and_track(
    encode_collision(
        sym, (X(b_prime), Y(b_prime), Z(b) + 2 * L), (X(b), Y(b), Z(b) + 2 * L), L
    ),
    "tube2",
)
check_solver(s)
s.pop()

print("checking tube3")
s.push()
s.assert_and_track(
    encode_collision(sym, (X(b), Y(b), Z(b) + L), (X(b), Y(b), Z(b) + 100 * L), L),
    "tube3",
)
check_solver(s)
s.pop()
