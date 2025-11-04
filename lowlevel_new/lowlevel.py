from z3 import *
from matplotlib import pyplot as plt

# --- Sorts and functions ---
# Box = DeclareSort("Box")
Box, (b9, b10, b11, b12, b13) = EnumSort("Box", ["b9", "b10", "b11", "b12", "b13"])
x = Function("x", Box, RealSort())
y = Function("y", Box, RealSort())
z = Function("z", Box, RealSort())

L, c, eps = Reals("L c eps")
b, bp, m, n, cbox, b0 = Consts("b bp m n cbox b0", Box)

abs_diff = lambda u, v: If(u - v >= 0, u - v, v - u)

# --- ON_star definition with reflexivity ---
ON_star = Function("ON_star", Box, Box, BoolSort())

geom_def = ForAll([m, n],
    ON_star(m, n) ==
    Or(m == n,
       And(abs_diff(x(m), x(n)) < L/2,
           abs_diff(y(m), y(n)) < L/2,
           z(m) >= z(n) + L))
)

# --- Order axioms ---
# Transitivity
axiom_trans = ForAll([m, n, cbox],
    Implies(And(ON_star(m, n), ON_star(n, cbox)), ON_star(m, cbox))
)

# Antisymmetry
axiom_antisym = ForAll([m, n],
    Implies(And(ON_star(m, n), ON_star(n, m)), m == n)
)

# Comparability among children
axiom_comparable = ForAll([m, n, cbox],
    Implies(And(ON_star(m, n), ON_star(m, cbox)),
            Or(ON_star(n, cbox), ON_star(cbox, n)))
)

# Reflexivity
axiom_refl = ForAll([m], ON_star(m, m))

# --- top(x) definition ---
# top = Function("top", Box, BoolSort())
# top_def = ForAll([m],
    # top(m) == ForAll([n], Implies(ON_star(n, m), n == m))
# )
def top(m):
    return ForAll([n], Implies(ON_star(n, m), n == m))

# --- Tube separation assumption ---
# Every box's tube is separated by at least L along x or y
tube_separation = ForAll([m, n], Implies(Not(ON_star(m, n)),
    Or(abs_diff(x(m), x(n)) >= L,
       abs_diff(y(m), y(n)) >= L)
))

# --- Setup solver ---
s = Solver()
s.add(ForAll([m], And(x(m) >= 0, x(m) <= 5, y(m) >= 0, y(m) <= 5, z(m) >= 0, z(m) <= 5)))
s.add(L == 1.0)
s.add(geom_def)
# s.add(axiom_trans, axiom_antisym, axiom_comparable, axiom_refl, top_def)
s.add(axiom_trans, axiom_antisym, axiom_comparable, axiom_refl)
# s.add(tube_separation)
s.add(ForAll([m], Or(top(m), ON_star(m, b0))))
s.add(ON_star(b, b0))

# --- Known world assumptions ---
s.add(top(b), top(bp))
s.add(bp != b)
s.add(L > 0, c > 0, eps > 0)
# s.add(b0 == b9)
# s.add(b == b10)
# s.add(bp == b11)

# --- Tube 1: vertical lift above bp ---
# tube1_clear = ForAll([m],
#     Implies(
#         And(abs_diff(x(m), x(bp)) < delta,
#             abs_diff(y(m), y(bp)) < delta,
#             z(bp) < z(m),
#             z(m) < z(b) + c),
#         m == bp
#     )
# )

# # --- Tube 2: horizontal move at z_target ---
# # Now provable using tube separation
# tube2_clear = ForAll([m],
#     Implies(
#         And(m != bp, abs_diff(z(m), z(b) + c) < eps),
#         Or(abs_diff(x(m), x(bp)) >= delta,
#            abs_diff(y(m), y(bp)) >= delta,
#            abs_diff(x(m), x(b)) >= delta,
#            abs_diff(y(m), y(b)) >= delta)
#     )
# )

# # --- Tube 3: lowering above b ---
# tube3_clear = ForAll([m],
#     Implies(
#         And(abs_diff(x(m), x(b)) < delta,
#             abs_diff(y(m), y(b)) < delta,
#             z(b) < z(m),
#             z(m) < z(b) + c),
#         m == bp
#     )
# )

# Add the tube-clearance constraints
# s.add(tube1_clear, tube2_clear, tube3_clear)

# --- Check for any violation in Tube 1, Tube 2, or Tube 3 ---
violation = Exists([m],
    Or(
        # Tube 1: vertical lift above bp
        And(m != bp,
            abs_diff(x(m), x(bp)) < L,
            abs_diff(y(m), y(bp)) < L,
            z(bp) < z(m),
            z(m) < z(b) + c),
        # Tube 2: horizontal move at z_target
        And(m != bp,
            abs_diff(z(m), z(b)+c) < eps,
            And(abs_diff(x(m), x(bp)) < L, abs_diff(y(m), y(bp)) < L)
        ),
        # # Tube 3: vertical lowering above b
        And(m != bp,
            abs_diff(x(m), x(b)) < L,
            abs_diff(y(m), y(b)) < L,
            z(b) < z(m),
            z(m) < z(b) + c)
    )
)

print("Solver result:", s.check())  # Should print 'unsat'
s.push()
s.add(violation)
print("Checking if any tube is blocked (expecting UNSAT)...")
print("Solver result:", s.check())  # Should print 'unsat'
# print(s.model())
if s.check() == sat:
    model = s.model()
    print(model)

    # Extract box positions
    boxes = [b9, b10, b11, b12, b13]
    xs, ys, zs = [], [], []

    for box in boxes:
        try:
            xs.append(float(model.evaluate(x(box)).as_fraction()))
            ys.append(float(model.evaluate(y(box)).as_fraction()))
            zs.append(float(model.evaluate(z(box)).as_fraction()))
        except:
            # if solver doesn't assign a value, set default 0
            assert False
            xs.append(0.0)
            ys.append(0.0)
            zs.append(0.0)
    # import pdb; pdb.set_trace()
    print(xs)
    print(ys)
    print(zs)

    # --- Plotting ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Box positions')

    for i, (cx, cy, cz) in enumerate(zip(xs, ys, zs)):
        # Draw cube: 3D box with edge length L
        r = [cx - 0.5*1.0, cx + 0.5*1.0]  # half L = 0.5
        X = [r[0], r[1], r[1], r[0], r[0], r[1], r[1], r[0]]
        Y = [r[0], r[0], r[1], r[1], r[0], r[0], r[1], r[1]]
        Z = [cz - 0.5*1.0, cz - 0.5*1.0, cz - 0.5*1.0, cz - 0.5*1.0,
             cz + 0.5*1.0, cz + 0.5*1.0, cz + 0.5*1.0, cz + 0.5*1.0]

        # list of sides' polygons of cube
        verts = [
            [ [X[0],Y[0],Z[0]], [X[1],Y[1],Z[1]], [X[2],Y[2],Z[2]], [X[3],Y[3],Z[3]] ],
            [ [X[4],Y[4],Z[4]], [X[5],Y[5],Z[5]], [X[6],Y[6],Z[6]], [X[7],Y[7],Z[7]] ],
            [ [X[0],Y[0],Z[0]], [X[1],Y[1],Z[1]], [X[5],Y[5],Z[5]], [X[4],Y[4],Z[4]] ],
            [ [X[2],Y[2],Z[2]], [X[3],Y[3],Z[3]], [X[7],Y[7],Z[7]], [X[6],Y[6],Z[6]] ],
            [ [X[1],Y[1],Z[1]], [X[2],Y[2],Z[2]], [X[6],Y[6],Z[6]], [X[5],Y[5],Z[5]] ],
            [ [X[4],Y[4],Z[4]], [X[7],Y[7],Z[7]], [X[3],Y[3],Z[3]], [X[0],Y[0],Z[0]] ]
        ]
        print(verts)

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        colors = ['red', 'blue', 'cyan', "brown", "yellow"]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=colors[i], linewidths=1, edgecolors='r', alpha=0.5))
        

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 5)
    plt.show()
else:
    print("No solution found.")
s.pop()
