from z3 import (
    And,
    BoolSort,
    Const,
    Consts,
    DeclareSort,
    Exists,
    ForAll,
    Function,
    If,
    Implies,
    Not,
    Or,
    Solver,
    unsat,
)


Goal = DeclareSort("goal")

# Universe symbols
null = Const("null", Goal)
h, i, j, u, v, w, z = Consts("h i j u v w z", Goal)

# Structural relations/functions from dtca_nested
d_star = Function("d_star", Goal, Goal, BoolSort())
r_star = Function("r_star", Goal, Goal, BoolSort())
l0 = Function("l0", Goal, Goal)

# Mutable marking predicate
Mark = Function("Mark", Goal, BoolSort())


def reflexive(rel):
    x = Const("x_refl", Goal)
    return ForAll([x], rel(x, x))


def transitive(rel):
    x, y, z0 = Consts("x_tr y_tr z_tr", Goal)
    return ForAll([x, y, z0], Implies(And(rel(x, y), rel(y, z0)), rel(x, z0)))


def lin(rel):
    x, y, z0 = Consts("x_lin y_lin z_lin", Goal)
    return ForAll([x, y, z0], Implies(And(rel(x, y), rel(x, z0)), Or(rel(y, z0), rel(z0, y))))


def antisymm(rel):
    x, y = Consts("x_as y_as", Goal)
    return ForAll([x, y], Implies(And(rel(x, y), rel(y, x)), x == y))


def reflisolated(rel, node):
    x = Const("x_ri", Goal)
    return ForAll([x], Implies(Or(rel(node, x), rel(x, node)), node == x))


def dtca(rel):
    return And(reflexive(rel), transitive(rel), lin(rel), antisymm(rel), reflisolated(rel, null))


def f_plus(rel, a, b):
    return And(rel(a, b), a != b)


def f_(rel, a, b):
    t = Const("t_f", Goal)
    return And(f_plus(rel, a, b), ForAll([t], Implies(f_plus(rel, a, t), rel(b, t))))


def f_tot(rel, a, b):
    t = Const("t_ft", Goal)
    return Or(f_(rel, a, b), And(b == null, ForAll([t], Not(f_plus(rel, a, t)))))


def dtot(a, b):
    return f_tot(d_star, a, b)


def rtot(a, b):
    return f_tot(r_star, a, b)


def dr_reach(x, y):
    # [d*r*](x,y) := d*(x,l0(y))
    return d_star(x, l0(y))


def flat_order(a, b):
    return If(l0(a) == l0(b), r_star(a, b), d_star(l0(a), l0(b)))


def flat_between(a, b, c):
    return And(flat_order(a, b), flat_order(b, c))


def _mk_mark_fn(mark_fn):
    return mark_fn if mark_fn is not None else (lambda t: Mark(t))


def P(h0, mark_fn=None):
    m = _mk_mark_fn(mark_fn)
    x = Const("x_P", Goal)
    return And(
        h0 != null,
        l0(h0) == h0,
        ForAll([x], Implies(dr_reach(h0, x), Not(m(x)))),
    )


def I1(h0, i0, mark_fn=None):
    m = _mk_mark_fn(mark_fn)
    x = Const("x_I1", Goal)
    scanned = If(i0 == null, dr_reach(h0, x), And(flat_between(h0, x, i0), x != i0))
    return And(
        h0 != null,
        l0(h0) == h0,
        Implies(i0 != null, d_star(h0, i0)),
        ForAll([x], Implies(scanned, m(x))),
    )


def I2(h0, i0, j0, mark_fn=None):
    m = _mk_mark_fn(mark_fn)
    x = Const("x_I2", Goal)
    inner_scanned = If(
        j0 == null,
        r_star(i0, x),
        And(r_star(i0, x), f_plus(r_star, x, j0)),
    )
    return And(
        h0 != null,
        l0(h0) == h0,
        d_star(h0, i0),
        Implies(j0 != null, r_star(i0, j0)),
        ForAll([x], Implies(And(flat_between(h0, x, i0), x != i0), m(x))),
        ForAll([x], Implies(inner_scanned, m(x))),
    )


def Q(h0, mark_fn=None):
    m = _mk_mark_fn(mark_fn)
    x = Const("x_Q", Goal)
    return ForAll([x], Implies(dr_reach(h0, x), m(x)))


def add_dtca_nested_axioms(solver):
    solver.add(dtca(d_star))
    solver.add(dtca(r_star))

    x, y, z0 = Consts("x_ax y_ax z_ax", Goal)
    solver.add(ForAll([x, y, z0], Implies(And(r_star(y, x), r_star(z0, x)), Or(r_star(y, z0), r_star(z0, y)))))

    # l0 axioms
    solver.add(ForAll([x], r_star(l0(x), x)))
    solver.add(ForAll([x, y], Implies(r_star(x, y), r_star(l0(y), x))))

    # nested-shape interaction
    solver.add(ForAll([x, y, z0], Implies(And(r_star(x, y), d_star(z0, y)), Or(x == y, z0 == y))))
    solver.add(ForAll([x, y], Implies(And(d_star(x, y), x != y), l0(x) == x)))


def prove_valid(solver, name, formula):
    solver.push()
    solver.add(Not(formula))
    result = solver.check()
    print(f"{name}: {result}")
    if result != unsat:
        print("Potential countermodel:")
        print(solver.model())
    solver.pop()


def main():
    solver = Solver()
    add_dtca_nested_axioms(solver)

    # ----- Inner loop VCs (while j != null { set_true; j := j.r }) -----
    # Inner-Init: after j := i, establish I2
    vc_inner_init = Implies(And(I1(h, i), i != null), I2(h, i, i))

    # Inner-Preserve: I2 & j!=null -> wp(set_true; j:=j.r, I2)
    i2_after_mark_then_move = I2(h, i, z, mark_fn=lambda t: Or(Mark(t), t == j))
    vc_inner_preserve = Implies(
        And(I2(h, i, j), j != null),
        And(
            j != null,
            ForAll([z], Implies(rtot(j, z), i2_after_mark_then_move)),
        ),
    )

    # Inner-Exit: I2 & j==null -> wp(i := i.d, I1)
    vc_inner_exit = Implies(
        And(I2(h, i, j), j == null, i != null),
        And(
            i != null,
            ForAll([z], Implies(dtot(i, z), I1(h, z))),
        ),
    )

    # ----- Outer loop VCs (while i != null { ... }) -----
    # Outer-Init: after i := h, establish I1
    vc_outer_init = Implies(P(h), I1(h, h))

    # Outer-Preserve: body preserves I1 under i!=null
    # Decomposed through the inner-loop obligations, as in standard VC generation.
    vc_outer_preserve = Implies(
        And(I1(h, i), i != null),
        And(vc_inner_init, vc_inner_preserve, vc_inner_exit),
    )

    # Outer-Exit: I1 & i==null -> Q
    vc_outer_exit = Implies(And(I1(h, i), i == null), Q(h))

    print("Checking all 6 verification conditions for nsll-mark-all...")
    prove_valid(solver, "VC_outer_init", vc_outer_init)
    prove_valid(solver, "VC_outer_preserve", vc_outer_preserve)
    prove_valid(solver, "VC_outer_exit", vc_outer_exit)
    prove_valid(solver, "VC_inner_init", vc_inner_init)
    prove_valid(solver, "VC_inner_preserve", vc_inner_preserve)
    prove_valid(solver, "VC_inner_exit", vc_inner_exit)


if __name__ == "__main__":
    main()
