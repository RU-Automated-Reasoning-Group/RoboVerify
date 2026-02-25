import pdb

from z3 import (
    And,
    BoolSort,
    Consts,
    DeclareSort,
    Exists,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Solver,
    sat,
    unsat,
    is_const,
    is_app,
    eq,
    is_quantifier,
    Const,
)

# pdb.set_trace()
# set_option("smt.mbqi", False)

solver = Solver()

BoxSort = DeclareSort("Box")


########################## rewriting ########

def match_pattern(expr, pattern, subst=None):
    """Try to match expr against pattern. Returns substitution dict if match succeeds, else None."""
    if subst is None:
        subst = {}

    # Pattern variable (placeholder)
    if is_const(pattern) and pattern.decl().arity() == 0 and pattern.sort() == expr.sort():
        if pattern in subst:
            return subst if subst[pattern] == expr else None
        new_subst = subst.copy()
        new_subst[pattern] = expr
        return new_subst

    # Applications
    if is_app(expr) and is_app(pattern):
        if expr.decl() != pattern.decl():
            return None
        if len(expr.children()) != len(pattern.children()):
            return None
        for e_child, p_child in zip(expr.children(), pattern.children()):
            subst = match_pattern(e_child, p_child, subst)
            if subst is None:
                return None
        return subst

    # Otherwise must be identical
    if eq(expr, pattern):
        return subst

    return None


def rewrite_expr(e, old_pattern, new_builder):
    """
    Recursively rewrite expressions.
    old_pattern: expression like ON_star(u, v)
    new_builder: function(subst) -> new expression
    """

    # Quantifiers
    if is_quantifier(e):
        bound_vars = [Const(e.var_name(i), e.var_sort(i)) for i in range(e.num_vars())]
        body = rewrite_expr(e.body(), old_pattern, new_builder)
        return ForAll(bound_vars, body) if e.is_forall() else Exists(bound_vars, body)

    # Applications
    if is_app(e):
        args = [rewrite_expr(arg, old_pattern, new_builder) for arg in e.children()]
        candidate = e.decl()(*args)

        subst = match_pattern(candidate, old_pattern)
        if subst is not None:
            return new_builder(subst)

        return candidate

    return e


# Define the rewrite rule generator
def make_rewriter_put(X_fixed, Y_fixed):
    u, v = Consts("u v", BoxSort)
    pattern = ON_star(u, v)

    def builder(subst):
        uu, vv = subst[u], subst[v]
        return Or(
            ON_star(uu, vv),
            And(ON_star(uu, X_fixed), ON_star(Y_fixed, vv))
        )

    return pattern, builder


def make_rewriter_put_tbl(X_fixed):
    u, v = Consts("u v", BoxSort)
    pattern = ON_star(u, v)

    def builder(subst):
        uu, vv = subst[u], subst[v]
        return And(
            ON_star(uu, vv),
            Or(Not(ON_star(uu, X_fixed)), ON_star(vv, X_fixed))
        )

    return pattern, builder

def wp_for_put_on_box(b_prime, b, Q):
    """Calculate the weakest precondition for Q when putting b_prime on table"""
    pattern, builder = make_rewriter_put(b_prime, b)
    return And(Not(ON_star(b, b_prime)), rewrite_expr(Q, pattern, builder))

def wp_for_put_on_tbl(b_prime, Q):
    pattern, builder = make_rewriter_put_tbl(b_prime)
    return rewrite_expr(Q, pattern, builder)

#######################################################


# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", BoxSort)
ON_star = Function("ON_star", BoxSort, BoxSort, BoolSort())
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
        #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #         print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        import pdb

        pdb.set_trace()
    else:
        # print("NOT satisfiable")
        print(x.check())


def on_table(x):
    return Not(Exists([y], And(Not(y == x), ON_star(x, y))))


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


def while_cond(b_prime):
    # return Exists([b_prime], And(top(b_prime), b_prime != b))
    return Exists([b_prime], Not(ON_star(b, b_prime)))


def while_cond_instantized(b_prime):
    # return And(top(b_prime), b_prime != b)
    return Not(ON_star(b, b_prime))


def precondition():
    return ForAll([a], top(a))

def postcondition():
    return ForAll([a], ON_star(a, b0))


# def loop_invariant(b):
#     return And(
#         ForAll([a], Or(top(a), ON_star(a, b0))),
#         ON_star(b, b0),
#         top(b),
#     )

# def loop_invariant(b):
#     # this is learned by the inference program with the tbl as one of the axiom
#     return And(
#         ForAll([x], Implies(ON_star(b, x), ON_star(x, b0))),
#         ForAll([x, y], Implies(ON_star(x, y), Or(ON_star(y, x), ON_star(b, x)))),
#         ForAll([x, y], Implies(And(ON_star(y, x), Not(ON_star(x, b0))), ON_star(x, y)))
#     )

def loop_invariant(b):
    # this is learned by the inference program WITHOUT the tbl as one of the axiom
    return And(
        ForAll([x], Implies(ON_star(x, b0), ON_star(b, x))),
        ForAll([x], Implies(ON_star(b, x), ON_star(x, b0))),
        ForAll([x, y], Implies(And(ON_star(y, x), Not(ON_star(x, b0))), ON_star(x, y)))
    )


def substituted_loop_invariant(x, b_prime, b):
    return And(
        ForAll(
            [a],
            Or(
                substituted_top(a, b_prime, b),
                Or(ON_star(a, b0), And(ON_star(a, b_prime), ON_star(b, b0))),
            ),
        ),
        Or(ON_star(x, b0), And(ON_star(x, b_prime), ON_star(b, b0))),
        substituted_top(x, b_prime, b),
    )


solver.push()
print("verying precondition")
solver.add(precondition())
solver.add(Not(loop_invariant(b0)))
check_solver(solver)
solver.pop()

# while loop correctness: verify while condition (true) + loop invariant implies another loop invariant
solver.push()
print("verifying inductive loop invariant")
solver.add(while_cond_instantized(b_prime))
solver.add(loop_invariant(b))
# wp = And(Not(ON_star(b, b_prime)), substituted_loop_invariant(b_prime, b_prime, b))
# wp = wp_for_put_on_tbl(b_prime, wp_for_put_on_box(b_prime, b, loop_invariant(b_prime)))
wp = wp_for_put_on_box(b_prime, b, loop_invariant(b_prime))
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
