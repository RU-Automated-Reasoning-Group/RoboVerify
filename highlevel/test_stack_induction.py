from z3 import *

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
    u, v = Consts("u v", IntSort())
    pattern = ON_star(u, v)

    def builder(subst):
        uu, vv = subst[u], subst[v]
        return Or(
            ON_star(uu, vv),
            And(ON_star(uu, X_fixed), ON_star(Y_fixed, vv))
        )

    return pattern, builder


# def make_rewriter_put_tbl(X_fixed):
#     u, v = Consts("u v", IntSort())
#     pattern = ON_star(u, v)

#     def builder(subst):
#         uu, vv = subst[u], subst[v]
#         return And(
#             ON_star(uu, vv),
#             Or(Not(ON_star(uu, X_fixed)), ON_star(vv, X_fixed))
#         )

#     return pattern, builder

def wp_for_put_on_box(b_prime, b, Q):
    """Calculate the weakest precondition for Q when putting b_prime on table"""
    pattern, builder = make_rewriter_put(b_prime, b)
    return And(Not(ON_star(b, b_prime)), rewrite_expr(Q, pattern, builder))

# def wp_for_put_on_tbl(b_prime, Q):
    # pattern, builder = make_rewriter_put_tbl(b_prime)
    # return rewrite_expr(Q, pattern, builder)

#######################################################

# -----------------------------
# Symbolic number of boxes
# -----------------------------
n = Int('n')  # symbolic number of boxes
solver = Solver()
solver.add(n > 0)  # finite but unspecified number of boxes

# -----------------------------
# Box indices and ON_star relation
# -----------------------------
x, y, z, b, b_prime, b0 = Ints("x y z b b_prime b0")
ON_star = Function('ON_star', IntSort(), IntSort(), BoolSort())

# Utility: restrict variable to valid box indices
def is_box(x, n):
    return And(x >= 0, x < n)

# -----------------------------
# Axioms
# -----------------------------
# Transitivity: x on y and y on z => x on z
solver.add(
    ForAll([x, y, z],
           Implies(
               And(is_box(x, n), is_box(y, n), is_box(z, n),
                   ON_star(x, y), ON_star(y, z)),
               ON_star(x, z)
           ))
)

# Reflexivity: every box is on itself
solver.add(
    ForAll([x],
           Implies(is_box(x, n), ON_star(x, x)))
)

# Antisymmetry: if x on y and y on x, then they are the same box
solver.add(
    ForAll([x, y],
           Implies(
               And(is_box(x, n), is_box(y, n),
                   ON_star(x, y), ON_star(y, x)),
               x == y
           ))
)

# Partial order property (from original code)
solver.add(
    ForAll([x, y, z],
           Implies(
               And(is_box(x, n), is_box(y, n), is_box(z, n),
                   ON_star(x, y), ON_star(x, z)),
               Or(ON_star(y, z), ON_star(z, y))
           ))
)
solver.add(
    ForAll([x, y, z],
           Implies(
               And(is_box(x, n), is_box(y, n), is_box(z, n),
                   ON_star(x, z), ON_star(y, z)),
               Or(ON_star(x, y), ON_star(y, x))
           ))
)

# -----------------------------
# Top-of-stack predicate
# -----------------------------
def top(x):
    """Box x is on top (nothing on top of it)"""
    return And(is_box(x, n), Not(Exists([y], And(is_box(y, n), y != x, ON_star(y, x)))))

def while_cond(b_prime):
    # return Exists([b_prime], And(top(b_prime), b_prime != b))
    return Exists([b_prime], And(is_box(b_prime, n), Not(ON_star(b, b_prime))))

def while_cond_instantized(b_prime):
    # return And(top(b_prime), b_prime != b)
    return Not(ON_star(b, b_prime))

def precondition():
    return ForAll([x], Implies(is_box(x, n), top(x)))

def postcondition():
    return ForAll([x], Implies(is_box(x, n), ON_star(x, b0)))

def loop_invariant(b):
    # this is learned by the inference program WITHOUT the tbl as one of the axiom
    return And(
        ForAll([x], Implies(And(is_box(x, n), ON_star(x, b0)), ON_star(b, x))),
        ForAll([x], Implies(And(is_box(x, n), ON_star(b, x)), ON_star(x, b0))),
        ForAll([x, y], Implies(And(is_box(x, n), is_box(y, n), ON_star(y, x), Not(ON_star(x, b0))), ON_star(x, y)))
    )

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

solver.add(is_box(b0, n))
solver.add(is_box(b, n))

print("verifying precondition")
solver.push()
solver.add(precondition())
solver.add(Not(loop_invariant(b0)))
check_solver(solver)
solver.pop()

print("verifying inductive loop invariant")
solver.push()
solver.add(is_box(b_prime, n))
solver.add(while_cond_instantized(b_prime))
solver.add(loop_invariant(b))
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