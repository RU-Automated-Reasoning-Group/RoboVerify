from z3 import *

# set_option("smt.core.minimize", "true")

solver = Solver()
solver.set(unsat_core=True)

Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13'])
# Box, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
# Box, (b9, b10) = EnumSort("Box", ["b9", "b10"])
x, y, c, n0, t, next_box, x1, tbl, new_x = Consts("x y c n0 t next_box x1 tbl new_x", Box)


def define_ON_func(func_name: str, solver: Solver):
    ON_star = Function(func_name, Box, Box, BoolSort())
    solver.assert_and_track(
        ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))),
        f"{func_name}_1",
    )
    solver.assert_and_track(ForAll([x], ON_star(x, x)), f"{func_name}_2")
    solver.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))
            ),
        ),
        f"{func_name}_3",
    )
    solver.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))
            ),
        ),
        f"{func_name}_4",
    )
    solver.assert_and_track(
        ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))),
        f"{func_name}_5",
    )
    solver.assert_and_track(
        ForAll(
            [x],
            Or(
                Exists(
                    [y],
                    And(
                        (y != x),
                        ForAll(
                            [x1], Implies(And(x1 != x, ON_star(x, x1)), ON_star(y, x1))
                        ),
                    ),
                ),
                on_table(x, ON_star),
            ),
        ),
        f"{func_name}_6",
    )
    solver.assert_and_track(
        ForAll([x], Implies(Or(ON_star(x, tbl), ON_star(tbl, x)), x == tbl)),
        f"{func_name}_tbl"
    )
    return ON_star


def func_equiv(func1, func2):
    return ForAll([x, y], func1(x, y) == func2(x, y))


def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # print(f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}")
        print("----------------")
        for name1, box1 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
            for name2, box2 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
                print(
                    f"ON_star({name1}, {name2}): {x.model().evaluate(ON_star(box1, box2))}"
                )

                                
        for name1, box1 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
            for name2, box2 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
                print(
                    f"ON_star_zero({name1}, {name2}): {x.model().evaluate(ON_star_zero(box1, box2))}"
                )
        solver = x
        for name1, box1 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
            for name2, box2 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
                x = box1
                y = box2
                b_prime = new_x
                b = t
                print(
                    f"""x={name1}, y={name2}: {solver.model().evaluate(
                        Implies(
                            And(x != tbl, y != tbl),
                            Or(
                                And(x != t, ON_func_substituted(x, n0, b_prime, b, ON_star), ON_func_substituted(x, y, b_prime, b, ON_star) == ON_star_zero(x, y)),
                                And(Not(ON_func_substituted(x, n0, b_prime, b, ON_star)), If(new_x != tbl, ON_func_substituted(new_x, x, b_prime, b, ON_star), False), ON_func_substituted(x, y, b_prime, b, ON_star) == ON_star_zero(y, x))
                            )
                        )
                )}"""
                )
        # for name1, box1 in zip(["b9", "b10", ], [b9, b10]):
        #     for name2, box2 in zip(["b9", "b10", ], [b9, b10]):
        #         print(
        #             f"ON_star({name1}, {name2}): {x.model().evaluate(ON_star(box1, box2))}"
        #         )
        # for name1, box1 in zip(["b9", "b10", ], [b9, b10]):
        #     for name2, box2 in zip(["b9", "b10", ], [b9, b10]):
        #         print(
        #             f"ON_star_zero({name1}, {name2}): {x.model().evaluate(ON_star_zero(box1, box2))}"
        #         )
        import pdb

        pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())


def precondition(ON_func):
    return And(
        ForAll([x], Implies(x != tbl, ON_func(x, n0))),
        n0 != tbl,
        Exists([x], And(top(x, ON_star), ON_star(x, n0)))
    )


def postcondition():
    return And(
        ForAll([x],Implies(x != tbl, ON_star(n0, x))),
        ForAll([x, y], Implies(And(x != tbl, y != tbl), ON_star_zero(x, y) == ON_star(y, x)))
    )


def postcondition_substituted(b_prime, b):
    if b == tbl:
        return And(
        ForAll([x], Implies(x != tbl, ON_func_substituted(n0, x, b_prime, b, ON_star))),
        ForAll([x, y], Implies(And(x != tbl, y != tbl), ON_star_zero(x, y) == ON_func_substituted(y, x, b_prime, b, ON_star)))
    )
    return And(
        Not(ON_star(b, b_prime)),
        ForAll([x], Implies(x != tbl, ON_func_substituted(n0, x, b_prime, b, ON_star))),
        ForAll([x, y], Implies(And(x != tbl, y != tbl), ON_star_zero(x, y) == ON_func_substituted(y, x, b_prime, b, ON_star)))
    )


# def postcondition_top():
# return ForAll([x], Implies(ON_star_zero(x, n0), top(x, ON_star)))


# def postcondition_on_table():
# return ForAll([x], Implies(ON_star_zero(x, n0), on_table(x, ON_star)))


def top(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))

def top_substituted(x, b_prime, b, ON_func):
    return Not(
        Exists(
            [y],
            And(Not(y == x), ON_func_substituted(y, x, b_prime, b, ON_func))
        )
    )

def top_substituted_for_null(x, b_prime, ON_func):
    return Not(
        Exists(
            [y],
            And(Not(y == x), ON_func_substituted_for_null(y, x, b_prime, ON_func))
        )
    )

def on_table(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))


def while_cond():
    return Exists([x], ForAll([c], Implies(Or(ON_star(n0, x), ON_star(c, n0)), ON_star(x, c))))
    # return Exists([x], And(top(x, ON_star), ON_star(x, n0), x != n0))


def while_cond_instance(x):
    return ForAll([c], Implies(Or(ON_star(n0, x), ON_star(c, n0)), ON_star(x, c)))
    # return And(top(x, ON_star), ON_star(x, n0), x != n0)


def loop_invariant(t):
    return And(
        ForAll(
            [x, y],
            Implies(
                And(x != tbl, y != tbl),
                Or(
                    And(x != t, ON_star(x, n0), ON_star(x, y) == ON_star_zero(x, y)),
                    And(Not(ON_star(x, n0)),  If(t != tbl, ON_star(t, x), False), ON_star(x, y) == ON_star_zero(y, x))
                    # If(t != tbl, And(Not(ON_star(x, n0)),  ON_star(t, x), ON_star(x, y) == ON_star_zero(y, x)), False)
                )
            )
        ),
        t != n0,
        # Exists([x], And(top(x, ON_star), ON_func_substituted(x, n0, new_x, t, ON_star)))
        Exists([x], And(top(x, ON_star), ON_star(x, n0)))
    )

def loop_invariant_substituted(new_x, b_prime, b):
    if b == tbl:
        return And(
            ForAll(
                [x, y],
                Implies(
                    And(x != tbl, y != tbl),
                    Or(
                        And(x != t, ON_func_substituted_for_null(x, n0, b_prime, ON_star), ON_func_substituted_for_null(x, y, b_prime, ON_star) == ON_star_zero(x, y)),
                        And(Not(ON_func_substituted_for_null(x, n0, b_prime, ON_star)), If(new_x != tbl, ON_func_substituted_for_null(new_x, x, b_prime, ON_star), False), ON_func_substituted_for_null(x, y, b_prime, ON_star) == ON_star_zero(y, x))
                    )
                )
            ),
            new_x != n0,
            Exists([x], And(top_substituted_for_null(x, b_prime, ON_star), ON_func_substituted_for_null(x, n0, b_prime, ON_star)))
        )
    else:
        return And(
            Not(ON_star(b, b_prime)),
            ForAll(
                [x, y],
                Implies(
                    And(x != tbl, y != tbl),
                    Or(
                        And(x != t, ON_func_substituted(x, n0, b_prime, b, ON_star), ON_func_substituted(x, y, b_prime, b, ON_star) == ON_star_zero(x, y)),
                        And(Not(ON_func_substituted(x, n0, b_prime, b, ON_star)), If(new_x != tbl, ON_func_substituted(new_x, x, b_prime, b, ON_star), False), ON_func_substituted(x, y, b_prime, b, ON_star) == ON_star_zero(y, x))
                    )
                )
            ),
            new_x != n0,
            Exists([x], And(top_substituted(x, b_prime, b, ON_star), ON_func_substituted(x, n0, b_prime, b, ON_star)))
        )


def not_loop_invar_instance(next_box):
    return And(
        Not(top_substituted(b11, next_box)), Not(ON_func_substituted(b11, n0, next_box))
    )


def ON_func_substituted(alpha, beta, b_prime, b, ON_func):
    return Or(
        ON_func(alpha, beta),
        And(ON_func(alpha, b_prime), ON_func(b, beta))
    )

def ON_func_substituted_for_null(alpha, beta, b_prime, ON_func):
    return And(
        ON_func(alpha, beta),
        Or(Not(ON_func(alpha, b_prime)), ON_func(beta, b_prime))
    )

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
    u, v = Consts("u v", Box)
    pattern = ON_star(u, v)

    def builder(subst):
        uu, vv = subst[u], subst[v]
        return Or(
            ON_star(uu, vv),
            And(ON_star(uu, X_fixed), ON_star(Y_fixed, vv))
        )

    return pattern, builder


def make_rewriter_put_tbl(X_fixed):
    u, v = Consts("u v", Box)
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

# def top_substituted(x, next_box, ON_func):
#     return Not(
#         Exists(
#             [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
#         )
#     )


# def on_table_substituted(x, next_box, ON_func):
#     return Not(
#         Exists(
#             [y], And(Not(y == x), ON_func_substituted(x, y, next_box, ON_func=ON_func))
#         )
#     )


# add ON_star and ON_star_zero
ON_star = define_ON_func("ON_star", solver)
ON_star_zero = define_ON_func("ON_star_zero", solver)
solver.assert_and_track(precondition(ON_star_zero), "pre_cond")

print("verifying precondition")
solver.push()
solver.assert_and_track(func_equiv(ON_star, ON_star_zero), "two_on_equiv")
solver.assert_and_track(Not(loop_invariant(tbl)), "not_loop_invar")
check_solver(solver)
solver.pop()

print("verifying loop invariant when t is tbl")
solver.push()
solver.assert_and_track(t == tbl, "t_is_tbl")
solver.assert_and_track(while_cond_instance(new_x), "while_cond")
solver.assert_and_track(loop_invariant(t), "loop_invar")


solver.push()
print("equivalence check 1", solver.check(wp_for_put_on_tbl(new_x, loop_invariant(new_x)) != loop_invariant_substituted(new_x, new_x, tbl)))
solver.pop()
# import pdb; pdb.set_trace()
# solver.assert_and_track(Not(loop_invariant_substituted(new_x, new_x, tbl)), "substituted_loop_invar")
solver.assert_and_track(Not(wp_for_put_on_tbl(new_x, loop_invariant(new_x))), "substituted_loop_invar")
check_solver(solver)
solver.pop()

print("verifying loop invariant when t is not tbl")
solver.push()
solver.assert_and_track(t != tbl, "t_neq_tbl")
solver.assert_and_track(while_cond_instance(new_x), "while_cond")
solver.assert_and_track(loop_invariant(t), "loop_invar")
# solver.assert_and_track(Not(loop_invariant_substituted(new_x, new_x, t)), "substituted_loop_invar")
solver.push()
print("equivalence check 2", solver.check(And(Not(ON_star(t, new_x)), wp_for_put_on_box(new_x, t, loop_invariant(new_x))) != loop_invariant_substituted(new_x, new_x, t)))
solver.pop()
solver.assert_and_track(Not(wp_for_put_on_tbl(new_x, wp_for_put_on_box(new_x, t, loop_invariant(new_x)))), "substituted_loop_invar")
check_solver(solver)
solver.pop()

# print("verifying post condition when t is tbl")
# solver.push()
# solver.assert_and_track(t == tbl, "t_is_tbl")
# solver.assert_and_track(Not(while_cond()), "not_while_cond")
# solver.assert_and_track(loop_invariant(t), "loop_invar")
# solver.assert_and_track(Not(postcondition_substituted(n0, tbl)), "not_post_cond")
# check_solver(solver)
# solver.pop()

print("verifying post condition when t is not tbl")
solver.push()
solver.assert_and_track(t != tbl, "t_neq_tbl")
solver.assert_and_track(Not(while_cond()), "not_while_cond")
solver.assert_and_track(loop_invariant(t), "loop_invar")

solver.push()
print("equivalence check 3", solver.check(wp_for_put_on_box(n0, t, postcondition()) != postcondition_substituted(n0, t)))
solver.pop()

solver.push()
print("tmp testing", solver.check(ON_star(t, n0)))
solver.pop()

# solver.assert_and_track(Not(postcondition_substituted(n0, t)), "not_post_cond")
solver.assert_and_track(Not(wp_for_put_on_box(n0, t, postcondition())), "not_post_cond")
check_solver(solver)
solver.pop()
