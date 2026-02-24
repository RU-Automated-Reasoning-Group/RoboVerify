from z3 import *

# set_option("smt.core.minimize", "true")

solver = Solver()
solver.set(unsat_core=True)

# BoxSort = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13, b14, b15, b16, b17, b18) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18'])
BoxSort, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])
x, y, c, n0, t, next_box, x1, fresh = Consts("x y c n0 t next_box x1 fresh", BoxSort)

# define ON_star
ON_star = Function("ON_star", BoxSort, BoxSort, BoolSort())
solver.assert_and_track(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))), "on1")
solver.assert_and_track(ForAll([x], ON_star(x, x)), "on2")
solver.assert_and_track(
    ForAll(
        [x, y, c],
        Implies(And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))),
    ),
    "on3"
)
solver.assert_and_track(
    ForAll(
        [x, y, c],
        Implies(And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))),
    ),
    "on4"
)
solver.assert_and_track(ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))), "on5")


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

def extract_direct_on(model, blocks, names, ON_star):
    direct_on = {n: "table" for n in names}

    for i, (a_name, a) in enumerate(zip(names, blocks)):
        for j, (b_name, b) in enumerate(zip(names, blocks)):

            if a_name == b_name:
                continue

            if not model.evaluate(ON_star(a,b)):
                continue

            # check if there exists an intermediate block
            has_middle = False

            for k, (c_name, c) in enumerate(zip(names, blocks)):
                if c_name in [a_name, b_name]:
                    continue

                if model.evaluate(ON_star(a,c)) and model.evaluate(ON_star(c,b)):
                    has_middle = True
                    break

            if not has_middle:
                direct_on[a_name] = b_name

    return direct_on

def build_stacks(direct_on):
    stacks = []
    visited = set()

    blocks = list(direct_on.keys())

    bases = [b for b,v in direct_on.items() if v == "table"]

    for base in bases:
        if base in visited:
            continue

        stack = [base]
        visited.add(base)
        top = base

        while True:
            above = None
            for b,v in direct_on.items():
                if v == top and b not in visited:
                    above = b
                    break

            if above:
                stack.append(above)
                visited.add(above)
                top = above
            else:
                break

        stacks.append(stack)

    return stacks

def print_stacks(stacks, title):
    print("\n" + title)

    for stack in stacks:
        for block in reversed(stack):
            print(f"   [{block}]")
        print("  --------")
        print("   table\n")

def draw_stacks(stacks, filename, title):
    width, height = 800, 400
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    block_w = 70
    block_h = 30
    gap = 60

    draw.text((10,10), title, fill="black", font=font)

    x = 50
    for stack in stacks:
        y = 350

        for block in stack:
            draw.rectangle([x, y-block_h, x+block_w, y], outline="black", width=2)
            draw.text((x+15, y-block_h+5), block, fill="black", font=font)
            y -= block_h + 5

        draw.line([x, y+5, x+block_w, y+5], fill="black", width=3)
        x += block_w + gap

    img.save(filename)

def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # print(f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}")
        # print("----------------")
        # for name1, box1 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
        #     for name2, box2 in zip(["b9", "b10", "b11", "b12"], [b9, b10, b11, b12]):
        #         print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")

        blocks = [b9, b10, b11, b12]
        names  = ["b9", "b10", "b11", "b12"]

        current_state = extract_direct_on(solver.model(), blocks, names, ON_star)

        current_stacks = build_stacks(current_state)

        print_stacks(current_stacks, "CURRENT STATE")

        draw_stacks(current_stacks, "current_state.png", "Current State")
        
        import pdb
        pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())

def precondition():
    return And(
        ForAll([x], ON_star(x, n0)),
        Exists([x], And(top(x), ON_star(x, n0)))
    )

def postcondition():
    return ForAll([x], top(x))

def on_table(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))

def top(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))

def while_cond():
    return Exists([x], ForAll([c], Implies(Or(ON_star(n0, x), ON_star(c, n0)), ON_star(x, c))))
    # return Exists([x], And(top(x), ON_star(x, n0), x != n0))

def while_cond_instance(x):
    return ForAll([c], Implies(Or(ON_star(n0, x), ON_star(c, n0)), ON_star(x, c)))
    # return And(top(x), ON_star(x, n0), x != n0)

def loop_invariant():
    return And(
        ForAll([x, y], Implies(And(ON_star(y, x), Not(ON_star(x, n0))), ON_star(x, y))),
        Exists([t], And(top(t), ON_star(t, n0)))
    )

def loop_invariant_substituted(next_box):
    # # return ForAll(
    #     # [x], 
    #     # Or(top_substituted(x, next_box), ON_func_substituted(x, n0, next_box))
    # # )

    # # return Not(
    #     # ForAll(
    #         # [x], 
    #         # Or(top_substituted(x, next_box), ON_func_substituted(x, n0, next_box))
    #     # )
    # # )

    # return Exists(
    #     [x],
    #     And(Not(top_substituted(x, next_box)), Not(ON_func_substituted(x, n0, next_box)))
    # )
    return And(
        wp_for_put_on_tbl(next_box, ForAll([x, y], Implies(And(ON_star(y, x), Not(ON_star(x, n0))), ON_star(x, y)))),
        Exists([t], And(top_substituted(t, next_box), ON_func_substituted(t, n0, next_box)))
    )

def not_loop_invar_instance(next_box):
    return And(Not(top_substituted(b11, next_box)), Not(ON_func_substituted(b11, n0, next_box)))

def ON_func_substituted(alpha, beta, next_box, ON_func=ON_star):
    return And(
        ON_func(alpha, beta), Or(Not(ON_func(alpha, next_box)), ON_func(beta, next_box))
    )

def top_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
        )
    )

def exists_next():
    return ForAll(
        [x],
        Or(
            Exists(
                [y],
                And(
                    (y != x),
                    ForAll([x1], Implies(And(x1 != x, ON_star(x, x1)), ON_star(y, x1)))
                ) 
            ),
            on_table(x)
        )
    )

# print("testing")
# solver.push()
# solver.assert_and_track(
#     Not(ForAll([t], Exists([x], Or(top(t), And(top(x), ON_star(x, t)))))),
#     "test"
# )
# solver.add(Not(exists_next()))
# check_solver(solver)
# solver.pop()
solver.assert_and_track(
    exists_next(),
    "exist_next"
)

print("verifying precondition")
solver.push()
solver.assert_and_track(precondition(), "pre_cond")
solver.assert_and_track(Not(loop_invariant()), "not_loop_invar")
check_solver(solver)
solver.pop()

print("verifying loop invariant")
solver.push()
solver.assert_and_track(while_cond_instance(fresh), "while_cond")
solver.assert_and_track(loop_invariant(), "loop_invar")
solver.assert_and_track(
    Not(loop_invariant_substituted(fresh)),
    "substituted_loop_invar"
)
check_solver(solver)
solver.pop()

print("verifying post condition")
solver.push()
solver.assert_and_track(Not(while_cond()), "not_while_cond")
solver.assert_and_track(loop_invariant(), "loop_invar")
solver.assert_and_track(Not(postcondition()), "not_post_cond")
check_solver(solver)
solver.pop()