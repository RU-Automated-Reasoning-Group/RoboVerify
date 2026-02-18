from z3 import *
import pdb

solver = Solver()
solver.set(unsat_core=True)

BoxSort = DeclareSort("Box")
# BoxSort, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])
ON_star = Function("ON_star", BoxSort, BoxSort, BoolSort())
ON_star_zero = Function("ON_star_zero", BoxSort, BoxSort, BoolSort())

b, b0 = Consts("b b0", BoxSort)

from PIL import Image, ImageDraw, ImageFont

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

def check_solver(solver):
    if solver.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(solver.model())
        blocks = [b9, b10, b11, b12]
        names  = ["b9", "b10", "b11", "b12"]

        start_state   = extract_direct_on(solver.model(), blocks, names, ON_star_zero)
        current_state = extract_direct_on(solver.model(), blocks, names, ON_star)

        start_stacks   = build_stacks(start_state)
        current_stacks = build_stacks(current_state)

        print_stacks(start_stacks, "START STATE")
        print_stacks(current_stacks, "CURRENT STATE")

        draw_stacks(start_stacks, "start_state.png", "Start State")
        draw_stacks(current_stacks, "current_state.png", "Current State")
        import pdb

        pdb.set_trace()
    else:
        print(solver.check())
        print("Unsat Core:", solver.unsat_core())


def add_axiom(s: Solver):
    x, y, c = Consts("x y c", BoxSort)
    s.assert_and_track(
        ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))),
        "on1",
    )
    s.assert_and_track(ForAll([x], ON_star(x, x)), "on2")
    s.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y))
            ),
        ),
        "on3",
    )
    s.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star(x, c), ON_star(y, c)), Or(ON_star(x, y), ON_star(y, x))
            ),
        ),
        "on4",
    )
    s.assert_and_track(
        ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))), "on5"
    )


def add_axiom_on_star_zero(s: Solver):
    x, y, c = Consts("x y c", BoxSort)
    s.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(And(ON_star_zero(x, y), ON_star_zero(y, c)), ON_star_zero(x, c)),
        ),
        "on_zero_1",
    )
    s.assert_and_track(ForAll([x], ON_star_zero(x, x)), "on_zero_2")
    s.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star_zero(x, y), ON_star_zero(x, c)),
                Or(ON_star_zero(y, c), ON_star_zero(c, y)),
            ),
        ),
        "on_zero_3",
    )
    s.assert_and_track(
        ForAll(
            [x, y, c],
            Implies(
                And(ON_star_zero(x, c), ON_star_zero(y, c)),
                Or(ON_star_zero(x, y), ON_star_zero(y, x)),
            ),
        ),
        "on_zero_4",
    )
    s.assert_and_track(
        ForAll(
            [x, y], Implies(ON_star_zero(x, y), Implies(ON_star_zero(y, x), x == y))
        ),
        "on_zero_5",
    )


def func_equiv(func1, func2):
    x, y = Consts("x y", BoxSort)
    return ForAll([x, y], func1(x, y) == func2(x, y))


def precondition():
    x, y = Consts("x y", BoxSort)
    return And(
        ForAll([x], (x != b) == ON_star(x, b0)),
        ForAll([x], Implies(ON_star(x, b), x == b)),
        ForAll([x], ON_star_zero(x, b0)),
        ForAll([x, y], Implies(And(x != b, y != b), ON_star(x, y) == ON_star_zero(x, y))),
        ForAll([x], Implies(ON_star_zero(x, b), x == b))
    )

def postcondition():
    x, y = Consts("x y", BoxSort)
    return And(
        ForAll([x], (x != b0) == ON_star(b, x)),
        ForAll([x], Implies(ON_star(x, b0), x == b0)),
        ForAll([x, y], Implies(And(x != b0, y != b0), ON_star(x, y) == ON_star_zero(y, x))),
    )

def while_cond():
    x, c = Consts("x c", BoxSort)
    return Exists([x], ForAll([c], Implies(Or(ON_star(b0, x), ON_star(c, b0)), ON_star(x, c))))
    # return Exists([x], And(top(x, ON_star), ON_star(x, n0), x != n0))


def while_cond_instance(new_x):
    c, = Consts("c", BoxSort)
    return ForAll([c], Implies(Or(ON_star(b0, new_x), ON_star(c, b0)), ON_star(new_x, c)))
    # return And(top(x, ON_star), ON_star(x, n0), x != n0)


def loop_invariant(solver: Solver):
    x, y = Consts("x y", BoxSort)
    solver.assert_and_track(ForAll([x], Or(ON_star(b, x), ON_star(x, b0))), "term0")
    solver.assert_and_track(
        ForAll([x], Or(Not(ON_star(b, x)), Not(ON_star(x, b0)))), "term1"
    )
    solver.assert_and_track(
        ForAll(
            [x, y], Implies(And(ON_star(x, b0), Not(ON_star_zero(y, x))), ON_star(x, y))
        ),
        "term2",
    )
    solver.assert_and_track(
        ForAll(
            [x, y],
            Implies(And(Not(ON_star(x, b0)), Not(ON_star_zero(x, y))), ON_star(x, y)),
        ),
        "term3",
    )
    solver.assert_and_track(
        ForAll([x, y], Or(Not(ON_star(x, b0)), Not(ON_star(x, y)), ON_star_zero(x, y))),
        "term4",
    )

def loop_invariant_combined(t):
    x, y = Consts("x y", BoxSort)
    return And(
        ForAll([x], Or(ON_star(t, x), ON_star(x, b0))),
        ForAll([x], Or(Not(ON_star(t, x)), Not(ON_star(x, b0)))),
        ForAll(
            [x, y], Implies(And(ON_star(x, b0), Not(ON_star_zero(y, x))), ON_star(x, y))
        ),
        ForAll(
            [x, y],
            Implies(And(Not(ON_star(x, b0)), Not(ON_star_zero(x, y))), ON_star(x, y)),
        ),
        ForAll([x, y], Or(Not(ON_star(x, b0)), Not(ON_star(x, y)), ON_star_zero(x, y))),
    )


add_axiom_on_star_zero(solver)
add_axiom(solver)

print("verifying precondition")
solver.push()
# loop_invariant(solver)
solver.assert_and_track(Not(loop_invariant_combined(b)), "negated_invariant")
# solver.assert_and_track(func_equiv(ON_star, ON_star_zero), "func_equiv")
solver.assert_and_track(precondition(), "precondition")
check_solver(solver)
solver.pop()

print("verifying inductiveness of loop invariant")
solver.push()

new_x, = Consts("newx", BoxSort)
solver.assert_and_track(loop_invariant_combined(b), "loop_invariant_combined")
solver.assert_and_track(while_cond_instance(new_x), "while_cond_instance")
solver.assert_and_track(Not(wp_for_put_on_tbl(new_x, wp_for_put_on_box(new_x, b, loop_invariant_combined(new_x)))), "substituted_loop_invar")
# solver.assert_and_track(wp_for_put_on_tbl(new_x, wp_for_put_on_box(new_x, b, loop_invariant_combined(new_x))), "substituted_loop_invar")

check_solver(solver)
solver.pop()

print("verifying postcondition")
solver.push()
x, y = Consts("x y", BoxSort)
solver.assert_and_track(Exists([x], And(ForAll([y], Implies(ON_star(y, x), x == y)), ON_star(x, b0))), "exists_top")

solver.assert_and_track(loop_invariant_combined(b), "loop_invariant_combined")
solver.assert_and_track(Not(while_cond()), "not_while_cond")
solver.assert_and_track(Not(postcondition()), "not_postcondition")
check_solver(solver)
solver.pop()
