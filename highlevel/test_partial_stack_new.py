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
    EnumSort
)
from PIL import Image, ImageDraw, ImageFont


# pdb.set_trace()
# set_option("smt.mbqi", False)

solver = Solver()

BoxSort = DeclareSort("Box")
# BoxSort, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])


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
solver.assert_and_track(
    ForAll(
        [x], Exists([y], And(ForAll([c], Implies(ON_star(c, y), c == y)), ON_star(y, x)))
    ),
    "on_exists_top"
)

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

        blocks = [b9, b10, b11, b12]
        names  = ["b9", "b10", "b11", "b12"]

        current_state = extract_direct_on(solver.model(), blocks, names, ON_star)

        current_stacks = build_stacks(current_state)

        print_stacks(current_stacks, "CURRENT STATE")

        draw_stacks(current_stacks, "current_state_partial.png", "Current State")
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
    return Exists([b_prime], And(top(b_prime), b_prime != b))
    # return Exists([b_prime], Not(ON_star(b, b_prime)))


def while_cond_instantized(b_prime):
    return And(top(b_prime), b_prime != b)
    # return Not(ON_star(b, b_prime))


def precondition():
    return And(top(b0), on_table(b0))

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
        ForAll([x, y], Implies(And(ON_star(y, x), ON_star(y, b0), Not(ON_star(x, b0))), ON_star(x, y)))
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
wp = wp_for_put_on_tbl(b_prime, wp_for_put_on_box(b_prime, b, loop_invariant(b_prime)))
# wp = wp_for_put_on_box(b_prime, b, loop_invariant(b_prime))
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
