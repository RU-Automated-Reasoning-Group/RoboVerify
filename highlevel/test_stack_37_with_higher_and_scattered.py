from z3 import (
    And,
    BoolSort,
    Consts,
    DeclareSort,
    EnumSort,
    Exists,
    ForAll,
    Function,
    Implies,
    Not,
    Or,
    Solver,
    is_true,
    sat,
    unsat,
)

# pdb.set_trace()
# set_option("smt.mbqi", False)
from PIL import Image, ImageDraw, ImageFont

solver = Solver()

Box, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])
# Box = DeclareSort("Box")
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)
ON_star = Function("ON_star", Box, Box, BoolSort())
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

higher = Function("higher", Box, Box, BoolSort())
solver.add(ForAll([x, y, c], Implies(And(higher(x, y), higher(y, c)), higher(x, c))))
solver.add(ForAll([x], higher(x, x)))
solver.add(
    ForAll(
        [x, y, c],
        Implies(And(higher(x, y), higher(x, c)), Or(higher(y, c), higher(c, y))),
    )
)

scattered = Function("scattered", Box, Box, BoolSort())
solver.add(ForAll([x, y], scattered(x, y) == scattered(y, x)))
solver.add(ForAll([x], Not(scattered(x, x))))

# ON_star–scattered coupling: blocks in the same vertical stack
# have identical scattered-profiles w.r.t. every other block.
m, n, z = Consts("m n z", Box)
solver.add(
    ForAll(
        [m, n, z],
        Implies(
            ON_star(m, n),
            scattered(m, z) == scattered(n, z),
        ),
    )
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

def _bool_str(b: bool) -> str:
    return "T" if b else "F"

def print_pairwise_relations(model, blocks, names):
    """
    Print pair-wise truth tables (rows=lhs, cols=rhs) for:
      - ON_star
      - higher
      - scattered
    """

    def print_matrix(title, pred):
        print(f"\n{title} (rows=lhs, cols=rhs)")
        print("      " + "  ".join(names))
        for i, ni in enumerate(names):
            row = []
            for j in range(len(names)):
                val = model.evaluate(pred(blocks[i], blocks[j]), model_completion=True)
                row.append(_bool_str(is_true(val)))
            print(f"{ni:>4}  " + "  ".join(row))

    print_matrix("ON_star", ON_star)
    print_matrix("higher", higher)
    print_matrix("scattered", scattered)

def check_solver(solver):
    if solver.check() == sat:
        print("constraints satisfiable")
        print("model is")
        model = solver.model()
        print(model)
        blocks = [b9, b10, b11, b12]
        names  = ["b9", "b10", "b11", "b12"]

        print_pairwise_relations(model, blocks, names)

        current_state = extract_direct_on(model, blocks, names, ON_star)

        current_stacks = build_stacks(current_state)

        print_stacks(current_stacks, "CURRENT STATE")

        draw_stacks(current_stacks, "current_state.png", "Current State")
        import pdb

        pdb.set_trace()
    else:
        print(solver.check())
        print("Unsat Core:", solver.unsat_core())



# def check_solver(s):
#     res = s.check()
#     print(res)
#     if res == sat:
#         print("model is")
#         print(s.model())


def on_table(x):
    return ForAll([y], higher(y, x))


def scattered_invariant():
    # High-level invariant learned from data:
    # all on-table blocks are pairwise scattered.
    return ForAll(
        [x, y],
        Implies(And(on_table(x), on_table(y), x != y), scattered(x, y)),
    )


def substituted_scattered_invariant(b_prime, b):
    # Post-state version: all blocks on the table after put are pairwise scattered.
    fresh_x, fresh_y = Consts("fresh_x fresh_y", Box)
    return ForAll(
        [fresh_x, fresh_y],
        Implies(
            And(
                substituted_on_table(fresh_x, b_prime, b),
                substituted_on_table(fresh_y, b_prime, b),
                fresh_x != fresh_y,
            ),
            Or(
                And(fresh_x != b_prime, fresh_x != b, fresh_y != b_prime, fresh_y != b, scattered(fresh_x, fresh_y)),
                And(fresh_x != b_prime, fresh_x != b, fresh_y == b, scattered(fresh_x, fresh_y)),
                And(fresh_x != b_prime, fresh_x != b, fresh_y == b_prime, scattered(fresh_x, b)),
                And(fresh_x == b_prime, fresh_y != b, fresh_y != b_prime, scattered(b, fresh_y)),
                And(fresh_x == b, fresh_y != b_prime, fresh_y != b, scattered(fresh_x, fresh_y)),
            )
        ),
    )


def top(x):
    return And(
        Not(Exists([y], And(Not(y == x), ON_star(y, x)))),
        ForAll([y, c], Implies(And(ON_star(y, c), ON_star(x, c)), higher(x, y)))
    )


def substituted_top(x, b_prime, b):
    return And(
        Not(
            Exists(
                [y],
                And(
                    Not(y == x), Or(ON_star(y, x), And(ON_star(y, b_prime), ON_star(b, x)))
                ),
            )
        ),
        ForAll(
            [y, c], 
            Implies(
                And(
                    Or(ON_star(y, c), And(ON_star(y, b_prime), ON_star(b, c))),
                    Or(ON_star(x, c), And(ON_star(x, b_prime), ON_star(b, c)))
                ),
                Or(higher(x, y), And(higher(x, b_prime), higher(b, y)))
            )
        )
    )


def substituted_on_table(x, b_prime, b):
    fresh_gg, = Consts("fresh_gg", Box)
    return ForAll(
        [fresh_gg],
        Or(higher(fresh_gg, x), And(higher(fresh_gg, b_prime), higher(b, x)))
    )


def while_cond(b_prime):
    return Exists([b_prime], And(top(b_prime), b_prime != b))


def while_cond_instantized(b_prime):
    return And(top(b_prime), b_prime != b)


def postcondition():
    return ForAll([a], ON_star(a, b0))


def loop_invariant(b):
    return And(
        ForAll([a], Or(And(top(a), on_table(a)), ON_star(a, b0))),
        scattered_invariant(),
        ON_star(b, b0),
        top(b),
    )


def substituted_loop_invariant(x, b_prime, b):
    return And(
        # ForAll(
        #     [a],
        #     Or(
        #         And(substituted_top(a, b_prime, b), substituted_on_table(a, b_prime, b)),
        #         Or(ON_star(a, b0), And(ON_star(a, b_prime), ON_star(b, b0))),
        #     ),
        # ),
        # Or(ON_star(x, b0), And(ON_star(x, b_prime), ON_star(b, b0))),
        # substituted_top(x, b_prime, b),
        substituted_scattered_invariant(b_prime, b),
    )


# while loop correctness: verify while condition (true) + loop invariant implies another loop invariant
solver.push()
print("verifying inductive loop invariant")
solver.add(on_table(b0))
solver.add(while_cond_instantized(b_prime))
solver.add(loop_invariant(b))
wp = And(Not(ON_star(b, b_prime)), substituted_loop_invariant(b_prime, b_prime, b))
solver.add(Not(wp))
# solver.add(wp)
check_solver(solver)
solver.pop()

print("verifying postcondition")
solver.push()
solver.add(loop_invariant(b))
solver.add(Not(while_cond(b_prime)))
solver.add(Not(postcondition()))
check_solver(solver)
solver.pop()

# Optional: check that the loop invariant itself is satisfiable with the axioms
# print("checking loop invariant satisfiability")
# solver.push()
# solver.add(loop_invariant(b))
# check_solver(solver)
# solver.pop()
