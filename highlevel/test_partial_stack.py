from z3 import *
from PIL import Image, ImageDraw, ImageFont

# set_option("smt.mbqi", False)
# set_option("sat.euf", True)

# set_option("verbose", 10)
# set_option("smt.mbqi.max_iterations", 1000000)

solver = Solver()

# Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13, b14, b15, b16) = EnumSort(
    # "Box", ["b9", "b10", "b11", "b12", "b13", "b14", "b15", "b16"]
# )
Box, (b9, b10, b11, b12) = EnumSort("Box", ["b9", "b10", "b11", "b12"])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)

# define ON_star
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

# define ON_star_zero
ON_star_zero = Function("ON_star_zero", Box, Box, BoolSort())
solver.add(
    ForAll(
        [x, y, c],
        Implies(And(ON_star_zero(x, y), ON_star_zero(y, c)), ON_star_zero(x, c)),
    )
)
solver.add(ForAll([x], ON_star_zero(x, x)))
solver.add(
    ForAll(
        [x, y, c],
        Implies(
            And(ON_star_zero(x, y), ON_star_zero(x, c)),
            Or(ON_star_zero(y, c), ON_star_zero(c, y)),
        ),
    )
)
solver.add(
    ForAll(
        [x, y, c],
        Implies(
            And(ON_star_zero(x, c), ON_star_zero(y, c)),
            Or(ON_star_zero(x, y), ON_star_zero(y, x)),
        ),
    )
)
solver.add(
    ForAll([x, y], Implies(ON_star_zero(x, y), Implies(ON_star_zero(y, x), x == y)))
)
# solver.add(ON_star_zero(b10, b9))
# solver.add(b0 == b9)

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

        start_state   = extract_direct_on(solver.model(), blocks, names, ON_star_zero)
        current_state = extract_direct_on(solver.model(), blocks, names, ON_star)

        start_stacks   = build_stacks(start_state)
        current_stacks = build_stacks(current_state)

        print_stacks(start_stacks, "START STATE")
        print_stacks(current_stacks, "CURRENT STATE")

        draw_stacks(start_stacks, "start_state_partial.png", "Start State")
        draw_stacks(current_stacks, "current_state_partial.png", "Current State")
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #         print(
        #             f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}"
        #         )
        # print("----------------")
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #     for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        #         print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        # import pdb

        # pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())


def ON_func_substituted(alpha, beta, next_box, ON_func=ON_star):
    return And(
        ON_func(alpha, beta), Or(Not(ON_func(alpha, next_box)), ON_func(beta, next_box))
    )


def top(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))


def top_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(y, x, next_box, ON_func=ON_func))
        )
    )


def on_table(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))


def on_table_substituted(x, next_box, ON_func=ON_star):
    return Not(
        Exists(
            [y], And(Not(y == x), ON_func_substituted(x, y, next_box, ON_func=ON_func))
        )
    )


def while_cond():
    return Exists([x], And(top(x, ON_star), ON_star(x, b0), x != b0))


def while_cond_instantized(x):
    return And(top(x, ON_star), ON_star(x, b0), x != b0)


def precondition():
    return ForAll(
        [a], Exists([x], Or(top(a, ON_star), And(top(x, ON_star), ON_star(x, a))))
    )


def postcondition():
    return ForAll(
        [x], Implies(ON_star_zero(x, b0), And(on_table(x, ON_star), top(x, ON_star)))
    )


def postcondition_before_on_table_n0():
    return ForAll(
        [x],
        Implies(
            ON_star_zero(x, b0),
            on_table_substituted(x, b0, ON_func=ON_star)
            # top_substituted(x, b0, ON_func=ON_star),
        ),
    )


# def postcondition_before_on_table_n0():
#     return ForAll(
#         [a, x],
#         Implies(
#             ON_star_zero(a, b0),
#             Or(
#                 x == a, Not(And(ON_star(a, x), Or(Not(ON_star(a, b0)), ON_star(x, b0))))
#             ),
#         ),
#     )


def loop_invariant():
    return ForAll(
        [x],
        Implies(
            ON_star_zero(x, b0),
            Or(ON_star(x, b0), top(x, ON_star)),
        ),
    )
    # return ForAll([x, y], And(Implies(ON_star_zero(b0, x), ON_star_zero(x, y) == ON_star(x, y)), Implies(ON_star_zero(x, b0), Or(ON_star(x, b0), on_table(x, ON_star)))))


def loop_invariant_substituted(next_box):
    return ForAll(
        [a],
        Implies(
            ON_star_zero(a, b0),
            Or(
                ON_func_substituted(a, b0, next_box, ON_func=ON_star),
                And(
                    on_table_substituted(a, next_box, ON_func=ON_star),
                    top_substituted(a, next_box, ON_star),
                ),
            ),
        ),
    )


# print("testing precondition")
# solver.push()
# solver.add(Not(precondition()))
# check_solver(solver)
# solver.pop()

# print("verifying precondition")
# solver.push()
# solver.add(ForAll([x, y], ON_star_zero(x, y) == ON_star(x, y)))
# solver.add(precondition())
# solver.add(Not(loop_invariant()))
# check_solver(solver)
# solver.pop()

print("verifying loop invariant")
solver.push()
solver.add(while_cond_instantized(x))
solver.add(loop_invariant())
solver.add(Not(loop_invariant_substituted(x)))
check_solver(solver)
solver.pop()

# print("verifying postcondition", flush=True)
# solver.push()
# solver.add(loop_invariant())
# solver.add(Not(while_cond()))
# solver.add(Not(postcondition_before_on_table_n0()))
# check_solver(solver)
# solver.pop()


# # def next_loop_invariant(b):
# #     return And(
# #         ForAll(
# #             [a],
# #             Exists(
# #                 [x],
# #                 Or(
# #                     ON_star(a, b0), top(a, ON_star), And(top(x, ON_star), ON_star(x, a))
# #                 ),
# #             ),
# #         ),
# #         ON_star(b, b0),
# #         top(b, ON_star),
# #         on_table(b0, ON_star),
# #     )


# # print("verifying postcondition implies the loop invariant of next loop")
# # solver.push()
# # solver.add(postcondition())
# # solver.add(Not(next_loop_invariant(b0)))
# # check_solver(solver)
# # solver.pop()
