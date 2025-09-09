from z3 import *

set_option("smt.core.minimize", "true")

solver = Solver()
solver.set(unsat_core=True)


Box = DeclareSort("Box")
# Box, (b9, b10, b11, b12, b13, b14, b15) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15'])
# Box, (b9, b10, b11) = EnumSort("Box", ["b9", "b10", "b11"])
x, y, c, n0, t, next_box, x1 = Consts("x y c n0 t next_box x1", Box)

# define ON_star
ON_star = Function("ON_star", Box, Box, BoolSort())
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


def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        # for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
        # print(f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}")
        print("----------------")
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        import pdb

        pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())

def precondition():
    return ForAll([x], ON_star(x, n0))

def postcondition():
    return ForAll([x], top(x))

def on_table(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))

def top(x, ON_func=ON_star):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))

def loop_invariant(t):
    return And(
        ForAll([x], Or(top(x), ON_star(x, n0))),
        top(t),
        ON_star(t, n0)
    )

def loop_invariant_substituted(next_box):
    return ForAll([x],
        Or(
            top_substituted(x, next_box),
            ON_func_substituted(x, n0, next_box)
        )
    )

def while_cond():
    # return (t != n0)
    return Not(top(n0))

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

print("testing the necessity of table")
solver.push()
solver.assert_and_track(precondition(), "all_on_n0")
solver.assert_and_track(Not(on_table(n0)), "not_n0_on_table")
check_solver(solver)
solver.pop()
# this indicates that there's no need to have on_table(n0)
# in precondition because it can be derived from the other part

print("testing the necessity of table (2)")
solver.push()
solver.assert_and_track(ForAll([x], Or(top(x), ON_star(x, n0))), "top_or_on_n0")
solver.assert_and_track(Not(on_table(n0)), "not_n0_on_table")
check_solver(solver)
solver.pop()
# this indicates there's no need to have on table for n0 in the loop invariant
# either.

print("verifying precondition")
solver.push()
solver.assert_and_track(precondition(), "pre_cond")
solver.assert_and_track(And(top(t), ON_star(t, n0)), "t_constraint")
solver.assert_and_track(Not(loop_invariant(t)), "not_loop_invar")
check_solver(solver)
solver.pop()

print("verifying loop invariant")
solver.push()
solver.assert_and_track(while_cond(), "while_cond")
solver.assert_and_track(loop_invariant(t), "loop_invar")
solver.assert_and_track(
    Not(loop_invariant_substituted(t)),
    "substituted_loop_invar"
)
check_solver(solver)
solver.pop()

print("verifying post condition")
solver.push()
solver.assert_and_track(Not(while_cond()), "not_while_cond")
solver.assert_and_track(loop_invariant(t), "loop_invar")
solver.assert_and_track(Not(postcondition()), "not_post_cond")
check_solver(solver)
solver.pop()