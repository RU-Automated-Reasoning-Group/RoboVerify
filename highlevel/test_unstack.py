from z3 import *
# set_option("smt.mbqi", False)
# set_option("sat.euf", True)

solver = Solver()

Box = DeclareSort('Box')
Box, (b9, b10, b11, b12, b13) = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13'])
# Box, (b9, b10, b11) = EnumSort('Box', ['b9', 'b10', 'b11'])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)

# define ON_star
ON_star = Function('ON_star', Box, Box, BoolSort())
solver.add(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))))
solver.add(ForAll([x], ON_star(x, x)))
solver.add(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y)))))
solver.add(ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))))

# define ON_star_zero
ON_star_zero = Function('ON_star_zero', Box, Box, BoolSort())
solver.add(ForAll([x, y, c], Implies(And(ON_star_zero(x, y), ON_star_zero(y, c)), ON_star_zero(x, c))))
solver.add(ForAll([x], ON_star_zero(x, x)))
solver.add(ForAll([x, y, c], Implies(And(ON_star_zero(x, y), ON_star_zero(x, c)), Or(ON_star_zero(y, c), ON_star_zero(c, y)))))
solver.add(ForAll([x, y], Implies(ON_star_zero(x, y), Implies(ON_star_zero(y, x), x == y))))
# solver.add(ON_star_zero(b10, b9))
solver.add(b0 == b9)

def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"{name1}, {name2}: {x.model().evaluate(ON_star_zero(box1, box2))}")
        print("----------------")
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        import pdb; pdb.set_trace()
    else:
        print(x.check())
        print("Unsat Core:", x.unsat_core())

def top(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(y, x))))

def on_table(x, ON_func):
    return Not(Exists([y], And(Not(y == x), ON_func(x, y))))

def while_cond():
    return Exists([x], And(top(x, ON_star), ON_star(x, b0)))

def postcondition():
    return ForAll([x], Implies(ON_star_zero(x, b0), on_table(x, ON_star)))

def loop_invariant():
    return ForAll([x], Implies(ON_star_zero(x, b0), Or(ON_star(x, b0), on_table(x, ON_star))))

print("verifying postcondition")
solver.push()
# solver.add(loop_invariant())
solver.add(Not(while_cond()))
# solver.add(postcondition())
check_solver(solver)
solver.pop()