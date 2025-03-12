from z3 import *
# set_option("smt.mbqi", False)
# set_option("sat.euf", True)
solver = Solver()

# Box = DeclareSort('Box')
Box, _ = EnumSort('Box', ['b9', 'b10', 'b11', 'b12', 'b13', "b14", "b15", 'b16', 'b17', 'b18'])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)
ON_star = Function('ON_star', Box, Box, BoolSort())
solver.add(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))))
solver.add(ForAll([x], ON_star(x, x)))
solver.add(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(x, c)), Or(ON_star(y, c), ON_star(c, y)))))
solver.add(ForAll([x, y], Implies(ON_star(x, y), Implies(ON_star(y, x), x == y))))

def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        for name1, box1 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
            for name2, box2 in zip(["b9", "b10", "b11"], [b9, b10, b11]):
                print(f"{name1}, {name2}: {x.model().evaluate(ON_star(box1, box2))}")
        import pdb; pdb.set_trace()
    else:
        print(solver.check())

def top(x):
    return Not(Exists([y], And(Not(y == x), ON_star(y, x))))

def precondition():
    return ForAll([a], Exists([x], Or(top(a), And(top(x), ON_star(x, a)))))

solver.push()
solver.add(Not(precondition()))
check_solver(solver)
solver.pop()