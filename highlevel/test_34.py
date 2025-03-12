from z3 import *
set_option("smt.mbqi", False)

solver = Solver()

Box = DeclareSort('Box')
# Box, (b0, b1, b2) = EnumSort('Box', ['b0', 'b1', 'b2'])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)
ON = Function('ON', Box, Box, BoolSort())
# solver.add(ForAll([x, y, c], Implies(And(ON_star(x, y), ON_star(y, c)), ON_star(x, c))))
# solver.add(ForAll([x, y], Implies(x != y, Not(And( ON(x, y),  ON(y, x) )))))
# solver.add(ForAll([x, y, c], Implies(And(ON(x, c), ON(y, c), x != c, y != c), x == y)))
ON_star = TransitiveClosure(ON)

def check_solver(x):
    if x.check() == sat:
        print("constraints satisfiable")
        print("model is")
        print(x.model())
        import pdb; pdb.set_trace()
    else:
        # print("NOT satisfiable")
        print(x.check())

def top(x):
    return Not(Exists([y], And(Not(y == x), ON_star(y, x))))

def substituted_top(x, b_prime, b):
    return Not(Exists([y], And(Not(y == x), Or(ON_star(y, x), And(ON_star(y, b_prime), ON_star(b, x))))))

def loop_invariant(b):
    return And(ForAll([a], Or(ON_star(a, b0), top(a))), ON_star(b, b0), top(b))

def substituted_loop_invariant(x, b_prime, b):
    return And(
        ForAll([a], Or(Or(ON_star(a, b0), And(ON_star(a, b_prime), ON_star(b, b0))), substituted_top(a, b_prime, b))),
        Or(ON_star(x, b0), And(ON_star(x, b_prime), ON_star(b, b0))),
        substituted_top(x, b_prime, b)
    )

def while_cond(b_prime):
    return Exists([b_prime], And(top(b_prime), b_prime != b))

def while_cond_instantized(b_prime):
    return And(top(b_prime), b_prime != b)

def precondition():
    return ForAll([a], top(a))

def postcondition():
    return ForAll([a], ON_star(a, b0))

# precondition correctness: precondition + b <- b0 implies loop invariant
print("verifying precondition")
solver.push()
solver.add(precondition())
solver.add(Not(loop_invariant(b0)))
check_solver(solver)
solver.pop()



# while loop correctness: verify while condition (true) + loop invariant implies another loop invariant
solver.push()
print("verifying inductive loop invariant")
solver.add(while_cond_instantized(b_prime))
solver.add(loop_invariant(b))
wp = And(Not(ON_star(b, b_prime)), substituted_loop_invariant(b_prime, b_prime, b))
solver.add(Not(wp))
check_solver(solver)
solver.pop()

# postcondition correctness: verify while condition (false) + loop invariant implies postcondition
print("verifying postcondition")
solver.push()
solver.add(loop_invariant(b))
solver.add(Not(while_cond(b_prime)))
solver.add(Not(postcondition()))
check_solver(solver)
solver.pop()




