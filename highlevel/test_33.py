from z3 import *

solver = Solver()

Box = DeclareSort('Box')
# Box, (b0, b1, b2) = EnumSort('Box', ['b0', 'b1', 'b2'])
x, y, c, a, b_prime, b, b0 = Consts("x y c a b_prime b b0", Box)
ON_star = Function('ON_star', Box, Box, BoolSort())
solver.add(ForAll([x], ON_star(x, x)))
# solver.add(ForAll([x, y], Implies(x != y, Not(And( ON(x, y),  ON(y, x) )))))
# solver.add(ForAll([x, y, c], Implies(And(ON(x, c), ON(y, c), x != c, y != c), x == y)))
# ON_star = TransitiveClosure(ON)

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
# solver.add(precondition())
# solver.add(Not(loop_invariant(b0)))



# while loop correctness: verify while condition (true) + loop invariant implies another loop invariant
# solver.add(while_cond_instantized(b_prime))
solver.add(while_cond_instantized(b_prime))
solver.add(loop_invariant(b))
solver.add(Not(substituted_loop_invariant(b_prime, b_prime, b)))
# solver.add(ON_star(b_prime, b))
# solver.add(Not(loop_invariant(b_prime)))


# postcondition correctness: verify while condition (false) + loop invariant implies postcondition
# solver.add(loop_invariant(b))
# solver.add(Not(while_cond(b_prime)))
# solver.add(Not(postcondition()))

if solver.check() == sat:
    print("constraints satisfiable")
    print("model is")
    print(solver.model())
    import pdb; pdb.set_trace()
else:
    print("NOT satisfiable")


