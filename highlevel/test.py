from z3 import *

# Create solver
solver = Solver()

# Declare sort for boxes
Box = DeclareSort('Box')
b0, b1, b2, b3, c = Consts('b0 b1 b2 b3 c', Box)

# Declare the uninterpreted function "ON"
ON = Function('ON', Box, Box, BoolSort())

ON_star = Function("ON_star", Box, Box, BoolSort())
solver.add(ForAll([b0], ON_star(b0, b0)))
solver.add(ForAll([b0, b1], Implies(Exists([c], And(ON(b0, c), ON_star(c, b1))) , ON_star(b0, b1))))

def top(x):
    return Not(Exists([c], ON(c, x)))

def on_or_top(b):
    return Or(ON_star(b, b0), top(b))

# precondition
solver.add([on_or_top(b) for b in [b0, b1, b2]])
solver.add(ON_star(b1, b0))
# solver.add(top(b1))

# put b2 on b2
solver.add(ON(b2, b1))

# postcondition
postcondition = And([ON_star(b, b0) for b in [b0, b1, b2]])
solver.add(Not(postcondition))
# solver.add(postcondition)

# precondition = ForAll([b0], top(b0))
# solver.add(precondition)
# solver.add(ON(b1, b0))
# solver.add(ON(b2, b1))
# solver.add(ON(b3, b2))


# postcondition = ForAll([c], ON_star(c, b0))
# postcondition = And(ON_star(b0, b0), ON_star(b1, b0), ON_star(b2, b0), ON_star(b3, b0))
# solver.add(Not(postcondition))

# Check satisfiability
if solver.check() == sat:
    print("SAT - The constraints are consistent!")
    print(solver.model())
    import pdb; pdb.set_trace()

else:
    print("UNSAT - The constraints are contradictory!")

