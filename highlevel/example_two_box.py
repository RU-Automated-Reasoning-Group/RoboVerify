from z3 import *

# Create solver
solver = Solver()

# Declare sort for boxes
Box = DeclareSort('Box')

# Declare the uninterpreted function "ON"
ON = Function('ON', Box, Box, BoolSort())

# Define boxes
b1, b2, c = Consts('b1 b2 c', Box)
b0 = Const('b0', Box)  # Top box

# Define ON_star (transitive closure)
ON_star = Function('ON_star', Box, Box, BoolSort())

# Properly define ON_star
BaseCase = ForAll([b1], ON_star(b1, b1))  # Reflexivity
solver.add(BaseCase)

# Explicitly add ON_star(b0, b0) (which follows from reflexivity)
solver.add(ON_star(b0, b0))

# Check satisfiability
if solver.check() == sat:
    print("SAT - The constraints are consistent!")
    print(solver.model())
else:
    print("UNSAT - The constraints are contradictory!")
