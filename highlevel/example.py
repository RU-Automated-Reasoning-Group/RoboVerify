from z3 import *

# Declare a sort for boxes (number of boxes is unspecified)
Box = DeclareSort('Box')

# Declare the uninterpreted function "ON"
ON = Function('ON', Box, Box, BoolSort())  # ON(x, y) means x is on y

# Declare an arbitrary box variable
x, y, c, b0 = Consts('x y c b0', Box)

# Define "top(x)" as a function: There's no box above x
def top(x):
    return Not(Exists([y], ON(y, x)))  # No box y is on top of x

# **Precondition: Every box has no box on top of it**
precondition = ForAll([x], top(x))

# Define ON_star (transitive closure of ON)
ON_star = RecFunction('ON_star', Box, Box, BoolSort())

# Base case: Reflexivity
base_case = ForAll([x], ON_star(x, x))

# Recursive case: If ON(x, c) and ON_star(c, y), then ON_star(x, y)
rec_case = ForAll([x, y], Implies(Exists([c], And(ON(x, c), ON_star(c, y))), ON_star(x, y)))

# **Postcondition: Ensure every box is connected to b0 via ON_star**
postcondition = ForAll([x], ON_star(x, b0))

# **Recursive program simulating stacking process**
def recursive_program(x, max_depth=10):
    if max_depth <= 0:
        return True  # Stop recursion after reaching depth limit
    y = Const('y', Box)
    return Exists([y], And(ON(x, y), recursive_program(y, max_depth - 1)))

# Create a solver
solver = Solver()

# Add the precondition
solver.add(precondition)

# Add recursive stacking logic (for an arbitrary number of boxes)
solver.add(recursive_program(Const('x0', Box)))  # Start from an arbitrary box x0

# Add ON_star rules (transitive closure)
solver.add(base_case)
solver.add(rec_case)

# Add the postcondition: Ensure every box is related to b0 via ON_star
solver.add(postcondition)

# Check satisfiability
if solver.check() == sat:
    print("The program satisfies the precondition and postcondition!")
    print(solver.model())  # Print a model satisfying the constraints
else:
    print("The program does not satisfy the conditions!")
