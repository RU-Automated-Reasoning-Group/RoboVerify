from z3 import *

# Create a solver
solver = Solver()

# Declare a sort for boxes (number of boxes is unspecified)
Box = DeclareSort('Box')

# Declare the uninterpreted function "ON"
ON = Function('ON', Box, Box, BoolSort())  # ON(x, y) means x is on y

# Declare arbitrary boxes
a, b, b0, b_prime, c = Consts('a b b0 b_prime c', Box)

# Define "top(x)" as a function: There's no box above x
def top(x):
    return Not(Exists([c], ON(c, x)))  # No box c is on top of x

# Define ON_star (transitive closure of ON)
ON_star = RecFunction('ON_star', Box, Box, BoolSort())

# Base case: Reflexivity (every box is related to itself)
solver.add(ForAll([a], ON_star(a, a)))

# Recursive case: If ON(a, c) and ON_star(c, b), then ON_star(a, b)
solver.add(ForAll([a, b], Implies(Exists([c], And(ON(a, c), ON_star(c, b))), ON_star(a, b))))

# **Precondition:**
precondition = And(
    ForAll([a], Or(ON_star(a, b0), top(a))),  # Every box is either connected to b0 or has no box on it
    ON_star(b, b0),  # The current top box b is connected to b0
    top(b)  # The current top box b has no box on it
)
solver.add(precondition)

# Define P_aux as a recursive function
P_aux = RecFunction('P_aux', Box, BoolSort())

P_aux = RecFunction('P_aux', Box, BoolSort())

# Base case: If there is no `b'` with `top(b')` and `b' != b`, do nothing (implicitly terminating)
solver.add(Implies(Not(Exists([b_prime], And(top(b_prime), b_prime != b))), P_aux(b)))

# Recursive case: If there exists a `b'` such that `top(b')` and `b' != b`,
# then `ON(b', b)` is set to `True`, and `P_aux(b')` is called recursively.
solver.add(Implies(
    Exists([b_prime], And(top(b_prime), b_prime != b)),  # There exists a free b' ≠ b
    And(ON(b_prime, b), P_aux(b_prime))  # Place b' on b and recurse
))

# **Postcondition: After executing P_aux(b), ON_star(a, b0) must hold for all a**
postcondition = ForAll([a], ON_star(a, b0))
solver.add(postcondition)

# Check satisfiability
if solver.check() == sat:
    print("The function P_aux satisfies the precondition and postcondition!")
    print(solver.model())  # Print a model satisfying the constraints
else:
    print("The function P_aux does not satisfy the conditions!")
