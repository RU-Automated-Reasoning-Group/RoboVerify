from z3 import *

# Create a solver
solver = Solver()

# Declare the recursive function S(k) to calculate the sum of first k natural numbers
k = Int('k')
S = Function('S', IntSort(), IntSort())  # S(k) returns the sum of first k natural numbers

# Base case: S(0) = 0
# solver.add(k >= 0)
solver.add(S(0) == 0)

# Recursive step: S(k) = S(k-1) + k, for k > 0
solver.add(ForAll([k], Implies(k > 0, S(k) == S(k-1) + k)))


# Now, we want to prove that S(k) = k * (k + 1) / 2 for all k
solver.add(ForAll(k, 2 * S(k) == (k * (k + 1))))
# solver.add(Not(ForAll(k, 2 * S(k) == (k * (k + 1)))))

# Check if the recursive function satisfies the formula
if solver.check() == sat:
    model = solver.model()
    print("Inductive proof holds:", model)
else:
    print("Inductive proof does not hold")
