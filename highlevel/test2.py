from z3 import *

# >>> ctx = Context()
# >>> fac = RecFunction('fac', IntSort(ctx), IntSort(ctx))
# >>> n = Int('n', ctx)
# >>> RecAddDefinition(fac, n, If(n == 0, 1, n*fac(n-1)))
# >>> simplify(fac(5))

# Define recursive function "factorial"
factorial = Function('factorial', IntSort(), IntSort())

# Create the solver
solver = Solver()

# Define the base case for the recursive factorial function
solver.add(factorial(0) == 1)  # factorial(0) = 1
x = Int('x')
# Define the recursive case: factorial(n) = n * factorial(n-1)
solver.add(ForAll([x], Implies(x > 0, factorial(x) == x * factorial(x - 1))))
# solver.add(RecAddDefinition(factorial, [IntSort()], factorial(Int('n')) == Int('n') * factorial(Int('n') - 1)))
# solver.add(RecAddDefinition(factorial, [n], factorial(n) == n * factorial(n - 1)))
# Now, let's assert a property we want to prove about the factorial function
# For example, let's check if factorial(3) equals 6
solver.push()
solver.add(factorial(3) != 6)

if solver.check() == unsat:
    print("Proven: factorial(3) equals 6")
else:
    print("Not Proven: There's a counterexample")

solver.pop()