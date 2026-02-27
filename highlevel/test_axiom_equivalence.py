from z3 import *

# Declare an uninterpreted sort for boxes
BoxSort = DeclareSort('BoxSort')

# Declare ON_star as a predicate: BoxSort × BoxSort → Bool
ON_star = Function('ON_star', BoxSort, BoxSort, BoolSort())

# Declare universally quantified variables
x, y = Consts('x y', BoxSort)

# Formula 1:
# ∀x,y: ON_star(x,y) → (ON_star(y,x) → x = y)
F1 = ForAll([x, y],
            Implies(ON_star(x, y),
                    Implies(ON_star(y, x),
                            x == y)))

# Formula 2:
# ∀x,y: (ON_star(x,y) ∧ ON_star(y,x)) → x = y
F2 = ForAll([x, y],
            Implies(And(ON_star(x, y),
                        ON_star(y, x)),
                    x == y))

# Check equivalence by checking satisfiability of their difference
s = Solver()
# s.add(F1)
# s.add(Not(F2))
s.add(F2)
s.add(Not(F1))

result = s.check()

print("Satisfiability of (F1 != F2):", result)

if result == unsat:
    print("✔ The formulas are logically equivalent.")
else:
    print("✘ The formulas are NOT equivalent.")
    print("Countermodel:")
    print(s.model())