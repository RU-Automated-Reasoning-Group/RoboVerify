"""
Prove by explicit induction in Z3:
  ∀b. ∃x. is_box(x) ∧ top(x) ∧ ON(x, b)

  i.e., in any partial order on n boxes {0,...,n-1} satisfying the chain
  property, EVERY box has a top element above it.

Uses the same top() and goal definitions as test_stack_induction.py.
"""

from z3 import *

x, y, z, b0, b, t, n = Ints('x y z b0 b t n')

def is_box(v, size):
    return And(v >= 0, v < size)

def top(v, ON, size):
    """v is a top box: no other valid box sits strictly above v."""
    w = Int('w')
    return And(is_box(v, size),
               Not(Exists([w], And(is_box(w, size), w != v, ON(w, v)))))

def goal_for(b, ON, size):
    """There exists a top element above the specific box b."""
    return Exists([x], And(is_box(x, size), top(x, ON, size), ON(x, b)))

def goal_forall(ON, size):
    """∀b. ∃x. top(x) ∧ ON(x, b)  — every box has a top above it."""
    return ForAll([b], Implies(is_box(b, size), goal_for(b, ON, size)))

def add_axioms(solver, ON, size):
    solver.add(ForAll([x], Implies(is_box(x, size), ON(x, x))))
    solver.add(ForAll([x, y, z],
        Implies(And(is_box(x, size), is_box(y, size), is_box(z, size),
                    ON(x, y), ON(y, z)),
                ON(x, z))))
    solver.add(ForAll([x, y],
        Implies(And(is_box(x, size), is_box(y, size), ON(x, y), ON(y, x)),
                x == y)))
    solver.add(ForAll([x, y, z],
        Implies(And(is_box(x, size), is_box(y, size), is_box(z, size),
                    ON(x, y), ON(x, z)),
                Or(ON(y, z), ON(z, y)))))
    solver.add(ForAll([x, y, z],
        Implies(And(is_box(x, size), is_box(y, size), is_box(z, size),
                    ON(x, z), ON(y, z)),
                Or(ON(x, y), ON(y, x)))))

# ===========================================================================
# BASE CASE: n = 1
# ===========================================================================
print("=" * 60)
print("BASE CASE: n = 1")
print("=" * 60)

ON1 = Function('ON1', IntSort(), IntSort(), BoolSort())
s_base = Solver()
add_axioms(s_base, ON1, 1)
# Negate the universal goal: ∃b. is_box(b) ∧ ¬∃x. top(x) ∧ ON1(x, b)
s_base.add(Not(goal_forall(ON1, 1)))

result_base = s_base.check()
print(f"Result: {result_base}")
if result_base == unsat:
    print("Base case PROVED.")
else:
    print("Base case FAILED.")
    print(s_base.model())
print()

# ===========================================================================
# INDUCTIVE STEP: assume P(n), prove P(n+1)
# ===========================================================================
print("=" * 60)
print("INDUCTIVE STEP: assume P(n), prove P(n+1)")
print("=" * 60)
print()

ON_s = Function('ON_s', IntSort(), IntSort(), BoolSort())  # order on {0,...,n-1}
ON_l = Function('ON_l', IntSort(), IntSort(), BoolSort())  # order on {0,...,n}

# Skolem function: the witness top above each box in the small order
wit_s = Function('wit_s', IntSort(), IntSort())

s_step = Solver()
s_step.add(n > 0)

add_axioms(s_step, ON_s, n)
add_axioms(s_step, ON_l, n + 1)

# Restriction: ON_l and ON_s agree on the shared domain {0,...,n-1}
s_step.add(ForAll([x, y],
    Implies(And(is_box(x, n), is_box(y, n)),
            ON_s(x, y) == ON_l(x, y))))

# Inductive hypothesis (Skolemized):
# wit_s(b) is a top above b in ON_s, for every b in {0,...,n-1}
w = Int('w')
s_step.add(ForAll([b0],
    Implies(is_box(b0, n), And(
        is_box(wit_s(b0), n),
        ON_s(wit_s(b0), b0),
        ForAll([w], Implies(
            And(is_box(w, n), ON_s(w, wit_s(b0))),
            w == wit_s(b0)))
    ))))

# Negation of the universal goal for the large order:
# ∃b. is_box(b, n+1) ∧ ¬∃x. top(x, ON_l) ∧ ON_l(x, b)
s_step.add(Not(goal_forall(ON_l, n + 1)))

print("Running Z3 on the inductive step...")
result_step = s_step.check()
print(f"Result: {result_step}")
if result_step == unsat:
    print("Inductive step PROVED.")
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("By induction on n:")
    print("  Base case (n=1):          PROVED")
    print("  Inductive step (n→n+1):   PROVED")
    print()
    print("THEOREM: In any finite partial order on n>0 boxes satisfying")
    print("the chain/stack property, every box has a top element above it.")
elif result_step == sat:
    print("Inductive step FAILED — counterexample found:")
    print(s_step.model())
else:
    print(f"Inductive step UNKNOWN: {result_step}")
