"""
Test whether the three unstack loop theorems are provable directly by Z3
using the uninterpreted Box sort (as in test_unstack_exists_top.py) combined
with the all-universal predicate formulation from test_unstack_induction.py.

No integer indexing, no is_box guards, no induction, no IH.
"""

import sys
from z3 import *

# ---------------------------------------------------------------------------
# Sort and variables
# ---------------------------------------------------------------------------
Box = DeclareSort("Box")
x, y, z, t, nb, w = Consts("x y z t nb w", Box)
n0 = Const("n0", Box)


# ---------------------------------------------------------------------------
# Axioms  (same as test_unstack_exists_top.py, minus the 6th "next" axiom)
# ---------------------------------------------------------------------------

def add_axioms(solver, ON):
    solver.add(ForAll([x], ON(x, x)))
    solver.add(ForAll([x, y, z],
        Implies(And(ON(x, y), ON(y, z)), ON(x, z))))
    solver.add(ForAll([x, y],
        Implies(And(ON(x, y), ON(y, x)), x == y)))
    solver.add(ForAll([x, y, z],
        Implies(And(ON(x, y), ON(x, z)), Or(ON(y, z), ON(z, y)))))
    solver.add(ForAll([x, y, z],
        Implies(And(ON(x, z), ON(y, z)), Or(ON(x, y), ON(y, x)))))


# ---------------------------------------------------------------------------
# All-universal predicates  (no Exists anywhere)
# ---------------------------------------------------------------------------

def ON_subst(a, b, nb_val, ON):
    """ON'(a,b) = ON(a,b) ∧ (¬ON(a,nb) ∨ ON(b,nb))"""
    return And(ON(a, b), Or(Not(ON(a, nb_val)), ON(b, nb_val)))


def loop_invariant(ON, ON_zero, n0v):
    """∀t w. ON_zero(t,n0) → (top(t) ∧ on_table(t)) ∨ ON(t,n0)
       top/on_table written as ∀w. w≠t → ¬ON(w,t) / ¬ON(t,w)"""
    return ForAll([t, w],
        Implies(ON_zero(t, n0v),
                Or(And(Implies(w != t, Not(ON(w, t))),
                       Implies(w != t, Not(ON(t, w)))),
                   ON(t, n0v))))


def loop_invariant_subst(nb_val, ON, ON_zero, n0v):
    """INV after removing nb_val."""
    return ForAll([t, w],
        Implies(ON_zero(t, n0v),
                Or(And(Implies(w != t, Not(ON_subst(w, t, nb_val, ON))),
                       Implies(w != t, Not(ON_subst(t, w, nb_val, ON)))),
                   ON_subst(t, n0v, nb_val, ON))))


def while_cond_instance(nb_val, ON, n0v):
    """nb_val is a top above n0: ON(nb,n0) ∧ nb≠n0 ∧ ∀w. w≠nb → ¬ON(w,nb)"""
    return And(ON(nb_val, n0v),
               nb_val != n0v,
               ForAll([w], Implies(w != nb_val, Not(ON(w, nb_val)))))


def not_while_cond(ON, n0v):
    """¬while: n0 is a top — nothing strictly above it.
       Equivalent to ¬(∃x. top(x) ∧ ON(x,n0) ∧ x≠n0) when the loop exits
       because the only remaining top above n0 is n0 itself."""
    return ForAll([w], Implies(w != n0v, Not(ON(w, n0v))))


def precondition(ON, ON_zero):
    return ForAll([x, y], ON(x, y) == ON_zero(x, y))


def postcondition(ON, ON_zero, n0v):
    """∀t w. ON_zero(t,n0) ∧ t≠n0 ∧ w≠t → ¬ON(w,t) ∧ ¬ON(t,w)"""
    return ForAll([t, w],
        Implies(And(ON_zero(t, n0v), t != n0v, w != t),
                And(Not(ON(w, t)), Not(ON(t, w)))))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def check_direct(label, solver):
    sys.stdout.flush()
    result = solver.check()
    if result == unsat:
        print(f"  {label}: PROVED")
    elif result == sat:
        print(f"  {label}: FAILED — counterexample exists")
        print(f"  {solver.model()}")
    else:
        print(f"  {label}: UNKNOWN — Z3 cannot decide")
    sys.stdout.flush()
    return result


# ---------------------------------------------------------------------------
# Direct check with Box sort — no induction, no integer indexing
# ---------------------------------------------------------------------------
ON  = Function("ON",  Box, Box, BoolSort())
ONz = Function("ONz", Box, Box, BoolSort())

solver = Solver()
add_axioms(solver, ON)
add_axioms(solver, ONz)

print("=" * 60)
print("DIRECT CHECK: Box sort, no induction, no integer indexing")
print("(all-∀ predicates, uninterpreted Box sort)")
print("=" * 60)
print()

# T1: pre → INV
print("T1 (pre → INV):")
solver.push()
solver.add(precondition(ON, ONz))
solver.add(Not(loop_invariant(ON, ONz, n0)))
check_direct("pre → INV", solver)
solver.pop()

# T2: INV ∧ while_inst → INV'
print("T2 (INV ∧ while → INV'):")
solver.push()
solver.add(loop_invariant(ON, ONz, n0))
solver.add(while_cond_instance(nb, ON, n0))
solver.add(Not(loop_invariant_subst(nb, ON, ONz, n0)))
check_direct("INV ∧ while → INV'", solver)
solver.pop()

# T3: INV ∧ ¬while → post
print("T3 (INV ∧ ¬while → post):")
solver.push()
solver.add(loop_invariant(ON, ONz, n0))
solver.add(not_while_cond(ON, n0))
solver.add(Not(postcondition(ON, ONz, n0)))
check_direct("INV ∧ ¬while → post", solver)
solver.pop()
