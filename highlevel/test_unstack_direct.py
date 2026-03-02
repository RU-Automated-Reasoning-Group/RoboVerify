"""
Test whether the three unstack loop theorems are provable DIRECTLY by Z3
(no induction, no IH) using the all-universal predicate formulation from
test_unstack_induction.py.

A single symbolic n represents the number of boxes.  If Z3 can prove all
three theorems here, induction is not needed — the all-∀ formulation is
already in the decidable fragment for arbitrary n.

If any theorem is UNKNOWN or times out, it means the all-∀ formulation alone
is not enough and induction (as in test_unstack_induction.py) is still
required.
"""

import sys
from z3 import *

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
x, y, z, t, n, nb, w = Ints("x y z t n nb w")
n0 = Int("n0")


# ---------------------------------------------------------------------------
# Axioms
# ---------------------------------------------------------------------------

def is_box(v, sz):
    return And(v >= 0, v < sz)


def add_axioms(solver, ON, sz):
    solver.add(ForAll([x], Implies(is_box(x, sz), ON(x, x))))
    solver.add(ForAll([x, y, z],
        Implies(And(is_box(x,sz), is_box(y,sz), is_box(z,sz), ON(x,y), ON(y,z)),
                ON(x, z))))
    solver.add(ForAll([x, y],
        Implies(And(is_box(x,sz), is_box(y,sz), ON(x,y), ON(y,x)),
                x == y)))
    solver.add(ForAll([x, y, z],
        Implies(And(is_box(x,sz), is_box(y,sz), is_box(z,sz), ON(x,y), ON(x,z)),
                Or(ON(y,z), ON(z,y)))))
    solver.add(ForAll([x, y, z],
        Implies(And(is_box(x,sz), is_box(y,sz), is_box(z,sz), ON(x,z), ON(y,z)),
                Or(ON(x,y), ON(y,x)))))


# ---------------------------------------------------------------------------
# All-universal predicates  (no Exists anywhere)
# ---------------------------------------------------------------------------

def ON_subst(a, b, nb_val, ON):
    return And(ON(a, b), Or(Not(ON(a, nb_val)), ON(b, nb_val)))


def loop_invariant(ON, ON_zero, sz, n0v):
    return ForAll([t, w],
        Implies(And(is_box(t, sz), is_box(w, sz), ON_zero(t, n0v)),
                Or(And(Implies(w != t, Not(ON(w, t))),
                       Implies(w != t, Not(ON(t, w)))),
                   ON(t, n0v))))


def loop_invariant_subst(nb_val, ON, ON_zero, sz, n0v):
    return ForAll([t, w],
        Implies(And(is_box(t, sz), is_box(w, sz), ON_zero(t, n0v)),
                Or(And(Implies(w != t, Not(ON_subst(w, t, nb_val, ON))),
                       Implies(w != t, Not(ON_subst(t, w, nb_val, ON)))),
                   ON_subst(t, n0v, nb_val, ON))))


def while_cond_instance(nb_val, ON, sz, n0v):
    return And(is_box(nb_val, sz),
               ON(nb_val, n0v),
               nb_val != n0v,
               ForAll([w], Implies(And(is_box(w, sz), w != nb_val),
                                   Not(ON(w, nb_val)))))


def not_while_cond(ON, sz, n0v):
    return ForAll([x, w],
        Implies(And(is_box(x, sz), is_box(w, sz),
                    ON(x, n0v), x != n0v, w != x),
                ON(w, x)))


def precondition(ON, ON_zero, sz, n0v):
    return ForAll([x, y],
        Implies(And(is_box(x, sz), is_box(y, sz)),
                ON(x, y) == ON_zero(x, y)))


def postcondition(ON, ON_zero, sz, n0v):
    return ForAll([t, w],
        Implies(And(is_box(t, sz), is_box(w, sz),
                    ON_zero(t, n0v), t != n0v, w != t),
                And(Not(ON(w, t)), Not(ON(t, w)))))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def check_direct(label, solver):
    sys.stdout.flush()
    result = solver.check()
    if result == unsat:
        print(f"  {label}: PROVED (no induction needed)")
    elif result == sat:
        print(f"  {label}: FAILED — counterexample exists")
        print(f"  {solver.model()}")
    else:
        print(f"  {label}: UNKNOWN — Z3 cannot decide without induction")
    sys.stdout.flush()
    return result


# ---------------------------------------------------------------------------
# Single symbolic n, two ON functions, no IH
# ---------------------------------------------------------------------------
ON  = Function("ON",  IntSort(), IntSort(), BoolSort())
ONz = Function("ONz", IntSort(), IntSort(), BoolSort())

solver = Solver()
solver.add(n > 0)
add_axioms(solver, ON,  n)
add_axioms(solver, ONz, n)
solver.add(is_box(n0, n), is_box(nb, n))

print("=" * 60)
print("DIRECT CHECK: no induction, symbolic n")
print("(same all-∀ predicates as test_unstack_induction.py)")
print("=" * 60)
print()

# T1: pre → INV
print("T1 (pre → INV):")
solver.push()
solver.add(precondition(ON, ONz, n, n0))
solver.add(Not(loop_invariant(ON, ONz, n, n0)))
check_direct("pre → INV", solver)
solver.pop()

# T2: INV ∧ while_inst → INV'
print("T2 (INV ∧ while → INV'):")
solver.push()
solver.add(loop_invariant(ON, ONz, n, n0))
solver.add(while_cond_instance(nb, ON, n, n0))
solver.add(Not(loop_invariant_subst(nb, ON, ONz, n, n0)))
check_direct("INV ∧ while → INV'", solver)
solver.pop()

# T3: INV ∧ ¬while → post
print("T3 (INV ∧ ¬while → post):")
solver.push()
solver.add(loop_invariant(ON, ONz, n, n0))
solver.add(not_while_cond(ON, n, n0))
solver.add(Not(postcondition(ON, ONz, n, n0)))
check_direct("INV ∧ ¬while → post", solver)
solver.pop()
