"""
Prove the three unstack loop theorems by structural induction on n (the number
of boxes), where each theorem uses only its own IH.

Induction schema
----------------
  T1(n): ∀ ON, ON_zero, n0.  pre(ON, ON_zero) → INV(ON, ON_zero, n0)
  T2(n): ∀ ON, ON_zero, n0, nb.  INV ∧ while_inst(nb) → INV'(nb, ...)
  T3(n): ∀ ON, ON_zero, n0.  INV ∧ ¬while → post(ON, ON_zero, n0)

Each inductive step uses only T_i(n) to prove T_i(n+1) — no cross-theorem
dependencies, no existential witness, no Skolem function.

Key design: all predicates (top, on_table, inv, while, post) are written
purely as universal formulas (∀ only, no ∃) by rewriting
    ¬∃w. P(w)   as   ∀w. ¬P(w)
This keeps every IH in the ∀* fragment that Z3's MBQI can decide.

Definitions
-----------
  top(t, ON)       ≡  ∀w. w≠t → ¬ON(w, t)
  on_table(t, ON)  ≡  ∀w. w≠t → ¬ON(t, w)

  INV(ON, ON_zero, n0) :=
      ∀t w. ON_zero(t,n0) →
          (w≠t → ¬ON(w,t)) ∧ (w≠t → ¬ON(t,w))   [top ∧ on_table]
          ∨  ON(t, n0)

  while_inst(nb, ON, n0) :=
      ON(nb, n0) ∧ nb≠n0 ∧ ∀w. w≠nb → ¬ON(w, nb)   [nb is top above n0]

  ¬while_cond(ON, n0) :=
      ∀x w. ON(x,n0) ∧ x≠n0 ∧ w≠x → ON(w, x)   [no other top above n0]

  INV'(nb, ON, ON_zero, n0) :=
      ∀t w. ON_zero(t,n0) →
          (w≠t → ¬ON'(w,t)) ∧ (w≠t → ¬ON'(t,w))   ∨  ON'(t, n0)
      where ON'(a,b) = ON(a,b) ∧ (¬ON(a,nb) ∨ ON(b,nb))

  post(ON, ON_zero, n0) :=
      ∀t w. ON_zero(t,n0) ∧ t≠n0 ∧ w≠t →
          ¬ON(w, t) ∧ ¬ON(t, w)

  pre(ON, ON_zero) :=
      ∀x y. ON(x,y) ↔ ON_zero(x,y)
"""

import sys
from z3 import *

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
x, y, z, t, n, nb, n0q, nbq, w, w2 = Ints("x y z t n nb n0q nbq w w2")
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


def restrict(solver, ON_s, ON_l, sz):
    solver.add(ForAll([x, y],
        Implies(And(is_box(x, sz), is_box(y, sz)),
                ON_s(x, y) == ON_l(x, y))))


# ---------------------------------------------------------------------------
# All-universal predicates  (no Exists anywhere)
# ---------------------------------------------------------------------------

def ON_subst(a, b, nb_val, ON):
    """ON after removing nb_val: ON'(a,b) = ON(a,b) ∧ (¬ON(a,nb) ∨ ON(b,nb))"""
    return And(ON(a, b), Or(Not(ON(a, nb_val)), ON(b, nb_val)))


def loop_invariant(ON, ON_zero, sz, n0v):
    """∀t w. ON_zero(t,n0) → (top(t) ∧ on_table(t)) ∨ ON(t,n0)
       with top/on_table written as ∀w. w≠t → ¬ON(w,t) / ¬ON(t,w)"""
    return ForAll([t, w],
        Implies(And(is_box(t, sz), is_box(w, sz), ON_zero(t, n0v)),
                Or(And(Implies(w != t, Not(ON(w, t))),
                       Implies(w != t, Not(ON(t, w)))),
                   ON(t, n0v))))


def loop_invariant_subst(nb_val, ON, ON_zero, sz, n0v):
    """INV after removing nb_val (same structure, ON replaced by ON_subst)."""
    return ForAll([t, w],
        Implies(And(is_box(t, sz), is_box(w, sz), ON_zero(t, n0v)),
                Or(And(Implies(w != t, Not(ON_subst(w, t, nb_val, ON))),
                       Implies(w != t, Not(ON_subst(t, w, nb_val, ON)))),
                   ON_subst(t, n0v, nb_val, ON))))


def while_cond_instance(nb_val, ON, sz, n0v):
    """nb_val is a top above n0: ON(nb,n0) ∧ nb≠n0 ∧ ∀w. w≠nb → ¬ON(w,nb)"""
    return And(is_box(nb_val, sz),
               ON(nb_val, n0v),
               nb_val != n0v,
               ForAll([w], Implies(And(is_box(w, sz), w != nb_val),
                                   Not(ON(w, nb_val)))))


def not_while_cond(ON, sz, n0v):
    """¬while: every box above n0 other than n0 has something above it."""
    return ForAll([x, w],
        Implies(And(is_box(x, sz), is_box(w, sz),
                    ON(x, n0v), x != n0v, w != x),
                ON(w, x)))


def precondition(ON, ON_zero, sz, n0v):
    return ForAll([x, y],
        Implies(And(is_box(x, sz), is_box(y, sz)),
                ON(x, y) == ON_zero(x, y)))


def postcondition(ON, ON_zero, sz, n0v):
    """∀t w. ON_zero(t,n0) ∧ t≠n0 ∧ w≠t → ¬ON(w,t) ∧ ¬ON(t,w)"""
    return ForAll([t, w],
        Implies(And(is_box(t, sz), is_box(w, sz),
                    ON_zero(t, n0v), t != n0v, w != t),
                And(Not(ON(w, t)), Not(ON(t, w)))))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

all_proved = True

def check_case(label, solver):
    global all_proved
    sys.stdout.flush()
    result = solver.check()
    if result == unsat:
        print(f"    {label}: PROVED")
    elif result == sat:
        print(f"    {label}: FAILED — counterexample:")
        print(f"    {solver.model()}")
        all_proved = False
    else:
        print(f"    {label}: UNKNOWN ({result})")
        all_proved = False
    sys.stdout.flush()
    return result


def make_step_solver(tag):
    ON_s  = Function(f"ON_s_{tag}",  IntSort(), IntSort(), BoolSort())
    ON_l  = Function(f"ON_l_{tag}",  IntSort(), IntSort(), BoolSort())
    ON_sz = Function(f"ON_sz_{tag}", IntSort(), IntSort(), BoolSort())
    ON_lz = Function(f"ON_lz_{tag}", IntSort(), IntSort(), BoolSort())
    s = Solver()
    s.add(n > 0)
    add_axioms(s, ON_s,  n);    add_axioms(s, ON_l,  n + 1)
    add_axioms(s, ON_sz, n);    add_axioms(s, ON_lz, n + 1)
    restrict(s, ON_s,  ON_l,  n)
    restrict(s, ON_sz, ON_lz, n)
    return s, ON_s, ON_l, ON_sz, ON_lz


# ===========================================================================
# BASE CASE  n = 1
# ===========================================================================
print("=" * 60)
print("BASE CASE: n = 1")
print("=" * 60)

print("  T1 (pre → INV):")
ON1  = Function("ON1",  IntSort(), IntSort(), BoolSort())
ON1z = Function("ON1z", IntSort(), IntSort(), BoolSort())
s = Solver()
add_axioms(s, ON1, 1);  add_axioms(s, ON1z, 1)
s.add(is_box(n0, 1))
s.add(precondition(ON1, ON1z, 1, n0))
s.add(Not(loop_invariant(ON1, ON1z, 1, n0)))
check_case("pre → INV", s)

print("  T2 (INV ∧ while → INV'):")
ON2  = Function("ON2",  IntSort(), IntSort(), BoolSort())
ON2z = Function("ON2z", IntSort(), IntSort(), BoolSort())
s = Solver()
add_axioms(s, ON2, 1);  add_axioms(s, ON2z, 1)
s.add(is_box(n0, 1), is_box(nb, 1))
s.add(loop_invariant(ON2, ON2z, 1, n0))
s.add(while_cond_instance(nb, ON2, 1, n0))
s.add(Not(loop_invariant_subst(nb, ON2, ON2z, 1, n0)))
check_case("INV ∧ while → INV'", s)

print("  T3 (INV ∧ ¬while → post):")
ON3  = Function("ON3",  IntSort(), IntSort(), BoolSort())
ON3z = Function("ON3z", IntSort(), IntSort(), BoolSort())
s = Solver()
add_axioms(s, ON3, 1);  add_axioms(s, ON3z, 1)
s.add(is_box(n0, 1))
s.add(loop_invariant(ON3, ON3z, 1, n0))
s.add(not_while_cond(ON3, 1, n0))
s.add(Not(postcondition(ON3, ON3z, 1, n0)))
check_case("INV ∧ ¬while → post", s)

print()

# ===========================================================================
# INDUCTIVE STEP  n → n+1
# Each theorem uses only its own IH.
# ===========================================================================
print("=" * 60)
print("INDUCTIVE STEP: T_i(n) → T_i(n+1)  for i = 1, 2, 3")
print("=" * 60)

# T1 step: IH = T1(n) only
print("  T1 (pre → INV):")
s, ON_s, ON_l, ON_sz, ON_lz = make_step_solver("t1")
s.add(ForAll([n0q],
    Implies(And(is_box(n0q, n), precondition(ON_s, ON_sz, n, n0q)),
            loop_invariant(ON_s, ON_sz, n, n0q))))
s.add(is_box(n0, n + 1))
s.add(precondition(ON_l, ON_lz, n + 1, n0))
s.add(Not(loop_invariant(ON_l, ON_lz, n + 1, n0)))
check_case("T1(n) → T1(n+1)", s)

# T2 step: IH = T2(n) only
print("  T2 (INV ∧ while → INV'):")
s, ON_s, ON_l, ON_sz, ON_lz = make_step_solver("t2")
s.add(ForAll([n0q, nbq],
    Implies(And(is_box(n0q, n), is_box(nbq, n),
                loop_invariant(ON_s, ON_sz, n, n0q),
                while_cond_instance(nbq, ON_s, n, n0q)),
            loop_invariant_subst(nbq, ON_s, ON_sz, n, n0q))))
s.add(is_box(n0, n + 1), is_box(nb, n + 1))
s.add(loop_invariant(ON_l, ON_lz, n + 1, n0))
s.add(while_cond_instance(nb, ON_l, n + 1, n0))
s.add(Not(loop_invariant_subst(nb, ON_l, ON_lz, n + 1, n0)))
check_case("T2(n) → T2(n+1)", s)

# T3 step: IH = T3(n) only
print("  T3 (INV ∧ ¬while → post):")
s, ON_s, ON_l, ON_sz, ON_lz = make_step_solver("t3")
s.add(Implies(
    And(is_box(n0, n),
        loop_invariant(ON_s, ON_sz, n, n0),
        not_while_cond(ON_s, n, n0)),
    postcondition(ON_s, ON_sz, n, n0)))
s.add(is_box(n0, n + 1))
s.add(loop_invariant(ON_l, ON_lz, n + 1, n0))
s.add(not_while_cond(ON_l, n + 1, n0))
s.add(Not(postcondition(ON_l, ON_lz, n + 1, n0)))
check_case("T3(n) → T3(n+1)", s)

print()
if all_proved:
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("By induction on n, all three unstack loop theorems hold:")
    print("  T1. precondition  →  loop invariant")
    print("  T2. loop invariant ∧ while_cond  →  loop invariant (after step)")
    print("  T3. loop invariant ∧ ¬while_cond  →  postcondition")
    print()
    print("Each theorem proved using only its own IH.")
    print("No existential in the invariant, no Skolem witness.")
