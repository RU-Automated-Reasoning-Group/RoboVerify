# Resolved paper / implementation findings

Resolved findings from [PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md).
Original numbers are retained so plan and audit references remain valid. These
are settled decisions and implementation records, not an active defect list.
The alignment proof and input assumptions in entry 12 remain part of the model.

## 5. The geometric translator conflated initial and current ON_star

**Status:** fixed in Phase D. The old `_translate_expr` mapped both `ON_star` and
`ON_star_zero` to the current coordinates. That could strengthen or contradict a
Reverse invariant that compares two different configurations. The translator now
uses separate frozen `X0/Y0/Z0` functions for `ON_star_zero`, with a regression in
which an initial on-relation holds and the current one does not. This follows the
already-settled initial-state semantics; it introduces no global link axiom.

## 6. Plan correction: an endpoint bounding box is not an equivalent collision check

**Status:** documentation corrected; the proposed fallback was never implemented.
This entry identifies a plan error, not an error in the paper's shared-parameter
collision formula or the code implementing that formula.

- **Paper:** §5.5, p. 29 defines collision using one segment parameter `t` shared
  across X, Y and Z. All three overlap conditions must hold at the same position
  along the straight trajectory.
- **Code:** `encode_collision_at` in
  `roboverify/synthesis/verification_lib/lowlevel_verification_lib.py` retains that
  shared-`t` query for the moving-cube model. No endpoint-box fallback is used.
- **Plan:** D2 originally proposed replacing the query with one axis-aligned box
  enclosing the entire movement and incorrectly treated this as equivalent. The
  original paragraph is now corrected; the Phase D spike did not require a fallback.

For diagonal motion, the enclosing box contains space outside the swept path.
An obstacle there can overlap the box without colliding with the moving cube.
A correctly enclosing box can therefore prove clearance when it is clear, but
overlap alone cannot establish a collision. The retained query is exact for its
straight-moving cube model; this does not establish physical-controller behavior
or resolve the separate Pick-contact issue in discrepancy 18.

## 11. Corrected plan/code restriction: loop guards may have multiple witnesses

**Status:** uniqueness restriction removed following the user's semantic correction.
The paper's arbitrary-witness semantics are appropriate here; this entry records
an incorrect plan/implementation restriction, not a paper error.

The paper's §5.3 requires correctness for every witness satisfying the existential
guard. Choosing one object in a demonstration does not make other choices wrong;
symmetric objects can be indistinguishable and equally valid. First-match runtime
selection is permitted because it chooses one of the witnesses covered by the proof.

Previously, guard learning labeled all unselected bindings negative, generated loops
required uniqueness, runtime raised on multiple matches, and a `guard_unique` VC
rejected such guards. Those restrictions could reject valid programs and are removed:

- Learning uses demonstrated bindings as positives. Unselected bindings at a
  continuing head are unlabeled; every binding at a demonstrated exit is negative.
- Runtime selects the first matching binding. No match exits a loop normally;
  standalone Get still requires a witness (entry 9).
- The preservation VC retains an arbitrary guard witness. Searching for a
  refutation can select any matching witness, including one never demonstrated;
  successful verification therefore covers all permitted choices.

Regressions cover indistinguishable alternatives, multi-variable witnesses,
witness-free exits, execution with multiple matches, acceptance when all choices
preserve the invariant, and rejection when an additional permitted choice breaks it.
This corrects the uniqueness restriction; matching demonstrations alone still does
not prove that a learned guard or its body is correct.

## 12. Root discovery and tight alignment premises — implemented with an explicit input assumption

**Status after the user decision:** implemented. Existing input towers are assumed
to satisfy tight root-relative alignment. Every new placement must establish that
bound by a motion VC. The agreed root-only induction remains valid; this resolves
the code gap and makes its geometric premise explicit, rather than deriving it
from the weaker ON* predicate.

**Original audit finding.** Section 5.5, equations (6)/(7), and Appendix J require
root-relative alignment. Phase D2 omitted this obligation. With block length .05,
root x=0, top x=.024, and the new block at x=.048, local placement and the original
motion checks passed although geometric ON*(new,root) was false. The added
alignment and complete-effect checks now reject this scene; see
`roboverify/synthesis/verification_lib/test_motion_verification.py`,
`test_root_drift_is_rejected_even_when_local_placement_passes`. That regression
now starts with a tightly aligned top at x=.012 and places at x=.036, so it
checks failure of the constructed tower within the newly declared input domain.

**Current implementation.** `verification_lib/root_selection.py` searches in-scope
names and proves the paper's universally quantified bottom-root criterion.
`cfg/verification.py` supplies the remaining symbolic body's WP and transports
established entry/invariant/guard facts through the symbolic prefix. It first
proves the WP applies in that context; it does not assume the desired invariant.
This contextual proof may use established facts in addition to P. Standalone
motion calls prove the root directly from their supplied entry conditions.
An unproved/inconsistent/unknown result cannot select a name fallback. The old
`b0`/first-name selection and conditional designated-reference check are removed.
`frame_base` is retained for compatibility and populated with the proved reference;
a caller's hint is never trusted as a proof.

**Paper root rule, p. 30, lines 1433–1439.** With P = wp(pi_phi, I), seek a named
r such that P implies: for every u below y, u is above r. Under reflexivity,
antisymmetry, a satisfiable P, and quantification over the relevant objects, this
is a valid bottom-root criterion. It need not have a named solution. Its use also
requires the actual entry context to imply P; failed/unknown checks cannot select
an arbitrary fallback. The main text writes u in O, whereas Appendix J writes
unrestricted u. O includes a fresh arbitrary witness, so finite instantiation can
cover unnamed objects only with the corresponding universal-validity argument;
checking a single concrete assignment is insufficient.

**Agreed root-alignment induction, pp. 29–32 and Appendix J.** Checking a new
block against the root is sufficient. There is no need to compare it separately
with every existing block, provided the following invariant is established and
maintained. For tower members S, a common root r, and each horizontal coordinate
F in {X, Y}, let

```text
Aligned(S, r) := for every a in S and F in {X, Y},
                    |F(a) - F(r)| <= delta_F
with 2 * delta_F <= N_F.
```

The proof is:

1. **Base.** A singleton tower S={r} satisfies Aligned because its displacement
   from its root is zero. For pre-existing towers, the user has explicitly
   selected Aligned as an input assumption defining the permitted configurations.
2. **Existing members.** Before insertion, assume Aligned(S,r). Preserve the root
   and existing members' positions, or otherwise prove that their bounds remain
   true after the operation.
3. **New member.** Check only `|F(x)-F(r)| <= delta_F` for the newly placed x.
   Together with step 2, this establishes Aligned(S union {x},r).
4. **Pairwise consequence.** For any two members a,b of the enlarged tower,
   the triangle inequality gives
   `|F(a)-F(b)| <= |F(a)-F(r)| + |F(b)-F(r)| <= 2*delta_F <= N_F`.
   This includes every new-to-old pair without individual placement checks.
5. **Iteration.** Step 3 preserves the SAME root-relative invariant, so the
   argument repeats for any finite number of insertions. With strict bounds,
   the corresponding strict triangle-inequality conclusion applies.

This is the paper's valid Lemma 5.5/J.1 mechanism: tight root-relative bounds
imply looser pairwise bounds. It does not require pairwise distances <= delta_F,
nor prove vertical ordering, collision freedom, or physical stability; those
remain separate obligations. Removing members preserves Aligned for the remaining
subset if its reference is retained. Replacing/moving the reference or merging
chains requires re-establishing the relevant bounds.

**Resolved obligations and implementation boundary.**

- **Root justification:** quantified proof covers arbitrary unnamed objects and
  scoped aliases; missing or inconclusive proofs prevent certification.
- **Establishment:** input towers are assumed Aligned, as requested by the user.
  `assume_input_alignment` adds a quantified constraint on fresh entry geometry
  for proved input roots. It does not claim ON* implies the tighter bound.
  Concrete inputs violating the assumption fail consistency. A Get/Assign may
  expose another input root before any Put; after a Put the verifier does not
  insert fresh assumptions on the constructed geometry.
- **Preservation:** `alignment_entry` checks the destination tower before motion;
  `alignment` checks the new member against its proved root after motion,
  including bounded noise. The frame VC fixes all other objects, including the
  root, and support checks reject moving an occupied support. Removing a top
  member preserves the remaining bounds; separated table placement starts a
  singleton. These justify carrying tight alignment into fresh loop contexts
  as an additional geometric invariant. A changed reference must pass root and
  entry-alignment checks again. The verifier does not accept arbitrary root
  replacement or chain merging on the strength of a name.
- **Tolerance consistency:** the implementation uses strict delta=L/4 and N=L/2
  in each horizontal coordinate, giving strict pairwise separation below N.
  Local direct-on, vertical support, collisions, and exact relation effects
  remain separate VCs; alignment alone does not certify the whole placement.

Regression tests cover an unrelated `b0`, unnamed lower objects, scoped roots,
WP applicability and assignments, solver unknown/inconsistency, bad input
alignment, local-success/global-drift rejection, bounded noise, and fresh loop
contexts. No saved demonstrations or experimental results are required.

**Why the initial tight premise matters.** The following example does NOT satisfy
Aligned initially, so it is not a counterexample to the agreed induction. A concrete
horizontal example, in units of delta with N=2, has existing
block centers bottom-to-top [0, -1, -2, -1, 0]. Every adjacent displacement is 1,
all old pairwise distances are at most 2, and the true root is the first block.
Place a new top block at +1: its distance to both target and root is 1, but its
distance to the old block at -2 is 3 > N. Heights can increase by one valid block
step, with the other horizontal coordinate fixed. Direct arithmetic checks confirm
all these inequalities. This refutes the sufficiency of the loose old-chain
bounds plus the new-element alignment check; it does not refute the lemma with
its full hypotheses or assert that the current complete-effect checker accepts
this scene. A sound implementation needs a justified stable reference and an
established/preserved tight alignment invariant, or another proof of the complete
required geometric effects. See A1 in `AUDIT-popl-alignment.md`.

## 13. Placement effects and block-only Scattered — implementation resolved

**Status:** resolved in code; the user confirmed the intended semantics. After
`Put(a, tbl)`, the symbolic action promises that `a` is scattered from every
other physical block. Motion verification must prove those relations from the
resulting geometry. `Scattered` applies between blocks, excluding the symbolic
`tbl` marker. The physical table height is a separate placement condition.

`check_abstract_effects` in
`roboverify/synthesis/verification_lib/motion_verification.py:421` checks the
actual ON*/Higher/Scattered results against the symbolic Put rewrites, including
an arbitrary unnamed object. `check_contract_realization` separately checks
release and direct placement or table height. A successful local contract does
not bypass the relation-effect obligations. CFG verification invokes both.

`rewrite_for_put_on_tbl_for_scattered` in
`roboverify/synthesis/api/program.py:684` excludes `tbl` from both arguments,
preserving the standing table-isolation decision. The earlier statement that the
implementation retained the conflicting unrestricted rewrite was obsolete.

The original counterexample placed a block at table height, 0.075 m from another
block, while Scattered requires at least 0.1 m separation along X or Y. That
scene now passes the local contract but fails `effect_Scattered`; verification
rejects it. Regression tests also ensure `Scattered(a, tbl)` remains false:

- `test_table_placement_requires_symbolic_separation_effect`
- `test_table_scattered_wp_preserves_isolation`

Both tests are in
`roboverify/synthesis/verification_lib/test_motion_verification.py:147` and passed
in the focused follow-up run (2 tests, 0.408 seconds). Plan item A1 records the
completed full-effect checks. No further implementation change is required for
this finding. The remaining paper notation clarification is retained as entry 13
in the active discrepancy document.
