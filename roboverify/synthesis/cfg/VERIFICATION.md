# Synthesis and verification

`synthesis.entry.synthesize_cfg` now runs the structured CFG synthesizer and
both verification stages on the same candidate and task conditions. It no
longer substitutes a tower fixture at the verification boundary.

## Verification workflow

`verified_synthesis.py` implements the bounded Algorithm 6 workflow:

1. Synthesize/refine the CFG and propose relational placement summaries.
2. Check finite universes for counterexamples, then run the unbounded symbolic
   proof. Unknown, inconsistent and bounded results cannot become verified.
3. For preservation failures, execute the abstract body over the concretized
   counterexample (enumerating Get witnesses), add an uncovered successor, and
   re-learn with an explicit monotonicity check. This is the documented abstract
   replay model of paper discrepancy 7, not a simulator trajectory.
4. For other symbolic failures, export `resynthesis_request.json`. Supplied
   `--additional-demos` recordings are validated, added to complete task demos,
   and repartitioned through synthesis. Without new recordings, the result is
   `needs_demonstrations`.
5. Motion counterexamples accumulate per CFG block. Straight-line synthesis uses
   their failure count alongside imitation distance. Physical instruction
   structure may change, but the entire symbolic program, Get bindings, and
   loop structure must remain unchanged. Every repair is rechecked.

Standalone CEGIS defaults to monotone Boolean-row learning; the integrated CLI
defaults to the legacy learner. Both require explicit invariant-progress checks.
False is logged before bootstrap from demos; unconditional convergence is not
promised. The standalone offset-repair API has a narrower contract than the
integrated structural-repair workflow above.

## Scope and model

A successful result is named **`verified_model`**: partial correctness in the
explicit geometric primitive model. It does not assert total loop termination,
MuJoCo controller refinement, settling, or hardware safety. `--initial-arm X Y Z`
adds an explicit initial-arm condition; without it, all arm positions are checked.
`--motion-noise` and the associated bounds are available through the shared
motion options. Table placements require `--table-surface-height`.

The tower-task scope covers Stack, Unstack, Reverse and ReStack/Partial; the
integrated CLI currently exposes Stack and Unstack. Branch synthesis,
nested/starred quotient, Grid/Pyramid, total-termination proofs and controller
refinement are outside scope. Existing code for other tasks is not evidence of
verified physical execution.

Supported towers have uniform upright blocks of height L, a common flat table,
exact support and complete layers. The tight XY input invariant and placement
VCs are described below. General solver height premises for all blocks, including
unnamed objects, are deferred until needed (entry 16). The two-height premise
belongs only to a regression fixture. Missing premises can cause rejection of
otherwise valid motion; failed/unknown checks are never silently accepted.

The primitive model follows §5.5's held-object and composed-position state.
Blocks share their current geometry, arm position, and held object. Loop bodies
start from fresh invariant/guard states; continuation uses a fresh
invariant/guard-false state. Frozen ON_star_zero geometry remains separate.
Empty-gripper paths use a point against block cubes; carried cubes use the swept
cube model. Intentional Pick/Release contact with the selected object is exempt,
explicitly resolving the paper's Pick self-collision contradiction. Release
requires physical support; a missing support causes a failed obligation and an
arbitrary falling position, never an assumed stable placement.

Motion retains the shared-t straight-segment collision query. An enclosing
endpoint box is not an equivalent replacement (entry 6). Noise is opt-in and off
by default. BMC verifies bounded goals, not collision freedom; solve/feasibility
modes are existential even with noise, and do not prove robustness.

## Placement effects and alignment

Placement summaries are proposed from the outgoing ON relation or the final
transport reference, then checked against all ON*/Higher/Scattered WP effects.
Root discovery follows §5.5: enumerate in-scope physical names `r` and prove
`forall u. ON*(target,u) => ON*(u,r)`, including unnamed objects. The integrated
path uses `P = wp(remaining symbolic body, postcondition)` under the established
entry/invariant/guard context transported through prior symbolic instructions.
It also proves that context establishes P; a desired invariant alone cannot
manufacture a root. Standalone motion checking uses its declared entry conditions.
No named root, an inconsistent context, or solver unknown prevents certification;
`b0`, name order, concrete coordinates, and `frame_base` hints are not evidence.

**Input assumption:** existing towers satisfy tight root-relative alignment in
both horizontal coordinates, `abs(F(member)-F(root)) < L/4`. This is explicit
quantified geometry, not a consequence inferred from ON*'s looser `L/2` bound.
Contradictory concrete input scenes fail consistency. Fresh loop contexts carry
this additional geometric invariant alongside the learned relational invariant.
Before each placement, `alignment_entry` checks the destination tower's bound;
after motion, `alignment` checks the placed block against the proved root for
all allowed noise. The input assumption is never inserted on a placement's
output. The frame VC preserves every non-manipulated object, including the root;
support checks reject removing a root with blocks above it. A separated table
placement creates a singleton. Together these preserve the tight invariant.
Changing references requires proving the new root and its entry alignment.
The triangle inequality gives strict pairwise distance `< L/2`; separate
all-pairs placement checks are unnecessary. See
[entry 12](../../../PAPER-DISCREPANCIES.md#12-root-discovery-and-tight-alignment-premises--implemented-with-an-explicit-input-assumption)
for the proof and user decision. Complete ON*/Higher/Scattered effect checks remain in force.

A transfer may span adjacent blocks. A Get/assignment/control boundary inside an
unfinished transfer, multiple placements in one unsplit block, unknown primitive,
or unsupported summary produces an explicit unsupported result.

## Demonstration and loop semantics

Demo segments use absolute inclusive indices and share their cut state. Recorded
snapshots or deterministic replay reproduce segment starts; replay is the default,
and observation-only demos need a faithful replay source. CFG invariant inference
includes terminal loop heads and frozen invocation-entry geometry.

A split replaces `P -> v0 -> Q` with `P -> v1 --C--> v2 -> Q`. Validate
`first(C) <= last(Q)` on the original unsplit segment, with P at the start, C's
first occurrence strictly interior and Q at the end. The boundary checks imply
the comparison. Entry and later blocks use the same rule; P and C need not persist
until the next condition. Whole-CFG boundary, binding and adjacency checks remain
atomic. See paper-review entry 17 for the notation correction.

Learned loop guards may have multiple witnesses. Demonstrated bindings are
positive examples; unselected bindings at continuing heads are unlabeled, while
all bindings at demonstrated exits are negative. Runtime selects the first match.
The symbolic preservation obligation covers every guard-satisfying witness, so
an unsafe alternative can refute verification even if it was never demonstrated.
No match exits the loop; standalone Get still requires witness existence.

Generated loops have no demonstration-derived execution cap. An explicit budget
raises `LoopBudgetExceeded` if a guard witness remains, reporting incomplete
execution instead of a normal loop exit.

Tests use synthetic scenes and scripted realization proposals to exercise the
real verification and feedback code. They do not depend on saved demos or on
the paper's experimental numbers. Real learning success still requires valid,
representative task demonstrations.
