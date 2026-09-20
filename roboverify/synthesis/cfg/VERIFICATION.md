# Synthesis and verification

`synthesis.entry.synthesize_cfg` now runs the structured CFG synthesizer and
both verification stages on the same candidate and task conditions. It no
longer substitutes a tower fixture at the verification boundary.

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

A successful result is named **`verified_model`**: partial correctness in the
explicit geometric primitive model. It does not assert total loop termination,
MuJoCo controller refinement, settling, or hardware safety. `--initial-arm X Y Z`
adds an explicit initial-arm condition; without it, all arm positions are checked.
`--motion-noise` and the associated bounds are available through the shared
motion options. Table placements require `--table-surface-height`.

The primitive model follows §5.5's held-object and composed-position state.
Blocks share their current geometry, arm position, and held object. Loop bodies
start from fresh invariant/guard states; continuation uses a fresh
invariant/guard-false state. Frozen ON_star_zero geometry remains separate.
Empty-gripper paths use a point against block cubes; carried cubes use the swept
cube model. Intentional Pick/Release contact with the selected object is exempt,
explicitly resolving the paper's Pick self-collision contradiction. Release
requires physical support; a missing support causes a failed obligation and an
arbitrary falling position, never an assumed stable placement.

Supported synthesis scope remains structured chains and recovered flat loops.
Placement summaries are proposed from the outgoing ON relation or the final
transport reference, then checked against all ON*/Higher/Scattered WP effects.
A transfer may span adjacent blocks. A Get/assignment/control boundary inside an
unfinished transfer, multiple placements in one unsplit block, unknown primitive,
or unsupported summary produces an explicit unsupported result. The existing
Higher abstraction can also be refuted even for reasonable geometry (discrepancy
16). Branch/nested-loop synthesis and controller dynamics remain outside the
agreed flat-loop remediation scope.

Tests use synthetic scenes and scripted realization proposals to exercise the
real verification and feedback code. They do not depend on saved demos or on
the paper's experimental numbers. Real learning success still requires valid,
representative task demonstrations.
