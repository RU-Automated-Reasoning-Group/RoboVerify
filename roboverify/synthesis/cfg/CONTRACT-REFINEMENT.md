# Contract-guided CFG refinement

The integrated synthesis CLI accepts `--contract-refinement` with
`--search-strategy contract` in `synthesize` or `full` mode. The option combines
bounded milestone-boundary refinement, shared numeric endpoint proposals and
empirical endpoint contracts. It is opt-in; existing search and WP selections
retain their defaults. This implements the proposal studied in the
[boundary experiments](../experiment/README.md#contract-guided-boundary-refinement-experiments).
Fresh integrated robustness results are recorded separately under
`runs/contract-integrated/20261110/`; earlier continuation experiments do not
establish successful synthesis from full demonstrations in this mode.

## Boundaries and loop recovery

Geometric conditions can become true while the gripper still holds a block.
`boundaries.py` searches a bounded, contiguous truth interval containing the
original milestone. A nonterminal cut must admit a shared next-action proposal
from the existing primitive grammar, using the **next** segment's bindings.
Positive explicit `ON` milestones additionally require a known released hand.
Continuing loop entries must also match the empty/occupied hand phase of their
demonstration's first iteration. A Move alone can be admissible in both phases;
that does not imply a shared Pick/Release continuation exists.
No teacher action labels, prescribed body or retreat detector are used.

Loop extraction refines its demonstration intervals before guard learning and
body search. Terminal iterations need no next action; their remaining suffix is
retained for synthesis. A failed body may try another bounded cut vector. Failed
straight-line blocks may also retry adjoining cuts before adding another CFG
split. Predicate labels, witnesses and frozen entry geometry are preserved.
`DemoSegment.milestone_index` records the original geometric endpoint; CFG
validation checks that the predicate remains true through the refined endpoint
and that neighboring segments cover every original transition without gaps.

With this option, ID-first synthesis can fold a learned repetition before its
unrolled controller is complete. Residual IDs are explicitly named for continued
search. Unresolved regions remain unresolved: the ordinary final naming and
lowering checks still require a complete executable.

Boundary limits are `--boundary-max-shift` (80 recorded steps),
`--boundary-first-actions` (32), `--boundary-proposals` (3), and
`--boundary-seconds` (15 seconds per proposal call). These are bounded heuristic
searches, not completeness claims. The total CLI wall-clock limit also applies.

## Shared numeric proposals

The nominal beam can extend held-state prefixes with generated Move/Release
operand pairs and solve a single shared Move XYZ offset across all demonstration
entries. The solver uses the maintained predicate language, including quantified
finite goals and frozen entry relations. It maximizes endpoint/support margin
inside a bounded offset neighborhood, then replays the unchanged nominal
transitions and endpoint predicate. No settling effect or stronger controller
capability is introduced.

`--contract-solver-timeout-ms` (250), `--contract-solver-attempts` (24), and
`--contract-offset-radius` (0.02 m) bound this additional proposal mechanism.
UNSAT, UNKNOWN and failed exact transition replay produce no proposal. This
extension is incomplete: it repairs generated terminal pairs, rather than
solving all parameters of arbitrary-length programs. Existing BMC screening
still precedes optional CEM and physical candidate execution.

After nominal goals are found, the opt-in mode explores single-primitive operand
substitutions while keeping their kinds, lengths and numeric offsets fixed.
Every alternative is replayed against the same nominal transitions and goal.
A small pool preserves different shared XY references at each Move site among
programs within 1% of the best nominal imitation score, then fills by rank.
No particular task, block name or supplied program is preferred. This addresses
the loss of distinct reference choices when nominally similar endpoints compete
for a beam slot; it remains an incomplete proposal heuristic.

The neighborhood has a 5-second, 4096-candidate and 100000-transition cap within
the existing proposal deadline. An exhausted optional expansion retains already
checked beam goals. `--contract-no-operand-neighborhood` disables this stage for
comparison; ordinary search defaults are unchanged. Physical imitation,
BMC and all endpoint acceptance conditions still apply afterward.

Beam diversity also distinguishes empty-hand states by which generated Picks
their existing transition contracts enable across all examples. This prevents
a low, unusable approach from removing a useful clear approach solely because
their block poses and hand occupancy agree. Readiness probes consume the same
transition and wall-clock budgets; caching is bounded. Pick's nominal and BMC
contracts require source clearance for lateral approach and retain direct grasps
from already aligned entries. This is a necessary enabling check, not a complete
swept-collision or physical-grasp guarantee.

## Endpoint acceptance and proof scope

The learned postcondition must hold at the actual body endpoint. An earlier
transient occurrence cannot satisfy acceptance. An explicit `ON(new, old)` uses
the physical-entry roles before trailing alias updates, requires observed release
and designated support contact, and receives finite checks of the production
old-Put ON*/Scattered effects and input/successor domain axioms. These checks
evaluate the observed tables directly; they do not assume the axioms while
testing their conclusions.

Audited execution records nested primitive convergence, contact-derived hand
state and transfer-wide sampled nonmover displacement. The default allowed
displacement is 0.005 m (`--contract-frame-tolerance`). Generic Higher and other
non-placement intermediate milestones may legitimately end holding a block;
their phase must match the recorded endpoint. A shared single-slot loop body
also checks its own first action against actual continuing endpoints with the
next bindings. This is nominal readiness, separate from physically executing the
complete recovered loop.

Contact observation follows the active MuJoCo constraint and its finite positive
normal force, using `efc_address` and `mj_contactForce`. It does not require surface
penetration: MuJoCo's configured margin can create force-bearing contacts at a
positive geometric gap. Two distinct fingers must still contact the same unique
object to establish holding; unavailable or nonfinite force data remains unknown.
See the [MuJoCo contact model](https://mujoco.readthedocs.io/en/2.1.2/computation.html#contact).
Per-obligation diagnostics distinguish a failed requirement from unknown evidence;
both reject acceptance when that evidence is required. A release's observed support
is checked independently from whether holding was observable at its entry.

These are finite empirical rejection checks. Controller-action observations do
not cover every physics substep. The checks do not prove unbounded collision
safety, Higher effects, arbitrary relational summaries, termination or all scene
assumptions of other WP backends. The `full` verification pipeline retains its
separate obligations. Search acceptance is never promoted to a formal proof.

Guard enumeration can prefer references unchanged by the inferred loop update.
This selects among training-consistent separators using a stated inductive bias;
it supplies no uniqueness or generalization guarantee.

## Complete-program acceptance

The opt-in search retains bounded alternatives that pass the same local endpoint,
handoff and imitation thresholds. It tests complete executables from the original
archived starts, including every actual block endpoint and 50 final settling
steps. The gate checks the task precondition at each restored start, primitive
convergence, observed contact/frame behavior, the final task condition, and the
learned block contracts with their pre-update roles. An inconclusive contact
state or exhausted budget cannot produce acceptance.

Selection rejects a candidate after its first failed complete start and tries
previously failing task/size/seed groups first on later candidates. Reports keep
requested and evaluated counts separate and list unexamined starts explicitly;
every requested start must be executed successfully for acceptance. Duplicate
seeds never merge distinct archived snapshots. `--candidate-max-total-actions`
(100000) is shared across candidate executions and settling, in addition to the
per-execution action cap and shared wall deadline.

`--candidate-validation-limit` (8) bounds combinations of locally accepted
alternatives; `--candidate-validation-seconds` (120) is their shared time budget.
`--candidate-max-actions` (5000), `--max-loop-iterations`, and
`--candidate-settle-steps` (50) bound each execution. Alternatives must preserve
the block's exported bindings. Selection visits combinations by ordinal ranks,
without adding imitation distances with different units. Rejection or an
exception restores the original candidate; only a fully accepted alternative is
committed. In `full` mode validation precedes invariant preparation and repeats
after physical repair.

Optional `--candidate-parameter-trials N` (default 0) adds bounded float repair
when the retained complete structures all fail. For each retained structure,
it probes every registered parameter at `+step`, `-step`, then `+2*step`,
`-2*step`, in registration order; `--candidate-parameter-step` defaults to 1 mm.
Each sample is centered on that structure's original vector and changes only
floats, including all Move coordinates. `N` is a per-structure cap; structure
count, wall time and action usage retain their shared outer bounds.

Every trial first passes fixed-offset BMC on the original entries. Complete-task
success then triggers the original local imitation, endpoint and handoff checks
for changed blocks, using the original metric/theta and original near-best pool's
minimum distance plus epsilon. Missing local certification, incomplete evidence,
timeouts and later local rejection cannot promote a candidate. Only a fully
tested sample is installed; exceptions or failed search restore the original
graph. Exact pre-execution samples and reports are saved under
`parameter-refinement-REVISION/structure-INDEX/`. This coordinate search is an
incomplete repair heuristic, separate from the existing CEM stage.

Optional `--validation-starts PATH ...` supplies additional full-state archives,
including other block counts. These starts participate in candidate selection;
they are never imitation examples or hidden action labels, and must not later be
reported as held-out evaluation. Environments use each archive's own block count.
The original training starts remain in the acceptance denominator.

Every tested executable and replay report is saved under
`training-validation-REVISION-candidate-TRIAL/`; `training-selection-REVISION.json`
records the bounded selection. Final `cfg.json` and `program.json` describe the
same selected program. A failed complete execution produces
`simulation_rejected` and exit code 2, even if all local block searches succeeded.
Attempted block artifacts also retain the exact target predicate AST, frozen
entry index, original milestone index and demonstration boundaries.
Every locally accepted alternative is saved under that block attempt's
`alternatives/candidate-RANK.json`, including its exact operands, float values
and physical imitation score, before complete-program selection begins.

These are empirical acceptance checks. They do not establish arbitrary-size
generalization, a complete physics-substep audit, or a formal refinement proof.

## Metrics and reproduction

The ordinary `--distance`, `--theta`, and `--epsilon` control whole-demo and CFG
fragment search. A recovered loop can use a separate metric with all three of
`--loop-body-distance`, `--loop-body-theta`, and `--loop-body-epsilon`. Thresholds
must be supplied explicitly in the chosen metric's units; no KL/MMD/Arc scale
conversion is inferred.

For the current independent-controller archives, from `roboverify/` with the
simulator environment configured as in [AGENTS.md](../../../AGENTS.md):

```bash
uv run python -m synthesis.entry.synthesize_cfg \
  --task stack --num-blocks 4 --mode synthesize \
  --demos runs/contract-end-to-end/20261110/data/stack-independent-accepted-5-holding.npz \
  --synthesis-approach id-first --search-strategy contract --contract-refinement \
  --slots 6 --refinements 20 --cem-iterations 0 \
  --distance mmd --theta .1 --epsilon .1 \
  --loop-body-distance arc --loop-body-theta .03 --loop-body-epsilon .003 \
  --contract-beam-width 12 --contract-pool-limit 8 --contract-seconds 60 \
  --wall-timeout-seconds 600
```

This is an experimental invocation, not a claim that this budget succeeds.
Run directories retain CFG cuts and boundary proposals, exact executable JSON
when available, endpoint rejection counts, solver statuses and bounded-search
termination reasons. Frozen programs must be evaluated independently on fresh
starts before reporting a simulation success rate.
