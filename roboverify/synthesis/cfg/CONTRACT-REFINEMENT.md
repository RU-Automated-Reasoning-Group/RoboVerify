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

These are finite empirical rejection checks. Controller-action observations do
not cover every physics substep. The checks do not prove unbounded collision
safety, Higher effects, arbitrary relational summaries, termination or all scene
assumptions of other WP backends. The `full` verification pipeline retains its
separate obligations. Search acceptance is never promoted to a formal proof.

Guard enumeration can prefer references unchanged by the inferred loop update.
This selects among training-consistent separators using a stated inductive bias;
it supplies no uniqueness or generalization guarantee.

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
