# Structural search with BMC and CEM

Both `synthesis.entry.synthesize_cfg` and `synthesis.experiment.mcmc.run` accept
`--search-strategy mcmc|enumeration|contract`. The default is `mcmc`. This choice is
independent of `--synthesis-approach relational|id-first` and
`--wp-system old|new|reduction` in the integrated CLI.

Enumeration replaces random mutations and Metropolis acceptance with a lazy,
deterministic stream of primitive structures. Each candidate is checked by BMC
before any rollout or CEM evaluation. Only a feasible candidate reaches the
existing CEM optimizer, which updates its continuous parameters. A rejected or
inconclusive BMC query skips CEM and cannot produce a successful search result.

## Candidate space and budgets

After checking the existing initial seed, enumeration visits lengths zero through
the configured maximum, ordered by instruction kind and sorted operand tuples.
The initial seed is a separate warm start and can also occur in the enumerated
space. The seed construction is unchanged and may depend on `--seed` in the CFG
driver; enumeration itself consumes no random numbers. CEM still samples floats.

The primitive alphabet is `Pick`, `Move`, `Release`, in that order. Every Move
independently chooses its X, Y and Z reference operands. All three Move offsets
and the Release height are initialized to zero and tuned by CEM. Instructions and
parameters are fresh for each candidate. No Skip padding or delay is enumerated;
the empty body represents a zero-step no-op.

| Setting | Integrated CFG search | Standalone search |
| --- | --- | --- |
| Strategy | `--search-strategy` | `--search-strategy` |
| Maximum enumerated primitive length | `--slots` | `--program-slots` |
| Candidate budget after the initial seed | `--iterations` per block search | `--iters` |
| CEM rounds per candidate | `--cem-iterations` | `--cem-iterations` |
| CEM sample / elite counts | `--cem-samples`, `--cem-elites` | `--cem-N`, `--cem-K` |

With N operands, there are N³+2N primitive choices and
Σ(k=0…L)(N³+2N)ᵏ structures through length L. The iterator generates them on
demand and stops at the candidate budget or the end of this finite space.
Exhaustion describes the structural search only; stochastic CEM is not a complete
solver for the float parameters.

CEM returns the best parameter vector it actually evaluated, including sampled
vectors and distribution means. Averaging elites can leave a feasible region;
the averaged vector therefore does not replace a better sample. The legacy and
instrumented standalone optimizers use the same evaluation and sampling order.
The integrated CFG optimizer also retains its best sampled vector.

## Integrated CFG search

Numeric enumeration uses physical block IDs. Relational enumeration uses the
names in the block's available scope, plus names introduced by the seed's fixed
leading Get bindings. That binding prefix is copied into each candidate and does
not count toward `--slots`. Table and observation-relative operands are outside
the enumeration alphabet. Get guards and loops are not enumerated by this
strategy; CFG refinement and quotienting retain their existing roles. The same
strategy is used for later block resynthesis and named loop-body searches.

BMC grounds names and leading Get selections at each demonstration segment's
entry. It fixes the referenced blocks' initial positions and checks whether the
action sequence has an existential trace in the existing BMC abstraction. A full
trajectory starts with an empty hand; interior segments leave holding state
unconstrained because the abstract holding state is not recorded there. Each
query has a 10-second limit. Offsets are existential during screening, independent
of the stored floats and potentially different across segments. CEM subsequently
tunes one shared parameter vector against all segments.

This is a necessary trace-feasibility filter, not a goal or collision proof.
Goals remain checked by the shared PostScore ranking of the epsilon candidate
pool and the later symbolic/motion verification stages. Search keeps the existing
imitation-distance objective and motion-refinement penalties. The CFG's MCMC
path retains its existing behavior, including its lack of this BMC prefilter.

For example, add `--search-strategy enumeration --slots 4 --iterations 2000` to
an existing `synthesis.entry.synthesize_cfg --mode full` invocation. The driver
records the strategy in configuration, BMC decisions in metrics, and checked /
rejected candidate counts in each `block_result` event. `--mode verify` skips
initial search and uses the selected strategy only if resynthesis is requested.

## Standalone search

### Contract-guided proposals

The integrated CLI additionally offers opt-in `--contract-refinement` for
contract-guided milestone cuts, shared numeric endpoint proposals and hard
physical endpoint acceptance. It retains the learned `ON` milestone through the
completed body, and can search recovered loops before finishing an incompatible
unrolled controller. See the [implementation and scope](../cfg/CONTRACT-REFINEMENT.md).
This option is separate from the existing standalone strategy switch.

`--search-strategy contract` uses the maintained `synthesis.search.contract`
backend. It grows straight-line prefixes using nominal Pick/Move/Release
preconditions, effects, nonmover framing, support clearance and supported release.
The finite proposal corpus combines coordinate grids with offsets estimated
from recorded observations. It does not read teacher commands, action labels,
controller targets or a supplied body template. History-sensitive trajectory
ranking and a diversity heuristic retain promising prefixes; exact successor
memoization avoids recomputing an identical state's transitions. Diversity and
beam truncation remain incomplete search heuristics.

Both CLIs expose `--contract-beam-width` (24),
`--contract-max-expansions` (300), `--contract-seconds` (35), and
`--contract-pool-limit` (12). The time limit covers proposal preparation and
search, not subsequent BMC/CEM or whole-pipeline verification. Ablations are
`--contract-grid-only`, `--contract-no-memo`, and `--contract-no-diversity`.
Primitive length is bounded by `--slots` in CFG or `--program-slots` standalone.

The CFG adapter preserves a leading Get/Assign/Skip prefix and its per-entry
bindings. ID-first search uses physical IDs; after quotienting, named searches
also admit the instruction-entry `Y_min`, `Z_m` and `gripper` references.
Standalone contract search uses physical IDs. Logical `tbl` is available in
formulas, but table coordinate operands require explicit runtime
`table_reference` metadata. Unknown interior holding state is rejected: supply
`holding_by_index` metadata or observations with open fingers. Full starts
default to empty unless `initial_holding` is recorded. Raw observations, including
gripper coordinates, are required; object geometry alone is insufficient.
At an observed held entry, nominal Move translates the payload while preserving
its measured offset from the gripper. Payload collision and support clearance
use the payload's centre. The CFG BMC screen enables the same
`preserve_grasp_offset` option in the shared Move kernel for these entries;
legacy noiseless BMC callers and noisy semantics are unchanged. Nominal Pick
still establishes zero grasp offset, so this does not model controller grasp
error after a newly synthesized Pick.

Every proposed body passes an entry-specific BMC feasibility check before CEM.
This screen includes all physical objects and exact scalar-reference semantics;
offsets remain existential separately at each entry. It is not a shared-parameter
goal or physical safety proof. The nominal contracts are proposal filters whose
physical validity still depends on successful controller execution. CFG keeps
its selected imitation objective, epsilon/theta scale, PostScore, quotienting,
invariant inference and later verification. The internal trajectory rank does
not replace those acceptance checks. Contract search retains CEM's evaluated
parameter trials in the same bounded near-best imitation pool before PostScore
ranking, using executable fingerprints so distinct float values remain distinct.
CEM still optimizes imitation alone; a nearby trial that reaches every
segment postcondition can therefore survive even when the best imitation trial
stops just short of a relational boundary. MCMC and enumeration retain their
existing candidate-pool behavior. Standalone keeps its MMD/CEM scorer and
uses the task postcondition, or `--goal-feature`, to select nominal candidates.

For example, add `--search-strategy contract --slots 8` to an existing CFG
invocation, or `--search-strategy contract --program-slots 8` to the standalone
driver. BMC must remain enabled. Proposal exhaustion, unsupported input and
deadlines are reported explicitly; none establishes successful verification.

For integrated CFG refinement, a failed contract search can retain non-goal
prefixes as separate refinement candidates. They still pass BMC and the configured
numeric stage, but cannot be accepted as synthesized blocks merely because their
imitation score is small. If nothing was evaluated, refinement receives an empty
body or binding prefix, rather than an unevaluated random physical seed. Exact
rollout caching avoids repeating the same deterministic simulation for objective,
distance and PostScore; it does not cache Monte Carlo distance estimates.

The integrated CLI also supports `--distance arc`. It compares each candidate
trajectory with its own demonstration after ordered arc-length resampling, then
averages spatial RMS in metres. Gripper and all block positions contribute;
moving blocks are inferred from observed displacement, without teacher action or
source-role labels. Both `--theta` and `--epsilon` are metres in this mode, unlike
KL or MMD. For example, `--theta 0.03 --epsilon 0.005` means a 3 cm convergence
threshold and 5 mm near-best pool. `--cem-std 0.004` sets the initial numeric
proposal standard deviation to 4 mm. These are explicit experiment settings,
not replacements for the defaults (KL, epsilon 0.1 and CEM std 0.05).

### Four-block Stack synthesis without verification

The repository includes
[`demonstrations.npz`](../../demos/stack/4-blocks-5-trajectories-pyeda-verification/demonstrations.npz)
and its original
[`collection.json`](../../demos/stack/4-blocks-5-trajectories-pyeda-verification/collection.json).
This collection contains five validated four-block Stack trajectories from the
provided `synthesis.examples.stack:build_program`, collected for the 2026-10-01
PyEDA verification recheck. Full simulator snapshots let search restore the
recorded starts without recollecting demonstrations.

With the project dependencies and MuJoCo installed, run from the repository root:

```bash
cd roboverify
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

uv run python -m synthesis.experiment.mcmc.run \
  --task stack --num-blocks 4 \
  --demos demos/stack/4-blocks-5-trajectories-pyeda-verification/demonstrations.npz \
  --search-strategy mcmc --no-bmc \
  --video-policy none --run-name stack-4-blocks-synthesis
```

This runs standalone straight-line MCMC/CEM imitation search over the archived
demonstrations. It performs no symbolic or motion verification; `--no-bmc` also
disables BMC feasibility pruning. It does not recover loops. The integrated
`synthesis.entry.synthesize_cfg --mode synthesize` instead runs integrated CFG
refinement and optional loop recovery without final verification, saving a
reloadable program candidate. The standalone command above still performs only
straight-line search.
The original collection's verification history does not verify a newly
synthesized candidate.

The command uses all five demonstration seeds and the default search budget of
2,000 MCMC iterations and 10 CEM rounds per candidate. Add `--smoke` for a smaller
diagnostic run (20 MCMC iterations, 2 CEM rounds and 2 seeds). These budgets do not
guarantee successful task synthesis. Videos are disabled by the command above;
use `--video-policy best` to render new best candidates.

Inspect progress or the completed result from `roboverify/`:

```bash
uv run python -m synthesis.experiment.report --run runs/stack-4-blocks-synthesis/latest
```

The run directory records the resolved configuration, metrics and best-program
artifacts. See [the environment and monitoring guide](../../../AGENTS.md).

### Enumeration with BMC pruning

The standalone driver uses exactly the same BMC goal check, CEM implementation,
imitation cost and optional goal reward as its MCMC path. Enumeration requires
`--goal-feature` and rejects `--no-bmc`:

```bash
uv run python -m synthesis.experiment.mcmc.run \
  --task stack --num-blocks 3 --demos path/to/demonstrations.npz \
  --search-strategy enumeration --program-slots 3 --iters 5000 \
  --goal-feature 'ON(1,0)' --video-policy none --run-name enumeration
```

Use a full-state archive printed by the demonstration collector. The BMC model
uses the existing scattered-table initialization over referenced blocks; CEM
restores the actual archived starts. Best-program checkpoints include tuned
floats. Results distinguish budget exhaustion, space exhaustion and absence of a
BMC-feasible candidate. A completed search does not establish task verification.
`MCMCConfig(search_strategy="enumeration")` selects the same path for Python
callers of the instrumented `MCMC` function; that public name is retained for
compatibility. The legacy uninstrumented `synthesis.mcmc.synthesis.MCMC` is unchanged.

Tests cover exhaustive small structure spaces, scope/alias handling, real BMC
screening followed by real CEM on synthetic objectives, rejection without rollout,
CLI routing into named loop bodies, and the existing simulator-backed MCMC parity
checks. These checks do not establish end-to-end task synthesis performance.

## Stack structure-only measurement

The [2026-11-09 BMC-only experiment](../../runs/enumeration-structure-only/20261109-stack-body/report.md)
called the enumerative search driver with CEM and simulation disabled. An explicit
stacking endpoint goal accepted the first transfer after 3,431 enumerated
candidates and 48.759 seconds across 45 recorded loop entries. It found a
three-instruction transfer. The supplied demo's exact five-instruction structure
has calculated position 4,581,985 in the same order; that position was not timed
by exhaustive search. This is a structure-search result, not a motion proof or
an end-to-end synthesis result.

For the standalone four-ID alphabet and goal `ON(1,0)`, the demo's first-iteration
five-motion-instruction body has exact rank **63,105,263**. Before it, the
production BMC filter would receive 63,105,263 candidates including its initial
seed; 59,388,189 would reach a Z3 check. These are calculated invocation counts,
not SAT/CEM counts or an exhaustive timed run. The target itself passes the
production BMC gate. See the [standalone rank calculation](../../runs/enumeration-structure-only/20261109-standalone-stack-demo-rank/report.md).

Among those preceding candidates, **28,254,518 satisfy the standalone BMC goal
and have float parameters**, so they would enter CEM if optimization were enabled
and search continued to the demo. This separate eligibility count was calculated
from the current encoder's semantics and validated with 1,249 production-checker
comparisons; no CEM was run. The linked report gives the counting argument,
including the encoder's permissive behavior when moving a support.
