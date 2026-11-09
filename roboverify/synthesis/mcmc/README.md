# Structural search with BMC and CEM

Both `synthesis.entry.synthesize_cfg` and `synthesis.experiment.mcmc.run` accept
`--search-strategy mcmc|enumeration`. The default is `mcmc`. This choice is
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

## Integrated CFG search

Numeric search enumerates physical block IDs. Relational search enumerates the
names in the block's available scope, plus names introduced by the seed's fixed
leading Get bindings. That binding prefix is copied into each candidate and does
not count toward `--slots`. Table and observation-relative operands are outside
this primitive search alphabet. Get guards and loops are not enumerated by this
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
