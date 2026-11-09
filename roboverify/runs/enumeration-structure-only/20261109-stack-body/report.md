# Stack loop-body structure search without CEM

The enumerative search driver found a BMC-feasible transfer at **candidate 3,431
in 48.759 seconds**, using the three loop-scope names `b`, `b0`, `b_prime`.
CEM, MuJoCo rollouts, imitation scoring and final verification were disabled.
This is a structure-search measurement, not a synthesized-controller result.

| Target | Enumerated candidate number, starting at 1 | Time |
| --- | ---: | --- |
| First transfer satisfying the BMC endpoint goal at all 45 recorded loop entries | 3,431 | 48.759 s, measured |
| Exact five-instruction primitive structure of the supplied demo, ignoring floats | 4,581,985 | Position calculated; search was not run to this candidate |

The driver also checked its empty initial seed once, for **3,432 total candidate
evaluations**. The first enumerated candidate is another empty program. It did
not receive the supplied body as a warm start.

## What was run

- Code: `2588b36`, on `codex/enumerative-search`.
- Driver: `synthesis.experiment.mcmc.search.MCMC`, configured with
  `search_strategy="enumeration"`, `program_slots=5`, `iters=10000`, no videos
  and no periodic checkpoints. A temporary BMC-only scorer replaced the normal
  BMC/CEM scoring callback and stopped at the first accepted structure. An
  assertion blocked any call to `optimize_program`. The production candidate
  generator and its order were unchanged.
- Demonstrations: `demos/pipeline-audit-20261023/stack/demonstrations.npz`, 15
  successful four-block traces, with three loop iterations each. Archive
  fingerprints match the current `synthesis.examples.stack:build_program`.
- Body boundaries: recorded instruction paths `1.0` through `1.4`; the carried
  assignment `b := b_prime` is part of the fixed loop context. Guard discovery,
  assignment enumeration and loop discovery were outside this experiment.
- Search limits: 300 seconds overall, 2 seconds per Z3 query. Neither was hit.
- Timer: includes candidate generation, BMC encoding/checking and search logging;
  excludes imports and archive loading. Python 3.10.22, Z3 4.15.4.

The normal CFG feasibility screen alone cannot define success here: it asks only
whether an action sequence has a trace, so even the empty body is feasible. The
temporary scorer therefore added an explicit stacking endpoint goal. This was
not an ordinary `synthesize_cfg --cem-iterations 0` invocation, which would still
execute candidates and score their untuned floats.

## BMC acceptance condition

For each of the 45 recorded loop heads, ground the named operands using that
head's bindings and use the existing BMC primitive encodings over all four
physical blocks. Fix all initial block positions and the initial gripper
position to the recording, and require an empty hand initially. At the end:

1. `b_prime` is ON `b` using the repository's X/Y tolerance (less than 0.025 m
   difference per axis), and its Z center is exactly one block height (0.05 m)
   above the original Z center of `b`.
2. Every other block retains its initial position.
3. The gripper holds no block.

Move and Release offsets are existential variables during search. A candidate
must be SAT at every recorded entry, with early rejection on the first
UNSAT/UNKNOWN query. This does not prove a property over all possible scenes.
The existing BMC endpoint abstraction does not check the controller's swept
collision path or require the demo's clearance waypoints.

The run issued **3,476 BMC queries: 3,431 UNSAT and 45 SAT**, with no UNKNOWN.
BMC encoding plus solver calls consumed 46.340 seconds. Candidate counts by
length, including the initial seed, were 2 empty programs, 33 length-one
programs, 1,089 length-two programs and 2,308 length-three programs.

## Returned structure and exact demo position

The first accepted structure was:

```text
Pick(b_prime)
Move(b, b, b)       # X, Y and Z references; offsets remain symbolic
Release(b)         # Release-height reference
```

The supplied demonstration instead has:

```text
Pick(b_prime)
Move(b_prime, b_prime, b)
Move(b0, b0, b)
Move(b0, b0, b)
Release(b_prime)
# The loop separately updates b := b_prime.
```

All operands are enumerated. Three names give 3 Picks, 3³ Moves and 3 Releases:
an alphabet of 33 instructions. Within each length, primitives are ordered
Pick/Move/Release and operands lexicographically as `b`, `b0`, `b_prime`.
The demo's five zero-based alphabet indices are `[2, 27, 15, 15, 32]`. Its
one-based position, excluding the separate initial seed, is therefore:

```text
(1 + 33 + 33² + 33³ + 33⁴)
  + 2·33⁴ + 27·33³ + 15·33² + 15·33 + 32 + 1
= 4,581,985
```

BMC accepts a shorter transfer before reaching that position. Recovering the
demo's precise waypoint structure needs an acceptance criterion that constrains
those waypoints; endpoint feasibility alone does not select it.

As a separate check, the discovered structure with fixed Move offsets
`[0, 0, 0.05]` and Release offset `0.15` was BMC-feasible at all 45 entries. These
were explicit witness values, not optimizer output. The demo body with its
original fixed parameters also passed all 45 endpoint queries. These checks took
0.546 and 1.135 seconds respectively, outside the search timer. Neither check
executes or certifies the physical controller.

The raw measurements and separate fixed-parameter BMC checks are in
[summary.json](summary.json). The one-off harness remains outside maintained
repository code at `/tmp/stack_structure_benchmark.py`.
