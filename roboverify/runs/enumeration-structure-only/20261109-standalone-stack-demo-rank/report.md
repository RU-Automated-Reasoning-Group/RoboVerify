# Standalone Stack search: position of the demo body

With the same standalone setting—four block operands `{0,1,2,3}`, goal
`ON(1,0)`, Pick/Move/Release enumeration and maximum length five—the demo's
first-iteration motion body is **enumerated candidate 63,105,263**. This is an
exact calculation, not a completed run of the full prefix. The previous
10,000-candidate budget cannot reach this position.

The target uses `b0=b=0`, `b_prime=1`, ignores float values, and contains:

```text
Pick(1)
Move(1, 1, 0)
Move(0, 0, 0)
Move(0, 0, 0)
Release(1)
```

The standalone grammar does not enumerate the final `Assign(b,b_prime)` or
the While guard. Thus this target is the five physical instructions of one
iteration, not a synthesized generic named loop.

| Count strictly before the target | Candidates |
| --- | ---: |
| Earlier enumerated candidates | 63,105,262 |
| Calls to the BMC candidate filter, including the initial all-Skip seed | 63,105,263 |
| Actual Z3 satisfiability checks | 59,388,189 |
| Rejections because the goal references an absent block | 3,717,072 |
| Empty-program rejections, including the initial seed | 2 |

The target itself adds one BMC filter call and one Z3 check. Every nonempty
candidate enters the BMC builder; candidates missing block 0 or 1 are rejected
before `solver.check()`. These counts describe checker invocations. The separate
SAT/CEM-eligibility count below measures which candidates pass that check.

The actual production `check_bmc_candidate` accepted the target in **0.0212 s**,
using the unchanged `ON(1,0)` goal and scattered-table initialization. This is
the target-only check time, not the time to enumerate and check its predecessors.
CEM and simulation were not called.

## Counting method

Four operand choices produce 4 Picks, 4³ Moves and 4 Releases, for an alphabet
of 72. The target's zero-based alphabet indices are `[1,24,4,4,69]`:

```text
rank = (1 + 72 + 72² + 72³ + 72⁴)
       + 1·72⁴ + 24·72³ + 4·72² + 4·72 + 69 + 1
     = 63,105,263
```

For Z3-call counts, dynamic programming counts the same lexicographic prefix
while recording whether each candidate references block 0, block 1, both, or
neither. Exactly the nonempty candidates referencing both reach a solver check.
The method was cross-checked against the first 79 actual generated candidates
and the earlier standalone measurement: 20 Z3 checks, 58 missing-block
rejections, and two empty checks including the separate initial seed.

Code version: `760fe7a`. Raw configuration, counts and the target BMC result
are in [summary.json](summary.json). The one-off counting script is outside
maintained repository code at `/tmp/standalone_demo_rank.py`.

## Candidates that pass BMC before the demo

**28,254,518 candidates before the target satisfy the current standalone BMC
condition and would be eligible for CEM.** The target itself also passes, making
28,254,519 eligible candidates through that point. Every passing candidate
contains a Move and therefore has float parameters. The separate initial seed
fails BMC and does not add to these counts.

| Candidate length | BMC-passing candidates strictly before the target |
| --- | ---: |
| 0 or 1 | 0 |
| 2 | 74 |
| 3 | 19,222 |
| 4 | 2,460,508 |
| 5, preceding the target | 25,774,714 |
| Total | 28,254,518 |

This is an exact combinatorial count under the current nominal BMC semantics,
not a completed solver run over all 63 million preceding candidates. No CEM or
simulation was invoked. The calculation used code version `432e7ff`, whose
production BMC and enumeration code are unchanged from the rank measurement.

### Why the acceptance condition can be counted exactly

The standalone start puts all referenced blocks at the same height and separate
X coordinates. Every Move has fresh, unconstrained real offsets, so its reference
operands restrict which blocks enter the model but do not restrict its possible
destination. Release does not change block positions. A candidate is satisfiable
exactly when all of the following hold:

1. Both blocks 0 and 1 occur in its operands; otherwise the production wrapper
   rejects the absent goal block.
2. Its Pick/Release sequence is consistent with an initially empty hand. Picks
   require an empty hand, and Releases require a held block.
3. It contains a Move while holding block 0 or 1, or it moves the same other
   block (2 or 3) at least twice.

For the direct-goal case, the unconstrained offsets allow placing either goal
block so `ON(1,0)` holds. Later actions can preserve that relation. The helper
case follows the encoder's explicit behavior when a support moves: after moving
block 2 or 3 below a goal block, another Move of that helper leaves the block
above it unconstrained, permitting the goal. The two Moves may occur in different
holding intervals for the same helper. This is a property of the current abstract
encoding, not evidence that such a controller is physically valid.

Conversely, if neither goal block is moved and neither helper is moved twice,
the goal blocks stay at their distinct initial positions. Each helper's first
Move starts at the common initial height, so neither goal block lies strictly
above it. The goal therefore remains false. Empty-hand Moves, Picks and Releases
do not change that conclusion.

Dynamic programming counts the lexicographic prefix while tracking the held
block, whether the goal can already be achieved, each helper's Move count, and
whether both goal blocks occur. Once the goal can be achieved it can be retained
through any further holding-consistent actions. This preserves the full original
operand enumeration and counts every candidate separately.

The classification was checked against **1,249 calls to the production
`check_bmc_candidate`**, all matching: 546 holding-valid core instruction patterns
through length five, 448 Move-operand variants, 250 deterministic random full-
alphabet programs, and five targeted cases including the demo. An independent
direct enumeration also agreed with the counts for every length through three.
The DP calculation took 0.0154 seconds; validation and the independent small-space
counts took 20.355 seconds. Raw counts are in [cem-eligibility.json](cem-eligibility.json).
The one-off script is `/tmp/count_standalone_bmc_passes.py`.
