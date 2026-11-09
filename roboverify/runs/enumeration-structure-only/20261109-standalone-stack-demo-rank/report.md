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
before `solver.check()`. These counts describe checker invocations, not the
number of SAT candidates or CEM calls. The latter was not calculated.

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
