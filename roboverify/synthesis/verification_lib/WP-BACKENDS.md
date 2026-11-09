# Selectable WP and symbolic verification backends

The CFG and counterexample-learning CLIs accept `--wp-system old|new|reduction`.
The default is `old`. The selection follows inference, rebuilt finite/unbounded
verification contexts, and the Put transformer used by motion effect checks.
It does not change executable controllers or demonstration fingerprints.

| Selection | Put rules | Symbolic domain and solver |
| --- | --- | --- |
| `old` | Existing guarded, detach-before-attach WP | Existing relational axioms and quantified Z3 queries. |
| `new` | Shared height representative around the whole postcondition | Existing relational axioms plus total height order and representative coverage; optional support completeness. |
| `reduction` | Same shared representative WP | Compile a supported query into a finite Boolean encoding of complete, separated current/frozen tower scenes. No unbounded background scene axioms are asserted. |

Both new choices require the `Higher` block vocabulary. `EqualHeight` and goal
predicates remain supported only by `old`. Assignment, sequence, Get and loop
rules retain their existing behavior. `new` supports quantified guards such as
Unstack's guard; it does not promise EPR membership or solver termination.

## Selecting a backend

Append one of these options to the existing
[supplied-program verification command](../cfg/VERIFICATION.md#provided-stack-verification)
or [counterexample-learning command](../experiment/invariant_learning/README.md):

```text
--wp-system old
--wp-system new
--wp-system new --wp-support-complete
--wp-system reduction
```

The optional `--wp-support-complete` flag is accepted only with `new`. It adds C1
below; it is off by default. `reduction` already encodes complete towers.

Python callers can select a context directly, or use a scope for a pipeline
that constructs its own contexts:

```python
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.wp_options import using_wp_system

context = HighLevelContext(use_tbl=True, wp_system="new", support_complete=True)
solver = context.new_solver(timeout_ms=10000)

with using_wp_system("reduction", reduction_max_slots=64):
    reduced_context = HighLevelContext(use_tbl=True)
    # Construct and verify programs within this scope.
```

Always obtain the solver through `context.new_solver()`. The reduction context
rejects `add_axiom()` because that interface cannot implement model reduction.
CLI run configuration records the selected options. Scope exit restores the
previous configuration, including on exceptions.

## Relational axioms for `new`

Write `R = ON_star`, `R0 = ON_star_zero`, `H = Higher`, `S = Scattered`,
and `P(x) = x != tbl`. Without a table marker, every object is physical.
All variables below range over physical blocks unless table behavior is stated
explicitly. The abbreviations introduce no new relation or function symbols:

```text
Strict(x,y) = H(x,y) and not H(y,x)
Clear(x)   = forall t. R(t,x) => t=x
Maximum(x) = forall t. H(x,t)
Bottom(u)  = forall t. H(t,u)
NextRep(b,u) = Strict(u,b)
               and forall t. Strict(t,b) => H(t,u)
```

The following existing assumptions are retained:

| Family | Axioms |
| --- | --- |
| Current order | `R(x,x)`; `R(x,y) and R(y,z) => R(x,z)`; `R(x,y) and R(y,x) => x=y`. |
| Columns | `R(x,y) and R(x,z) => R(y,z) or R(z,y)`; `R(x,z) and R(y,z) => R(x,y) or R(y,x)`. |
| Frozen order | The same five order/column axioms for `R0`, independently of `R`. |
| Height | `H(x,x)`; `H(x,y) and H(y,z) => H(x,z)`. The old common-upper-bound comparability clause is also retained, now redundant under totality. |
| Height links | `R(x,y) => H(x,y)`; `R(x,y) and x!=y => not H(y,x)`. |
| Separation | `S(x,y) = S(y,x)`; `not S(x,x)`; `R(x,y) => (S(x,z) = S(y,z))`. |
| Optional existing top assumption | With `exists_top=True`, `forall x. exists t. R(t,x) and Clear(t)`, and the corresponding frozen formula. This option is independent of `wp_support_complete`. |

For `R`, `R0` and `H`, any relation involving the table is true exactly when both
operands are the table. `S` is false whenever either operand is the table.
No global `R0 = R` axiom is added: tasks may equate them in an entry condition.

The added assumptions are:

```text
H3: forall x,y. H(x,y) or H(y,x)
W1: (exists x. P(x)) => exists u. P(u) and Bottom(u)
W2: forall b. Maximum(b) or exists u. NextRep(b,u)

C1 (optional): forall x,y. H(x,y) =>
                exists z. R(x,z) and H(z,y) and H(y,z)
```

W1 explicitly permits an empty physical domain when the table is the only
object. W2, C1 and their existential witnesses are restricted to physical
objects. Totality makes `not H(b,u)` imply `H(u,b)` on physical objects; the
implementation retains both terms in `Strict` to keep its meaning explicit.

These are succinct relational assumptions, not an exact characterization of
finite complete towers. In particular, separation is not required between all
distinct columns. C1 strengthens support but does not make this axiom system
identical to the reduction's model. SAT results can therefore differ between
`new` and `reduction`. W1/W2 and optional top existence/C1 contain quantifier
alternation; no EPR claim is made for this background theory.

## Shared Put transformer

Public `wp(Put(a,b), Q)` retains the source/target legality condition. A source
must be a clear physical block; a block target must be distinct and clear.
The source is detached before attachment. Table aliases dispatch by value.

For a physical block target the new effect is:

```text
(Maximum(b) => Qmax)
and forall u. (P(u) and NextRep(b,u)) => Qu
```

For a table target it is:

```text
forall u. (P(u) and Bottom(u)) => Qtable_u
```

Each `Q...` is a simultaneous atom substitution in the **whole** postcondition,
including beneath negation and quantifiers. The same representative is used by
every rewritten atom. Legality is conjoined outside these effects by the public
WP. `put_effect` supplies just the effects for motion checking.

For physical operands, `R` uses detach/attach, `S` inherits the target column
(or separates the source on table placement), and `H(a,y)` uses `H(u,y)` for
`y!=a`. The maximum case uses `True` for that row. The incoming height column
`H(x,a)`, for `x!=a`, becomes `Strict(x,b)` for a block target and `True` for a
table target. The diagonal stays reflexive. Table pairs, equality and `R0` are
framed. If Q has no H atoms, the representative-independent rewrite omits the
redundant selector quantifiers; representative coverage justifies this step.

See [shared_wp.py](../api/shared_wp.py) and the
[Lean shared-witness rules](../../proofs/wp_soundness/README.md#shared-height-witnesses).
The whole WP can still contain quantifiers. Moving selectors around Q is what
enables the restricted AER result; it does not make every arbitrary Python VC
an EPR formula.

## Direct model reduction

For an unbounded `DeclareSort` context, the compiler normalizes the **entire
solver query**, including supplied premises and the negated VC, into a
function-free `exists* forall*` formula. It accepts `R`, `R0`, `H`, `S`, object
equality, Boolean connectives and object quantifiers. Independent quantified
branches can be reordered safely; a remaining universal above an existential
is rejected. Scene reflexivity/table identities and logical simplification may
first eliminate irrelevant clauses. This is a conservative fragment recognizer.

Let k be the number of remaining free object constants other than `tbl`, plus
the normalized existential variables. The
[proved small-model bound](../../proofs/wp_soundness/README.md#exact-validity-over-the-block-model)
is N = k(k+1) physical blocks. A context without a table adds one seed to ensure
its physical sort is nonempty. The compiler then encodes all scene sizes up to
N, symbolically, using Boolean choices for:

1. Active objects and name/witness bindings; names may alias.
2. Current and frozen roots and natural height levels on the same object set.
3. A unique object at each occupied root/level, with every lower level present.
4. The exact relation tables and universal instances of the query.

Current H compares levels, R also requires the same root, and S means different
roots. Frozen R0 uses a separately chosen complete scene. The table remains
isolated. There are no additional integer or object-valued function symbols in
the query sent to Z3: levels and roots are finite Boolean circuits. The result
is propositional, hence a special case of EPR. This compresses the finite scene
choices in Lean's explicit `GroundEPR` enumeration; it does not enumerate a
user-chosen block bound or solve the query first and replace it by True/False.

SAT is decoded into a native finite relational model for existing counterexample
consumers. UNSAT excludes every scene allowed by the formula-derived bound.
The small-model result is the reason this can establish unbounded validity for
the supported fragment. Premise consistency is checked separately; valid,
invalid, vacuous and unknown remain distinct.

For an `EnumSort` context, the compiler instead encodes exactly the declared
domain size, supports arbitrary object-quantifier alternation by finite
expansion, and reports a result for that size only. The CFG's separate unbounded
check remains required for an unbounded proof.

### Guard scope and limits

The checked restricted VC theorem uses the adjusted Put rules, AF
specifications/invariants and selection guards `exists v. G(v)` with QF G.
The Python compiler checks the resulting query's fragment directly; it does not
certify that a program was produced by that restricted Lean generator.

In particular, an Unstack guard with a quantified matrix remains supported by
`new`. Its negation in an exit VC can introduce `forall exists`, so the unbounded
reduction may return `UNKNOWN`. No guard is silently rewritten or weakened.
RelationalSummary actions (quantified successor relations), EqualHeight,
goal predicates, arithmetic and additional predicates/functions are outside
the reduction's vocabulary. Summary construction rejects this mode explicitly;
unsupported solver queries return `UNKNOWN` with a reason.

`--reduction-max-slots` (default 64) and `--reduction-max-instances` (default
200000) limit compilation effort. The latter bounds universal expansion and
formula-construction work. If the derived bound or required work exceeds a
limit, the answer is `UNKNOWN`, **never UNSAT at a smaller bound**. The symbolic
query timeout includes reduction preprocessing, construction, solving and SAT
model reconstruction. Expensive formulas may remain unknown even within these
caps. Increasing a cap does not change the modeled block counts or semantics.

The reduction does not transform motion's geometric/arithmetic obligations
into EPR. Motion still checks the selected WP effects using its existing
geometric solver and explicit premises.

## Validation and proof boundary

`test_wp_backends` checks all three flags, scope restoration, finite/unbounded
context propagation, the new axiom consequences, quantified guards and illegal
actions. It independently compares every legal relocation in every three-block
tower scene with the new predicate effects, including table aliases and frozen
geometry. `test_scene_reduction` exhausts all two-block R/H/S relation tables,
checks current/frozen scene pairs, normalization and binding scope, model
decoding, unbounded placement/frame VCs, and failure on unsupported queries or
insufficient budgets.

The small-model and WP theorems are kernel-checked in Lean. The Python
transformer, normalization and Boolean compiler are tested implementations;
there is no mechanically checked Python-to-Lean correspondence. These changes
do not establish a new full Stack/Unstack/Reverse synthesis or motion result.
Existing recorded benchmark results used the default `old` backend.
