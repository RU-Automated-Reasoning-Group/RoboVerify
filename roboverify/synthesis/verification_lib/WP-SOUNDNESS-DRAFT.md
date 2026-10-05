# Draft proof of weakest precondition soundness for tower operations

This is an initial mathematical proof for review. It establishes exact atomic
rewrites in a deliberately restricted relational model, then gives a conditional
partial correctness theorem for programs. The proof has not been checked in a
proof assistant. It does not establish that every call accepted by the current
implementation meets the theorem's hypotheses.

The main distinction is between attaching a singleton block and relocating a
top block that may already have a support. The current block-placement rewrite
describes the former. A final corollary specifies the latter by deletion followed
by attachment. No physical visit to the table is required by that corollary.

The proof follows the semantic strategy of Theorem 1 in
[the linked-list technical report](../../../cav2013tr.pdf), Appendix A.1:
define execution independently, derive exact relation updates, and lift those
updates to assertions. The formulas here follow the corrected implementation,
not the uncorrected Higher rules in Table 7 of
[the RoboVerify paper](../../../POPL2027.pdf).

## Model and scope

Fix an arbitrary finite nonempty set of physical blocks \(B\). The assertion
domain is \(D=B\cup\{\mathit{tbl}\}\), where the table is a distinguished object,
not a vertex in a block support chain. The argument is uniform in \(|B|\).

A support state is a map

\[
p:B\longrightarrow B\cup\{\bot\}
\]

whose edges \(x\to p(x)\), when \(p(x)\ne\bot\), form disjoint finite chains.
Thus every block has at most one direct support, every block supports at most
one block directly, and there are no cycles. The marker \(\bot\) denotes the
absence of a block support; it is not the relational table object.

Let \(r(x)\) be the root of \(x\)'s chain, and define the level recursively:

\[
\ell(x)=
\begin{cases}
0 & p(x)=\bot,\\
1+\ell(p(x)) & p(x)\in B.
\end{cases}
\]

Levels are nonnegative integers. Every block of positive level has a support
at the preceding level. This complete support structure is an assumption of
the model, not a consequence of the currently encoded height inequalities.

A state \(\sigma\) also contains a valuation \(\eta\) of mutable program names into \(D\)
and a frozen entry support state \(p_0\) on the same blocks. Placement changes
only \(p\); assignment changes only \(\eta\); \(p_0\) remains fixed.
The distinguished constant \(\mathit{tbl}\) is immutable. At invocation entry
we may require \(p_0=p\); the local lemmas also cover later states where they differ.

For physical blocks, interpret the assertion vocabulary as

\[
\begin{aligned}
R(x,y)&\equiv ON^*(x,y)\equiv x\text{ reaches }y\text{ through zero or more }p
\text{ edges},\\
H(x,y)&\equiv Higher(x,y)\equiv \ell(x)\ge\ell(y),\\
S(x,y)&\equiv Scattered(x,y)\equiv r(x)\ne r(y).
\end{aligned}
\]

Interpret \(R_0=ON^*_0\) by reachability in \(p_0\). Equality is object identity.
For pairs involving the table, \(R\), \(R_0\), and \(H\) are true exactly for
\((\mathit{tbl},\mathit{tbl})\); \(S\) is always false.

The definition of \(S\) restricts this draft to separated towers: different
chains are separated columns. It is stronger than merely requiring the
abstract Scattered axioms. It is not a claim that every pair of distinct towers
in a simulator is geometrically Scattered.

Assertions are first-order formulas over this vocabulary and named objects,
with Boolean connectives and quantifiers over the fixed domain \(D\). No
restriction to AFR is needed for semantic soundness. Decidability of the
resulting verification conditions is a separate question.

All formulas below refer to the initial state unless marked with a prime.
Quantification over \(B\) abbreviates quantification over \(D\) restricted by
\(x\ne\mathit{tbl}\). We use the following mathematical abbreviations; they
do not add predicates to the implementation's vocabulary:

\[
\begin{aligned}
Clear(a)&\equiv a\in B\land\forall x\in B.\ R(x,a)\Rightarrow x=a,\\
Singleton(a)&\equiv Clear(a)\land
                  \forall x\in B.\ R(a,x)\Rightarrow x=a.
\end{aligned}
\]

Different program names may alias. The side conditions below constrain their
denoted objects, not their spellings.

## Primitive semantics and their side conditions

We distinguish two existing forms of Put and a proposed derived operation.

**Attach(a,b)** denotes the current block-target Put rule, under the condition

\[
E_{\rm attach}(\sigma,a,b)\equiv
Singleton(a)\land Clear(b)\land a\ne b.
\]

Execution sets \(p'(a)=b\), leaves \(p'(x)=p(x)\) for \(x\ne a\), and preserves
the valuation and frozen state.

**Separate(a)** denotes table-target Put under the condition

\[
E_{\rm separate}(\sigma,a)\equiv Clear(a).
\]

Execution sets \(p'(a)=\bot\), leaves other support edges unchanged, and preserves
the valuation and frozen state. Its relational meaning includes separation from
all other towers. Availability of an actual sufficiently separated table
location is outside this finite graph model.

An operation whose side condition is false is invalid; it is not a blocking
assume and does not silently succeed. Atomic equivalence below is conditional
on legality. A verifier claiming safety must establish legality at every call.

**Lemma 1 (model preservation).** Legal Attach and Separate preserve the class
of support states. They leave the root and level of every block other than
\(a\) unchanged. Attach gives \(\ell'(a)=\ell(b)+1\) and \(r'(a)=r(b)\);
Separate gives \(\ell'(a)=0\) and \(r'(a)=a\).

**Proof.** For Attach, the source is an entire singleton chain and the target
has no block above it. Adding the edge joins that singleton to the top of a
different chain, creating neither a branch nor a cycle. For Separate, no other
block reaches \(a\), so deleting its support edge detaches only \(a\).
Remaining chains keep their supports and roots. The stated levels and roots
follow from their definitions. The domain, valuation, and entry state do not
change. This also covers Separate applied to an existing singleton. \(\square\)

## Exact reachability updates

**Lemma 2 (attachment).** Under \(E_{\rm attach}\), for all \(u,v\in D\),

\[
R'(u,v)\iff R(u,v)\lor\bigl(R(u,a)\land R(b,v)\bigr). \tag{1}
\]

**Proof.** No old support edge is removed because \(p(a)=\bot\). Every old path
therefore survives. A path using the new edge consists of an old path to \(a\),
the edge \(a\to b\), and an old path from \(b\). Conversely, those pieces form
a new path. Paths avoiding the new edge are old paths. The table is isolated:
if either argument is the table, the added disjunct is false, and its old
interpretation is preserved. \(\square\)

The guard \(\neg R(b,a)\) conjoined by the current block-target WP is true
under \(E_{\rm attach}\): a distinct block cannot reach a singleton source.
This guard alone does not imply the full side condition.

**Lemma 3 (separation).** Under \(E_{\rm separate}\), for all \(u,v\in D\),

\[
R'(u,v)\iff
R(u,v)\land\bigl(\neg R(u,a)\lor R(v,a)\bigr). \tag{2}
\]

**Proof.** If \(u\ne a\), then \(R(u,a)\) is false by Clear(a) and table
isolation; no path from \(u\) uses the removed edge, so its row is unchanged.
If \(u=a\), the right side becomes \(R(a,v)\land R(v,a)\), equivalent to
\(v=a\) by acyclicity. After separation \(a\) reaches only itself. These cases
exhaust \(D\), including the table. \(\square\)

## Exact height updates

This section uses levels derived from support edges. It does not define a
continuous robot trajectory or assume arbitrary real-valued heights.

**Lemma 4 (attachment heights).** Under \(E_{\rm attach}\), the Higher update
is given by the following exhaustive cases. Here \(c\in B\setminus\{a,b\}\).

| Arguments | Value after attachment |
| --- | --- |
| \(m,n\ne a\) | \(H(m,n)\) |
| \(a,a\) | true |
| \(a,b\) | true |
| \(b,a\) | false |
| \(c,a\) | \(H(c,b)\land\neg H(b,c)\) |
| \(a,c\) | \(H(b,c)\lor\bigl(H(c,b)\land U(c,b)\bigr)\) |

The first row includes table pairs not involving \(a\). The additional pairs
\((a,\mathit{tbl})\) and \((\mathit{tbl},a)\) are false, where

\[
U(c,b)\equiv
\forall t\in B.\quad
\bigl(H(c,t)\land\neg H(t,c)\bigr)\Rightarrow H(b,t). \tag{3}
\]

**Proof.** Lemma 1 gives \(k'_a=k_b+1\), writing \(k_x=\ell(x)\), and leaves
other levels unchanged. This proves the frame, reflexive, and \(a,b\) cases.
For \(c,a\), integrality yields

\[
k_c\ge k_b+1
\iff k_c>k_b
\iff H(c,b)\land\neg H(b,c). \tag{4}
\]

For \(a,c\), the desired condition is \(k_c\le k_b+1\).
If \(k_c\le k_b\), the first disjunct \(H(b,c)\) holds.
If \(k_c=k_b+1\), every strictly lower level than \(k_c\) is at most \(k_b\),
so both \(H(c,b)\) and \(U(c,b)\) hold.
If \(k_c\ge k_b+2\), the direct support \(t=p(c)\) exists and has
\(k_t=k_c-1>k_b\). It witnesses the failure of \(U(c,b)\), and \(H(b,c)\)
is false. These three cases are exhaustive and establish equivalence.

Table cases follow from the isolated interpretation. In the implementation's
quantification over \(D\), the additional witness \(t=\mathit{tbl}\) makes the
antecedent of (3) false and does not affect the universal. When \(c\) itself
is the table, both \(H(b,c)\) and \(H(c,b)\) are false. \(\square\)

The existence of \(p(c)\) at exactly the preceding level is essential.
Ordering axioms or lower bounds on support spacing alone do not establish
that fact.

**Lemma 5 (separation heights).** Under \(E_{\rm separate}\), for physical
\(c\ne a\),

\[
\begin{aligned}
H'(a,a)&=\mathit{true},\\
H'(c,a)&=\mathit{true},\\
H'(a,c)&\iff\forall t\in B.\ H(t,c),\\
H'(m,n)&\iff H(m,n)\quad(m,n\ne a).
\end{aligned} \tag{5}
\]

Pairs involving \(a\) and the table are false.

**Proof.** The new level of \(a\) is zero, all other levels are unchanged,
and all levels are nonnegative. Thus \(H'(a,c)\) means \(k_c=0\).
The initial finite nonempty forest has a root of level zero. Consequently,

\[
\forall t\in B.\ k_t\ge k_c
\iff k_c=0.
\]

The other physical cases follow immediately. The implementation restricts
the universal to non-table objects. If its second argument is the table,
any physical block refutes that universal, so the mixed pair is false.
The other mixed direction is explicitly excluded. \(\square\)

## Exact separation and frame updates

**Lemma 6 (Scattered).** Under the corresponding legality condition,
attachment has, for physical \(c\ne a\),

\[
S'(a,c)\iff S(b,c),\qquad
S'(c,a)\iff S(c,b), \tag{6}
\]

where \(c=b\) is allowed. Separation has

\[
S'(a,c)=S'(c,a)=\mathit{true}\quad(c\in B,\ c\ne a). \tag{7}
\]

In both operations \(S'(a,a)\) is false, pairs not involving \(a\) are
unchanged, and pairs involving the table are false.

**Proof.** For attachment, Lemma 1 gives \(r'(a)=r(b)\) while preserving every
other root. Thus \(r'(a)\ne r'(c)\) exactly when \(r(b)\ne r(c)\).
For separation, \(a\) becomes a singleton root. No other block has root \(a\):
otherwise that block would have been above \(a\), contradicting Clear(a).
The remaining cases follow from root inequality and table isolation.
\(\square\)

**Lemma 7 (unchanged symbols).** Placement preserves equality and the
interpretation of \(R_0\), as well as all program-name values.

**Proof.** The identity of blocks, the domain, the valuation, and \(p_0\)
are unchanged. In particular, the entry relation is not recomputed from the
new support state. \(\square\)

## Lifting atomic identities to assertions

Let \(\rho_A(Q)\) replace every occurrence of a current-state relation in \(Q\)
by its corresponding formula from Lemmas 2 through 6, using initial-state
relations on the right sides. Equality and \(R_0\) are left unchanged.
Substitution is simultaneous in this semantic definition and capture avoiding;
replacement formulas are not recursively rewritten as if they were new
postconditions.

The three implementation traversals can be compared with this definition:
each replacement uses only its own current relation, equality, and fresh
bound variables. A later traversal for a different relation therefore does
not reinterpret an earlier replacement. This observation does not by itself
verify the Python traversal implementation.

**Theorem 1 (exact atomic weakest preconditions).** For every support state
\(\sigma\), every legal operation \(A\in\{Attach,Separate\}\), every assertion
\(Q\) in the language above, and every valuation of its logical free variables,

\[
\sigma\models\rho_A(Q)
\iff
\operatorname{exec}(A,\sigma)\models Q. \tag{8}
\]

On these legal states, the mathematical formulas implemented by the current
Put rewrites have the same property, including the attachment guard.

**Proof.** Fix \(\sigma\) and a legal operation, and put
\(\sigma'=\operatorname{exec}(A,\sigma)\). Induct on the syntax of \(Q\).

For relation atoms use Lemmas 2 through 7; equality uses the unchanged
valuation. For negation, the induction hypothesis gives equivalence of the
truth values before negation, hence equivalence afterward. Conjunction and
disjunction follow component by component. Other Boolean connectives may
be reduced to these.

For \(Q=\forall x.\ F\), the domain \(D\) is unchanged. For each \(d\in D\),
apply the induction hypothesis to \(F\) under the logical valuation \(x=d\).
The resulting equivalences hold for every \(d\), establishing the universal
case. The existential case uses the same argument for a witnessing \(d\).
Fresh binders in replacement formulas prevent variable capture; bound
variables are distinct from mutable program-name symbols.

The current attachment guard is true on legal states by the observation after
Lemma 2, and therefore does not change the equivalence. \(\square\)

The forward direction of (8) is soundness; the reverse is completeness on the
specified legal domain. Define a guarded atomic transformer

\[
W_A(Q)\equiv E_A\land\rho_A(Q). \tag{9}
\]

On model states, (9) is true exactly when the operation is legal and its result
satisfies \(Q\). The implemented Put WP does not include all of \(E_A\).
Consequently (9) is a proof specification, not a claim about its current
behavior on illegal inputs.

## Relocating a top block by deletion and attachment

Define a proposed general relocation operation by

\[
Relocate(a,b)\equiv Separate(a);Attach(a,b). \tag{10}
\]

Its precondition requires \(Clear(a)\), \(b\in B\), \(a\ne b\), and that \(b\)
be clear in the state after Separate. Checking the target afterward also allows
putting a block back onto its original direct support.

**Corollary 1 (relocation).** Under this precondition, (10) has the same final
support state as replacing \(p(a)\) directly by \(b\). Its assertion transformer
is

\[
W_{\rm relocate}(Q)=W_{\rm separate}
                         \bigl(W_{\rm attach}(Q)\bigr). \tag{11}
\]

**Proof.** Separate changes only \(p(a)\) to \(\bot\), making \(a\) a singleton.
Attach is then legal by the stated condition and changes only \(p(a)\) to \(b\).
All other support values, bindings, and frozen state agree with direct
replacement. Since the relations are independently defined from those data,
all final predicates agree. Apply Theorem 1 to the two steps in execution order,
or equivalently apply their transformers in reverse order. \(\square\)

For a clear source, the resulting reachability update simplifies to

\[
R'(u,v)\iff
(u\ne a\land R(u,v))
\lor
\bigl(u=a\land(v=a\lor R(b,v))\bigr). \tag{12}
\]

Rows other than \(a\)'s are unchanged, and \(a\)'s row is replaced by itself
and the target's downward chain. This formula removes old support links.

Equation (10) is an abstract decomposition. It establishes equivalence of
final relational states, not equivalence of intermediate robot poses or
collision behavior. Here it covers Higher and Scattered as well as ON*
because both final support states are identical. A geometric implementation
must establish that combined endpoint contract through its own motion proof.

The singleton restriction is necessary for the current addition formula.
For example, initially let a be above d and let b be a separate singleton.
The source a is clear, and the current acyclicity guard holds. Formula (1)
retains R(a,d), while direct relocation removes it. This is outside the
hypotheses of Theorem 1 and is not repaired by the source being clear.

The current block-target Put WP uses the addition formula (1), not the
general relocation transformer (11) or (12). This draft proposes no code change.
The distinction is the existing
[attachment model limit](../../../PAPER-DISCREPANCIES.md#30-stack-invariant-vocabulary-and-attachment-semantics).

## Composition and loop verification

The following extension states the logical obligations needed for a program
theorem. It does not assert that the current verifier generates every legality
obligation in (9).

For Skip, use \(W(Q)=Q\). For program-name assignment \(x:=y\), use
\(W(Q)=Q[y/x]\). The ordinary substitution lemma is exact because assignment
updates the name valuation and preserves both support states.

For an executable Get that binds a tuple of distinct names \(\bar x\) to any
tuple satisfying \(G\), use capture-avoiding binding. Distinct names may bind
the same object unless the guard excludes that choice:

\[
W_{\rm Get}(Q)=
(\exists\bar x.\ G)\land
\forall\bar x.\bigl(G\Rightarrow Q\bigr). \tag{13}
\]

The first conjunct excludes failure from having no witness; the second covers
every permitted witness. This is exact for arbitrary-witness semantics. It is
sufficient, potentially stronger than weakest, for the runtime's particular
lowest-ID selection policy.

For sequence, set

\[
W_{C_1;C_2}(Q)=W_{C_1}(W_{C_2}(Q)).
\]

For finite loop-free programs these rules describe legal execution and
correctness of every permitted outcome. The proof is induction on program
structure using Theorem 1, the substitution lemma, (13), and composition.
Model preservation is supplied by Lemma 1.

For an annotated loop, define \(W_{\rm while}(Q)=I\), where \(I\) is its
supplied invariant, and let \(G_{\exists}\equiv\exists\bar x.\ G(\bar x)\).
Require all recursive body obligations and

\[
\begin{aligned}
P&\Rightarrow I,\\
I\land G(\bar x)&\Rightarrow W_{\rm body}(I)
                   &&\text{for every binding }\bar x,\\
I\land\neg G_{\exists}&\Rightarrow Q.
\end{aligned} \tag{14}
\]

Here witness names are local to the guard/body and do not occur freely in \(I\)
or the outside continuation. More general scoping requires an explicit
condition that rebinding witnesses preserves the invariant. This draft does
not audit that condition for every generated program.

To make the obligation set precise, let \(V_C(Q)\) be true for atomic
commands, Skip, assignment, and Get, and define

\[
\begin{aligned}
V_{C_1;C_2}(Q)&=V_{C_1}(W_{C_2}(Q))\land V_{C_2}(Q),\\
V_{\rm while}(Q)&=V_{\rm body}(I)\land
  \bigl(\forall\bar x.\ I\land G(\bar x)\Rightarrow W_{\rm body}(I)\bigr)
  \land\bigl(I\land\neg G_{\exists}\Rightarrow Q\bigr),\\
VC(P,C,Q)&=(P\Rightarrow W_C(Q))\land V_C(Q).
\end{aligned} \tag{15}
\]

Validity here means truth on every model state and every permitted valuation
of the remaining free names and logical variables. Primitive legality is
included through the guarded transformers (9), not assumed implicitly.

**Theorem 2 (conditional program partial correctness).** For the stated
language and scoping convention, if \(VC(P,C,Q)\) is valid in this sense, then
execution of \(C\) from any model state satisfying \(P\) cannot encounter a
modeled illegal operation or failed Get, and every normally terminating
execution satisfies \(Q\).

**Proof.** The loop-free cases follow from the preceding exactness result.
At a loop, establishment gives \(I\). If a witness exists, universal
preservation applies to whichever witness is chosen. Inductively, every
finite completed iteration preserves \(I\), and recursive obligations protect
execution inside the body. If no witness exists, the exit obligation yields
\(Q\). Apply structural induction for surrounding composition and any nested
annotated bodies. An infinite execution need not produce a final state;
nothing here proves termination. \(\square\)

The theorem does not claim completeness for arbitrary supplied invariants.
A learned invariant is only a candidate until the obligations are established.

## Compatibility with solver axioms and geometry

The model satisfies the tower axioms used by the symbolic context.
Reachability is reflexive, transitive, and antisymmetric. Blocks above or below
a common block lie in one chain and are comparable. Finiteness supplies a
clear top above each block. Integer level ordering satisfies Higher's ordering
axioms; a strict support path strictly increases level. Blocks in one chain
have the same root and therefore the same Scattered relationships. The
frozen relation satisfies the corresponding reachability axioms independently.
The table conventions satisfy the isolation axioms.

These are one-way model inclusion claims: every state in this draft satisfies
the relevant solver axioms. The reverse is not required for soundness and is
not claimed. Abstract models may have incomparable heights or lack complete
layers. A valid implication over all axiom models applies to the draft's
states, but atomic exactness has only been proved on the draft's states.

A geometric interpretation of this model can assign each chain one XY column,
separate distinct columns by at least \(2L\) in one horizontal coordinate,
and set \(z(x)=h_0+L\ell(x)\) for a common bottom-center height \(h_0\).
Then the repository's numeric ON*, Higher, and Scattered definitions agree
with the relations above. For exact grid heights, a Higher tolerance
\(0\le\varepsilon<L\) preserves the same integer ordering. The configured
tolerance lies in that interval for the standard block size.

This is an interpretation of exact states, not a proof of tolerance to
arbitrary placement error. Complete support, exact columns, and a common
height grid are stronger than arbitrary simulator scenes and stronger than
the current supported-tower height inequalities. Controller refinement,
collision freedom, finite table capacity, and availability of safe motions
remain separate obligations.

## Validation of this draft

A bounded review check enumerated all 591 labeled chain forests on one through
five physical blocks. It loaded the six existing predicate rewrite helper
bodies from program.py and evaluated their generated formulas over each finite
domain, including the table and every ordered argument pair. Expected
relations were independently recomputed from the support map after mutation.

All 778 legal attachments and 1,207 legal separations agreed with those
relations. The composed relocation transformer and the simplified formula
(12) were also checked on all 274 legal relocations through four blocks.
Across these checks, all 224,106 ON*, Higher, and Scattered comparisons agreed.

These checks exercise the actual predicate helper bodies but use a separate
finite formula evaluator. They do not verify the full WP dispatcher, quantifier
hygiene on arbitrary input formulas, program VCs, or arbitrary block counts.
They are bounded supporting evidence, not a mechanization of either theorem.

An independent Z3 query over unbounded nonnegative integer levels also found
no disagreement in (4), after separately confirming satisfiable premises.
Replacing the integers by arbitrary nonnegative reals produced the expected
counterexample: base height 0 and other height 1/2, with unit block height.
This checks one height identity and demonstrates why its domain restriction
matters; it is not an SMT proof of all the lemmas.

## Correspondence and review boundaries

| Mathematical component | Implementation reference | Scope of this draft |
| --- | --- | --- |
| Equations (1) and (2) | ON_star rewrites in [program.py](../api/program.py) | Exact under the respective primitive side conditions |
| Lemmas 4 and 5 | Higher rewrites in [program.py](../api/program.py) | Exact on complete support chains with integer levels |
| Lemma 6 | Scattered rewrites in [program.py](../api/program.py) | Exact for separated columns represented by distinct chains |
| Formula induction | Rewriters in [program.py](../api/program.py) and hygiene helpers in [symbols.py](../util/symbols.py) | Mathematical proof; Python implementation is not mechanically verified |
| Get and loop obligations | WP and VC generation in [program.py](../api/program.py) | Requires the stated binding scope and additional legality obligations |
| Relocation | Current attachment rewrite versus (11) | Proposed derived semantics; general relocation is not implemented by the current attachment rewrite |
| Geometry | [Motion verification guide](README.md) | Separate refinement task; model-level proof only |

The first review should settle these points:

1. Is attaching a singleton the intended primitive, or should block-target Put
   denote general relocation? The latter requires the composed symbolic update.
2. Is the complete, separated-chain model an acceptable initial theorem domain?
   If a broader domain is intended, the height and separation lemmas must be
   revisited.
3. Where will each program establish the primitive legality conditions? The
   current attachment guard alone does not establish them.
4. Are the witness-scope restrictions satisfied by the programs to which
   Theorem 2 will be applied?
5. Should the next formalization establish the remaining unbounded SMT lemmas,
   stronger implementation correspondence, or the formula induction in a proof
   assistant?

Existing findings remain open where recorded:
[Higher assumptions](../../../PAPER-DISCREPANCIES.md#16-the-higher-rewrite-can-disagree-with-geometric-placement),
[termination](../../../PAPER-DISCREPANCIES.md#15-termination-is-not-established-by-the-stated-vcs-runtime-cap-mismatch-fixed),
[Higher tolerance](../../../PAPER-DISCREPANCIES.md#32-higher-tolerance-for-contact-induced-height-differences),
and [Scattered placement error](../../../PAPER-DISCREPANCIES.md#33-scattereds-sharp-xy-boundary-exposes-held-block-placement-error).
This draft does not close those findings or change verification status.
