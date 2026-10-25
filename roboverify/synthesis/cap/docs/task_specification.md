# Task specification: block counts and pre/postconditions

- **Purpose:** record the number of blocks and the pre/postconditions of each task, so that CaP experiments over the six tasks ([Revision 5](revision_5_add_tasks.md)) use the same scale as RoboVerify's reported experiments.
- **Source:** [POPL2027.pdf](../../../../POPL2027.pdf). Line numbers below are the paper's margin numbers. The paper is an artifact under test, not a specification (repository [AGENTS.md](../../../../AGENTS.md)); this document records what it shows, not a requirement.
- **Recorded:** 2026-10-25. Counts were checked against the rendered Fig. 10 panels (PDF page 36).

## 1. Block counts

| Task | Blocks | Initial scene in the paper | Evidence |
| --- | ---: | --- | --- |
| Stack | 4 | 4 scattered blocks | Fig. 10a shows 4 blocks. Fig. 7 (line 938) illustrates a 4-block loop-head state: a tower x₁, x₂, x₃ with base b₀ = x₁ plus a block x₄ on the table. §5.7 (lines 1659–1665): after the loop is found, "an execution on n = 4 blocks" is added to D_V for invariant inference. |
| Unstack | 4 | One tower of 4 | Fig. 10b |
| Reverse | 4 | One tower of 4 | Fig. 10c |
| ReStack (PartialStack) | 5 | One tower plus loose blocks; Fig. 10d draws a tower of 3 and 2 loose blocks | Fig. 10d is captioned "PartialStack"; the text (line 1752) and the Fig. 11 panel call the same task ReStack. The text gives no tower height; 3 is counted from the drawing only. |
| Grid | 6 | 6 scattered blocks; 6 goal markers | Fig. 10e (brown dots are goal markers, not collision objects; lines 1755–1758) |
| Pyramid | 6 | 6 scattered blocks; 6 goal markers for a 3 + 2 + 1 pyramid | Fig. 10f |

### 1.1 Cross-check with Tables 2 and 3

- **Table 2** (lines 1814–1817) gives the number of counterexample-guided learning iterations, not block counts:

  | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 4 | 5 | 5 | 5 | 7 | 7 |

- **Table 3** (lines 1819–1830) lists the Stack iterations. The counterexample in iteration *i* has *i* blocks (1, 2, 3), and the last iteration has none ("N/A"). With that reading, the largest counterexample has (Table 2 − 1) blocks:
  - **Unstack, Reverse: 4.** Matches Fig. 10b/c.
  - **Grid, Pyramid: 6.** Matches Fig. 10e/f.
  - **Stack: 3.** Fig. 10a and §5.7 show 4. The Table 2 count reflects the counterexample sequence of the RQ2 experiment, not the scene size.
  - **ReStack: 4.** Fig. 10d shows 5.
- This mapping is an inference from Table 3, which covers only Stack. Use Fig. 10 as the primary source for scene size.

### 1.2 Other experiment scale in the paper

- **Demonstrations:** 15 per environment for RQ1 training (line 1833).
- **Seeds:** RoboVerify and all baselines are trained in each environment with 5 random seeds (line 1762).

## 2. Preconditions and postconditions

Two sources:

- **Paper:** Table 8, Appendix B (lines 2164–2186). The notation m⟨on*⟩n is written ON*(m, n) here.
- **Code:** the shared [`task_spec`](../../cfg/tasks.py) used by RoboVerify collection and verification, and by CaP's `task_spec_success`.
  - **Domain:** Unstack, Reverse and ReStack quantify over the physical blocks plus the relational table `tbl`. Stack quantifies over physical blocks only.
  - **Predicates:** evaluated on simulator coordinates by `util/on.py`. Higher has a 1 mm tolerance.
  - **Grid and Pyramid:** no formula. They use the numeric goal-occupancy check `goal-occupancy-v1` (`task_identity`, [`goal_collection.py`](../../cfg/goal_collection.py)).

| Task | Source | Precondition φ_pre | Postcondition φ_post |
| --- | --- | --- | --- |
| Stack | Paper | ∀m,n. (m = n ∨ ¬ON*(n, m)) ∧ ∀m,n. Higher(n, m) ∧ ∀m,n. (m ≠ n ⇒ Scattered(m, n)) | ∀m. ON*(m, b0) |
| | Code | ∀x,y. (x ≠ y ⇒ ¬ON*(x, y) ∧ Scattered(x, y)) ∧ ∀x,y. Higher(x, y) | ∀x. ON*(x, b0) |
| Unstack | Paper | ∀m. ON*(m, b0) | ∀m,n. (m = n ∨ ¬ON*(m, n)) |
| | Code | ∀x. (x = tbl ∨ ON*(x, b0)) | ∀x,y. (ON*(x, y) ⇒ x = y) |
| Reverse | Paper | ∀m. ON*(m, b0) | ∀m,n. (ON*(m, n) ⇔ ON*₀(n, m)) |
| | Code | ∀x ≠ tbl. ON*(x, b0) ∧ ∀x,y. (ON*₀(x, y) ⇔ ON*(x, y)) | ∀x,y ≠ tbl. (ON*₀(x, y) ⇔ ON*(y, x)) |
| ReStack | Paper | ∀m,n. ((m ≠ n ∧ ON*(m, n)) ⇒ Scattered(m, n)) | ∀m. ON*(m, b0) |
| | Code | b0 ≠ tbl ∧ ∀x,y ≠ tbl. (ON*(x, y) ∨ ON*(y, x) ∨ Scattered(x, y)) | ∀x ≠ tbl. ON*(x, b0) |
| Grid | Paper | ∀g : Goal. (reach(h0, x) ⇒ ¬HasBlock(g)) ∧ ∀m,n. Scattered(m, n) | ∀g : Goal. (reach(h0, x) ⇒ HasBlock(g)) |
| | Code | Every goal empty; physical blocks pairwise Scattered; one block per goal | Every goal occupied by a physical block (centre within 20 mm, checked in 3D) |
| Pyramid | Paper | ∀g : Goal. (reach(h0, g) ⇒ ¬HasBlock(g)) ∧ ∀m,n. Scattered(m, n) | ∀g : Goal. (reach(h0, g) ⇒ HasBlock(g)) |
| | Code | Same as Grid | Same as Grid |

Symbols:

- **ON*₀** is ON* on the frozen entry geometry (`ON_star_zero`), used by Reverse.
- **h0** is the head goal; reach(x, y) ≜ ∃z. x⟨next_c⟩z ∧ z⟨next_r⟩y (Appendix B).
- **HasBlock(g)** holds when a block is at g's location.

Where the code departs from Table 8:

- **Stack:** equivalent. The code states both ordered pairs of the same conditions; Higher uses a 1 mm tolerance ([discrepancies 31–32](../../../../PAPER-DISCREPANCIES.md#31-stack-resets-equal-height-assumption-belongs-in-the-task-precondition)).
- **Unstack and Reverse:** the domain includes `tbl`.
  - The physical quantifiers exclude `tbl`, or the formula admits it explicitly (Unstack's `x = tbl`).
  - Reverse's precondition also fixes ON*₀ = ON* at entry, which ties the frozen geometry to the start state.
- **ReStack:**
  - **Precondition.** Read literally, Table 8's precondition contradicts the intended start: ON* requires XY alignment and Scattered requires XY separation, so no two distinct blocks could be stacked. The code instead requires every pair of physical blocks to be tower-related or Scattered ([discrepancy 41](../../../../PAPER-DISCREPANCIES.md#41-restack-needs-a-terminating-first-loop-and-an-executable-lateral-layout)).
  - **b0 inside the source tower.** CaP's start check additionally requires this (Revision 5, §4.1).
- **Grid and Pyramid:**
  - Table 8's Grid row writes `reach(h0, x)` where `g` is bound; this reads as a typo for `reach(h0, g)`.
  - Literal `∀m,n. Scattered(m, n)` includes m = n, which no block satisfies; the code requires distinct blocks to be Scattered.
  - Goal semantics in the code are numeric occupancy, not a verified formula ([discrepancy 40](../../../../PAPER-DISCREPANCIES.md#40-grid-and-pyramid-execution-and-verification-have-separate-support)).

## 3. Settings for CaP experiments

To match the paper's scale, use these task settings ([`RunConfig`](../config.py) / `run_cap` options; the resolvers are the upstream ones, see [Revision 5, §4.1](revision_5_add_tasks.md#41-shared-changes)):

| Task | `num_blocks` | Other options | Repository default (`collect_demos`) |
| --- | ---: | --- | --- |
| Stack | 4 | — | 3 |
| Unstack | 4 | — | 3 |
| Reverse | 4 | — (the reference program supports 2–4 blocks) | 3 |
| ReStack | 5 | `initial_tower_blocks: 3` (as drawn in Fig. 10d; not stated in the text) | 4 blocks, tower `max(2, n − 1)` |
| Grid | 6 | `grid_rows`, `grid_cols` with rows × cols = 6 | 3 × 2 |
| Pyramid | 6 | `bottom_row_blocks: 3` (layers 3, 2, 1) | `bottom_row_blocks: 3` |

- **Repository defaults differ from the paper for four tasks.** Stack, Unstack and Reverse default to 3 blocks, and ReStack to 4. Set the counts explicitly in every experiment configuration and in RoboVerify demonstration collection, so that both systems use the same scenes.
- **Revision 4.x CaP Stack results used 3 blocks** (for example the [Revision 4.3 report](revision_4_3_report.md#11-configuration)). They are not at the paper's Stack scale.
- **Grid orientation is not fixed by the figure.** Fig. 10e shows 6 goals, but which side is "rows" depends on the camera. The upstream default 3 × 2 also has 6 goals. Choose one layout, record it, and use it for both systems.
- **ReStack tower height.** The only evidence is the 3-block tower drawn in Fig. 10d; the text says only that the start "may contain existing towers" with b0 possibly inside one (lines 1752–1754). 3 equals the resolver default `max(2, n − 1)` only for n = 4. For n = 5, set `initial_tower_blocks: 3` explicitly; the default would give 4. Confirm with the authors before treating it as the paper's setting.
