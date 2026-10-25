# Task specification: block counts in the paper's experiments

- **Purpose:** record the number of blocks the paper uses for each task, so that CaP experiments over the six tasks ([Revision 5](revision_5_add_tasks.md)) use the same scale as RoboVerify's reported experiments.
- **Source:** [POPL2027.pdf](../../../../POPL2027.pdf). Line numbers below are the paper's margin numbers. The paper is an artifact under test, not a specification (repository [AGENTS.md](../../../../AGENTS.md)); this document records what it shows, not a requirement.
- **Recorded:** 2026-10-25. Counts were checked against the rendered Fig. 10 panels (PDF page 36).

## 1. Block counts

| Task | Blocks | Initial scene in the paper | Evidence |
| --- | ---: | --- | --- |
| Stack | 4 | 4 scattered blocks | Fig. 10a shows 4 blocks. Fig. 7 (line 938) illustrates a 4-block loop-head state: a tower x₁, x₂, x₃ with base b₀ = x₁ plus a block x₄ on the table. §5.7 (lines 1659–1665): after the loop is found, "an execution on n = 4 blocks" is added to D_V for invariant inference. |
| Unstack | 4 | One tower of 4 | Fig. 10b |
| Reverse | 4 | One tower of 4 | Fig. 10c |
| ReStack (PartialStack) | 5 | A tower of 3 plus 2 loose blocks | Fig. 10d, captioned "PartialStack"; the text calls the task ReStack (line 1752) |
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

## 2. Settings for CaP experiments

To match the paper's scale, use these task settings ([`RunConfig`](../config.py) / `run_cap` options; the resolvers are the upstream ones, see [Revision 5, §4.1](revision_5_add_tasks.md#41-shared-changes)):

| Task | `num_blocks` | Other options | Repository default (`collect_demos`) |
| --- | ---: | --- | --- |
| Stack | 4 | — | 3 |
| Unstack | 4 | — | 3 |
| Reverse | 4 | — (the reference program supports 2–4 blocks) | 3 |
| ReStack | 5 | `initial_tower_blocks: 3` | 4 blocks, tower `max(2, n − 1)` |
| Grid | 6 | `grid_rows`, `grid_cols` with rows × cols = 6 | 3 × 2 |
| Pyramid | 6 | `bottom_row_blocks: 3` (layers 3, 2, 1) | `bottom_row_blocks: 3` |

- **Repository defaults differ from the paper for four tasks.** Stack, Unstack and Reverse default to 3 blocks, and ReStack to 4. Set the counts explicitly in every experiment configuration and in RoboVerify demonstration collection, so that both systems use the same scenes.
- **Revision 4.x CaP Stack results used 3 blocks** (for example the [Revision 4.3 report](revision_4_3_report.md#11-configuration)). They are not at the paper's Stack scale.
- **Grid orientation is not fixed by the figure.** Fig. 10e shows 6 goals, but which side is "rows" depends on the camera. The upstream default 3 × 2 also has 6 goals. Choose one layout, record it, and use it for both systems.
- **ReStack tower height.** Fig. 10d has a 3-block tower, which equals the resolver default `max(2, n − 1)` only for n = 4. For n = 5, set `initial_tower_blocks: 3` explicitly; the default would give 4.
