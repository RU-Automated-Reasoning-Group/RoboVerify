# Revision 4.3: ablation experiment plan

- **Status:** design agreed except the open points in Section 10. The experiment starts after the [prompt factorial](revision_4_3_prompt_ablation_plan.md) has finished and its selected prompt set is recorded in the [report](revision_4_3_report.md); every Stage A and Stage B job uses that prompt set.
- **Model:** Qwen3.8 27B only (Section 6). The Qwen3.5 9B was dropped on 2026-10-22.
- **Prompt set:** `p0001`, i.e. only `explain_motion_planning_limits` on (`--prompt-flags 0001`), selected by the prompt factorial ([report, Section 1.5.4](revision_4_3_report.md#154-selected-prompt-set)).
- **Implementation it evaluates:** [Revision 4.3: configurable feedback and full executed-policy history](revision_4_3_configurable_feedback_and_full_policy_history.md).
- **Question:** which recovery-feedback items, and which history mode, change the chance that a failed policy is repaired?

## 1. Why the Revision 4.2 design cannot answer this

- **Feedback acts only after a failed rollout.** In independent trials, every trial that succeeds in round 0 is identical across feedback profiles but still enters the denominator.
  - Example: OpenRouter Qwen3.8 27B succeeded initially in 12, 16 and 17 of 20 trials. Only 3–8 trials per group could be affected by feedback.
- **Final success is near the ceiling.** Many groups finish 18–20/20, so differences are a few trials.
- **Approximate significance tests (not run formally):**

  | Comparison | Fisher two-sided p (approx.) |
  | --- | --- |
  | 18/20 vs 20/20 | 0.49 |
  | 14/20 vs 20/20 | 0.02; unlikely to survive correction across dozens of comparisons |
  | Haiku initial nonconvergence 1/20 vs 10/20 | 0.003 |

  The prompt effect is supported. For the feedback enrichments, Revision 4.2 can state only "no difference detected", not "no difference".
- **More trials alone are inefficient.** Per-group sample size for two independent proportions, 80% power, α = 0.05:

  | Difference to detect | Trials per group |
  | --- | ---: |
  | 90% vs 100% | ~70 |
  | 85% vs 95% | ~140 |
  | 90% vs 95% | ~400+ |

- **The shared 8,192-byte cap confounded the comparison.** Enabled items displaced each other, so an enabled item was sometimes not delivered.

## 2. Design: shared round 0, branched recovery

**Decided.** Remove the largest source of variance, "which program round 0 produced", by pairing every profile on the same failed program.

1. **Stage A: failure pool.**
   - For each method, generate fresh round-0 attempts from the saved initial state.
   - The selected prompt set is used; the initial prompt is identical across profiles.
   - Keep every attempt; classify each failed one (Section 3).
2. **Stage B: branched recovery.**
   - Select failed round-0 attempts from the pool, stratified by failure class.
   - Run recovery rounds 1–4 once under **every** profile from the same frozen round-0 attempt: same source, bindings, trajectory and initial state.
   - Feedback for round 1 is built from that frozen attempt under the profile being tested.
   - Collection is identical across profiles; only the rendered request differs.
- **Unit of analysis:** one failed round-0 attempt, the "seed". Each seed yields one recovery chain per profile.
- **Byproducts:**
  - Stage A also measures initial success with prompts on.
  - Stage A attempts that succeed or stop in preparation are reported but not branched. A preparation stop has no trajectory to give feedback on.
- **Implementation needed:**
  - A seed loader that starts `run_recovery` from an archived round-0 attempt instead of generating one.
  - The seed's recorded usage is charged to every branch's trial ledger, so budgets stay identical.
  - Seed identity and hash are recorded in every branch.
  - An audit checks that all branches of a seed share byte-identical round-0 source, bindings and initial-state hash.

## 3. Failure-class stratification

**Decided:** test recovery on different failure types, not only primitive nonconvergence.

| Class | Source label |
| --- | --- |
| Move nonconvergence | `primitive_failed`, Move |
| Pick nonconvergence | `primitive_failed`, Pick |
| Release nonconvergence | `primitive_failed`, Release |
| No-witness `get` | Query failure |
| Execution/query budget | `budget_exhausted` |
| Goal failure | Normal completion, goal false |
| Other runtime error | NameError, TypeError, argument errors |

- **Only natural model-generated failures are used.** Scripted failures would be a different population.
- **Rare classes are allowed to fall short of their quota.** For example, no-witness `get` occurred in only 4 of 325 Revision 4 transitions. Report actual counts per class rather than filling them artificially.
- **Selection rule:**
  - Deterministic: seeded random sampling within each class.
  - Fixed before Stage B starts.
  - Recorded in an immutable seed manifest.

## 4. Profiles

**Decided:**
- The primitive prompt switches are fixed in every profile at the set selected by the [prompt ablation](revision_4_3_prompt_ablation_plan.md), which runs first. They are not feedback.
- `scene_metadata` and `dsl_diagnostics` are always on and not ablated:
  - scene metadata is the core handle-to-block mapping;
  - DSL diagnostics are too rare to test (17 of 485 Revision 4.2 feedbacks).
- `event_counts` and `selection_metadata` are tested as one "metadata" group.
- `query_candidates` is implemented independently of `failure_events`, but is kept inside the failure-events group for this experiment because no-witness failures are rare.

| # | Profile | Change from `full` |
| --- | --- | --- |
| 1 | `full` | Reference: every item on, three primitive enrichments on, history `executed_programs` |
| 2 | `no_current_source` | Current executed source off, and history falls back to `summaries`. `executed_programs` requires the current source, so this profile removes the latest program and all earlier programs together; earlier failures remain as one-line summaries. |
| 3 | `no_failure_location` | Failure location off |
| 4 | `no_failure_events` | Failure events and query candidates off |
| 5 | `no_keyframes` | Keyframes off |
| 6 | `no_goal_clauses` | Goal clauses off |
| 7 | `no_supporting_events` | Supporting events off |
| 8 | `no_metadata` | Event counts and selection metadata off |
| 9 | `no_target` | Target derivation off |
| 10 | `no_phases` | Controller phases off |
| 11 | `no_context` | Action context off |
| 12 | `history_summaries` | History `summaries` |
| 13 | `history_none` | History `none` |
| 14 | `minimal` | Anchor: only the always-on core, the current source, and history `none` |
| 15 | `legacy_v1` | Anchor: Revision 4 items with summaries, primitive enrichments off. Same item set as Revision 4.2 `only_prompt`, but sized under `fixed_items` (Section 5), so it is not byte-identical to that cohort. |

- **Leave-one-out misses redundancy.**
  - If two items carry overlapping information, removing either one alone can show no effect, e.g. failure events vs target derivation, keyframes vs action context.
  - The `minimal` anchor measures the total value of feedback. Large `full`–`minimal` differences with flat leave-one-out results indicate redundancy.
- **Dependencies:**
  - History is one three-level factor: `none` / `summaries` / `executed_programs`. Edit ledger and diff exist only under `executed_programs`.
  - `executed_programs` requires the current source.
  - Invalid combinations are rejected by configuration validation, never run.

## 5. Evidence limits

**Decided:** the byte cap must not decide whether an enabled item is delivered.

- **Decided:** every profile, including `legacy_v1`, uses the `fixed_items` evidence layout. Within-experiment fairness takes priority over byte identity with Revision 4.2.

- **Each item's content is set by its own count limit, not by leftover space:**
  - supporting events: most recent N;
  - action context: at most 4 actions;
  - goal clauses: at most 16;
  - keyframes: at most 3;
  - query candidates: at most 16.
- **The total evidence cap becomes a safety guard sized above the measured maximum.**

  Measured maxima over all 485 ready Revision 4.2 feedbacks:

  | Item | Max bytes |
  | --- | ---: |
  | Supporting events, last 80 | 25,195 |
  | Supporting events, last 40 | 12,800 |
  | Primitive enrichments (action context as delivered under the cap) | ~6.4 KB total |
  | Failure events | 2,711 |
  | Keyframes | 953 |
  | Everything else combined | < 2 KB |

  With every item on, the worst case is about 26 KB at N = 40 and about 38 KB at N = 80. Proposed guard: 32 KiB or 48 KiB respectively.
- **Delivery audit.**
  - Every request records, per enabled item, whether it was delivered complete.
  - Any incomplete delivery is a reported protocol violation, not a silent omission.
- **History** keeps its own 16,384-byte allowance; all 379 measured historical requests fit.
- **Token budgets scale with the larger requests.**
  - Inputs grow from about 4k to about 15–25k tokens, which could otherwise create new budget stops.
  - Proposed: 200k tokens per recovery round and 900k per trial; call and time limits unchanged.
  - Record the values; any token-budget stop is reported per profile.

## 6. Model and local serving

**Decided (2026-10-22):** one local model, Qwen3.8 27B, served by vLLM on dbar.

| Model | Checkpoint (pinned revision) | Precision | GPUs |
| --- | --- | --- | --- |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` | Official FP8, weight-only Marlin kernels on A100 | 6 and 7 (`GPU-4e45a91e…`, `GPU-9e9f9166…`), one instance per GPU behind a local least-in-flight balancer |

- **Why the 9B was dropped.** In the prompt factorial (stopped after 291 completed trials), the Qwen3.5 9B succeeded in round 0 in 1 of 291 trials and recovered 20 of 290 initially failed trials (about 7%).
  - At that recovery rate, 50 seeds per method give about 3 recoveries per profile, so no profile difference is detectable.
  - 96 of its 291 trials stopped with `preparation_failure_no_rollout`: 32k-token length stops, SyntaxErrors and similar errors.
  - The Qwen3.8 27B recovered 57–71% of its initially failed trials, far from both floor and ceiling.
- **Consequence:** results are for one model. Cross-model generality is not claimed.
- **A second model may be added later,** only after a short pilot (for example 30 trials per method) shows round-0 success of about 20–60% and recovery away from the floor. It would form a separate stratum with its own seeds.

- **Server:**
  - vLLM 0.32.0 in a separate virtual environment; OpenAI-compatible endpoint on localhost only, protected by an API key stored outside the repository.
  - `--max-model-len 98304`.
  - `--gpu-memory-utilization 0.85`, leaving room for MuJoCo EGL workers.
  - Prefix caching on.
  - Qwen reasoning parser.
  - Image/video inputs disabled.
- **Generation:**
  - Thinking enabled through the chat template, with `reasoning_effort=medium`. The template default, `xhigh`, exhausted the output cap without code in every smoke round 0 ([report](revision_4_3_report.md#12-pre-run-findings)).
  - 32,768 output tokens; no thinking-token budget.
  - Sampling from the checkpoint's generation config; resolved values recorded.
- **Client changes:** a local provider mode with no request pacing or free-quota admission. vLLM version, checkpoint revision and sampling parameters are recorded in every call's provenance.
- **Comparability:** local results form new cohorts. They are not comparable with the OpenRouter Qwen3.8 27B batch, which differs in serving precision, sampling defaults and reasoning control.

## 7. Outcomes and analysis

- **Primary outcome:** recovered within four rounds, per seed × profile.
- **Secondary outcomes:**
  - rounds to recovery;
  - input/output tokens and calls;
  - repeated or reintroduced defects;
  - new stops (preparation, budget, history size);
  - physical, logical and DSL outcomes, reported separately.
- **Analysis:**
  - Paired comparisons of each profile against `full` on the same seeds: McNemar's test, with Holm correction across the 14 contrasts.
  - Pooled model: mixed-effects logistic regression `recovered ~ profile + method + failure_class + (1 | seed)`, rather than dozens of separate 2×2 tests.
  - Per-class results are descriptive unless a class has enough seeds.
- **Claims of "no effect" need an equivalence test (TOST)** with a margin fixed before the run. Non-significance alone is reported as "not detected".
- **Precision guide.**
  - Power in a paired design depends on discordant pairs.
  - With roughly 20% discordance, about 160 seeds detect a 10-point pooled difference.
  - With 70 seeds per method, every profile has 210 seeds pooled over 3 methods, enough for a 10-point pooled difference at roughly 20% discordance. Per method, 70 seeds detect differences of roughly 20 points or more.
  - The count was raised from 50 after the 9B was dropped (decided 2026-10-23).

## 8. Decided details

| Item | Decision |
| --- | --- |
| Seeds | 70 failed round-0 attempts per method (Qwen3.8 27B), 210 in total; 15 profiles give 3,150 Stage B chains. Seeds are method-specific. (Raised from 50 on 2026-10-23.) |
| Class quotas | The 50-seed quotas (Move 15, Pick 8, Release 8, execution/query budget 6, no-witness `get` up to 5, goal failure 4, other runtime 4), scaled to 70 by largest remainder: Move 21, Pick 11, Release 11, budget 8, no-witness up to 7, goal failure 6, other runtime 6 (`select-seeds --seeds-per-method 70`). Shortfalls are refilled proportionally from the other classes; actual counts are reported. |
| Stage A limit | At most 300 round-0 attempts per method, stopping early once quotas are met. The 50 round-0 attempts per method from the selected prompt-factorial cell `p0001` are imported first; with `p0001` failure rates (Move 8.8%, Pick 5.8%, Release 4.6% of attempts), about 250 new attempts per method are needed. Pilot attempts are not used. |
| Replicates | One recovery chain per seed × profile |
| Equivalence margin | 10 percentage points, for TOST claims pooled over methods only |
| Token budgets | 200k per recovery round, 900k per trial; call and time limits unchanged |
| Pilot | Stage A 30 round-0 attempts per method, then Stage B 3 seeds per method under all 15 profiles (135 chains). Pilot data are excluded from the main analysis. The seeded-branch validation ([report §2.2](revision_4_3_report.md#22-seeded-branch-validation-2026-10-22)) already checked the mechanics with 9B seeds. |
| Sampling | Checkpoint `generation_config` defaults; resolved values recorded per call |
| Output cap | 32,768 tokens per request, doubled from 16,384. The server context length must cover the largest input plus 32,768. |

## 9. Launcher implementation

[revision43_feedback_experiment.py](../experiments/revision43_feedback_experiment.py) implements the design; no live run has been made with it.

| Subcommand | Behavior |
| --- | --- |
| `stage-a` | Round-0-only batch (recovery on, `max_rounds=0`) for the model and all three methods, with a fixed prompt set (`--prompt-flags 1010` or JSON) |
| `select-seeds` | Reads round-0 attempts from Stage A or existing batches (`--source BATCH[:VARIANT]`, read only), filters by model and prompt set, classifies them (Section 3; preparation failures counted and excluded), samples per method with seeded quotas and proportional refill, and writes a read-only manifest plus its SHA-256 |
| `stage-b` | One job per seed × profile (15 profiles, `fixed_items`, supporting events `--supporting-events 40`, guard `--evidence-guard 32768`); every job starts from its seed through the Revision 4.3 seed loader |
| `audit` | Checks that all branches of a seed share one round-0 attempt and initial state, that profile settings match, and that every enabled item was delivered complete |
| `summarize` | Per-profile recovery with paired exact McNemar tests against `full`, counts by failure class, tokens and stop reasons |

- **`no_current_source` uses history `summaries`.** `executed_programs` requires the current source, so this profile removes the latest source and the earlier programs together; earlier failures remain as summaries.
- **Stage A early stopping** is done with incremental Stage A batches rather than an in-batch gate. `select-seeds` pools every source and reports per-class availability and shortfalls.
- **Seeds produced by the prompt-factorial runs** come from the Revision 4.2 runtime (pinned worktree). Stage B replays them with the Revision 4.3 code. The 4.3 seed loader and byte-identity replay already cover Revision 4.2 artifacts.

## 10. Open points

1. **Supporting events:** N = 40 with a 32 KiB guard (proposed), or N = 80 with a 48 KiB guard.
2. ~~Seed count with one model~~ — decided 2026-10-23: 70 per method (Section 8).
