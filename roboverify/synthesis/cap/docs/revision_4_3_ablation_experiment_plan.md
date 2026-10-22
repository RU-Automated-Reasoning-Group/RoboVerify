# Revision 4.3: ablation experiment plan

- **Status:** design in progress; no experiment authorized yet. Decisions agreed so far are marked **Decided**. Section 8 lists the open points.
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
   - For each model × method, generate fresh round-0 attempts from the saved initial state.
   - All four primitive prompt switches are on; the initial prompt is identical across profiles.
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
- The four primitive prompt switches are fixed **on** in every profile. They are not feedback, and their effect is already established.
- `scene_metadata` and `dsl_diagnostics` are always on and not ablated:
  - scene metadata is the core handle-to-block mapping;
  - DSL diagnostics are too rare to test (34 of 970 Revision 4.2 feedbacks).
- `event_counts` and `selection_metadata` are tested as one "metadata" group.
- `query_candidates` is implemented independently of `failure_events`, but is kept inside the failure-events group for this experiment because no-witness failures are rare.

| # | Profile | Change from `full` |
| --- | --- | --- |
| 1 | `full` | Reference: every item on, three primitive enrichments on, history `executed_programs` |
| 2 | `no_current_source` | Current executed source off |
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

  Measured maxima over 970 Revision 4.2 feedbacks:

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
- **History** keeps its own 16,384-byte allowance; all 758 measured historical requests fit.
- **Token budgets scale with the larger requests.**
  - Inputs grow from about 4k to about 15–25k tokens, which could otherwise create new budget stops.
  - Proposed: 200k tokens per recovery round and 900k per trial; call and time limits unchanged.
  - Record the values; any token-budget stop is reported per profile.

## 6. Models and local serving

**Decided:** two local models served by vLLM on dbar.

| Model | Checkpoint (pinned revision) | Precision | GPU |
| --- | --- | --- | --- |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` | Official FP8, weight-only Marlin kernels on A100 | 6 (`GPU-4e45a91e…`) |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` @ `c2022362` | Official BF16; no official FP8 exists | 7 (`GPU-9e9f9166…`) |

- **Server:**
  - vLLM 0.32.0 in a separate virtual environment; OpenAI-compatible endpoint on localhost only, protected by an API key stored outside the repository.
  - `--max-model-len 65536`.
  - `--gpu-memory-utilization 0.85`, leaving room for MuJoCo EGL workers.
  - Prefix caching on.
  - Qwen reasoning parser.
  - Image/video inputs disabled.
- **Generation:**
  - Thinking enabled through the chat template.
  - Qwen has no effort levels, so "medium" does not apply.
  - 16,384 output tokens.
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
  - Pooled model: mixed-effects logistic regression `recovered ~ profile + method + model + failure_class + (1 | seed)`, rather than dozens of separate 2×2 tests.
  - Per-class results are descriptive unless a class has enough seeds.
- **Claims of "no effect" need an equivalence test (TOST)** with a margin fixed before the run. Non-significance alone is reported as "not detected".
- **Precision guide.**
  - Power in a paired design depends on discordant pairs.
  - With roughly 20% discordance, about 160 seeds detect a 10-point pooled difference.
  - Pooled over 3 methods × 2 models with 50 seeds each, every profile has 300 seeds. Per model × method stratum, 50 seeds detect only differences of roughly 25 points or more.

## 8. Decided details

| Item | Decision |
| --- | --- |
| Seeds | 50 failed round-0 attempts per model × method. Seeds are method-specific. |
| Class quotas | Move 15, Pick 8, Release 8, execution/query budget 6, no-witness `get` up to 5, goal failure 4, other runtime 4. Shortfalls are refilled proportionally from the other classes; actual counts are reported. |
| Stage A limit | At most 300 round-0 attempts per model × method, stopping early once quotas are met |
| Replicates | One recovery chain per seed × profile |
| Equivalence margin | 10 percentage points, for pooled (all models and methods) TOST claims only |
| Token budgets | 200k per recovery round, 900k per trial; call and time limits unchanged |
| Pilot | Stage A 30 round-0 attempts per model × method, then Stage B 3 seeds per stratum under all 15 profiles (270 chains). Pilot data are excluded from the main analysis. |
| Sampling | Checkpoint `generation_config` defaults; resolved values recorded per call |
| Output cap | 32,768 tokens per request, doubled from 16,384. The server context length must cover the largest input plus 32,768. |

## 9. Open points

1. **Supporting events:** N = 40 with a 32 KiB guard (proposed), or N = 80 with a 48 KiB guard.
2. **Prompt study first:** the four primitive prompt blocks are to be optimized in a separate study before this ablation; this plan then uses the selected prompt set instead of all four.
