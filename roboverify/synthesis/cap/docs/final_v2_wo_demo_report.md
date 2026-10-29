# Final CaP results without demonstrations: `final_v2_wo_demo`, Claude Opus 5.5, 50 trials per task

- **Date:** 2026-10-28; ReStack seed 3 rerun on 2026-10-29 with the raised analysis budget (Section 4.2).
- **Run:** `runs/cap-final-v2-wo-demo/20261028-final-50/` (seeds 0–49 for every task). Its `report.md` lists every trial and round with its env steps.
- **Configuration:** [`config_final_v2_wo_demo`](../configs/config_final_v2_wo_demo.py). This is [`config_final_v2`](../configs/config_final_v2.py) with the demonstration block off; nothing else changes. The experiment's static DSL analysis budget is 1,000,000 steps, by the user's decision after the run (Section 4.2).
- **Comparison arm:** the demonstrations-on run of the [final_v1 report](final_v1_report.md) (`config_final_v2`, the same seeds; seed 50 replaces seed 39 for Unstack and Reverse there).
- **Tables, data and figures:** `python -m synthesis.cap.reports.final_report` (Section 9). Partial reward follows [TASK_PARTIAL_REWARD.md](TASK_PARTIAL_REWARD.md).

## 1. Summary

- **Final success:**
  - 300/300 trials succeeded under the frozen criterion `task-spec-v2+final-release-v1`, within at most four recovery rounds. Every task is at 50/50.
  - Under the checker's original 100,000-step analysis budget, ReStack seed 3 ended without a rollout because the analysis of its first program did not finish. The user raised the budget to 1,000,000 for the whole experiment. This changes no other analysis in the run, and the rerun of that trial succeeded in round 2 (Section 4.2).
- **Round 0:** removing the demonstrations lowers first-program success on every tower task and on Pyramid:

  | | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | No demonstrations | 35/50 | 23/50 | 21/50 | 11/50 | 50/50 | 43/50 |
  | Demonstrations (final_v1) | 50/50 | 50/50 | 23/50 | 34/50 | 50/50 | 50/50 |

  Recovery makes up the difference: mean recovery rounds rise from 0–0.78 to 0–1.38 per task.
- **Env steps:** reaching the successful program takes more environment steps:
  - 1.8× as many for Stack, 1.6× for Unstack, 1.5× for ReStack;
  - 1.4× for Pyramid, 1.15× for Grid, 1.06× for Reverse.

  Section 3.2 has the tables, and Fig. 11-style curves are in Section 3.3.
- **Failures:**
  - The dominant failure is `Move` non-convergence: 90 of the 117 failed round-0 attempts across tasks.
  - ReStack also exhausts the execution/query tick budget in 17 round-0 attempts.
  - No attempt failed only on the release check.
- **Cost:**
  - The 300 scored trials cost $57.08, of which ReStack accounts for $26.98.
  - Without the demonstration block, prompts are short, so the cache saves less: 42% of prompt tokens were read from cache, against 95% with demonstrations. Output tokens dominate the cost.
  - The key's total was $58.27, including requeued trials and the two new calls of the ReStack seed 3 rerun ($0.33). The smoke cost $1.01 on the earlier key.
- **Run interruptions and reruns:** the run was stopped twice by request (Section 6). The 13 trials interrupted at those stops were rerun with their original seeds. No trial was dropped. One further trial, ReStack seed 3, was rerun after the analysis budget change; it is the only trial whose result depends on that budget (Sections 4.2 and 6.4).

## 2. Configuration

| Setting | Value |
| --- | --- |
| Model | Claude Opus 5.5 (`anthropic/claude-opus-5.5`) on OpenRouter, reasoning effort medium, 64k output tokens; routing Anthropic, then Claude Platform on AWS (477 of 478 scored calls served by Anthropic) |
| Method | `cap_method_2`, `dsl_only`, `typed_v5`, primitive prompt `primitive-prompt-v2` (`p0101`) |
| Instructions | `task-instructions-v3`, as in final_v1 |
| Demonstrations | **Off.** No demonstration block, and no measurement of one |
| Recovery | On, up to 4 rounds, feedback profile `full`, RDP keyframes; success field `task_spec_success` |
| Success criterion | `task-spec-v2+final-release-v1` |
| Budgets per trial | 10,000 execution/query ticks, 10,000 simulation steps, 200,000 effort tokens, 12 model calls, 3,600 s |
| Static DSL analysis | Work budget 1,000,000 steps per analysis (`RunConfig.max_analysis_steps`; checker default `MAX_ANALYSIS_STEPS = 100000`). The run used the default; ReStack seed 3, the only trial affected, was rerun with 1,000,000. No other analysed program in the run needs more than 73,810 steps (Section 4.2) |
| Blocks | Stack, Unstack, Reverse 4; ReStack 5 (3-block source tower); Grid 2 × 3; Pyramid 3-2-1 |
| Scenes | Fresh resets with seeds 0–49, settled for 50 steps. With no demonstrations there is no information-boundary rejection, so seed 39 is included for every task |
| Prompt caching | `prompt_cache: "5m"`, first-response warm-up, as in final_v2; only the round-0 breakpoint exists |

The prompt text is final_v1's demonstrations-off text, byte for byte.

## 3. Task success

![Task success per task](plots/final_v2_wo_demo_success.png)

| Task | Blocks | Trials | Round-0 success | 95% CI (%) | Final success | 95% CI (%) | Mean recovery rounds | Mean calls | Total env steps |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| Stack | 4 | 50 | 35/50 (70%) | [56, 81] | 50/50 (100%) | [93, 100] | 0.30 | 1.32 | 11,238 |
| Unstack | 4 | 50 | 23/50 (46%) | [33, 60] | 50/50 (100%) | [93, 100] | 0.62 | 1.64 | 12,863 |
| Reverse | 4 | 50 | 21/50 (42%) | [29, 56] | 50/50 (100%) | [93, 100] | 0.86 | 1.88 | 16,702 |
| ReStack | 5 | 50 | 11/50 (22%) | [13, 35] | 50/50 (100%) | [93, 100] | 1.38 | 2.46 | 37,626 |
| Grid | 6 | 50 | 50/50 (100%) | [93, 100] | 50/50 (100%) | [93, 100] | 0.00 | 1.10 | 19,172 |
| Pyramid | 6 | 50 | 43/50 (86%) | [74, 93] | 50/50 (100%) | [93, 100] | 0.14 | 1.16 | 21,980 |

"Total env steps" sums the policy's `env.step` calls over every round of every trial. The 50 settling steps before and after each rollout are not included.

### 3.1 Recovery

![Cumulative success by recovery round](plots/final_v2_wo_demo_recovery.png)

| Task | ≤ round 0 | ≤ round 1 | ≤ round 2 | ≤ round 3 | ≤ round 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stack | 35 | 50 | 50 | 50 | 50 |
| Unstack | 23 | 46 | 50 | 50 | 50 |
| Reverse | 21 | 38 | 48 | 50 | 50 |
| ReStack | 11 | 28 | 42 | 50 | 50 |
| Grid | 50 | 50 | 50 | 50 | 50 |
| Pyramid | 43 | 50 | 50 | 50 | 50 |

- **Stack and Pyramid:** one recovery round solves every round-0 failure.
- **Unstack:** needs up to two rounds.
- **Reverse:** needs up to three.
- **ReStack:** needs up to three; 8 trials succeed only in round 3.

### 3.2 Env steps until the final result

The step count is taken until the final program succeeded, or until the trial finally failed. It covers all rounds, with settling excluded.

| Task | Succeeded | Mean | Median | Min | Max | Failed | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stack | 50 | 225 | 201 | 118 | 583 | 0 | – |
| Unstack | 50 | 257 | 236 | 114 | 566 | 0 | – |
| Reverse | 50 | 334 | 299 | 154 | 823 | 0 | – |
| ReStack | 50 | 753 | 688 | 297 | 1,756 | 0 | – |
| Grid | 50 | 383 | 375 | 295 | 563 | 0 | – |
| Pyramid | 50 | 440 | 410 | 285 | 927 | 0 | – |

Compared with the demonstrations-on run (successful trials):

| Task | Mean steps, no demonstrations | Mean steps, demonstrations | Ratio | Median, no demonstrations | Median, demonstrations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stack | 225 | 124 | 1.81 | 201 | 124 |
| Unstack | 257 | 160 | 1.61 | 236 | 159 |
| Reverse | 334 | 316 | 1.06 | 299 | 260 |
| ReStack | 753 | 496 | 1.52 | 688 | 388 |
| Grid | 383 | 332 | 1.15 | 375 | 332 |
| Pyramid | 440 | 318 | 1.38 | 410 | 318 |

Two effects add up here:
- **More failed rounds.** Each failed round adds its rollout's steps.
- **Longer successful programs.** Without demonstrations, successful programs also take more steps: they use higher clearances and extra waypoints. For example, all 50 Grid trials succeed in round 0 in both arms, yet the median Grid trial takes 375 instead of 332 steps.

### 3.3 Partial reward against env steps (Fig. 11 style)

![Partial reward against cumulative env steps](plots/final_v2_wo_demo_partial_reward.png)

How to read the figure:
- Each trial contributes a step function. Its value is the partial reward of its most recent rollout, at that rollout's cumulative env-step count: 0 before the first rollout, then held after the trial stops.
- The solid curve is the mean over the task's 50 trials, and the band is ± one standard deviation.
- Orange is the demonstrations-on run; blue is this run.
- Partial reward follows [`partial-reward-v1`](TASK_PARTIAL_REWARD.md), scored on the settled state of each rollout.

Mean partial reward after each round (trials that already stopped keep their last value):

| Task | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stack | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 |
| Unstack | 0.83 | 0.95 | 1.00 | 1.00 | 1.00 |
| Reverse | 0.59 | 0.85 | 0.99 | 1.00 | 1.00 |
| ReStack | 0.62 | 0.86 | 0.97 | 1.00 | 1.00 |
| Grid | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Pyramid | 0.87 | 1.00 | 1.00 | 1.00 | 1.00 |

- **Unstack round 0:** partial reward (0.83) is far higher than round-0 success (46%). Many failed Unstack attempts had already placed most blocks on the table when a later `Move` failed to converge.
- **ReStack:** the mean reaches 1.00 in round 3, when the last trial succeeds.

## 4. Failures

![Round-0 outcome per task](plots/final_v2_wo_demo_round0_outcomes.png)

### 4.1 Failed attempts

Round-0 attempts that did not reach `task_spec_success`:

| Task | `Move` non-convergence | `Release` non-convergence | Tick budget exhausted | Other | Total |
| --- | ---: | ---: | ---: | --- | ---: |
| Stack | 14 | 0 | 1 | | 15 |
| Unstack | 27 | 0 | 0 | | 27 |
| Reverse | 24 | 2 | 1 | 1 `goal_failed` (completed, postcondition false); 1 other primitive failure | 29 |
| ReStack | 19 | 0 | 17 | 2 completed with the postcondition false; 1 `get` without a witness | 39 |
| Grid | 0 | 0 | 0 | | 0 |
| Pyramid | 6 | 0 | 1 | | 7 |

- **`Move` non-convergence:**
  - A primitive did not reach its target within its 50-step budget. This usually happens because the target lies inside another block or the carried block collides.
  - With demonstrations, this was the main failure only for Reverse (23 of 27). Without them, it is the main failure of every task.
  - The demonstrations show the clearances and waypoint heights that the reference programs use.
- **Tick budget:**
  - Execution/query ticks count every query, guard and predicate evaluation as well as every simulation step. The budget is 10,000 per round.
  - The 31 ReStack attempts that exhausted it over all rounds had used only 41–546 simulation steps. Their programs evaluate thousands of quantified conditions, and the budget ran out mid-execution, right after a primitive, rather than in a non-progressing loop.
  - The 41-step attempt is round 0 of the ReStack seed 3 rerun. Its program moves the gripper toward each target in small increments through a helper, so each of its 40 primitives cost about 250 ticks. The other 30 attempts used 296–546 steps.
  - No demonstrations-on attempt hit this budget.
- **Release check:** no attempt failed only because of the final release check. Attempts that ended holding a block (129 over all rounds) had all already stopped on an exception.

### 4.2 Deviation: ReStack seed 3 rerun with a raised analysis budget

**Under the original budget.**
- **Status:** `analysis_failed`; stop reason `preparation_failure_no_rollout`.
- **What happened:** the model's first program had 156 lines, 9 helpers and 3 loops. The static DSL checker's type, provenance and effect analysis exceeded its work budget (`MAX_ANALYSIS_STEPS = 100000`, `validation.py`) before reaching a verdict.
- **Why the trial ended:** an unfinished analysis is neither a violation nor an approval. The program was therefore not executed and not sent to DSL repair, so the trial ended without a rollout and without recovery.
- **Offline check:** the same source was checked again with a budget of 1,000,000. The analysis finished in 0.24 s after 181,327 steps and reported no violations.
- **Comparison:** this did not occur in the 300 demonstrations-on trials. Programs written without demonstrations tend to be longer.

**Decision.** The user raised the analysis budget to 1,000,000 steps for the whole experiment. Only this trial was rerun; the experiment was not repeated.

**Why no other trial is affected.**
- **Deterministic analysis:** the analysis is a deterministic function of the source and the task's DSL contract. The budget only decides whether it finishes, not the verdict it reaches.
- **Recheck:** every analysed source in the run was checked again with the 1,000,000-step budget. That is 481 checks: 476 in the scored trials as they stood before the rerun (one per model call, including the 13 that failed the check) and 5 in requeued trials.
- **No changed verdicts:** the step counts were not recorded during the run, so they were recomputed. Every verdict and diagnostic matched the recorded check report, except for this program (error → pass).
- **Step counts:** no other source needed more than 73,810 steps, which is ReStack seed 29. The maximum per task is:
  - Stack 11,409; Unstack 9,599; Reverse 27,487;
  - ReStack 73,810; Grid 7,690; Pyramid 12,465.

  Only four sources exceeded 50,000 steps, all ReStack: 181,327 (seed 3), 73,810, 65,348 and 61,307.
- **Same checker:** all 481 recorded reports carry the same checker source hash (`38c82d20…`). No analysis came close to the 10-second checker deadline; the longest took 0.24 s.

**Rerun procedure.**
- **Requeue:** the trial was requeued as in Section 6.1:
  - its directory was moved to `replaced/cap_method_2-dsl_only-004-1`;
  - `requeued.json` records the reason and the previous state;
  - its finished video job was moved to `replaced/render_queue/`.
- **Job spec:** the job is the manifest's job with two explicit additions:
  - `run_config.max_analysis_steps: 1000000`;
  - `replay_calls`, the recorded round-0 request and response.
- **Round 0 is not resampled:** the failure was in the checker, not in the model. The worker therefore returns the recorded response only after checking that the new round-0 request is identical to the recorded one, which it was (all 100 recorded ReStack round-0 requests are identical). This round sends no provider request.
- **Same scene:** the seed and the settled initial state are unchanged. `initial_state.npz` is byte-identical to the original trial's.
- **Recovery rounds:** later rounds made live model calls under the unchanged configuration. They used one worker and a GPU from the batch's allowlist (GPUs 6 and 7).
- **Code:**
  - the budget option is `65a6499`;
  - the replay is `6fb6ddd`;
  - the report's handling of replayed calls is `2a0f3c9`.

  The worker ran at `6fb6ddd`. The worktree also held a concurrent session's uncommitted edits: an opt-in relaxed reset for Revision 9, off by default. They did not change this scene, as the identical initial state shows.

**Rerun outcome.**

| Round | Program | Outcome | Env steps |
| --- | --- | --- | ---: |
| 0 | Original response, replayed; analysis passes (1,000,000-step budget) | Tick budget exhausted (10,001 ticks) | 41 |
| 1 | New call | `Move` non-convergence | 169 |
| 2 | New call | Success (`task_spec_success`) | 412 |

- **Totals:** 2 recovery rounds, 3 model calls (1 replayed, 2 new), 68,962 effort tokens and 622 env steps.
- **Cost:** $0.72, including the replayed call's $0.39. The new spend is $0.33.
- **Wall time:** 128 s. This excludes the 205 s that generating the round-0 response originally took.
- **How it is counted:** the rerun replaces the trial, so final success is 50/50. Its round 0 failed, so ReStack's round-0 success is still 11/50. The failure is now a tick-budget exhaustion instead of `analysis_failed`.

## 5. Cost and prompt caching

![Cost per trial](plots/final_v2_wo_demo_cost.png)

![Prompt tokens by cache use](plots/final_v2_wo_demo_cache.png)

| Task | Calls | Prompt tokens | Read from cache | Written | Output tokens | Cost | Mean cost per trial | Mean wall time per trial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stack | 66 | 423,320 | 231,651 (55%) | 7,354 | 170,505 | $4.23 | $0.085 | 44 s |
| Unstack | 82 | 738,787 | 302,965 (41%) | 7,670 | 339,334 | $8.60 | $0.172 | 81 s |
| Reverse | 94 | 948,833 | 360,360 (38%) | 12,012 | 423,332 | $10.90 | $0.218 | 98 s |
| ReStack | 123 | 1,593,410 | 450,660 (28%) | 11,655 | 1,115,202 | $26.98 | $0.540 | 242 s |
| Grid | 55 | 249,910 | 215,424 (86%) | 8,976 | 123,232 | $2.65 | $0.053 | 35 s |
| Pyramid | 58 | 353,408 | 249,040 (70%) | 9,056 | 161,749 | $3.71 | $0.074 | 45 s |
| **Scored trials** | **478** | **4,307,668** | **1,810,100 (42%)** | **56,723** | **2,333,354** | **$57.08** | | |

- **Why caching saves less here:** without demonstrations, the round-0 prompt is only about 3.7–4.5k tokens. The cacheable prefix is therefore small. Recovery requests add long feedback that is new each time, so their uncached share is large.
- **Replayed call:** the scored figures count the rerun's replayed round-0 call once, with the rerun. It was paid once, by the original trial.
- **Output dominates the cost:** 2.3M output tokens at $20/M account for about $47 of the $57.08.
  - ReStack alone produced 1.1M output tokens, about 9k per call.
  - With demonstrations, the same 300 trials produced 1.0M output tokens over all tasks.
- **Rest of the key's spend:** the key's total of $58.27 also covers:
  - the calls of the 13 requeued trials ($0.83);
  - requests still in flight at the two stops.

  It was $57.94 before the rerun; the rerun's two new calls added $0.33.

## 6. Run log and deviations

### 6.1 First stop (requested)

- **Start:** the run started at 2 workers per task (12 concurrent trials).
- **First stop:** after 52 trials had finished (all successful), the run was stopped with SIGINT on request. 11 running trials were marked `interrupted`.
- **Restart:** it was later resumed on request.
- **Requeue:** before the resume, the 11 interrupted trials were requeued with their original seeds:
  - each directory was moved to `<batch>/replaced/`;
  - `<batch>/requeued.json` records the reason and the previous state;
  - their queued video jobs were moved to `replaced/render_queue/`.
- **Selection:** the same procedure as final_v1, Section 6.2. It never looks at whether a trial succeeded.

### 6.2 More ReStack workers

- **Why:** ReStack trials take about four minutes each, so ReStack became the tail of the run.
- **Change:** after Stack, Grid and Pyramid had finished, only the ReStack batch was stopped. Its 2 interrupted trials were requeued in the same way, and the batch was resumed alone with 6 workers.
- **Effect:** total concurrency stayed at about 10. No request returned HTTP 429 or 402 during the run.

### 6.3 Code versions

- **Session commits:**
  - the run was created at `f1c6e1f` (clean);
  - resumed sessions ran at `e8bca1e`;
  - the ReStack 6-worker session ran at `6b386f6`.
- **No runtime difference:** the commits between them change only documentation and the report module. Runtime, prompt and checker code are identical across sessions.
- **Earlier change:** `8cdf4d7`, before the run, makes demonstrations-off trials record their provider cache summary (`prefix_cache`). It does not change requests or execution.

### 6.4 ReStack seed 3 rerun (analysis budget)

- **When:** on 2026-10-29, after the run had finished, the user raised the experiment's analysis budget from 100,000 to 1,000,000 steps.
- **What was rerun:** ReStack seed 3 was the only trial whose result depended on the budget, and only it was rerun (Section 4.2).
- **How:** the requeue record follows Section 6.1. Its reason names the budget change, so the requeue table of the generated report lists it under "Other".
- **Videos:** the batch was resumed afterwards with no pending trials, only to render the rerun's three deferred videos (119 of 119 videos completed).

## 7. Smoke

One trial per task (seed 0) under `runs/cap-final-v2-wo-demo/20261028-smoke/`:
- **Result:** 6/6 succeeded.
- **Rounds:** Stack, Unstack and Reverse needed one recovery round each.
- **Cost:** $1.01.
- **Finding:** the smoke found the missing `prefix_cache` summary for demonstrations-off trials, which was fixed before the run (Section 6.3).

## 8. Interpretation and limits

- **What demonstrations buy:** final success is at or near the ceiling in both arms (300/300 here), so the demonstrations' effect shows in three places:
  - first-program success, for example ReStack 22% against 68% and Unstack 46% against 100%;
  - recovery depth;
  - env steps and cost. ReStack costs $0.54 per trial against $0.25.

  The recovery loop compensates for missing demonstrations through executed feedback. Each failed rollout is an environment interaction that the demonstrations would have avoided.
- **Grid:** the only task unaffected at round 0 (50/50 in both arms). Its reference structure (nested goal loops with `HasBlock`) follows directly from the goal API.
- **Statistics:** at n = 50, each task's 50/50 final success bounds its true rate only at ≥ 93% (Wilson 95%). Round-0 differences between arms are much larger than their intervals for Stack, Unstack, ReStack and Pyramid. For Reverse (42% against 46%) they are not distinguishable.
- **Analysis budget:** the 1,000,000-step budget was set after the run, in response to one failure. It changes no other verdict in this run. A future program could still exhaust it.
- **Scope:** one model, one reasoning effort and one block count per task. Generalization to other block counts is the subject of [Revision 9](revision_9_generalization_checker.md). Whether these programs verify is the subject of [Revision 8](revision_8_verification.md).

## 9. Reproduction

```bash
# from roboverify/, with the simulator environment of AGENTS.md
export OPENROUTER_API_KEY=...   # never on argv; read from the key file
python -m synthesis.cap.experiments.revision6_experiment \
    --root runs/cap-final-v2-wo-demo/20261028-final-50 --model claude-opus-5.5 \
    --frozen final_v2_wo_demo --trials 50 --seed-start 0 --workers 2 \
    --max-analysis-steps 1000000
python -m synthesis.cap.reports.batch_set_report runs/cap-final-v2-wo-demo/20261028-final-50
python -m synthesis.cap.reports.final_report runs/cap-final-v2-wo-demo/20261028-final-50 \
    --output runs/cap-final-v2-wo-demo/20261028-final-50/analysis \
    --plots synthesis/cap/docs/plots --prefix final_v2_wo_demo \
    --label "No demonstrations (final_v2_wo_demo)" \
    --compare "Demonstrations (final_v2)" runs/cap-final-v2/20261027-final-50 \
    --compare "Demonstrations (final_v2)" runs/cap-final-v2/20261027-final-50-supplement
```

- **Analysis budget:** `--max-analysis-steps 1000000` sets the experiment's analysis budget. The recorded run used the default, 100,000, except for the ReStack seed 3 rerun.
- **The rerun:** a one-off driver, not kept in the repository, ran `synthesis.cap.batch_worker` on the job spec of Section 4.2. That spec is `trials/cap_method_2-dsl_only-004/job.json`, with `max_analysis_steps` and `replay_calls`. The driver then wrote the summary and the batch report with `synthesis.cap.batch`'s `collect_job` and `write_report`.
- **Concurrency:** use at most about 2–3 workers per task on one OpenRouter key; a single remaining batch can use more.
- **Generated outputs:** the report module writes `final_v2_wo_demo_data.json`, `final_v2_wo_demo_tables.md` and the comparison arm's progress data to `--output`, and six figures to `docs/plots/final_v2_wo_demo_*.png`.
