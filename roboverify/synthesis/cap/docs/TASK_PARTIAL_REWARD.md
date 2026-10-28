# Task partial reward (`partial-reward-v1`)

Partial reward measures how much of a task one rollout completed, as a number in [0, 1]. It is used for progress curves in the style of the paper's Fig. 11 (partial reward against environment steps).

- **Reporting only:** no reward reaches the model, the DSL checker or the recovery loop. Task success remains `task_spec_success` (`task-spec-v2+final-release-v1`).
- **Implementation:** [`reports/partial_reward.py`](../reports/partial_reward.py).
- **Tests:** [`tests/test_partial_reward.py`](../tests/test_partial_reward.py).

## 1. Source

The RoboVerify environments report only binary success (`compute_reward` and `_is_success` in `environment/cee_us_env/fpp_construction_env.py`). No per-block reward exists in the code. The paper describes the reward only in prose (Section 6.1, below Fig. 11):

> The partial reward for Stack, Unstack, Reverse are assigned uniformly for each block that is successfully stacked on the base block b0, placed on the table, or reversed, respectively. For ReStack, we assign a reward of 0.5 to each of the unstack and stack subtasks, with the reward further distributed uniformly among the blocks successfully unstacked or stacked. For Grid and Pyramid, partial rewards are assigned uniformly for each block at the goal location.

`partial-reward-v1` makes these sentences concrete. It uses the shared geometric predicates (`util/on.py`) and the goal-occupancy check (`environment/goal_layout.py`) that the success criterion uses. Where the prose leaves a choice open, the choices below are this implementation's, not the paper's.

## 2. Which state is scored

- **The settled state.** A rollout is scored on the state that decides success: after the 50 unrecorded settling steps. Its block positions are the checker's `task_success.positions`, stored in each round's `result.json`.
- **Entry geometry** comes from the round's `initial_state.npz`, together with the base binding `b0`.
- **ReStack** also reads the rollout's recorded states (`execution/rollout_*/trajectory.npz`), because its unstack subtask is judged over the whole rollout (Section 3.4).
- **Rounds without a rollout** have no reward. This covers a round whose program never passed the DSL check and a trial rejected before any request.

## 3. Definitions

Notation:

- N is the number of blocks and b0 the base block given to the policy.
- ON\*(x, y) is `on_star_implementation`: the XY centres lie within half a block of each other, and x is not lower than y.
- below(x) = { y ≠ x : ON\*(x, y) } is the set of blocks under x.
- x **rests on the table** when below(x) is empty and the centre of x is at most half a block above the resting height. The resting height is the lowest block-centre height at entry; every entry scene has a block on the table. The height condition excludes a block held in the air.

### 3.1 Stack

reward = |{ x ≠ b0 : ON\*(x, b0) }| / (N − 1)

The entry scene scores 0, because the precondition requires every pair of blocks to be Scattered.

### 3.2 Unstack

reward = |{ x ≠ b0 : x rests on the table }| / (N − 1)

- b0 starts at the bottom of the tower, on the table, and earns nothing, so the entry scene scores 0.
- A block counts as soon as it rests on the table, even while other blocks are still stacked above it.

### 3.3 Reverse

Order the entry tower from the top, t1 (top) to tN = b0 (bottom). A correct result is a new tower t1, t2, …, tN from the bottom up.

reward = k / N, where k is the largest index such that:

- t1 rests on the table, and
- for every i ≤ k, below(ti) = { t1, …, t(i−1) }.

In words: the new tower counts as built up to block k, in reversed order. A block placed out of order stops the count, even if later blocks are correct. The entry scene scores 0, because t1 starts on top of t2.

### 3.4 ReStack

reward = 0.5 × unstacked share + 0.5 × stacked share

- **Unstack subtask.** U is b0 together with the blocks above it in the entry tower: { x : ON\*(x, b0) at entry }. This is what the supplied two-phase program moves in its first loop (`examples/restack.py`). A block x ∈ U counts as unstacked once some recorded state, or the settled state, has x above none of the blocks it was above at entry (its entry `below(x)`).
  - The share is (unstacked blocks in U) / |U|.
  - The whole rollout is checked because a correctly restacked block above b0 ends on b0 again, so the final state alone cannot show it was ever moved.
- **Stack subtask.** If b0 rests on the table in the settled state, the share is |{ x ≠ b0 : ON\*(x, b0) }| / (N − 1); otherwise it is 0. This keeps the blocks above b0 in the entry tower from earning stack credit at entry.

The entry scene scores 0.

### 3.5 Grid and Pyramid

reward = occupied goals / goals

A goal is occupied when some block centre lies within the 20 mm, 3-D occupancy tolerance of it (`goal_occupied` from the shared checker). This does not depend on block identity, just as the success criterion does not. The precondition requires every goal to be empty at entry.

## 4. Relation to task success

- A successful rollout scores 1.0. This was checked on every successful attempt in the final_v1 run root `runs/cap-final-v2/20261027-final-50`, excluding the seed-50 supplement. Every scored entry scene of the four tower tasks in that root scores 0. Grid and Pyramid entry scenes have no occupied goals by their precondition.
- A score of 1.0 does not imply success. Success also requires:
  - normal completion;
  - every primitive converged;
  - no block touching the gripper at the end.

  For example, two failed Reverse attempts in final_v1 built the reversed tower and then raised `No object satisfies get condition`. Each scores 1.0 with `task_spec_success` false.

## 5. Progress curves

For a trial with recovery rounds 0, 1, …, the curve is a step function of cumulative environment steps:

- **x-axis:** the sum of `simulation_steps` (policy `env.step` calls) over the rounds so far. The 50 initial and 50 post-rollout settling steps of each round are not counted. LLM calls, DSL checks and repairs use no environment steps.
- **y-axis:** the partial reward of the most recent rollout. It is 0 before the first rollout ends and is held after the trial stops.
- **Aggregation:** curves are averaged over a task's trials on a common step grid; the shaded band is ± one standard deviation, as in Fig. 11.
