# Demonstration compactor

Turns a RoboVerify demonstration archive into one deterministic text block that is placed at the start of every LLM request. It runs once per trial. Off by default; on in `config_frozen_v3` with demonstrations enabled.

## Input and output

- **Input:** a `demonstrations.npz` archive written by RoboVerify's collector (`synthesis.entry.collect_demos`), plus [`DemonstrationConfig`](../../config.py#L234).
- **Output:** a [`DemonstrationContext`](../../demonstrations.py#L518): the rendered text, its token count, and a manifest (archive hash, trace ids, seeds, settings, hashes).

## Steps

1. **Load and validate** ([`load_demonstrations`](../../demonstrations.py#L110)): checks the format, task and block count, optional archive hash, and the selected trace ids.
2. **Compact each demonstration** ([`compact`](../../demonstrations.py#L423)):
   - **Event frames** ([`event_frames`](../../demonstrations.py#L277)): first and last step, and every step where the gripper command switches open/closed. They carry every block position and the DSL relations, as a full set at the first event and as `+`/`-` changes afterward ([`relations`](../../demonstrations.py#L352)).
   - **Waypoint frames** ([`motion_waypoints`](../../demonstrations.py#L321), [`rdp`](../../demonstrations.py#L297)): the gripper path between events, simplified with Ramer–Douglas–Peucker at `waypoint_tolerance` (default 1 cm).
   - Each frame also records the gripper command and which block is held.
3. **Render** ([`render_context`](../../demonstrations.py#L542)): header and frame format from the prompt package (`typed_v5`, serialization `demo-keyframes-v1`). Numbers and order are deterministic.
4. **Measure** ([`count_tokens`](../../demonstrations.py#L611)): the server's `/tokenize` endpoint, or a UTF-8 byte bound offline.

[`prepare`](../../demonstrations.py#L654) runs steps 1–4.

## What the LLM does not see

The program that produced the demonstrations, the raw event log, the task spec, and simulator snapshots. The header tells the model that the evaluation scene is not among the demonstrations.

## Guards and accounting

- **No overlap with evaluation:** [`check_evaluation_seed`](../../demonstrations.py#L674) and [`check_evaluation_scene`](../../demonstrations.py#L689) reject an evaluation scene whose seed or block positions match a demonstration.
- **Size limit:** a block over `max_tokens` fails the trial before any request ([`runner.py`](../../runner.py#L302-L307)).
- **Tokens are charged separately:** each request adds the block's tokens to `demonstration_tokens`; the effort budgets see only the rest ([`budget.effort_tokens`](../../budget.py#L10), [`Budget.account_tokens`](../../budget.py#L76)). Demonstrations on and off therefore have the same effort budget.
