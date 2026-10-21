# MiniMax Method 1: four recoveries ending in primitive nonconvergence

- Setting: `cap_method_1-dsl_only-018-recovery-on`, Stack with three blocks, `typed_v3`.
- Parent run: `20261020-170025-29473fe-cap_method_1-recovery-7e945f9d`.
- Model: `minimax/minimax-m3.1-flash:free`, medium reasoning, 16,384 output-token cap.
- Outcome: five unsuccessful rollouts, including all four allowed recoveries; final `primitive_failed`, stop reason `max_recovery_rounds`.
- Usage: five model calls, 43,177 reported input/output tokens, 574 simulation steps. Each round has one policy/recovery-policy request; there are no separate helper-generation or static-repair requests. Helpers appearing below were emitted inline in those policy responses.
- All five candidates pass Method 1's pre-execution DSL checker. Both physical task success and the logical postcondition are false after every attempt.
- This study reads existing responses, prompts, feedback, timelines, and archived trajectories. It makes no model requests and performs no new rollout.

The sections distinguish **information delivered to the LLM**, a **brief high-level summary of the model's recorded explanation**, **the actual code changes**, and **retrospective execution analysis**. Explanation summaries are based on the saved API `choices[0].message.reasoning` field, not inferred solely from code edits. They do not reproduce private step-by-step reasoning. The duplicate `reasoning_details` text was checked for agreement. A model's explanation or diagnosis is not treated as a verified physical cause.

[Parent result](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/result.json) · [Parent recovery artifacts](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/artifacts/trial_001) · [Generated transition trace](../../../../../runs/cap-recovery-analysis/20261020-history-review/trials/minimax-cap_method_1-018.md)

## 1. Round-by-round outcome

| Round | While iterations | Failing primitive | Successful / failed primitive calls | Failure steps | Residual (m) | Simulation steps | Tokens |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 1 | Move | 1 / 1 | 50 | 1.438950 | 65 | 5,199 |
| 1 | 3 | Move | 16 / 1 | 50 | 0.021080 | 156 | 10,090 |
| 2 | 2 | Move | 8 / 1 | 50 | 0.039724 | 125 | 9,675 |
| 3 | 3 | Release | 11 / 1 | 50 | 0.013029 | 155 | 9,085 |
| 4 | 1 | Move | 2 / 1 | 50 | 1.407469 | 73 | 9,128 |

“Failure steps” is the budget used by the failing primitive, not the number of loop iterations or the whole rollout's simulation steps. The successful-call column counts only events with `converged=true`; the final nonconverged call is counted separately.

## 2. Shared start, API, and feedback audit

- Every rollout restores the exact same initial simulator snapshot and starts with fresh execution state. All five restoration events report `exact_match=true`, with SHA-256 `57036b6496d1f1eb7b736c032b136b71561b8803f1949d3be72563b3706a3b60`.
- `Move(x_ref, y_ref, z_ref, offsets)` samples the reference coordinates once and adds the offsets to form a **fixed gripper target**. `Pick` does not automatically lift. `Release(reference, offset)` opens, then retreats to the invocation-time reference Z plus the offset.
- The controller retains its 50-step primitive limit and 0.002 m positional tolerance throughout. This case does not establish that increasing those limits would repair the policies.
- The initial prompt provides task/API/mode text and the designated base b0, but no actual initial coordinate observation, image, or demonstration. Programs can read coordinates at runtime. Recovery additionally supplies selected recorded coordinate snapshots.
- All five saved prompts match the actual HTTP request contents. All four prepared feedback objects match their next-round delivered payloads; the prompt's source/binding and evidence JSON also matches those objects.
- Each recovery request contains the complete latest policy module, including its inline helpers. Older rounds have outcome summaries, not their program bodies. There is no separately generated helper missing feedback in this case.

| Feedback transition | Older rounds summarized | Selected state indices | Selected / omitted events | Evidence byte bound |
| --- | --- | --- | --- | ---: |
| 0 → 1 | None | 0, 15, 65 | 21 / 142 | 7,901 |
| 1 → 2 | 0 | 0, 106, 156 | 15 / 744 | 8,039 |
| 2 → 3 | 0, 1 | 0, 75, 125 | 15 / 416 | 8,034 |
| 3 → 4 | 0, 1, 2 | 0, 105, 155 | 13 / 443 | 7,834 |

The limit is 8,192 under `utf8-byte-upper-bound`, not a measured provider input-token count. Source is appended separately. Every payload has `missing_fields=[]`, but the explicit omitted-event counts show that the full trajectory is not delivered. Later sections label observations obtained from the full archived timeline separately from the smaller feedback excerpt.

| Object | Initial XYZ (world metres) |
| --- | --- |
| Gripper | (1.344643, 0.749116, 0.531461) |
| block_0 / b0 | (1.301914, 0.900809, 0.424892) |
| block_1 | (1.243266, 0.578952, 0.424892) |
| block_2 | (1.250824, 0.800068, 0.424892) |

## 3. Round 0: Initial policy double-counts reference coordinates

[Incoming prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_000/cap/20261020-170026-29473fe-cap_method_1-stack-90913f6b/artifacts/trial_001/lmp/call_0001/prompt.txt) · [API response](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_000/cap/20261020-170026-29473fe-cap_method_1-stack-90913f6b/artifacts/trial_001/lmp/call_0001/response.json) · [Executed policy](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_000/cap/20261020-170026-29473fe-cap_method_1-stack-90913f6b/artifacts/trial_001/checkpoints/cp_0003/policy.py) · [Timeline](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_000/cap/20261020-170026-29473fe-cap_method_1-stack-90913f6b/artifacts/trial_001/timeline.jsonl) · [Result](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_000/cap/20261020-170026-29473fe-cap_method_1-stack-90913f6b/result.json) · [Video](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_000/cap/20261020-170026-29473fe-cap_method_1-stack-90913f6b/artifacts/trial_001/execution/rollout_001/video.mp4)

### 3.1 Information received

| Original input | Supplied information |
| --- | --- |
| Task | Stack into one tower rooted at b0, which must remain on the table; cube side 0.05 m; declared initial relational conditions |
| Interface | Typed primitive/predicate descriptions, relative Move offsets, position getters, DSL-only conditions, primitive limits |
| Failure evidence/history | None: this is the initial request |


### 3.2 Recorded model explanation: high-level summary

The recorded explanation aims to build a tower by selecting blocks outside the b0 column, maintaining a support, and compensating for the measured block–gripper displacement. It recognizes the need for 0.05 m centre separation and DSL-only guards. The final response nevertheless encodes reference coordinates inside offsets, inconsistent with the API's addition of those references.

Source: the API response linked above, `choices[0].message.reasoning`; provider-reported reasoning tokens: 2,920. This is a concise summary of the recorded explanation, not an independently validated diagnosis.

### 3.3 Actual code changes

- The initial code adds support XYZ inside the offset expressions and adds 0.05 to X/Y as well as Z. Since Move itself adds the reference XYZ, the eventual gripper target contains the support position twice.
- The +0.05 X/Y terms also do not describe an aligned tower-centre target. No placement or Release is reached, so this case does not test that later placement geometry.

Complete executed source:

```python
support = b0

while exists(lambda x: not eq(x, b0) and not ON_star(x, b0)):
    candidate = get(lambda x: not eq(x, b0) and not ON_star(x, b0))

    Pick(candidate)

    block_pos = get_block_position(candidate)
    gripper_pos = get_gripper_position()
    support_pos = get_block_position(support)

    offset_x = support_pos[0] + 0.05 - (block_pos[0] - gripper_pos[0])
    offset_y = support_pos[1] + 0.05 - (block_pos[1] - gripper_pos[1])
    offset_z = support_pos[2] + 0.05 - (block_pos[2] - gripper_pos[2])

    Move(support, support, support, [offset_x, offset_y, offset_z])
    Release(support, 0.1)

    if exists(lambda x: not eq(x, b0) and ON_star(x, b0) and not exists(lambda y: ON(y, x))):
        support = get(lambda x: not eq(x, b0) and ON_star(x, b0) and not exists(lambda y: ON(y, x)))
    else:
        support = b0
```

### 3.4 Execution evidence and retrospective diagnosis

For any axis, with support coordinate `s` and block–gripper displacement `r`, the code passes `s + 0.05 - r` as an offset. The API adds `s` again, producing `2*s + 0.05 - r`. This arithmetic defect is established independently of any physical diagnosis. Pick converges, but the first Move requests `(2.654309, 1.850790, 0.898942)` and fails with a 1.438950 m residual. The large displacement is consistent with a grossly miscomputed target; the artifacts do not include a formal reachability test.

| # | Primitive | Source line | State interval | Steps | Converged | Fixed target / retreat XYZ | Residual (m) |
| ---: | --- | ---: | --- | ---: | --- | --- | ---: |
| 1 | Pick | 6 | 0 → 15 | 15 | True | — | — |
| 2 | Move | 16 | 15 → 65 | 50 | False | (2.654309, 1.850790, 0.898942) | 1.438950 |

### 3.5 Feedback passed to round 1

The next request receives this round's complete policy, including inline helper definitions, and the selected failure context summarized in Section 4.1. It includes the actual arguments/target, failed phase, 50-step exhaustion and residual, plus state indices 0, 15, 65. Terminal clauses: ON_star(block_0,block_0)=true; ON_star(block_1,block_0)=false; ON_star(block_2,block_0)=false.

The arithmetic/selection diagnoses developed in this case study are retrospective; the collector does not inject them as expert corrections. Older attempts remain short summaries.

[Prepared feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/artifacts/trial_001/rounds/000/feedback.json) · [Delivered feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/recovery_feedback.json) · [Next actual prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/lmp/call_0001/prompt.txt)

## 4. Round 1: Corrected offsets, but later transfer still fails

[Incoming prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/lmp/call_0001/prompt.txt) · [API response](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/lmp/call_0001/response.json) · [Executed policy](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/checkpoints/cp_0003/policy.py) · [Timeline](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/timeline.jsonl) · [Result](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/result.json) · [Video](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_001/cap/20261020-170115-29473fe-cap_method_1-stack-8c23986a/artifacts/trial_001/execution/rollout_001/video.mp4)

### 4.1 Information received

| Feedback from round 0 | Supplied information |
| --- | --- |
| Source/failure | Full initial source; failed Move at policy line 16; DSL pass and false task goals |
| Geometry/control | Offsets `(1.352395, 0.949981, 0.474050)`; fixed target `(2.654309, 1.850790, 0.898942)`; 50 steps and 1.438950 m residual |
| State/context | States 0, 15, 65; successful Pick; coordinate reads and offset assignments preceding the Move |
| History | No earlier summaries |


### 4.2 Recorded model explanation: high-level summary

The recorded explanation initially considers motion distance and staging, then explicitly identifies the doubled reference coordinates in the failed target. It proposes a helper converting desired block-centre positions into relative Move offsets, together with higher transport and intermediate horizontal targets. It expects successful placements to justify updating `support` to the selected block.

Source: the API response linked above, `choices[0].message.reasoning`; provider-reported reasoning tokens: 4,646. This is a concise summary of the recorded explanation, not an independently validated diagnosis.

### 4.3 Actual code changes

| Change | What the code actually does |
| --- | --- |
| Introduce `move_carried` | Subtract the reference position from `desired_block_position - measured_block_gripper_displacement`; this fixes the previous coordinate double counting |
| Add high/interpolated targets | First command already targets support XY at support Z+0.20, combining lifting with horizontal transfer. The 0.33/0.66 interpolation calls follow it, rather than splitting the original approach from the source |
| Simplify selection | Use `not ON_star(x,b0)` and update `support = candidate` after Release, without independently confirming the placement |
| Change retreat | Use `Release(support, 0.15)` |

Complete executed source:

```python
support = b0

def move_carried(block, reference, target_x, target_y, target_z):
    block_pos = get_block_position(block)
    gripper_pos = get_gripper_position()
    reference_pos = get_block_position(reference)
    relative_x = block_pos[0] - gripper_pos[0]
    relative_y = block_pos[1] - gripper_pos[1]
    relative_z = block_pos[2] - gripper_pos[2]
    Move(
        reference,
        reference,
        reference,
        [
            target_x - relative_x - reference_pos[0],
            target_y - relative_y - reference_pos[1],
            target_z - relative_z - reference_pos[2]
        ]
    )

while exists(lambda x: not ON_star(x, b0)):
    candidate = get(lambda x: not ON_star(x, b0))

    Pick(candidate)

    support_pos = get_block_position(support)
    block_pos = get_block_position(candidate)
    gripper_pos = get_gripper_position()

    move_carried(
        candidate,
        support,
        support_pos[0],
        support_pos[1],
        support_pos[2] + 0.20
    )

    block_pos = get_block_position(candidate)
    support_pos = get_block_position(support)
    move_carried(
        candidate,
        support,
        block_pos[0] + (support_pos[0] - block_pos[0]) * 0.33,
        block_pos[1] + (support_pos[1] - block_pos[1]) * 0.33,
        support_pos[2] + 0.20
    )

    block_pos = get_block_position(candidate)
    support_pos = get_block_position(support)
    move_carried(
        candidate,
        support,
        block_pos[0] + (support_pos[0] - block_pos[0]) * 0.66,
        block_pos[1] + (support_pos[1] - block_pos[1]) * 0.66,
        support_pos[2] + 0.20
    )

    support_pos = get_block_position(support)
    move_carried(
        candidate,
        support,
        support_pos[0],
        support_pos[1],
        support_pos[2] + 0.20
    )

    support_pos = get_block_position(support)
    move_carried(
        candidate,
        support,
        support_pos[0],
        support_pos[1],
        support_pos[2] + 0.05
    )

    Release(support, 0.15)
    support = candidate
```

### 4.4 Execution evidence and retrospective diagnosis

The coordinate conversion is now correct: for reference `s`, desired block centre `t`, and measured block–gripper displacement `r`, the helper passes `t-r-s`, so Move targets `t-r`.

The full timeline shows candidate IDs **1 → 2 → 1** over three iterations. After the first iteration, block_1 is aligned above b0. During the second, block_1 shifts away from b0; by state 83, blocks 1 and 2 form an off-base arrangement around Y=1.004 while b0 remains around Y=0.901. The next iteration picks block_1 again while support remains block_2.

The first Move of this third iteration raises both nonbase blocks in the recorded scene. At state 106, support block_2 is at Z=0.703923. The next helper request uses its new height plus 0.20; with measured relative Z=-0.023072, the fixed gripper target becomes approximately 0.926995. It fails after 50 steps. Thus the terminal Move failure follows disturbed support geometry and repeated selection; it is no longer the original doubled-reference formula.

The model's proposed staged path is only partly realized: the first target already transfers to support XY. In the first iteration, two subsequent Move calls take zero simulation steps because their targets are already within tolerance. These are concrete execution facts, not proof of a particular collision or attachment mechanism.

| # | Primitive | Source line | State interval | Steps | Converged | Fixed target / retreat XYZ | Residual (m) |
| ---: | --- | ---: | --- | ---: | --- | --- | ---: |
| 1 | Pick | 24 | 0 → 15 | 15 | True | — | — |
| 2 | Move | 10 | 15 → 27 | 12 | True | (1.302395, 0.899981, 0.624050) | 0.000669 |
| 3 | Move | 10 | 27 → 27 | 0 | True | (1.301344, 0.899701, 0.622806) | 0.001242 |
| 4 | Move | 10 | 27 → 28 | 1 | True | (1.300294, 0.899752, 0.622806) | 0.000966 |
| 5 | Move | 10 | 28 → 28 | 0 | True | (1.299321, 0.900077, 0.622684) | 0.001963 |
| 6 | Move | 10 | 28 → 34 | 6 | True | (1.299321, 0.900077, 0.472684) | 0.000834 |
| 7 | Release | 76 | 34 → 40 | 6 | True | (1.299471, 0.900106, 0.574892) | 0.000925 |
| 8 | Pick | 24 | 40 → 53 | 13 | True | — | — |
| 9 | Move | 10 | 53 → 62 | 9 | True | (1.302854, 0.900187, 0.674398) | 0.001593 |
| 10 | Move | 10 | 62 → 65 | 3 | True | (1.307051, 0.935559, 0.628705) | 0.001039 |
| 11 | Move | 10 | 65 → 68 | 3 | True | (1.311319, 0.980205, 0.625540) | 0.000969 |
| 12 | Move | 10 | 68 → 71 | 3 | True | (1.313484, 1.003696, 0.625826) | 0.000548 |
| 13 | Move | 10 | 71 → 77 | 6 | True | (1.313463, 1.003467, 0.475777) | 0.000952 |
| 14 | Release | 76 | 77 → 83 | 6 | True | (1.313512, 1.003480, 0.574892) | 0.000743 |
| 15 | Pick | 24 | 83 → 97 | 14 | True | — | — |
| 16 | Move | 10 | 97 → 106 | 9 | True | (1.318766, 1.004682, 0.678507) | 0.001080 |
| 17 | Move | 10 | 106 → 156 | 50 | False | (1.318978, 1.004596, 0.926995) | 0.021080 |

### 4.5 Feedback passed to round 2

The next request receives this round's complete policy, including inline helper definitions, and the selected failure context summarized in Section 5.1. It includes the actual arguments/target, failed phase, 50-step exhaustion and residual, plus state indices 0, 106, 156. Terminal clauses: ON_star(block_0,block_0)=true; ON_star(block_1,block_0)=false; ON_star(block_2,block_0)=false.

The arithmetic/selection diagnoses developed in this case study are retrospective; the collector does not inject them as expert corrections. Older attempts remain short summaries.

[Prepared feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/artifacts/trial_001/rounds/001/feedback.json) · [Delivered feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/recovery_feedback.json) · [Next actual prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/lmp/call_0001/prompt.txt)

## 5. Round 2: Source lift added; selection now picks the base

[Incoming prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/lmp/call_0001/prompt.txt) · [API response](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/lmp/call_0001/response.json) · [Executed policy](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/checkpoints/cp_0003/policy.py) · [Timeline](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/timeline.jsonl) · [Result](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/result.json) · [Video](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_002/cap/20261020-170211-29473fe-cap_method_1-stack-89a7c72b/artifacts/trial_001/execution/rollout_001/video.mp4)

### 5.1 Information received

| Feedback from round 1 | Supplied information |
| --- | --- |
| Source/failure | Full module including `move_carried`; Move fails at helper line 10, with caller context for the 0.33 interpolation |
| Geometry/control | Fixed target `(1.318978, 1.004596, 0.926995)`; 50 steps; residual 0.021080 m |
| State/context | States 0, 106, 156; both nonbase blocks high and displaced from b0; latest coordinate reads and helper arithmetic |
| Task/history | Both nonbase ON* goal clauses false; a short round-0 failure summary, not round-0 code |
| Not supplied as a full history | The complete sequence of earlier selections and placements within round 1; this is reconstructed below from the archived timeline |


### 5.2 Recorded model explanation: high-level summary

The recorded explanation considers disturbed support geometry and repeatedly selecting a previously moved block. It proposes separating source lift from transfer and changing selection to be relative to the current support. It incorrectly treats that condition as preserving exclusion of b0 after support changes; the actual next rollout selects b0.

Source: the API response linked above, `choices[0].message.reasoning`; provider-reported reasoning tokens: 4,068. This is a concise summary of the recorded explanation, not an independently validated diagnosis.

### 5.3 Actual code changes

| Change | What the code actually does |
| --- | --- |
| Introduce `move_block_to` | Use the selected block as the Move reference and subtract that block's coordinates; the absolute-to-relative conversion remains algebraically consistent |
| Separate lift and transfer | First target retains the candidate's XY; subsequent targets use support XY, first high and then at placement height |
| Change loop and selection anchor | `not eq(x, support) and not ON_star(x, support)` replaces the fixed-base condition. Once support becomes block_1, this permits b0, which is below block_1 |
| Preserve unconditional update | After Release, set support to candidate without verifying that the original task's base-rooted structure remains intact |

Complete executed source:

```python
def move_block_to(block, target_x, target_y, target_z):
    block_position = get_block_position(block)
    gripper_position = get_gripper_position()
    relative_x = block_position[0] - gripper_position[0]
    relative_y = block_position[1] - gripper_position[1]
    relative_z = block_position[2] - gripper_position[2]
    Move(
        block,
        block,
        block,
        [
            target_x - relative_x - block_position[0],
            target_y - relative_y - block_position[1],
            target_z - relative_z - block_position[2]
        ]
    )


support = b0

while exists(lambda x: not eq(x, support) and not ON_star(x, support)):
    candidate = get(lambda x: not eq(x, support) and not ON_star(x, support))

    Pick(candidate)

    candidate_position = get_block_position(candidate)
    support_position = get_block_position(support)

    move_block_to(
        candidate,
        candidate_position[0],
        candidate_position[1],
        support_position[2] + 0.20
    )

    candidate_position = get_block_position(candidate)
    support_position = get_block_position(support)

    move_block_to(
        candidate,
        support_position[0],
        support_position[1],
        support_position[2] + 0.20
    )

    support_position = get_block_position(support)

    move_block_to(
        candidate,
        support_position[0],
        support_position[1],
        support_position[2] + 0.05
    )

    Release(support, 0.15)
    support = candidate
```

### 5.4 Execution evidence and retrospective diagnosis

The first iteration places block_1 above b0. The code then changes `support` to block_1. Since b0 is a different object and is below that support, `not eq(b0,support)` and `not ON_star(b0,support)` both hold. The next `get` selects **physical block_0 (b0)** and the next Pick acts on it. This directly contradicts the recorded explanation's expectation that b0 remains excluded.

From state 46 onward, the source is b0 and the reference support is block_1 above it. Recorded heights rise together as the program recomputes targets relative to that changing support: the three Move targets have Z approximately 0.666539, 0.914307, and 1.012764. The third Move fails with 0.039724 m residual. The nominal placement approach actually requests a higher target because the measured support has risen.

The helper's reference subtraction is not the demonstrated defect here. The verified policy error is selecting and lifting the designated base; the full physical mechanism by which the blocks move together is not diagnosed from coordinates alone.

| # | Primitive | Source line | State interval | Steps | Converged | Fixed target / retreat XYZ | Residual (m) |
| ---: | --- | ---: | --- | ---: | --- | --- | ---: |
| 1 | Pick | 24 | 0 → 15 | 15 | True | — | — |
| 2 | Move | 7 | 15 → 23 | 8 | True | (1.243665, 0.578771, 0.624050) | 0.001047 |
| 3 | Move | 7 | 23 → 34 | 11 | True | (1.301129, 0.900527, 0.622941) | 0.001773 |
| 4 | Move | 7 | 34 → 40 | 6 | True | (1.299276, 0.899795, 0.471197) | 0.000978 |
| 5 | Release | 55 | 40 → 46 | 6 | True | (1.299399, 0.899822, 0.574892) | 0.000928 |
| 6 | Pick | 24 | 46 → 56 | 10 | True | — | — |
| 7 | Move | 7 | 56 → 65 | 9 | True | (1.302135, 0.900855, 0.666539) | 0.000969 |
| 8 | Move | 7 | 65 → 75 | 10 | True | (1.303569, 0.900805, 0.914307) | 0.000939 |
| 9 | Move | 7 | 75 → 125 | 50 | False | (1.304754, 0.900611, 1.012764) | 0.039724 |

### 5.5 Feedback passed to round 3

The next request receives this round's complete policy, including inline helper definitions, and the selected failure context summarized in Section 6.1. It includes the actual arguments/target, failed phase, 50-step exhaustion and residual, plus state indices 0, 75, 125. Terminal clauses: ON_star(block_0,block_0)=true; ON_star(block_1,block_0)=true; ON_star(block_2,block_0)=false.

The arithmetic/selection diagnoses developed in this case study are retrospective; the collector does not inject them as expert corrections. Older attempts remain short summaries.

[Prepared feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/artifacts/trial_001/rounds/002/feedback.json) · [Delivered feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/recovery_feedback.json) · [Next actual prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/lmp/call_0001/prompt.txt)

## 6. Round 3: Base-relative selection restored; Release retreat fails

[Incoming prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/lmp/call_0001/prompt.txt) · [API response](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/lmp/call_0001/response.json) · [Executed policy](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/checkpoints/cp_0003/policy.py) · [Timeline](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/timeline.jsonl) · [Result](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/result.json) · [Video](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_003/cap/20261020-170249-29473fe-cap_method_1-stack-4a9a6bd4/artifacts/trial_001/execution/rollout_001/video.mp4)

### 6.1 Information received

| Feedback from round 2 | Supplied information |
| --- | --- |
| Source/failure | Complete module including `move_block_to`; Move fails at helper line 7; caller context for the placement approach |
| Geometry/control | Move references are b0; target `(1.304754, 0.900611, 1.012764)`; 50 steps; residual 0.039724 m |
| State evidence | States 0, 75, 125 show b0 elevated: its centre Z is 0.890911 before the failure and 0.955912 afterward |
| Task/history | Physical/logical goals false; ON*(block_1,b0) true but ON*(block_2,b0) false; summaries of rounds 0–1 without their source |
| Diagnosis boundary | The payload does not explicitly explain that changing the selection anchor admitted the base. That is verified below from source and query/Pick events |


### 6.2 Recorded model explanation: high-level summary

The recorded explanation notices the elevated base in the feedback, considers both coordinate calculations and selection of an earlier support, and proposes restoring selection relative to b0. It also simplifies movement to direct support-relative offsets and changes Release to reference the candidate. Its suspicion that the previous helper formula was intrinsically wrong is not established: that helper's reference subtraction is algebraically consistent.

Source: the API response linked above, `choices[0].message.reasoning`; provider-reported reasoning tokens: 3,804. This is a concise summary of the recorded explanation, not an independently validated diagnosis.

### 6.3 Actual code changes

| Change | What the code actually does |
| --- | --- |
| Restore fixed-base selection | `not ON_star(x,b0)` excludes b0 through reflexivity and selects blocks outside its column |
| Inline support-relative offsets | Use `[-relative_x, -relative_y, h-relative_z]`, with h=0.20 and 0.05; reference coordinates are not double-counted |
| Remove source-only lift | The first Move again targets support XY and height together, rather than retaining the separate source lift from round 2 |
| Change Release | Use `Release(candidate, 0.15)` instead of `Release(support, 0.15)` |
| Wrap the policy | Define and invoke `stack_policy`; no separate helper-generation request occurs |

Complete executed source:

```python
def stack_policy():
    support = b0

    while exists(lambda x: not ON_star(x, b0)):
        candidate = get(lambda x: not ON_star(x, b0))

        Pick(candidate)

        candidate_position = get_block_position(candidate)
        gripper_position = get_gripper_position()
        support_position = get_block_position(support)

        relative_x = candidate_position[0] - gripper_position[0]
        relative_y = candidate_position[1] - gripper_position[1]
        relative_z = candidate_position[2] - gripper_position[2]

        Move(
            support,
            support,
            support,
            [
                -relative_x,
                -relative_y,
                0.20 - relative_z
            ]
        )

        candidate_position = get_block_position(candidate)
        gripper_position = get_gripper_position()
        support_position = get_block_position(support)

        relative_x = candidate_position[0] - gripper_position[0]
        relative_y = candidate_position[1] - gripper_position[1]
        relative_z = candidate_position[2] - gripper_position[2]

        Move(
            support,
            support,
            support,
            [
                -relative_x,
                -relative_y,
                0.05 - relative_z
            ]
        )

        Release(candidate, 0.15)
        support = candidate


stack_policy()
```

### 6.4 Execution evidence and retrospective diagnosis

The selection now protects b0. The full timeline again selects **1 → 2 → 1**: block_1 is initially placed over b0, then the nonbase arrangement moves away from the base while adding block_2, and block_1 is selected again. b0 stays at table height. The restored fixed-base guard repairs the round-2 selection defect but does not ensure stable placement.

All six Move calls in this attempt converge. The final failure occurs in the third iteration's Release, after the preceding Move reaches its target. At Release invocation, candidate block_1 has Z=0.741522, so the fixed retreat target is Z=0.891522. Opening finishes in one simulation step; the subsequent retreat uses the remaining budget and leaves a 0.013029 m positional error. This is **retreat nonconvergence after opening**, not failure to open the gripper.

By the terminal state, block_1 has fallen near table height and block_2 is above it, both away from b0. The frozen retreat target does not follow that falling reference. The records establish the target and phase failure; they do not prove that the target is globally unreachable or identify a contacting body.

| # | Primitive | Source line | State interval | Steps | Converged | Fixed target / retreat XYZ | Residual (m) |
| ---: | --- | ---: | --- | ---: | --- | --- | ---: |
| 1 | Pick | 7 | 0 → 15 | 15 | True | — | — |
| 2 | Move | 17 | 15 → 27 | 12 | True | (1.302395, 0.899981, 0.624050) | 0.000669 |
| 3 | Move | 36 | 27 → 33 | 6 | True | (1.299211, 0.899804, 0.472806) | 0.000884 |
| 4 | Release | 47 | 33 → 40 | 7 | True | (1.299378, 0.899835, 0.626207) | 0.001790 |
| 5 | Pick | 7 | 40 → 55 | 15 | True | — | — |
| 6 | Move | 17 | 55 → 64 | 9 | True | (1.302878, 0.899691, 0.674001) | 0.001586 |
| 7 | Move | 36 | 64 → 72 | 8 | True | (1.310072, 1.045050, 0.482182) | 0.000333 |
| 8 | Release | 47 | 72 → 79 | 7 | True | (1.310079, 1.044756, 0.632321) | 0.001411 |
| 9 | Pick | 7 | 79 → 91 | 12 | True | — | — |
| 10 | Move | 17 | 91 → 100 | 9 | True | (1.316142, 1.037983, 0.667174) | 0.000935 |
| 11 | Move | 36 | 100 → 105 | 5 | True | (1.319309, 1.037888, 0.765323) | 0.000834 |
| 12 | Release | 47 | 105 → 155 | 50 | False | (1.319107, 1.037797, 0.891522) | 0.013029 |

### 6.5 Feedback passed to round 4

The next request receives this round's complete policy, including inline helper definitions, and the selected failure context summarized in Section 7.1. It includes the actual arguments/target, failed phase, 50-step exhaustion and residual, plus state indices 0, 105, 155. Terminal clauses: ON_star(block_0,block_0)=true; ON_star(block_1,block_0)=false; ON_star(block_2,block_0)=false.

The arithmetic/selection diagnoses developed in this case study are retrospective; the collector does not inject them as expert corrections. Older attempts remain short summaries.

[Prepared feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/artifacts/trial_001/rounds/003/feedback.json) · [Delivered feedback](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/recovery_feedback.json) · [Next actual prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/lmp/call_0001/prompt.txt)

## 7. Round 4: Release repair is proposed, but an earlier Move regresses

[Incoming prompt](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/lmp/call_0001/prompt.txt) · [API response](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/lmp/call_0001/response.json) · [Executed policy](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/checkpoints/cp_0003/policy.py) · [Timeline](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/timeline.jsonl) · [Result](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/result.json) · [Video](../../../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_1-dsl_only-018-recovery-on/cap/20261020-170025-29473fe-cap_method_1-recovery-7e945f9d/attempts/round_004/cap/20261020-170350-29473fe-cap_method_1-stack-80798e8e/artifacts/trial_001/execution/rollout_001/video.mp4)

### 7.1 Information received

| Feedback from round 3 | Supplied information |
| --- | --- |
| Source/failure | Complete `stack_policy` module; Release fails at line 47 |
| Geometry/control | `Release(candidate, 0.15)`; open and retreat phase records; fixed retreat target `(1.319107, 1.037797, 0.891522)`; 50 steps and 0.013029 m residual |
| State/context | States 0, 105, 155; preceding converged Move with target Z=0.765323; both nonbase blocks away from the b0 column |
| Task/history | Both nonbase ON* clauses false; summaries of rounds 0–2. Only round 3's full code is supplied |
| Missing diagnosis | No validated collision/reachability explanation and no explicit warning that the round-0 arithmetic defect could be reintroduced |


### 7.2 Recorded model explanation: high-level summary

The recorded explanation attributes the latest failure mainly to a high Release retreat target, while considering contact or reach limitations as possibilities. It proposes a separate source lift, high transfer, descent, and a shorter retreat referenced to support. The generated transfer/descent offsets nevertheless reintroduce reference-coordinate double counting; the intended Release change is never tested.

Source: the API response linked above, `choices[0].message.reasoning`; provider-reported reasoning tokens: 3,819. This is a concise summary of the recorded explanation, not an independently validated diagnosis.

### 7.3 Actual code changes

| Change | What the code actually does |
| --- | --- |
| Add a source lift | First Move references the candidate, with `[0,0,0.20-relative_z]`; this Move converges |
| Rewrite transfer/descent offsets | Include support XYZ inside the offsets while still referencing support. This reintroduces the defect fixed in round 1 |
| Shorten/change Release | Replace `Release(candidate,0.15)` with `Release(support,0.10)`, but execution fails before reaching it |
| Retain fixed-base selection | The latest failure is not caused by the round-2 selection of b0 |

Complete executed source:

```python
def stack_policy():
    support = b0

    while exists(lambda x: not ON_star(x, b0)):
        candidate = get(lambda x: not ON_star(x, b0))

        Pick(candidate)

        candidate_position = get_block_position(candidate)
        gripper_position = get_gripper_position()

        relative_x = candidate_position[0] - gripper_position[0]
        relative_y = candidate_position[1] - gripper_position[1]
        relative_z = candidate_position[2] - gripper_position[2]

        Move(
            candidate,
            candidate,
            candidate,
            [
                0.0,
                0.0,
                0.20 - relative_z
            ]
        )

        support_position = get_block_position(support)
        candidate_position = get_block_position(candidate)
        gripper_position = get_gripper_position()

        relative_x = candidate_position[0] - gripper_position[0]
        relative_y = candidate_position[1] - gripper_position[1]
        relative_z = candidate_position[2] - gripper_position[2]

        Move(
            support,
            support,
            support,
            [
                support_position[0] - relative_x,
                support_position[1] - relative_y,
                support_position[2] + 0.20 - relative_z
            ]
        )

        support_position = get_block_position(support)
        candidate_position = get_block_position(candidate)
        gripper_position = get_gripper_position()

        relative_x = candidate_position[0] - gripper_position[0]
        relative_y = candidate_position[1] - gripper_position[1]
        relative_z = candidate_position[2] - gripper_position[2]

        Move(
            support,
            support,
            support,
            [
                support_position[0] - relative_x,
                support_position[1] - relative_y,
                support_position[2] + 0.05 - relative_z
            ]
        )

        Release(support, 0.10)
        support = candidate


stack_policy()
```

### 7.4 Execution evidence and retrospective diagnosis

The source lift succeeds in 8 simulation steps. The subsequent transfer uses a well-typed offset containing the support coordinates themselves. For X/Y the target is `2*support_coordinate - relative_coordinate`; for Z it is `2*support_z + 0.20 - relative_z`.

At the transfer's entry, support is still b0 at `(1.301914, 0.900809, 0.424892)`. The supplied offset is `(1.301126, 0.900519, 0.622922)`, so the fixed gripper target becomes `(2.603040, 1.801328, 1.047814)`. It fails after 50 steps with a 1.407469 m residual. The descent, modified Release, and support update are never executed.

This is a concrete recurrence of the **reference-double-counting defect**, rather than merely another Move-failure label: round 0 introduced it, round 1 removed it, and round 4 reintroduced it in a different source program. The latest round-3 source had a correct relative-offset form. The fact that earlier source was absent may motivate a history ablation, but does not establish that history inclusion would have prevented this regression.

| # | Primitive | Source line | State interval | Steps | Converged | Fixed target / retreat XYZ | Residual (m) |
| ---: | --- | ---: | --- | ---: | --- | --- | ---: |
| 1 | Pick | 7 | 0 → 15 | 15 | True | — | — |
| 2 | Move | 16 | 15 → 23 | 8 | True | (1.243184, 0.579599, 0.624412) | 0.001056 |
| 3 | Move | 35 | 23 → 73 | 50 | False | (2.603040, 1.801328, 1.047814) | 1.407469 |

### 7.5 No further recovery

Four recovery rounds have been used. The parent records `max_recovery_rounds`; there is no next model request receiving this final failure. The changed Release strategy is therefore untested, and no corrected coordinate formula is executed as part of this study.

## 8. What this case adds to the failure analysis

| Finding | Evidence and interpretation |
| --- | --- |
| Feedback can support useful diagnosis without yielding a successful policy | Round 1's recorded explanation identifies reference double counting, and its helper fixes that arithmetic. The later rollout still fails |
| Model explanations can be wrong | Round 2 expects the new selection to exclude b0, but actual query/Pick records select b0. Round 3 suspects a helper-arithmetic defect even though its reference conversion is consistent |
| A plausible repair plan can be implemented incorrectly | Round 1's planned staged transfer starts with a move to support XY; round 4 proposes a lower retreat but introduces an earlier invalid transfer target |
| Repair can reintroduce a previously corrected defect | Double-counted reference coordinates appear in rounds 0 and 4, with correct conversion in the intervening candidates |
| The final failure can hide earlier loss of task structure | Rounds 1/3 involve an off-base support arrangement before terminal nonconvergence; round 2 picks the designated base before its high-target failure |
| All five DSL passes are compatible with all five physical failures | The checker validates the allowed language/types/effects, not witness usefulness, correct numeric targets, stable physical support, or task achievement |
| Missing helper-request feedback does not explain this case | Each response includes the complete policy and inline helpers; no downstream helper-generation call occurs |

The failure evidence was delivered successfully. This case supports investigating explicit target arithmetic, selected-object/support progression, and a historical source/edit ledger. It does not establish that current feedback is unusable, that a particular extra observation would guarantee recovery, or that longer primitive budgets would solve the task. Model-stated collision/reach hypotheses remain hypotheses without a physical diagnosis.

The successful Haiku Method 1 trial 4 in [recovery report Section 4.5.1](../../revision_4_recovery_report.md#451-existing-feedback-can-support-targeted-repair-haiku-method-1-trial-4) illustrates that staged motion can help. These different-model cases are not a controlled comparison of feedback profiles. A future comparison should hold the failed starting program/state and model settings fixed and separately vary current feedback, semantic diagnostics, and full historical source.
