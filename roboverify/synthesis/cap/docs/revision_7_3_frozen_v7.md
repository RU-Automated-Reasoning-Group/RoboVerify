# Revision 7.3: `config_frozen_v7` for hosted Claude models

- **Status:** implemented and smoke-tested 2026-10-26 (Section 4).
- **Decisions (2026-10-26):** later CaP experiments use Claude Haiku 5.5 and Claude Opus 5.5. Raise `max_output_tokens` to 64k, and make RDP recovery keyframes ([Revision 7.2](revision_7_2_recovery_key_frame.md)) the default of the next version. Budgets may grow so that prompt items fit.

## 1. Changes from `config_frozen_v6`

| Setting | v6 | v7 | Why |
| --- | --- | --- | --- |
| Model | local Qwen3.8 27B FP8 (vLLM); hosted models by driver override | Claude Haiku 5.5 on OpenRouter (default), Opus 5.5 in `MODELS`; reasoning effort medium | decision above |
| `max_output_tokens` | 32,768 | **65,536** | Revision 7.1: 13 first responses without demonstrations hit the 32,768 limit |
| Demonstration token counter | `server_tokenize` | `provider_usage` | hosted models have no local tokenizer (the driver already overrode it) |
| Round-0 `max_tokens` | 100,000 | **200,000** | each request reserves its prompt bytes plus the output cap; with 64k, 100,000 leaves no room for a repair call |
| Recovery `round_tokens` / `total_tokens` | 200,000 / 900,000 | **300,000 / 1,400,000** | a recovery request with full evidence reserves up to ~165k |
| Recovery `context_tokens` | 262,144 | **524,288** | reservation plus the demonstration block (ReStack ~72k); both models accept 1,000,000 |
| Recovery `keyframe_method` | `legacy` | **`rdp`** (`recovery-keyframes-v2`) | Revision 7.2 |
| `keyframe_bytes` / `evidence_bytes` | – / 32,768 | **65,536 / 98,304** | every measured rollout fits (largest RDP evidence 43,943 bytes) |

Prompts, task instructions (`task-instructions-v2`), primitive prompt (`primitive-prompt-v2`), demonstration archives, success criterion (`task-spec-v2`, 50 settling steps), initial settling, method (`cap_method_2`, `dsl_only`), call, time and step limits, recovery rounds and block counts are v6's. [`test_config_frozen_v7.py`](../configs/test_config_frozen_v7.py) pins this list.

Budgets are larger than v6's, so v7 numbers are not directly comparable with v1–v6 numbers, and v7 does not fit the local 262,144-token vLLM window.

## 2. Recovery keyframe layout (`recovery-keyframes-v2`)

In the Revision 7.2 experiment (`recovery-keyframes-v1`, git `3a7dba1`), the RDP frames were an item of the evidence JSON object. The object is serialized as compact JSON with sorted keys, so each frame was a quoted string with escaped `\n`, all frames were on one line, and the summary and `inputs` lines came after the frames. The content matched the demonstrations, but the layout did not.

`recovery-keyframes-v2` renders the frames as a plain-text section after the evidence object, laid out like one demonstration:

```text
FAILED EXECUTION KEYFRAMES (recovery-keyframes-v2)
<format notes>
FAILED EXECUTION (round 0): restack, 5 blocks o0..o4, 77 states (t=0..76), 3 events, 4 waypoints
inputs: b0=o0
t=0 event open gripper=(1.345,0.749,0.531) holding=none
  blocks: o0=(1.328,0.949,0.474) ...
  holds: ON(o0,o3) ON(o4,o0) ...
t=8 waypoint open gripper=(1.329,0.948,0.531) holding=none
...
END OF FAILED EXECUTION
```

The evidence object keeps an index entry (`keyframes`: method, version, delivered and available frames). The section's bytes count toward the keyframe budget and the evidence cap. An omitted span is a marker line (`... N frames omitted (t=a..b) to fit the keyframe budget ...`). The summary and frame lines are produced by the same functions as the demonstration block (`summary_line`, `render_inputs`, `render_frame`). v1 is not kept; the Revision 7.2 results remain reproducible at git `3a7dba1`.

## 3. Experiment driver

`experiments/revision6_experiment.py --frozen v7 --model claude-haiku-5.5|claude-opus-5.5`. For v7 the driver records only overrides that differ from the frozen version: none for Haiku, and the model name for Opus.

## 4. Smoke

[`runs/cap-revision7-3/20261026-smoke-v7/report.md`](../../../runs/cap-revision7-3/20261026-smoke-v7/report.md): Haiku 5.5, demonstrations on, scene seed 0, one trial per task, git `02335bb` (clean). Cost about $0.2.

| Task | Result | Recovery rounds | Calls | Effort tokens (+ demonstration block) |
| --- | --- | ---: | ---: | --- |
| Stack | success, round 0 | 0 | 1 | 8,206 (+26,668) |
| Unstack | success, round 0 | 0 | 1 | 9,288 (+34,199) |
| Reverse | goal not reached after 5 rounds | 4 | 5 | 125,908 (+230,545) |
| ReStack | not recovered after 5 rounds (Pick nonconvergence ×3, Release nonconvergence, `get` found no object; last: Pick) | 4 | 6 | 195,775 (+433,518) |
| Grid | success, round 0 | 0 | 1 | 7,838 (+76,593) |
| Pyramid | success, round 0 | 0 | 1 | 7,342 (+81,503) |

- All 8 recovery requests carried the `recovery-keyframes-v2` section, complete (19–27 KB evidence including the section), after the evidence object.
- No attempt was cut at the output limit, and no request was refused by a token budget.
- The two failures are task outcomes, as in the v6 smoke (Reverse) and the Revision 7.2 arms.
