# Logger and artifacts

Every component writes to one logger per attempt, so each trial can be audited and replayed from its files.

## Components

| Component | Role | Code |
| --- | --- | --- |
| `RunLogger` | Run directory contract shared with RoboVerify: `config.json`, `status.json`, `metrics.jsonl`, `events.jsonl`, `result.json` | [`run_logger.py`](../../../experiment/run_logger.py#L101) |
| `CapLogger` | CaP timeline and artifacts under `artifacts/trial_001/`: [`emit`](../../logger.py#L61) appends an event to `timeline.jsonl`; [`artifact`](../../logger.py#L99) saves a file | [`logger.py`](../../logger.py#L25) |
| `Checkpoints` | Saves every candidate source (initial, repaired, executed) with its check report | [`checkpoints.py`](../../checkpoints.py#L11) |
| `Recorder` | Trajectory (`trajectory.npz`), per-step log and video | [`recording.py`](../../recording.py#L12) |
| Deferred video | Renders videos on a shared GPU queue after the trial | [`deferred_video.py`](../../deferred_video.py) |

## Layout of one recovery trial

```text
<trial run dir>/
  recovery_config.json, recovery_result.json
  rounds/<k>/result.json, identity.json, feedback.json
  attempts/round_<k>/cap/<attempt run dir>/
    config.json, result.json
    artifacts/trial_001/
      timeline.jsonl                    # every event: requests, checks, primitives, queries
      lmp/call_NNNN/                    # prompt, request, response, usage per LLM call
      checkpoints/cp_NNNN/policy.py     # each candidate and its check report
      execution/rollout_001/            # trajectory.npz, steps.jsonl, video.mp4
      demonstrations/                   # rendered block and manifest (if on)
```

The feedback builder reads these files ([`collect_failure`](../../recovery.py#L160)), and batch reports and audits are computed from them.
