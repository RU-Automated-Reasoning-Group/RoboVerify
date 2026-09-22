# Learning invariants from execution traces

The four tower verification entry points require a `DemoStore` of loop-head states.
They no longer learn from literal examples embedded in `inference.py`. The legacy
examples live in `golden_tower_fixtures.py` for regression tests only.

Run from `roboverify/`, with the simulator environment configured as in `AGENTS.md`:

```bash
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
uv run python -m synthesis.entry.collect_stack_loop_traces \
  --output /tmp/stack-loop-traces.json --num-blocks 4 --seeds 0 1 --max-iters 3
uv run python -m synthesis.entry.inference \
  --demo-store /tmp/stack-loop-traces.json --task stack --loop-id 1
uv run python -m synthesis.entry.verify_stack_with_learned_invariant \
  --demo-store /tmp/stack-loop-traces.json --loop-id 1 \
  --verification-mode finite --num-blocks 4 --disable-scene-viz
```

The collector runs the existing physical Stack program and reports task success
separately. Its traces are observations, not a certificate of successful execution
or an inductive invariant. A rollout can hit its iteration limit or fail its task
and still supply observed states. The existing verifier must check the resulting
candidate; [counterexample-guided refinement](../verification_lib/CEGIS.md)
provides the standalone feedback workflow.

For another executable tower program, pass the callback directly:

```python
from synthesis.inference_lib.demo_store import DemoStore, InvInference, tower_vocabulary

store = DemoStore()
trajectory = physical_program.eval(env, on_loop_head=store.add)
store.save("loop-traces.json")
invariant, clauses = InvInference(store, "1", tower_vocabulary("reverse"), context)
```

Use the vocabulary for the program's task and its corresponding high-level context
(`use_tbl=True` for Unstack and Reverse). The collector CLI currently supplies the
existing Stack program; other tasks use their own executable program and the same
callback. Unstack end-to-end runs remain capped at 60 seconds.

`Program.eval_from_observation` and `run_program_rollouts` accept the same callback.
Calling `While.eval` directly uses loop ID `"loop"` by default, or an explicit
`loop_id`. Through `Program`, IDs are zero-based instruction paths: `"1"` is the
second top-level instruction and `"1.0"` its first nested instruction. The four
tower tasks use flat loops.

Each row stores every physical block under a stable ID (`x1` for block 0), the
current symbolic bindings, and a copy of the geometry at entry to that loop
invocation. All iterations from one invocation share that entry geometry;
a later rollout captures a fresh entry. Guard witnesses are bound before the
callback, so `b_prime` describes the iteration being entered. Exit states, false
guards, and iterations beyond `max_iters` do not produce callback rows.
The separate CFG adapter in `cfg/invariants.py` includes terminal loop heads from
recovered iteration segments; it does not rely on this callback to record exits.

The adapter emits three equally sized lists for `compute_dataset`, resolves
constant names in the supplied Z3 context, and projects the relational `tbl`
marker according to `context.use_tbl`. It preserves every physical block even
when no instruction names that block. Missing bindings and empty datasets raise
errors. Recording is optional and does not change evaluation's return value.

The legacy Stack example has four empty initial dictionaries but two current
states. The golden test compares its two consumed current states and bindings
exactly and proves the two learned invariants equivalent with Z3. Recorded rows
instead contain complete initial geometry; Stack does not use `ON_star_zero`, so
this changes no truth-table values. Reverse's regression separately checks that
`ON_star_zero` uses entry geometry with current bindings.
