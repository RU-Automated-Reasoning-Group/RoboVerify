# Synthesizer: Method 2

[`Method2`](../../cap_method_2/method.py#L10) generates one complete program (policy plus every helper) and repairs it as a whole until the [DSL checker](dsl_checker.md) passes. This is Loop 1 of the framework.

## Loop

[`Method2.generate`](../../cap_method_2/method.py#L58-L106):

1. Request a complete program with the `policy` template ([Prompt builder](prompt_builder.md)).
2. For each attempt, up to `max_attempts` (frozen: 5):
   - save the candidate as a checkpoint;
   - run [`check_bundle`](../../checker.py#L128) on the complete candidate;
   - **pass:** return the checked bundle;
   - **checker error:** raise `AnalysisFailure` (no repair);
   - **violations:** request a whole-program repair with the `repair` template and the diagnostics.
3. If the last attempt still fails, raise `ConstraintFailure` (status `constraint_failed`). No rollout happens in that round.

Then [`Method2.__call__`](../../cap_method_2/method.py#L108-L110) passes the bundle to the interpreter ([Rollout](rollout.md)).

## Properties

- The repair loop sees only static checker diagnostics, never simulator results.
- There is no separate helper generation step: missing helpers are reported by the checker and fixed in the next whole-program repair.
- Every call and repair is charged to the current round's budget.

Other methods (Baseline, Method 1) are constructed by [`make_method`](../../runner.py#L80) and are not used by the frozen baseline.
