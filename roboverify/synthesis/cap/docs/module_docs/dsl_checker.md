# DSL checker

A static analyzer that decides whether a generated Python program stays inside the DSL contract. It uses no simulator and makes no LLM calls.

## Interface

- [`DSLContract`](../../checker.py#L36): condition mode, task API names and input types. Built per task by [`DSLContract.for_task`](../../checker.py#L55).
- [`check_bundle(bundle, contract)`](../../checker.py#L128-L193) returns a [`CheckReport`](../../checker.py#L79) with status `pass`, `fail` (with diagnostics) or `error` (checker-internal), under a 10-second analysis deadline.
- [`require_check`](../../checker.py#L196) is the same check used as an execution gate; the interpreter calls it again before running ([`Runtime.execute`](../../runtime.py#L46-L66)).

## What it checks

Implemented in [`validation.py`](../../validation.py) ([`parse`](../../validation.py#L157), [`Checker`](../../validation.py#L335), [`validate`](../../validation.py#L842)):

- only supported Python syntax;
- calls to known DSL primitives, predicates, queries and getters with the right arity and argument kinds (object, goal, position, number);
- every helper is defined, with consistent argument types across call sites;
- loop, branch and selection conditions are allowed by the condition mode; in `dsl_only` they may use DSL predicates, object identity and Boolean combinations, but no numeric comparisons;
- conditions and queries have no physical actions, tracked through helpers, assignments and loop-carried values.

## What it does not check

Passing does not mean the program completes the task, terminates, or that its Move targets are reachable. Those show up only in the rollout.

Rule-level design: [shared DSL checker](../SHARED-DSL-CHECKER.md).
