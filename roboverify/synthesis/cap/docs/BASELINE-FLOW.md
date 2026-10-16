# Google baseline: observations and program flow

This describes the implemented `cap_google_baseline` path, reviewed on
2026-10-16. It distinguishes **text sent to the LLM**, **values available to the
generated Python policy**, and **environment data retained by the host**.
See [method 1](METHOD-1-FLOW.md) for the checked generation path and
[the shared DSL](DSL.md) for predicate and action semantics.

## What the LLM receives

The model generates Python source from text. It does not receive a simulator
observation vector or a symbolic scene table. The standard runner creates a fresh
method instance and invokes it once per trial.

| Model request | Text actually assembled by the implementation |
| --- | --- |
| Initial policy | Code-generation instruction, task-filtered API description, the context `b0 is the input block.`, and the task instruction from `TASK_INSTRUCTIONS` or the caller's override. |
| Missing helper | Helper-definition instruction, the same API description, and an unresolved call expression, possibly its enclosing assignment, extracted by the upstream parser. |
| Later policy invocation on the same instance | Upstream `maintain_session=True` adds previous executed policy source to the prompt. The standard runner makes only one invocation, so this history is initially empty. |

For Stack, the default task instruction is:

```text
Stack all blocks into a single tower rooted at the input block b0.
```

The API description supplies signatures and semantic guidance, including Move's
four arguments, reference-relative offsets, relational ON, the default Higher
tolerance, and Scattered's separation rule. This is interface knowledge, not an
observation of the sampled scene. The first policy prompt is identical to method
1's for the same instruction, context, and available APIs.

The runner does not automatically insert `num_blocks`, a roster of all handles,
initial coordinates, predicate truth tables, seed, snapshot, or actual controller
configuration into the prompt. In particular, the API text states the default
Higher tolerance; changing the runtime tolerance does not interpolate its value
into that text. User-supplied instructions or direct-call context can contain
additional text, so these statements describe automatic inputs rather than a
filter on arbitrary user text.

Helper requests contain source-level expressions such as `ready(b0)`, not the
runtime value of `b0`. They do not automatically include the entire parent
policy, task instruction, other helper bodies, or environment state. Each HTTP
request carries its assembled prompt as one user text message; the transport does
not add previous responses or a conversation history. Upstream policy history,
when applicable, is explicitly assembled text.

## What the generated policy can observe

The shared adapter installs `b0` and task-available API callables in the Python
namespace. Other blocks are obtained through relational queries; the adapter
keeps the physical block enumeration and handle-to-ID mapping internally.

| Policy-facing interface | Returned information |
| --- | --- |
| `ON(a,b)`, `ON_star(a,b)`, `Higher(a,b)`, `Scattered(a,b)`, `eq(a,b)` | Boolean result computed from the current scene and object identities. |
| `ON_star_zero(a,b)` | Boolean result from frozen trial-entry geometry; exposed only for Reverse. |
| `exists(lambda ...: condition)`, `forall(lambda ...: condition)` | Boolean over the physical-block domain; multiple lambda parameters are allowed. |
| `get(lambda x: condition)` | One opaque block handle: the first matching witness in internal physical-ID order. Absence raises an error. |
| `Pick(b)`, `Move(u,v,w,[dx,dy,dz])`, `Release(b,h)` | `None` after convergence; primitive failure raises an exception. They do not return a perception vector or a success Boolean. |

Queries read current state without stepping physics. The adapter refreshes
observations for predicate evaluation; Reverse's entry observation remains
frozen. Query lambdas cannot execute actions through the adapter. For example,
`if Higher(b0,b0): ...` obtains its Boolean during Python execution. The LLM
writes that expression but never receives its evaluated truth value through the
current generation path.

This is the **advertised API boundary**, not a static compliance guarantee for
the baseline. It executes ordinary Python with upstream `exec_safe` filtering
(`import` and `__`, with `exec`/`eval` replaced); it does not run method 1's type,
provenance, attribute-access, or condition-role checker. Thus passing a rollout
must not be described as proof that its code obeyed the DSL or that the Python
namespace is a secure isolation boundary. Both methods are offered the same
adapter; their enforcement differs.

## Program flow and edge data

Every arrow is labeled with the information or outcome it carries. The LLM
node receives only generation requests. Runtime query results return to the
Python policy, and terminal results go to the host report.

```mermaid
flowchart TD
    T["Trial runner"] -->|"Task, seed or full snapshot, block count"| E["MuJoCo environment and shared controllers"]
    T -->|"Task API names, default b0 context, instruction"| G["Upstream LMP prompt assembly"]
    G -->|"Policy prompt text via shared model transport"| L["Text LLM"]
    L -->|"Policy Python source"| F["Upstream LMPFGen: discover missing helpers"]
    F -->|"API description and missing call or assignment text"| L
    L -->|"Helper definition source"| B["Upstream exec_safe: bind helper definition"]
    B -->|"Helper body source; recursively resolve child calls"| F
    F -->|"Resolved child functions; rebind parent if needed"| B
    F -->|"Policy source and resolved callable namespace"| P["Upstream exec_safe: execute policy Python"]
    T -->|"Adapter wrappers and initial b0 handle"| P
    P -->|"Predicate operands or query lambda"| A["Shared RoboVerify adapter"]
    E -->|"Current observation; entry observation captured once"| A
    A -->|"Predicate truth, quantified truth, or selected handle"| P
    P -->|"Pick, Move, Release operands and finite offsets"| A
    A -->|"Named primitive invocation using shared controller"| E
    E -->|"Controller convergence or exhaustion"| A
    A -->|"None on success; exception on primitive failure"| P
    P -->|"Normal completion or runtime exception"| R["Host terminal evaluator and trial report"]
    F -->|"Generation, binding, or budget failure"| R
    E -->|"Terminal observation and environment goal check"| R
    L -->|"Request/response artifacts and usage through logger"| O["Host logs, checkpoints, trajectories, video"]
    B -->|"Helper source and actual binding checkpoints"| O
    P -->|"Executed source, Python trace, adapter events"| O
    E -->|"Actions, full snapshots, frames"| O
    R -->|"Outcome, failure category, cost totals"| O
```

The repeated LLM node represents separate policy/helper requests, identified by
their input and output labels. Helper definitions are bound before recursive
child resolution, and parents can be rebound after children exist; this preserves
the upstream algorithm. It is not a validate-all-before-binding pipeline.

The baseline has no checker-guided repair request. Original helper generation is
not a repair attempt. Transport retries resend generation requests within the
shared budget; runtime failures do not trigger policy regeneration. The runner
registers no auxiliary query LMPs, so executing a predicate or ordinary helper
does not invoke the model.

## Host-only observations and outcomes

Raw observation arrays, XYZ positions, full simulator snapshots, physical IDs,
goals, rewards, controller errors, images/video, and terminal success are used
by the adapter, recorder, or evaluator. They are not automatically supplied to
the model. Saved artifacts are not automatically read back into prompts.

Physical success requires normal completion and the environment's terminal
success check; unconverged primitives fail execution. Stack/Unstack DSL
postconditions are separate diagnostics. Failures remain failures even when a
partial final scene satisfies the goal. Grid/Pyramid return `adapter_gap` before
any model request. No invariant inference, formal verification feedback, CEM,
or rollout-feedback repair is part of this baseline.

## Implementation references

- [Shared API text and task filtering](../contracts.py).
- [Baseline setup](../cap_google_baseline/method.py),
  [model and execution shims](../cap_google_baseline/compatibility.py), and
  [pinned LMP/LMPFGen source](../vendor/google_cap/core.py).
- [Adapter queries and primitive results](../adapter.py).
- [Trial lifecycle](../runner.py), [task instructions and scoring](../tasks.py),
  and [text request construction](../llm.py).
- [Prompt parity and generation tests](../test_methods.py).
