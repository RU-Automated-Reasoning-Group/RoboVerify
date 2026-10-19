# Revision 4: failure recovery

- **Status:** placeholder; design under discussion. Not implemented.
- **Goal:** investigate recovery from execution failures without conflating framework defects, policy errors, and primitive limitations.
- **Open design choices:**
  - Local recovery in the executing policy versus a new LLM generation/replanning step.
  - What failure information and current observations become available, and to which component.
  - Whether recovery continues from the failed state or restores an explicitly recorded checkpoint.
  - Supported failure classes, recovery budgets, and how attempts count toward task success and resource use.
  - How to compare recovery-enabled variants while preserving the original Google CaP baseline.
- **Current behavior:** primitives raise on nonconvergence; Method 1/2 repairs use static DSL diagnostics, not rollout feedback. No recovery implementation is authorized by this placeholder.
- **Discussion status:** recovery-mode selection is pending. Subsequent design decisions will be recorded here before implementation.
