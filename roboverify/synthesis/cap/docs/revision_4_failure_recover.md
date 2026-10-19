# Revision 4: failure recovery

- **Status:** placeholder; design under discussion. Not implemented.
- **Goal:** use failed execution trajectories to debug the current policy, while distinguishing framework defects, policy errors, and primitive limitations.
- **Open design choices:**
  - How the debugger uses the current policy and failed trajectory to produce a revised candidate.
  - What failure information and current observations become available, and to which component.
  - Supported failure classes, recovery budgets, and how attempts count toward task success and resource use.
  - How to compare recovery-enabled variants while preserving the original Google CaP baseline.
- **Current behavior:** primitives raise on nonconvergence; Method 1/2 repairs use static DSL diagnostics, not rollout feedback. No recovery implementation is authorized by this placeholder.
- **Agreed rollout protocol:**
  - Failed trajectories are debugging input, not continuation states for a revised policy.
  - Every complete policy rollout starts from an initial state. Each revised candidate is invoked from its entry point after restoring the same trial's saved initial simulator snapshot.
  - When evaluating multiple scenes, restore each scene's own initial snapshot for every policy version; do not resume evaluation at a failure or intermediate state.
  - Retain all attempts, trajectories, and resource costs separately.
- **Discussion status:** the rollout/reset rule is settled; the debugging mechanism, feedback content, and budgets remain open. No recovery code has been implemented.
