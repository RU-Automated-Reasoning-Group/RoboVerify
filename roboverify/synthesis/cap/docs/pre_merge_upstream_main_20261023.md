# Pre-merge review: upstream `main` into `feature/cap` (2026-10-23)

- **Purpose:** bring `upstream/main` (RU-Automated-Reasoning-Group/RoboVerify) into `feature/cap` without disturbing running experiments or breaking the provenance of recorded results.
- **Decision:** merge, not rebase. Prepare it in a separate worktree now; move `feature/cap` to the merge commit only after the running feedback ablation (Stage B) has finished. This is option A of the discussion on 2026-10-23.
- **Status:** merge prepared and validated on `merge/upstream-main-20261023` (Section 7). `feature/cap` moves to it after Stage B finishes.

## 1. State before the merge

| Item | Value |
| --- | --- |
| `feature/cap` (pushed to `origin`) | `8d404f7` |
| `upstream/main` | `67d48dc` (2026-10-23 10:00 EDT) |
| Merge base | `1f55b1b` "docs: align WP draft with checked Lean contracts" (2026-10-12) |
| Commits since the base | ours 151, upstream 19; no merge commits on either side |
| Files changed since the base | ours 129; upstream 67 (+4,458 / −3,522 lines) |

**Running experiment.** Revision 4.3 feedback-ablation Stage B, `runs/cap-revision43/20261023-feedback-stage-b`, with 3,150 chains, recorded at `git_sha` `b61c5d0`.
- It runs from the main worktree (`/common/home/lz729/RoboVerify`).
- Every new chain starts a worker process that imports `synthesis/*` from that worktree.
- Changing the main worktree's code before Stage B ends would therefore mix two code versions within one experiment.

The pinned prompt-factorial worktree (`RoboVerify-exp43`, branch `exp/revision43-prompt`) is unaffected.

## 2. What upstream changed

The 19 upstream commits fall into five groups:

- **New and reworked tasks.**
  - Lateral Unstack as the only Unstack environment, plus right-to-left groundtruth validation.
  - A lateral Reverse environment with a primitive groundtruth program.
  - Restack replaces PartialStack: the `RoboVerifyPartialStack` case and `make_roboverify_partial_stack_env` are removed, and `make_roboverify_env("restack")` is added.
  - Grid and Pyramid goal-linked programs, with groundtruth demonstration collection.
  - A demonstration audit for all tasks.
- **Shared runtime references.**
  - `PrimitiveController.reference_position` resolves block IDs, goal references (`GoalRef`), observation references (`ObservationReference`) and the table (`"tbl"`).
  - `move_relative` and `release` now use it. For a block ID it returns the same `box_position` as before.
  - `box_position` additionally rejects non-integer IDs.
- **Environment layout** (`fpp_construction_env.py`, 218 lines): Restack reset, grid/pyramid workspace and spacing, and a goal-occupancy success test for goal tasks. No Stack reset or Stack success path is changed in the diff.
- **Removed experiment runners:**
  - `compare_stack_control`, `compare_stack_paths`, `controller_baseline`, `diagnose_release_drift`, `id_first_continuation`, `render_stack_paths`, `tune_pick_tolerance`;
  - `verify_partial_with_learned_invariant.py`, `verify_reverse_with_learned_invariant.py`.

  Four tests were moved from `synthesis/experiment/` to `synthesis/cfg/` and `synthesis/environment/`.
- **Documentation:** `AGENTS.md`, `README.md`, `PAPER-DISCREPANCIES.md`, scoped Unstack findings.

## 3. Overlap and conflicts

A trial merge in a throwaway worktree (aborted, nothing committed) gave:

| File | Ours | Upstream | Result |
| --- | --- | --- | --- |
| `AGENTS.md` | CaP guidance | Task and status updates | Auto-merged |
| `README.md` | CaP index | Status | Auto-merged |
| `synthesis/api/control.py` | Observer hooks (`_cap_control_event`) in `move` and `gripper`; an isort blank-line change | New imports, `reference_position`, `TABLE_REFERENCE_MODE` | **One conflict**, in the import block only |

**Semantic dependencies of CaP on upstream changes** (no textual conflict):

- `synthesis/cap/tasks.py` and `runner.py` map the CaP task `restack` to the removed `partial_stack` environment, so after the merge CaP Restack would fail at environment creation. Fix: use `restack` directly and drop the alias. This also bears on the deferred Reverse/ReStack specification question in [Revision 5, Section 2.1](revision_5_add_tasks.md); upstream now defines both tasks.
- Two CaP documents link to removed files:
  - [DSL.md](DSL.md): the Reverse/Partial verifier entry points, and `PartialStack` naming;
  - [revision_5_add_tasks.md](revision_5_add_tasks.md): the Reverse/Partial verifier links.

  Update the links and mark the historical statements as pre-merge.
- No CaP module imports a removed runner.

## 4. Why merge rather than rebase

1. **Provenance.** Every recorded CaP batch stores the `git_sha` it ran from, and the Revision 4.x documents cite commit IDs as evidence (for example `b1a6ea9`, `9cbada5`, `bcd80f5`, `ec035a4`). Rebasing would rewrite all 151 commits, and those IDs would no longer be on the branch. The pinned experiment worktree also relies on cherry-picks of these commits.
2. **Published history.** `feature/cap` is pushed to `origin`. A rebase would need a force push.
3. **Effort and risk.** A merge resolves one import-block conflict once. A rebase replays 151 commits, re-resolving `control.py` at `555bedf` and possibly at later commits.

## 5. Procedure (option A)

1. Commit this document on `feature/cap`. A documentation-only change does not affect the running experiment.
2. Create the worktree `/common/home/lz729/RoboVerify-merge` on a new branch `merge/upstream-main-20261023` from `feature/cap`. Link `.venv`, `.runtime` and `runs` to the main worktree's, as for the pinned experiment worktree.
3. `git merge --no-ff upstream/main` and resolve `control.py`: keep upstream's imports and functions and our observer hooks; order the imports with isort.
4. Follow-up commits on the merge branch:
   - CaP `restack` uses the upstream `restack` environment;
   - update the stale CaP document links.
5. Validation in the merge worktree (Section 6).
6. Push `merge/upstream-main-20261023` to `origin`.
7. **After Stage B has finished and its audit and summary are written:** fast-forward `feature/cap` to the merge branch in the main worktree, and push. Also update the serving `PYTHONPATH` if needed; it points at the main worktree, so the guard follows automatically.

## 6. Post-merge checks

- **Full unit suite** from the repository guide (all `synthesis/**/test_*.py`), plus the serving tests in the vLLM environment.
- **Stack behavior is unchanged.** Recorded Stack round-0 attempts are replayed with their recorded model responses (no model calls) on the pre-merge code and on the merged code, from the same saved initial state. The recorded status, primitive outcomes and final block positions must match exactly. A difference would make post-merge CaP results incomparable with Revision 4.x results and must be explained before any new experiment.
- **CaP Restack** environment creation works with the new mapping (smoke test).

## 7. Outcome

**Branch.** `merge/upstream-main-20261023` in the worktree `/common/home/lz729/RoboVerify-merge`, pushed to `origin`:

| Commit | Content |
| --- | --- |
| `c6845a5` | `Merge upstream main (67d48dc) into feature/cap`. The only conflict was the `control.py` import block, resolved to upstream's imports; upstream's `reference_position` and CaP's observer hooks are both kept. |
| `6233ef4` | CaP `restack` creates the upstream Restack environment. Archives recorded as `partial_stack` still match. |
| `f28e9b5` | [DSL.md](DSL.md) and [Revision 5](revision_5_add_tasks.md) updated for Restack, the removed verifiers, and the shared Reverse/Restack `task_spec`. |

**Checks (2026-10-23):**

| Check | Result |
| --- | --- |
| Full unit suite (all `synthesis/**/test_*.py`) | 621 tests, OK (4 skipped), 269 s |
| Guard tests in the vLLM environment | OK |
| Stack replay: 30 Stage A round-0 attempts (5 successes and 5 failures per method) re-executed with their recorded responses, once on the pre-merge code (main worktree) and once on the merged code | Pre-merge replay = original recording: 30/30. Merged = pre-merge on status, reason, success flags and step counts: 30/30. Identical per-step action sequences: 30/30. Outcomes covered: 15 completed, 11 primitive failures, 3 goal failures, 1 budget exhaustion; 50–2,314 steps each. |
| CaP Restack environment creation | `RoboVerifyRestack` created |

- **Conclusion.** The merge leaves CaP Stack execution unchanged, so post-merge Stack results remain comparable with Revision 4.x results.
- **Replay artifacts:** `runs/cap-merge-check/20261023-stack-replay/{old,new}`.
- **Fast-forward (2026-10-24).** Stage B finished at 03:59 EDT (audit: no issues). `feature/cap` was then fast-forwarded in the main worktree from `f74a9c9` to `60fb26f`, the last validated commit pushed on this branch, and pushed.
- Later commits on `merge/upstream-main-20261023` (Revision 5 work in the merge worktree) reach `feature/cap` by an ordinary merge.
