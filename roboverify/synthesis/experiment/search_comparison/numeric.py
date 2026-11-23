"""One physical evaluator and CEM implementation shared by both search arms."""

from __future__ import annotations

import contextlib
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .metrics import ArcMetric, objective
from .protocol import candidate_seed


class EvaluationBudgetExceeded(TimeoutError):
    pass


def program_key(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def with_parameters(rows, parameters):
    values = np.asarray(parameters, float)
    result, index = [], 0
    for row in rows:
        count = len(row["offsets"])
        result.append(dict(row, offsets=values[index : index + count].tolist()))
        index += count
    if index != len(values):
        raise ValueError("Parameter dimension differs from instruction offsets")
    return result


def bmc_screen(rows, examples, *, deadline, timeout_ms=250):
    """Production transition screen at every input, before any physical rollout."""
    from synthesis.api.program import Program
    from synthesis.cfg.search_feasibility import check_scene_feasibility
    from synthesis.predicates.scene import Scene
    from synthesis.util.on import TABLE

    from .safety import decode

    candidate = Program(len(rows), decode(rows))
    reports = []
    for example in examples:
        remaining = int(1000 * (deadline - perf_counter()))
        if remaining <= 1:
            raise EvaluationBudgetExceeded("Budget exhausted during BMC screening")
        # Archives retain the relational table binding even when this experiment's
        # action alphabet excludes table coordinates. Preserve that binding with
        # its sentinel; do not invent a runtime table reference.
        scene_positions = dict(example["initial_positions"])
        if "tbl" in example["bindings"].values():
            scene_positions["tbl"] = TABLE
        result = check_scene_feasibility(
            candidate,
            Scene(scene_positions, example["bindings"]),
            gripper=example["observations"][0, :3],
            initial_holding=None,
            fixed_offsets=False,
            timeout_ms=min(timeout_ms, remaining),
        )
        reports.append(asdict(result))
        if not result:
            break
    return {
        "feasible": len(reports) == len(examples)
        and all(r["feasible"] for r in reports),
        "scenes": reports,
        "scope": "Per-entry existential offsets, goal=True; no shared-parameter, physical-safety or postcondition certificate",
    }


class SharedEvaluator:
    """Retains only complete shared-vector batches; interrupted work cannot win."""

    def __init__(
        self,
        problem,
        examples,
        config,
        output,
        *,
        deadline,
        start,
        rollout_fn=None,
        env_factory=None,
        metric=None,
    ):
        self.problem, self.examples, self.config = problem, examples, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline, self.start = deadline, start
        self.rollout_fn, self.env_factory = rollout_fn, env_factory
        self.metric = metric or ArcMetric(examples)
        self.env = None
        self.cache, self.records = {}, []
        self.best = None
        self.last_complete_batch_seconds = None
        self.counts = dict(
            objective_calls=0,
            cache_hits=0,
            nominal_rejected=0,
            radius_rejected=0,
            simulator_rollouts=0,
            completed_simulator_rollouts=0,
            completed_vector_batches=0,
            interrupted_vector_batches=0,
            bmc_candidates=0,
            bmc_rejected=0,
        )

    def remaining(self):
        return self.deadline - perf_counter()

    def ensure_budget(self, *, complete_batch=False):
        if self.remaining() <= 0.5:
            raise EvaluationBudgetExceeded("Total trial deadline reached")
        if complete_batch:
            if (
                self.counts["simulator_rollouts"] + len(self.examples)
                > self.config.max_rollouts
            ):
                raise EvaluationBudgetExceeded(
                    "Insufficient shared rollout budget for all examples"
                )
            # Once measured, avoid beginning a batch unlikely to finish. The first
            # batch has no data-dependent prediction and is protected by deadline.
            if (
                self.last_complete_batch_seconds is not None
                and self.remaining() <= 1.05 * self.last_complete_batch_seconds + 0.5
            ):
                raise EvaluationBudgetExceeded(
                    "Insufficient time for another complete vector batch"
                )

    def _environment(self):
        if self.env is None:
            if self.env_factory is None:
                from synthesis.mcmc.synthesis import (
                    make_roboverify_env,
                    preserved_global_rng,
                )

                with preserved_global_rng():
                    np.random.seed(0)
                    self.env = make_roboverify_env(
                        self.examples[0]["task"],
                        num_blocks=self.examples[0]["num_blocks"],
                    )
                    self.env.reset()
            else:
                self.env = self.env_factory()
        return self.env

    def _record(self, record):
        record["elapsed_seconds"] = perf_counter() - self.start
        self.records.append(record)
        with (self.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def evaluate(self, base_rows, parameters):
        self.ensure_budget()
        self.counts["objective_calls"] += 1
        rows = with_parameters(base_rows, parameters)
        key = program_key(rows)
        record = dict(instructions=rows, program_sha256=key, complete=False)
        initial = np.asarray([x for row in base_rows for x in row["offsets"]])
        if (
            not np.isfinite(parameters).all()
            or np.max(np.abs(np.asarray(parameters) - initial), initial=0)
            > self.config.parameter_radius
        ):
            self.counts["radius_rejected"] += 1
            record.update(
                status="parameter_domain_rejected",
                objective=-1000.0,
                accepted_examples=0,
            )
            # Radius is relative to this structural proposal, not just the
            # resulting row vector. Never share a domain rejection or bypass a
            # new proposal's radius check through the physical-result cache.
            self._record(record)
            return record["objective"]
        elif key in self.cache:
            self.counts["cache_hits"] += 1
            return self.cache[key]["objective"]
        else:
            from synthesis.search.contract import Action

            node = self.problem.replay(
                tuple(
                    Action(row["kind"], tuple(row["operands"]), tuple(row["offsets"]))
                    for row in rows
                )
            )
            if node is None:
                self.counts["nominal_rejected"] += 1
                record.update(
                    status="nominal_rejected", objective=-100.0, accepted_examples=0
                )
            else:
                self.ensure_budget(complete_batch=True)
                batch_start = perf_counter()
                trajectories, diagnostics = [], []
                if self.rollout_fn is None:
                    from .safety import run_rows

                    rollout = run_rows
                else:
                    rollout = self.rollout_fn
                env = self._environment()
                try:
                    for example in self.examples:
                        self.ensure_budget()
                        self.counts["simulator_rollouts"] += 1
                        observations, checks = rollout(
                            env, rows, example, settle_steps=50, drift_tolerance_m=0.005
                        )
                        self.counts["completed_simulator_rollouts"] += 1
                        for field in ("error", "callback_error"):
                            if checks.get(field) and "Timeout" in checks[field]:
                                raise EvaluationBudgetExceeded(checks[field])
                        restore_error = float(
                            np.max(np.abs(observations[0] - example["observations"][0]))
                        )
                        if restore_error != 0:
                            raise AssertionError(
                                "Physical rollout entry differs from recorded snapshot"
                            )
                        checks["restore_error"] = restore_error
                        trajectories.append(
                            observations[: checks["execution_end_index"] + 1]
                        )
                        diagnostics.append(checks)
                        with (self.output / "rollouts.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(
                                    dict(
                                        program_sha256=key,
                                        example=example["id"],
                                        accepted=checks["eligible"],
                                        elapsed_seconds=perf_counter() - self.start,
                                    )
                                )
                                + "\n"
                            )
                    self.ensure_budget()
                except TimeoutError as exc:
                    self.counts["interrupted_vector_batches"] += 1
                    record.update(
                        status="incomplete_vector_batch",
                        completed_examples=len(diagnostics),
                        error=str(exc),
                    )
                    self._record(record)
                    raise
                distance = self.metric(trajectories)
                passed = sum(d["eligible"] for d in diagnostics)
                score = objective(distance, len(self.examples) - passed)
                self.last_complete_batch_seconds = perf_counter() - batch_start
                self.counts["completed_vector_batches"] += 1
                record.update(
                    status="physically_evaluated",
                    complete=True,
                    objective=score,
                    accepted_examples=passed,
                    arc_rms_m=distance,
                    strict_1mm_examples=sum(
                        d["strict_1mm_eligible"] for d in diagnostics
                    ),
                    diagnostics=diagnostics,
                    batch_seconds=self.last_complete_batch_seconds,
                )
                if self.best is None or score > self.best["objective"]:
                    self.best = deepcopy(record)
                    self.best["discovery_seconds"] = perf_counter() - self.start
                    with (self.output / "anytime-best.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    k: self.best[k]
                                    for k in (
                                        "program_sha256",
                                        "objective",
                                        "accepted_examples",
                                        "arc_rms_m",
                                        "discovery_seconds",
                                    )
                                }
                            )
                            + "\n"
                        )
        self.cache[key] = record
        self._record(record)
        return record["objective"]

    def fit_candidate(self, rows):
        """Same production CEM and proposal-keyed noise, independent of search order."""
        from synthesis.mcmc.cem import cem_optimize
        from synthesis.mcmc.synthesis import preserved_global_rng

        self.ensure_budget()
        screen = bmc_screen(
            rows,
            self.examples,
            deadline=self.deadline,
            timeout_ms=self.config.solver_timeout_ms,
        )
        self.counts["bmc_candidates"] += 1
        with (self.output / "bmc.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(program_sha256=program_key(rows), **screen)) + "\n"
            )
        if not screen["feasible"]:
            self.counts["bmc_rejected"] += 1
            return
        initial = np.asarray([value for row in rows for value in row["offsets"]], float)
        with preserved_global_rng(), (self.output / "cem.log").open(
            "a"
        ) as stream, contextlib.redirect_stdout(stream):
            np.random.seed(candidate_seed(self.config.seed, rows))
            cem_optimize(
                lambda p: self.evaluate(rows, p),
                len(initial),
                iterations=self.config.cem_iterations if len(initial) else 0,
                N=self.config.cem_samples,
                K=self.config.cem_elites,
                init_mu=initial,
                init_std=self.config.cem_std,
                num_workers=0,
            )

    def close(self):
        if self.env is not None:
            self.env.close()
