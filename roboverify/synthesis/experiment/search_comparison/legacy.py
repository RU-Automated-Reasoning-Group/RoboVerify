"""Separate production CFG-MCMC reference, not the controlled matched arm.

The unchanged scoped mutation, KL objective, CEM and Metropolis implementation
consume public training observations. Physical audits are recorded independently;
they never replace the production search objective or select a parameter vector.
"""

from time import perf_counter

_MODULE_STARTED = perf_counter()

import argparse
import hashlib
import json
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LegacyConfig:
    task: str
    seed: int = 0
    wall_seconds: float = 48.0
    max_rollouts: int = 45
    slots: int = 8
    distance: str = "kl_kde"
    theta: float = 0.0

    def __post_init__(self):
        if self.task not in ("stack", "unstack") or self.distance not in (
            "kl_kde",
            "mmd",
        ):
            raise ValueError("Unsupported legacy reference task or metric")
        if not 0 < self.wall_seconds <= 49 or self.max_rollouts < 1 or self.slots < 1:
            raise ValueError(
                "Positive legacy budgets within the 49-second maximum required"
            )

    def search_budget(self):
        from synthesis.cfg.straightline import SearchBudget

        return SearchBudget(seed=self.seed, distance=self.distance, theta=self.theta)


def encode(program):
    from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName, Skip

    rows = []
    for instruction in program.instructions:
        operands = [operand["val"] for operand in instruction.get_operand()]
        if isinstance(instruction, Skip):
            rows.append(
                dict(
                    kind="Skip",
                    operands=[],
                    offsets=[],
                    skip_steps=instruction.skip_steps,
                )
            )
            continue
        if type(instruction) is PickByName:
            kind, parameters = "Pick", []
        elif type(instruction) is MoveByName:
            kind, parameters = "Move", instruction.target_offset
        elif type(instruction) is ReleaseByName:
            kind, parameters = "Release", [instruction.target_z_offset]
        else:
            raise ValueError(
                f"Unsupported legacy instruction: {type(instruction).__name__}"
            )
        rows.append(
            dict(
                kind=kind,
                operands=operands,
                offsets=[p.concrete_float("legacy reference") for p in parameters],
            )
        )
    return rows


def adapt_examples(examples):
    from synthesis.cfg.demos import DemoSegment, DemoTrace

    segments = []
    for index, example in enumerate(examples):
        observations = example["observations"][: example["execution_end_index"] + 1]
        trace = DemoTrace(
            tuple(observations),
            snapshots=(example["initial_snapshot"],),
            seed=example["seed"],
            task=example["task"],
            num_blocks=example["num_blocks"],
        )
        segments.append(
            DemoSegment(index, 0, len(observations) - 1, trace, example["bindings"])
        )
    return segments


def local_post(task):
    from synthesis.predicates.term import (
        atom,
        conjunction,
        disjunction,
        forall,
        implies,
        ref,
    )

    source = ref("b_prime")
    if task == "stack":
        return atom("ON", source, ref("b"))
    obj = ref("other")
    return conjunction(
        # The relational table is an isolated sentinel. Table surface height
        # belongs only to the independent physical acceptance audit below.
        forall(
            ["other"], implies(atom("ON_star", source, obj), atom("eq", source, obj))
        ),
        forall(
            ["other"],
            disjunction(
                atom("eq", obj, ref("tbl")),
                atom("eq", obj, source),
                atom("Scattered", source, obj),
            ),
        ),
    )


class LegacyRollouts:
    """Bound and audit real production rollout requests without caching them."""

    def __init__(
        self,
        examples,
        config,
        output,
        *,
        deadline,
        start,
        rollout_fn=None,
        env_factory=None,
    ):
        from synthesis.cfg.straightline import _features
        from synthesis.mcmc.distance import make_distance

        from .metrics import ArcMetric

        self.examples, self.config = examples, config
        self.output, self.deadline, self.start = Path(output), deadline, start
        self.output.mkdir(parents=True, exist_ok=True)
        self.rollout_fn, self.env_factory, self.env = rollout_fn, env_factory, None
        self.budget = config.search_budget()
        self.metric = make_distance(
            config.distance,
            np.concatenate(
                [
                    _features(
                        e["observations"][: e["execution_end_index"] + 1],
                        e["num_blocks"],
                    )
                    for e in examples
                ]
            ),
        )
        self.arc = ArcMetric(examples)
        self.counts = dict(
            simulator_rollouts=0,
            completed_vector_batches=0,
            interrupted_vector_batches=0,
            mutation_calls=0,
            bmc_candidates=0,
            audit_metric_seconds=0.0,
        )
        self.pending = None
        self.best = None

    def _write(self, name, row):
        with (self.output / name).open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _environment(self):
        if self.env is None:
            if self.env_factory is None:
                from synthesis.mcmc.synthesis import (
                    make_roboverify_env,
                    preserved_global_rng,
                    set_np_seed,
                )

                with preserved_global_rng():
                    set_np_seed(0)
                    self.env = make_roboverify_env(
                        self.config.task, num_blocks=self.examples[0]["num_blocks"]
                    )
                    self.env.reset()
            else:
                self.env = self.env_factory()
        return self.env

    def __call__(self, program, segment):
        from synthesis.cfg.straightline import _features

        from .numeric import EvaluationBudgetExceeded, program_key
        from .safety import run_rows

        if perf_counter() >= self.deadline:
            raise EvaluationBudgetExceeded("Legacy wall budget exhausted")
        rows, index = encode(program), segment.demo_idx
        key = program_key(rows)
        if index == 0:
            if self.pending is not None:
                raise ValueError("Legacy caller abandoned a partial vector batch")
            if (
                self.counts["simulator_rollouts"] + len(self.examples)
                > self.config.max_rollouts
            ):
                raise EvaluationBudgetExceeded(
                    "Legacy cap cannot cover another complete vector batch"
                )
            self.pending = dict(
                instructions=rows, program_sha256=key, trajectories=[], diagnostics=[]
            )
        if (
            self.pending is None
            or self.pending["program_sha256"] != key
            or len(self.pending["trajectories"]) != index
        ):
            raise ValueError("Legacy vector batch changed parameters or example order")
        example = self.examples[index]
        self.counts["simulator_rollouts"] += 1
        observations, checks = (self.rollout_fn or run_rows)(
            self._environment(), rows, example, settle_steps=50, drift_tolerance_m=0.005
        )
        if any(
            "Timeout" in str(checks.get(field, ""))
            for field in ("error", "callback_error")
        ):
            raise EvaluationBudgetExceeded(
                "Physical rollout was interrupted by the wall deadline"
            )
        if not np.array_equal(observations[0], example["observations"][0]):
            raise AssertionError(
                "Legacy rollout did not restore the archived entry exactly"
            )
        states = observations[: checks["execution_end_index"] + 1]
        self.pending["trajectories"].append(states)
        self.pending["diagnostics"].append(checks)
        self._write(
            "rollouts.jsonl",
            dict(
                program_sha256=key,
                example=example["id"],
                accepted=checks["eligible"],
                elapsed_seconds=perf_counter() - self.start,
            ),
        )
        if index + 1 == len(self.examples):
            trajectories = self.pending["trajectories"]
            metric_started = perf_counter()
            # This independent ledger uses exactly the production inner metric.
            # It neither supplies scores to the search nor changes its incumbent.
            imitation = self.metric(
                np.concatenate(
                    [
                        _features(states, e["num_blocks"])
                        for states, e in zip(trajectories, self.examples)
                    ]
                ),
                samples=self.budget.inner_samples,
            )
            arc = self.arc(trajectories)
            audit_seconds = perf_counter() - metric_started
            self.counts["audit_metric_seconds"] += audit_seconds
            record = {k: v for k, v in self.pending.items() if k != "trajectories"}
            record.update(
                complete=True,
                imitation_distance=imitation,
                arc_rms_m=arc,
                audit_metric_seconds=audit_seconds,
                accepted_examples=sum(d["eligible"] for d in record["diagnostics"]),
                strict_1mm_examples=sum(
                    d["strict_1mm_eligible"] for d in record["diagnostics"]
                ),
                elapsed_seconds=perf_counter() - self.start,
            )
            self.counts["completed_vector_batches"] += 1
            self._write("evaluations.jsonl", record)
            if self.best is None or imitation < self.best["imitation_distance"]:
                self.best = deepcopy(record)
            self.pending = None
        return states

    def close(self):
        if self.pending is not None:
            self.counts["interrupted_vector_batches"] += 1
            self._write(
                "evaluations.jsonl",
                dict(
                    complete=False,
                    program_sha256=self.pending["program_sha256"],
                    completed_examples=len(self.pending["trajectories"]),
                    status="incomplete_vector_batch",
                ),
            )
            self.pending = None
        if self.env is not None:
            self.env.close()


def source_hashes():
    from .run import sha
    from .run import source_hashes as shared_sources

    result = shared_sources()
    root = Path(__file__).resolve().parents[2]
    for name in (
        "cfg/straightline.py",
        "cfg/bindings.py",
        "mcmc/distance.py",
        "mcmc/synthesis.py",
    ):
        path = root / name
        result[str(path.relative_to(root.parent))] = sha(path)
    return result


def freeze_protocol(path, roots, *, seeds=tuple(range(10)), config=None):
    from .data import load_training

    config = config or LegacyConfig("stack")
    if set(roots) != {"stack", "unstack"}:
        raise ValueError("Legacy protocol requires both declared task inputs")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Distinct nonnegative run seeds required")
    training_inputs = {}
    for task, root in roots.items():
        examples = load_training(root, task)
        if [example["seed"] for example in examples] != list(range(15)):
            raise ValueError(
                "Legacy reference requires exactly training seeds 0 through 14"
            )
        training_inputs[task] = {e["id"]: e["public_sha256"] for e in examples}
    common = {k: v for k, v in asdict(config).items() if k not in ("task", "seed")}
    record = dict(
        schema="rv-legacy-cfg-protocol-v1",
        tasks=["stack", "unstack"],
        seeds=list(seeds),
        config=common,
        search_budget=asdict(config.search_budget()),
        source_hashes=source_hashes(),
        training_inputs=training_inputs,
        external_cap_seconds=55,
        pilot_included=False,
        scope="Separate legacy production reference; not the matched algorithm comparison.",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def run_trial(config, data_root, output, *, protocol=None, pilot=False, started=None):
    from synthesis.api.instructions import Skip
    from synthesis.api.program import Program
    from synthesis.cfg.bindings import mutate_scoped
    from synthesis.cfg.collection import execution_deadline
    from synthesis.cfg.straightline import straight_line_synthesize
    from synthesis.mcmc.synthesis import preserved_global_rng, set_np_seed

    from .data import load_training
    from .numeric import EvaluationBudgetExceeded, program_key

    start = perf_counter() if started is None else started
    deadline = start + config.wall_seconds
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite a legacy trial")
    out.mkdir(parents=True, exist_ok=True)
    budget = config.search_budget()
    report = dict(
        schema="rv-legacy-cfg-v1",
        method="legacy_cfg_mcmc",
        config=asdict(config),
        search_budget=asdict(budget),
        pilot=bool(pilot),
        status="running",
        heldout_read=False,
        source_hashes=source_hashes(),
        formal_verification="not_run",
        production_returned=False,
        production_converged=False,
        body_template_supplied=False,
        initial_program=[dict(kind="Skip", skip_steps=20)] * config.slots,
    )
    (out / "config.json").write_text(json.dumps(report, indent=2) + "\n")
    audit = None
    try:
        if (protocol is None) == (not pilot):
            raise ValueError(
                "Specify exactly one excluded pilot or frozen legacy protocol"
            )
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise EvaluationBudgetExceeded("Imports consumed the legacy budget")
        with execution_deadline(remaining), preserved_global_rng():
            set_np_seed(config.seed)
            examples = load_training(data_root, config.task)
            report["training_inputs"] = {e["id"]: e["public_sha256"] for e in examples}
            if protocol:
                frozen = json.loads(Path(protocol).read_text())
                common = {
                    k: v for k, v in asdict(config).items() if k not in ("task", "seed")
                }
                if (
                    frozen["schema"] != "rv-legacy-cfg-protocol-v1"
                    or config.task not in frozen["tasks"]
                    or config.seed not in frozen["seeds"]
                    or common != frozen["config"]
                    or frozen["source_hashes"] != source_hashes()
                    or frozen["training_inputs"][config.task]
                    != report["training_inputs"]
                ):
                    raise ValueError(
                        "Legacy sources, budgets or training inputs differ from frozen protocol"
                    )
                report["protocol_sha256"] = hashlib.sha256(
                    Path(protocol).read_bytes()
                ).hexdigest()
            segments = adapt_examples(examples)
            scope = set.intersection(*(set(e["bindings"]) for e in examples))
            audit = LegacyRollouts(
                examples, config, out, deadline=deadline, start=start
            )

            def propose(candidate, rng):
                audit.counts["mutation_calls"] += 1
                return mutate_scoped(candidate, scope)

            initial = Program(config.slots, [Skip() for _ in range(config.slots)])
            result = straight_line_synthesize(
                segments,
                local_post(config.task),
                initial,
                propose,
                rollout=audit,
                budget=budget,
            )
            report.update(
                status=result.status,
                production_returned=True,
                production_converged=result.ok,
                production_distance=result.distance,
                production_post_score=result.post_score,
                production_instructions=encode(result.program),
            )
    except TimeoutError as exc:
        report.update(status="budget_exhausted", termination_reason=str(exc))
    except Exception as exc:
        report.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        if audit is not None:
            audit.close()
            report.update(
                evaluation_counts=audit.counts, best_complete_vector=audit.best
            )
            report["training_accepted"] = bool(
                audit.best and audit.best["accepted_examples"] == len(audit.examples)
            )
            if audit.best is not None:
                # Skip records observations without advancing physics. Preserve
                # it in the fitting ledger; the physical export omits only those.
                rows = [r for r in audit.best["instructions"] if r["kind"] != "Skip"]
                candidate = dict(
                    task=config.task,
                    instructions=rows,
                    program_sha256=program_key(rows),
                    source_trial_method="legacy_cfg_mcmc",
                    source_trial_seed=config.seed,
                    training_accepted=report["training_accepted"],
                    complete_training_batch=True,
                    frozen_before_heldout=True,
                    formal_verification="not_run",
                    selection="Lowest production inner imitation distance among complete vector batches; not a successful production return",
                    physical_export_omits_observation_only_skips=True,
                )
                (out / "candidate.json").write_text(
                    json.dumps(candidate, indent=2) + "\n"
                )
        report.setdefault("training_accepted", False)
        report["elapsed_seconds"] = perf_counter() - start
        report["scope"] = (
            "Actual CFG mutate_scoped and straight_line_synthesize, including production CEM, KL/MMD, "
            "epsilon pool and PostScore. Eight mutable Skip slots; no supplied body. No nominal gate, "
            "no BMC in this existing CFG-MCMC route, no rollout cache and no observation scalar operands "
            "in its mutation grammar. Physical audits do not affect its objective. The complete-vector "
            "ledger is separate from production convergence. Different grammar/metric/initialization "
            "make this a separate reference, not the controlled algorithm comparison."
            " Unstack's symbolic local post requires detached/scattered source geometry; physical "
            "table height is checked only in the independent acceptance audit. Independent ledger "
            "KL/arc computation consumes the same wall budget and its overhead is recorded."
        )
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["stack", "unstack"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--protocol")
    args = parser.parse_args(argv)
    report = run_trial(
        LegacyConfig(args.task, args.seed),
        args.data,
        args.output,
        protocol=args.protocol,
        pilot=args.pilot,
        started=_MODULE_STARTED,
    )
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "training_accepted",
                    "elapsed_seconds",
                    "evaluation_counts",
                    "error",
                )
            }
        )
    )
    return int(report["status"] == "error")


if __name__ == "__main__":
    raise SystemExit(main())
