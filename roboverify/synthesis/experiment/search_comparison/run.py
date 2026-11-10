"""Run one bounded, paired-protocol structural-search trial.

Example (training-only implementation pilot):
  python -m synthesis.experiment.search_comparison.run --pilot --task stack \
    --method beam --data PATH --output NEW_DIRECTORY

Main experiments require a frozen --protocol file produced before any main
trial. Every result reports a candidate's empirical training acceptance separately
from search termination; no result here is formal verification.
"""

from time import perf_counter

_MODULE_STARTED = perf_counter()

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict, fields
from pathlib import Path

from .protocol import TrialConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    # Include transitive verification, reset, scoring and geometry code rather
    # than just the entry points. Tests and generated runtime artifacts are not
    # executable inputs to this experiment.
    paths = [p for p in root.rglob("*.py") if not p.name.startswith("test_")]
    assets = root / "environment/cee_us_env/fpp_construction/assets"
    paths += [assets / "fetch/robot.xml", assets / "fetch/shared.xml"]
    paths += list((assets / "stls").rglob("*.stl"))
    paths += [root.parent / "pyproject.toml", root.parent / "uv.lock"]
    return {str(p.relative_to(root.parent)): sha(p) for p in sorted(paths)}


def validate_protocol(path, config, examples):
    frozen = json.loads(Path(path).read_text())
    if (
        config.task not in frozen["tasks"]
        or config.method not in frozen["methods"]
        or config.seed not in frozen["seeds"]
    ):
        raise ValueError("Trial is outside the frozen protocol")
    common = {
        k: v for k, v in config.as_dict().items() if k not in {"task", "method", "seed"}
    }
    if common != frozen["config"]:
        raise ValueError("Trial budgets differ from the frozen protocol")
    if source_hashes() != frozen["source_hashes"]:
        raise ValueError("Runtime sources changed after protocol freeze")
    current = {e["id"]: e["public_sha256"] for e in examples}
    if current != frozen["training_inputs"][config.task]:
        raise ValueError("Training files differ from the frozen protocol")
    return sha(path)


def run_trial(config, data_root, output, *, protocol=None, pilot=False, started=None):
    start = perf_counter() if started is None else started
    from synthesis.cfg.collection import execution_deadline
    from synthesis.search.contract import (
        SearchConfig,
        TransitionBudgetExceeded,
        beam_search,
    )

    from .data import load_training, make_problem
    from .matched import MatchedConfig
    from .matched import search as matched_search
    from .numeric import EvaluationBudgetExceeded, SharedEvaluator

    deadline = start + config.wall_seconds
    structure_deadline = start + config.structure_seconds
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite an existing trial directory")
    out.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema="rv-matched-search-v1",
        status="running",
        config=config.as_dict(),
        pilot=bool(pilot),
        formal_verification="not_run",
        heldout_read=False,
        initial_program=[],
        source_hashes=source_hashes(),
    )
    (out / "config.json").write_text(json.dumps(report, indent=2) + "\n")
    evaluator = None
    try:
        if (protocol is None) == (not pilot):
            raise ValueError(
                "Specify exactly one of a frozen protocol or an excluded pilot"
            )
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise EvaluationBudgetExceeded("Imports consumed the trial budget")
        with execution_deadline(remaining):
            examples = load_training(data_root, config.task)
            report["training_inputs"] = {e["id"]: e["public_sha256"] for e in examples}
            if protocol:
                report["protocol_sha256"] = validate_protocol(
                    protocol, config, examples
                )
            report["data_seconds"] = perf_counter() - start
            problem = make_problem(examples, deadline=structure_deadline)
            report["contract_config"] = asdict(problem.config)
            problem.transition_limit = config.max_transition_trials
            report["action_corpus"] = [a.row() for a in problem.actions]
            report["action_corpus_sha256"] = hashlib.sha256(
                json.dumps(report["action_corpus"], sort_keys=True).encode()
            ).hexdigest()
            report["setup_seconds"] = perf_counter() - start
            if config.method == "beam":
                result = beam_search(
                    problem,
                    SearchConfig(
                        max_length=config.max_length,
                        beam_width=config.beam_width,
                        max_expansions=config.max_expansions,
                        max_transition_trials=config.max_transition_trials,
                        seconds=config.structure_seconds,
                        deadline=structure_deadline,
                        memoize=True,
                        diversity=True,
                        pool_limit=config.shortlist,
                    ),
                )
                candidates, stats, status = (
                    result.candidates,
                    result.stats,
                    result.status,
                )
            else:
                candidates, stats, status = matched_search(
                    problem,
                    MatchedConfig(
                        max_length=config.max_length,
                        max_transition_trials=config.max_transition_trials,
                        seconds=config.structure_seconds,
                        seed=config.seed,
                        temperature=config.temperature,
                        decay=config.decay,
                        floor=config.temperature_floor,
                        shortlist=config.shortlist,
                    ),
                    deadline=structure_deadline,
                )
            report["structure"] = dict(
                status=status,
                stats=stats,
                elapsed_seconds=perf_counter() - start,
                candidates=[
                    dict(
                        instructions=[a.row() for a in n.actions], nominal_rank=n.score
                    )
                    for n in candidates
                ],
            )
            (out / "structure.json").write_text(
                json.dumps(report["structure"], indent=2) + "\n"
            )
            if not candidates:
                report["status"] = "no_nominal_goal_candidate"
            else:
                problem.deadline = deadline
                problem.transition_limit = None
                numeric_transition_start = problem.counts["transition_trials"]
                evaluator = SharedEvaluator(
                    problem, examples, config, out, deadline=deadline, start=start
                )
                evaluator.numeric_transition_start = numeric_transition_start
                for candidate in candidates:
                    evaluator.fit_candidate([a.row() for a in candidate.actions])
                report["status"] = "shortlist_exhausted"
    except (TimeoutError, TransitionBudgetExceeded) as exc:
        report.update(
            status=(
                "budget_exhausted_with_candidate"
                if evaluator is not None and evaluator.best is not None
                else "budget_exhausted_without_candidate"
            ),
            termination_reason=str(exc),
        )
    except Exception as exc:
        report.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        if evaluator is not None:
            report["evaluation_counts"] = evaluator.counts
            report["evaluation_counts"]["numeric_transition_trials"] = (
                evaluator.problem.counts["transition_trials"]
                - evaluator.numeric_transition_start
            )
            report["selected"] = evaluator.best
            if evaluator.best is not None:
                selected = evaluator.best
                report["training_accepted"] = selected["accepted_examples"] == len(
                    evaluator.examples
                )
                candidate = dict(
                    task=config.task,
                    instructions=selected["instructions"],
                    program_sha256=selected["program_sha256"],
                    source_trial_method=config.method,
                    source_trial_seed=config.seed,
                    complete_training_batch=True,
                    training_accepted=report["training_accepted"],
                    arc_rms_m=selected["arc_rms_m"],
                    frozen_before_heldout=True,
                    formal_verification="not_run",
                )
                (out / "candidate.json").write_text(
                    json.dumps(candidate, indent=2) + "\n"
                )
            evaluator.close()
        report.setdefault("training_accepted", False)
        report["elapsed_seconds"] = perf_counter() - start
        report["scope"] = (
            "Budgeted empirical body search. Best fully evaluated shared vector retained; no partial vector promotion and no formal verification. Matched MCMC is a controlled baseline with shared proposals/contracts, not unchanged legacy MCMC."
        )
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["stack", "unstack"], required=True)
    p.add_argument("--method", choices=["beam", "matched_mcmc"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--protocol")
    group.add_argument("--pilot", action="store_true")
    defaults = TrialConfig("stack", "beam")
    for field in fields(defaults):
        if field.name not in {"task", "method", "seed"}:
            value = getattr(defaults, field.name)
            p.add_argument(
                "--" + field.name.replace("_", "-"), type=type(value), default=value
            )
    a = p.parse_args(argv)
    config = TrialConfig(**{f.name: getattr(a, f.name) for f in fields(defaults)})
    result = run_trial(
        config,
        a.data,
        a.output,
        protocol=a.protocol,
        pilot=a.pilot,
        started=_MODULE_STARTED,
    )
    print(
        json.dumps(
            {
                k: result.get(k)
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
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
