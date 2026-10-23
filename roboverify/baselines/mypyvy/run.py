"""Reproduce a mypyvy baseline with bounded subprocesses and explicit outcomes."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_COMMIT = "cd80557ae11f5958c7c81dbecf3b24f7a3b16c1f"
PROJECT = Path(__file__).resolve().parents[2]
MODELS = Path(__file__).with_name("models")


def classify(mode, returncode, text, *, timed_out=False):
    """A zero upstream exit alone is insufficient evidence of successful UPDR."""
    if timed_out:
        return "timeout"
    lowered = text.lower()
    if "unknown" in lowered:
        return "unknown"
    if returncode != 0 or any(
        marker in lowered
        for marker in (
            "traceback (most recent call last)",
            "error ",
            "error:",
            "errors found",
            "found 1 errors",
            "failed",
            "not inductive",
        )
    ):
        return "error"
    if mode == "updr":
        if "updr found abstract counterexample!" in text:
            return "abstract_counterexample"
        if (
            "frame is safe and inductive. done!" in text
            and "updr found inductive invariant!" in text
        ):
            return "inferred"
    elif mode == "verify" and "all ok!" in text:
        return "verified"
    elif mode == "trace" and "checked 3 traces and found 0 errors" in text:
        return "consistent"
    elif mode == "typecheck":
        return "typechecked"
    return "inconclusive"


def bounded_process(command, log_path, deadline, heartbeat=lambda: None):
    """Kill the entire process group, including the solver, on timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return dict(returncode=None, timed_out=True, elapsed_seconds=0.0)
    started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                heartbeat()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 0)
                try:
                    proc.wait(timeout=min(2, remaining))
                    timed_out = False
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    return dict(
        returncode=proc.returncode,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
    )


def probe_input(model):
    """Positive initial SAT and negative domain-carrier probes (no invariant)."""
    return model + """
# Initial states must exist; the domain carrier must actually constrain them.
sat trace { assert init }
unsat trace { assert init assert !on_star(b0,b0) }
unsat trace { assert init assert !domain_axioms }
"""


def run(args):
    started = time.monotonic()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("Timeouts must be finite and positive")
    if args.task == "unstack" and args.timeout_seconds > 50:
        raise ValueError(
            "Unstack baseline budget must be at most 50 seconds (60-second invocation limit)"
        )
    if args.query_timeout_ms <= 0:
        raise ValueError("Timeouts must be positive")
    deadline = started + args.timeout_seconds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out = (
        args.output_dir
        or PROJECT / "runs/baselines/mypyvy" / f"{stamp}-{args.task}-{args.solver}"
    ).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version=1,
        family="mypyvy",
        task=args.task,
        solver=args.solver,
        status="error",
        proof_scope="unbounded_abstract_partial_safety",
        supplied_supporting_invariants=False,
        phases=[],
        output_directory=str(out),
    )
    model = (MODELS / f"{args.task}.pyv").read_text()
    (out / "input.pyv").write_text(model)
    config = dict(
        task=args.task,
        solver=args.solver,
        timeout_seconds=args.timeout_seconds,
        query_timeout_ms=args.query_timeout_ms,
        seed=args.seed,
        mode=args.mode,
        model_sha256=hashlib.sha256(model.encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(
            Path(__file__).with_name("worker.py").read_bytes()
        ).hexdigest(),
    )

    def status(phase, state="running"):
        snapshot = dict(
            schema_version=1,
            family="mypyvy",
            task=args.task,
            solver=args.solver,
            status=state,
            current_phase=phase,
            pid=os.getpid(),
            elapsed_seconds=time.monotonic() - started,
            heartbeat_utc=datetime.now(timezone.utc).isoformat(),
            phases=result["phases"],
        )
        temporary = out / "status.json.tmp"
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.replace(out / "status.json")

    status("preflight")
    try:
        source = args.mypyvy_root.resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=3
        ).strip()
        config["mypyvy_commit"] = revision
        if revision != UPSTREAM_COMMIT:
            raise ValueError(
                f"mypyvy must be pinned at {UPSTREAM_COMMIT}; found {revision}"
            )
        if subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "src",
                "script",
            ],
            timeout=3,
        ).returncode:
            raise ValueError("Pinned mypyvy source has local modifications")
        versions_code = "import json,sys,z3,importlib.metadata as m; print(json.dumps(dict(python=sys.version.split()[0], z3=z3.get_version_string(), networkx=m.version('networkx'), flask=m.version('flask'))))"
        versions = subprocess.check_output(
            [args.python, "-c", versions_code], text=True, timeout=5
        )
        config["versions"] = json.loads(versions)
        if config["versions"]["z3"] != "4.16.0":
            raise ValueError(
                "The pinned baseline requires z3-solver==4.16.0.0, including for cvc5 serialization"
            )
        command = [
            args.python,
            "-u",
            "-m",
            "baselines.mypyvy.worker",
            "--mypyvy-root",
            str(source),
        ]
        if args.solver == "cvc5":
            if args.cvc5 is None:
                raise ValueError("--cvc5 must point to the cvc5 1.3.1 executable")
            binary = args.cvc5.resolve()
            version = subprocess.check_output(
                [str(binary), "--version"], text=True, timeout=3
            )
            if not version.startswith("This is cvc5 version 1.3.1 "):
                raise ValueError("The pinned baseline requires cvc5 1.3.1")
            config["cvc5_version"] = version.splitlines()[0]
            config["cvc5_binary_sha256"] = hashlib.sha256(
                binary.read_bytes()
            ).hexdigest()
            command += ["--cvc5", str(binary)]
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        def phase(mode, path, *, certificate=None):
            log = out / f"{mode}.log"
            cmd = command.copy()
            if certificate:
                cmd += ["--certificate", str(certificate)]
            if args.solver == "cvc5":
                cmd += ["--solver-evidence", str(out / f"{mode}-solver.smt2")]
            cmd += [
                "--",
                mode,
                "--timeout",
                str(args.query_timeout_ms),
                "--seed",
                str(args.seed),
                str(path),
            ]
            (out / f"{mode}-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            observed = bounded_process(
                cmd, log, deadline, heartbeat=lambda: status(mode)
            )
            outcome = classify(
                mode,
                observed["returncode"],
                log.read_text() if log.exists() else "",
                timed_out=observed["timed_out"],
            )
            result["phases"].append(dict(phase=mode, status=outcome, **observed))
            result["status"] = outcome
            return outcome

        if (
            phase("typecheck", out / "input.pyv") != "typechecked"
            or args.mode == "typecheck"
        ):
            return result
        probe = out / "consistency.pyv"
        probe.write_text(probe_input(model))
        if phase("trace", probe) != "consistent":
            return result
        certificate = out / "inferred.pyv"
        if phase("updr", out / "input.pyv", certificate=certificate) != "inferred":
            return result
        if not certificate.is_file():
            raise ValueError("UPDR reported inference without writing its certificate")
        # This is a fresh process checking the automatically generated clauses.
        phase("verify", certificate)
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result.update(status="error", reason=str(exc))
        return result
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        status(None, result["status"])


def main(argv=None):
    def interrupted(signum, frame):
        raise InterruptedError(f"Baseline interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["stack", "unstack", "reverse"], required=True
    )
    parser.add_argument("--solver", choices=["z3", "cvc5"], default="cvc5")
    parser.add_argument("--mypyvy-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python with the pinned baseline dependencies",
    )
    parser.add_argument("--cvc5", type=Path)
    parser.add_argument("--mode", choices=["infer", "typecheck"], default="infer")
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--query-timeout-ms", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("verified", "typechecked") else 2


if __name__ == "__main__":
    raise SystemExit(main())
