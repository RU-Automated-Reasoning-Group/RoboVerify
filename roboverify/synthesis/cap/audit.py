"""Artifact-only final-output compliance; never changes baseline execution."""

import ast
import copy
import hashlib
import json
import time
from pathlib import Path

from .checker import DSLContract, analysis_deadline, check_bundle
from .contracts import API_NAMES, Bundle, ConstraintFailure, Diagnostic
from .validation import definitions, parse


def read_final_output(run_dir):
    root = Path(run_dir).resolve()
    pointer = root / "artifacts/trial_001/final_output.json"
    if not pointer.exists():
        return {"status": "not_available", "reason": "No final-output manifest"}
    target = (root / json.loads(pointer.read_text())["artifact"]).resolve()
    if not target.is_relative_to(root) or not target.exists():
        return {"status": "incomplete", "reason": "Missing final-output artifact"}
    output = json.loads(target.read_text())
    if output.get("schema") not in {"cap-final-output-v1", "cap-final-output-v2"}:
        return {"status": "unsupported", "reason": "Unknown final-output schema"}
    return output


def normalize_output(output):
    """Alpha-rename final function bindings for analysis, retaining original sources."""
    policy = definitions(Bundle(output["policy"]))[0]["policy"]
    nodes, roots = output.get("functions", {}), output.get("bindings", {})
    occupied = {n.id for n in ast.walk(policy) if isinstance(n, ast.Name)}
    originals = {}
    for key, info in nodes.items():
        if hashlib.sha256(info["source"].encode()).hexdigest() != info["source_sha256"]:
            raise ValueError("Final source hash mismatch")
        if info["source_kind"] == "helper":
            trees, _ = definitions(Bundle("", {info["name"]: info["source"]}))
            tree = trees[info["name"]]
        else:
            tree = parse(info["source"])
        occupied.update(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
        matches = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == info["name"]
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous final function source")
        originals[key] = matches[0]
    aliases = {}
    for key in nodes:
        name = f"cap_audit_function_{len(aliases)}"
        while name in occupied:
            name += "x"
        occupied.add(name)
        aliases[key] = name

    class RenameCalls(ast.NodeTransformer):
        def __init__(self, bindings, blocked=()):
            self.bindings = bindings
            self.blocked = set(blocked)
            self.lambda_depth = 0

        def visit_Lambda(self, node):
            old = self.blocked
            self.blocked = old | {arg.arg for arg in node.args.args}
            self.lambda_depth += 1
            self.generic_visit(node)
            self.lambda_depth -= 1
            self.blocked = old
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                output.get("schema") == "cap-final-output-v1"
                and self.lambda_depth
                and isinstance(node.func, ast.Name)
                and node.func.id not in API_NAMES
                and node.func.id not in self.blocked
                and node.func.id not in self.bindings
            ):
                raise ValueError(
                    "Legacy final-output capture omitted or cannot establish the "
                    f"query-lambda helper binding: {node.func.id}"
                )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in self.bindings
                and node.func.id not in self.blocked
            ):
                node.func.id = aliases[self.bindings[node.func.id]]
            return node

    def local_names(node):
        return {a.arg for a in node.args.args} | {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }

    normalized, helpers, inline_keys = [], {}, set()
    for node in policy.body:
        if isinstance(node, ast.FunctionDef):
            key = roots.get(node.name)
            if (
                key is None
                or nodes[key]["source_kind"] != "policy"
                or ast.dump(originals[key]) != ast.dump(node)
            ):
                raise ValueError(
                    "Final inline definition does not match captured binding"
                )
            inline_keys.add(key)
            definition = RenameCalls(nodes[key]["bindings"], local_names(node)).visit(
                copy.deepcopy(node)
            )
            definition.name = aliases[key]
            normalized.append(definition)
        else:
            normalized.append(RenameCalls(roots).visit(copy.deepcopy(node)))
    for key, node in originals.items():
        if key not in inline_keys:
            definition = RenameCalls(nodes[key]["bindings"], local_names(node)).visit(
                copy.deepcopy(node)
            )
            definition.name = aliases[key]
            helpers[aliases[key]] = ast.unparse(definition)
    source = ast.unparse(ast.Module(body=normalized, type_ignores=[]))
    return Bundle(source, helpers), aliases


def audit_run(run_dir, *, timeout=10.0):
    """Idempotent postprocessing; original result/checkpoints stay immutable."""
    root = Path(run_dir)
    destination = root / "artifacts/trial_001/dsl_compliance.json"
    started = time.monotonic()
    record = {
        "scope": "final_output",
        "status": "error",
        "audit_version": "final-output-v2",
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        with analysis_deadline(timeout):
            config = json.loads((root / "config.json").read_text())["config"]
            saved_contract = config.get("dsl_contract")
            if saved_contract is None:
                raise ValueError("Recorded DSL contract is unavailable")
            contract = DSLContract(
                **{
                    k: saved_contract[k]
                    for k in (
                        "condition_mode",
                        "api_names",
                        "input_types",
                        "checker_version",
                        "observation_api_version",
                    )
                }
            )
            if (
                contract.identity()["checker_source_sha256"]
                != saved_contract["checker_source_sha256"]
            ):
                raise ValueError(
                    "Recorded checker implementation differs; explicit migration is required"
                )
            record["contract"] = contract.identity()
            record["method"] = config["method"]
            output = read_final_output(root)
            record["output_status"] = output["status"]
            record["final_output_sha256"] = hashlib.sha256(
                json.dumps(output, sort_keys=True).encode()
            ).hexdigest()
            if output["status"] == "unsupported" and output.get("policy"):
                try:
                    parse(output["policy"])
                except ConstraintFailure as exc:
                    output = {
                        **output,
                        "status": "fail",
                        "diagnostics": [d.to_dict() for d in exc.diagnostics],
                    }
            if output["status"] != "complete":
                record.update(
                    status=output["status"],
                    reason=output.get("reason", ""),
                    diagnostics=output.get("diagnostics", []),
                )
            else:
                try:
                    bundle, aliases = normalize_output(output)
                except ConstraintFailure as exc:
                    record.update(
                        status="fail",
                        diagnostics=[d.to_dict() for d in exc.diagnostics],
                    )
                except ValueError as exc:
                    record.update(status="unsupported", reason=str(exc))
                else:
                    report = check_bundle(
                        bundle,
                        contract,
                        timeout=max(0.001, timeout - (time.monotonic() - started)),
                    )
                    record.update(report.to_dict())
                    record["scope"] = "final_output"
                    record["diagnostic_source"] = "normalized_source_and_helpers"
                    record["normalized_bindings"] = aliases
                    record["normalized_source"] = bundle.source
                    record["normalized_helpers"] = bundle.helpers
    except Exception as exc:
        record.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    record["elapsed_seconds"] = time.monotonic() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(destination)
    return record
