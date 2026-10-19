"""Observational final-output capture for Google's unchanged execution path."""

import ast
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType


def referenced_global_names(code):
    """Include globals used inside query lambdas compiled as nested code objects."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(referenced_global_names(constant))
    return sorted(names)


class FinalOutputRecorder:
    def __init__(self, logger):
        self.logger = logger
        self.policy = ""
        self.ready = False
        self.code_sources = {}
        self.counter = 0

    def publish(self, value):
        self.counter += 1
        value = {"schema": "cap-final-output-v2", **value}
        path = self.logger.artifact(f"outputs/output_{self.counter:04d}.json", value)
        pointer = self.logger.root / "final_output.json"
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"artifact": self.logger.relative(path)}, indent=2) + "\n"
        )
        temporary.replace(pointer)
        self.logger.register(pointer)

    def generated(self, source, helper):
        if not helper:
            self.policy, self.ready = source, False
            self.publish(
                {
                    "status": "incomplete",
                    "policy": source,
                    "reason": "Helper generation has not completed",
                }
            )

    def capture(self, namespace):
        """Read function/code objects only. Never call a generated function."""
        try:
            nodes, seen = {}, {}

            def visit(fn):
                if (
                    not isinstance(fn, FunctionType)
                    or fn.__code__.co_filename not in self.code_sources
                ):
                    return None
                if id(fn) in seen:
                    return seen[id(fn)]
                if len(nodes) >= 256:
                    raise ValueError("Final output exceeds 256 function bindings")
                if fn.__closure__:
                    raise ValueError(
                        "Closure bindings cannot be represented by the current DSL bundle"
                    )
                key = f"function_{len(nodes)}"
                seen[id(fn)] = key
                source, is_policy = self.code_sources[fn.__code__.co_filename]
                nodes[key] = {
                    "name": fn.__code__.co_name,
                    "source": source,
                    "source_kind": "policy" if is_policy else "helper",
                    "bindings": {},
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                }
                for name in referenced_global_names(fn.__code__):
                    child = visit(fn.__globals__.get(name))
                    if child:
                        nodes[key]["bindings"][name] = child
                return key

            bindings = {}
            for name, value in namespace.items():
                key = visit(value)
                if key:
                    bindings[name] = key
            self.publish(
                {
                    "status": "complete",
                    "policy": self.policy,
                    "bindings": bindings,
                    "functions": nodes,
                }
            )
        except Exception as exc:
            self.publish(
                {
                    "status": "unsupported",
                    "policy": self.policy,
                    "reason": f"Final output capture: {type(exc).__name__}: {exc}",
                }
            )
