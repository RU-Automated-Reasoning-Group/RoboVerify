"""Dependencies injected into extracted upstream definitions, not their algorithm."""

import ast
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

from ..contracts import API_NAMES, api_description
from ..final_output import FinalOutputRecorder
from ..prompt_package import package


class CompletionShim:
    def __init__(self, model, on_source):
        self.model = model
        self.on_source = on_source

    def create(self, *, prompt, **kwargs):
        # Generation settings belong to the shared configured transport.
        text = self.model.complete(prompt, purpose="google_generation")
        self.on_source(text)
        return {"choices": [{"text": text}]}


class NeverRetriedByUpstream(Exception):
    """Shared transport handles bounded retries; upstream's infinite loop is unused."""


class Unparse:
    @staticmethod
    def unparse(node):
        # astunparse accepts statement lists; stdlib ast.unparse expects a node.
        return (
            ast.unparse(ast.Module(body=node, type_ignores=[]))
            if isinstance(node, list)
            else ast.unparse(node)
        )


class _TraversalStatements(ast.NodeTransformer):
    """Give RoboVerify's MoveRight(x)/MoveDown(x) their in-place semantics.

    exec'd Python cannot rebind a caller's variable from inside a call, so the
    statement form is rewritten to ``x = MoveRight(x)`` before execution. Only
    plain-name statements are rewritten; any other use runs unchanged.
    """

    def __init__(self, names):
        self.names = names

    def visit_Expr(self, node):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in self.names
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and not call.keywords
        ):
            target = ast.Name(id=call.args[0].id, ctx=ast.Store())
            return ast.copy_location(ast.Assign(targets=[target], value=call), node)
        return node


def load_core(model, logger, checkpoints, budget, *, traversals=frozenset()):
    core_path = Path(__file__).parents[1] / "vendor" / "google_cap" / "core.py"
    functions, execution_paths = {}, set()
    sources = {}
    policy_source = ""
    output = FinalOutputRecorder(logger)

    def generated(source):
        nonlocal policy_source
        helper = logger.context.get("helper")
        output.generated(source, helper)
        if helper:
            sources[helper] = source
        else:
            policy_source = source
        checkpoints.save(
            policy_source,
            sources,
            reason="generated_source",
            validation="not_run",
            bound_helpers=sorted(functions),
        )

    globals_ = {
        "ast": ast,
        "astunparse": Unparse,
        "openai": SimpleNamespace(Completion=CompletionShim(model, generated)),
        "RateLimitError": NeverRetriedByUpstream,
        "APIConnectionError": NeverRetriedByUpstream,
        "highlight": lambda text, *args: text,
        "PythonLexer": lambda: None,
        "TerminalFormatter": lambda: None,
        "print": lambda *args, **kwargs: logger.text(" ".join(str(a) for a in args)),
        "sleep": lambda _: budget.remaining(),
    }

    def observed_exec(source, gvars, lvars):
        budget.tick()
        try:
            tree = ast.parse(source)
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        except SyntaxError:
            names = []
        # Save actual binding/execution order. There is intentionally no DSL check.
        checkpoint = checkpoints.save(
            policy_source,
            sources,
            reason="execution_bundle",
            validation="not_run",
            bound_helpers=sorted(functions),
        )
        path = logger.root / "checkpoints" / checkpoint / "executed.py"
        logger.artifact(f"checkpoints/{checkpoint}/executed.py", source)
        filename = str(path)
        execution_paths.add(filename)
        is_policy = output.ready and source.strip() == policy_source.strip()
        if is_policy:
            logger.emit("policy_rollout_start", source_path=logger.relative(path))
        output.code_sources[filename] = (source, is_policy)
        original_trace = sys.gettrace()

        def trace(frame, event, arg):
            if frame.f_code.co_filename in execution_paths:
                budget.tick()
                if event in {"call", "line", "return", "exception"}:
                    fields = dict(
                        source_path=logger.relative(frame.f_code.co_filename),
                        source_line=frame.f_lineno,
                        function=frame.f_code.co_name,
                    )
                    logger.context.update(fields)
                    if event == "return":
                        fields["result"] = arg
                    logger.emit("python_" + event, **fields)
            return trace

        try:
            sys.settrace(trace)
            code = source
            if traversals:
                try:
                    tree = _TraversalStatements(traversals).visit(ast.parse(source))
                    code = ast.fix_missing_locations(tree)
                except SyntaxError:
                    pass  # compile() reports it as for any other source
            with budget.timings.measure("execution"):
                builtins.exec(compile(code, filename, "exec"), gvars, lvars)
        finally:
            sys.settrace(original_trace)
            if is_policy:
                output.capture(gvars | lvars)
        for name in names:
            functions[name] = source

    globals_["exec"] = observed_exec
    builtins.exec(compile(core_path.read_text(), str(core_path), "exec"), globals_)
    return SimpleNamespace(**globals_, final_output=output)


def module_config(
    *,
    helper=False,
    api_names=API_NAMES,
    condition_mode="dsl_only",
    prompt_version="legacy_v1",
    higher_tolerance=0.001,
    primitive_prompt=None,
):
    return {
        "prompt_text": package(prompt_version)["baseline_prefixes"][
            "helper" if helper else "policy"
        ]
        + api_description(
            api_names,
            condition_mode,
            prompt_version=prompt_version,
            higher_tolerance=higher_tolerance,
            primitive_prompt=primitive_prompt,
        ),
        "query_prefix": "\n# " if helper else "\n# Instruction: ",
        "query_suffix": "",
        "stop": [],
        "temperature": 0,
        "engine": "shared-model-transport",
        "max_tokens": 4096,
        "maintain_session": True,
        "include_context": False,
        "debug_mode": False,
        "has_return": False,
        "return_val_name": "ret_val",
    }
