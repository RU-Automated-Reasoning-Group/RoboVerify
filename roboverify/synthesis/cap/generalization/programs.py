"""Final CaP programs of an experiment, deduplicated (Revision 9, Section 6).

Every successful trial contributes one program: the executed checkpoint of
its successful recovery round
(`attempts[k].run_dir/artifacts/trial_001/checkpoints/<active>/policy.py`,
with `k = task_spec_success_round`; Revision 8, Section 3.1). Programs are
deduplicated on an alpha-renamed AST hash, which ignores comments,
docstrings and the names of locally defined functions, parameters and
variables.
"""

import ast
import builtins
import glob
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialProgram:
    pool: str
    task: str
    trial: str
    seed: int
    round: int
    source_path: str
    sha256: str
    alpha_sha256: str


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def _defined_names(tree):
    """Names the program binds itself, in order of first definition."""
    names = []

    def add(name):
        if name not in names:
            names.append(name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            add(node.id)
    return names


def alpha_hash(source):
    """Hash of the AST with comments/docstrings dropped and own names renamed.

    Only names the program binds are renamed; API names (`Pick`, `b0`, `tbl`,
    predicates) and builtins keep their identity. The entry point `policy`
    is kept because the program must invoke it by name."""
    tree = _strip_docstrings(ast.parse(source))
    keep = set(dir(builtins)) | {"policy"}
    mapping = {
        name: f"_v{i}"
        for i, name in enumerate(n for n in _defined_names(tree) if n not in keep)
    }

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_arg(self, node):
            node.arg = mapping.get(node.arg, node.arg)
            return node

        def visit_FunctionDef(self, node):
            node.name = mapping.get(node.name, node.name)
            self.generic_visit(node)
            return node

    tree = Rename().visit(tree)
    dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()


def trial_programs(pool, run_roots, exclude=()):
    """Programs of every successful trial under `run_roots` (batch roots that
    contain `<task>/trials/<trial>/`). `exclude` holds (task, seed) pairs."""
    programs = []
    for root in run_roots:
        pattern = str(
            Path(root) / "*" / "trials" / "*" / "cap" / "*recovery*" / "result.json"
        )
        for path in sorted(glob.glob(pattern)):
            path = Path(path)
            task, trial = path.parents[4].name, path.parents[2].name
            job = json.loads((path.parents[2] / "job.json").read_text())
            seed = job["run_config"]["seed"]
            if (task, seed) in exclude:
                continue
            result = json.loads(path.read_text())
            if not result.get("task_spec_success"):
                continue
            k = result["task_spec_success_round"]
            attempt = result["attempts"][k]
            checkpoint = attempt.get("active_checkpoint_id") or attempt["checkpoint_id"]
            source = (
                Path(attempt["run_dir"])
                / (attempt.get("artifacts_path") or "artifacts/trial_001")
                / "checkpoints"
                / checkpoint
                / "policy.py"
            )
            text = source.read_text()
            programs.append(
                TrialProgram(
                    pool=pool,
                    task=task,
                    trial=trial,
                    seed=seed,
                    round=k,
                    source_path=str(source),
                    sha256=hashlib.sha256(text.encode()).hexdigest(),
                    alpha_sha256=alpha_hash(text),
                )
            )
    return programs


def distinct(programs):
    """{(task, alpha_sha256): [TrialProgram, ...]} in first-seen order."""
    groups = {}
    for program in programs:
        groups.setdefault((program.task, program.alpha_sha256), []).append(program)
    return groups


def as_rows(programs):
    return [asdict(p) for p in programs]
