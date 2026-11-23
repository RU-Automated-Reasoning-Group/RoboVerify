"""Run pinned upstream mypyvy with a narrow cvc5 compatibility adapter.

Upstream still calls its SMT-LIB interface CVC4. This adapter changes only the
solver command and the printed model format; it does not rewrite queries or
change the UPDR algorithm. Unsupported responses raise instead of claiming proof.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

MODEL_PREFIX = "__rv_cvc5_model_"


def certificate_text(source, clauses):
    """Keep immutable learned clauses checkable, never turn them into axioms.

    Upstream rejects invariants containing no mutable symbol. Conjoining a
    tautology mentioning the existing ``done`` relation preserves each clause
    while satisfying that syntactic restriction. Fresh verification still
    proves initialization and preservation for every clause.
    """
    return (
        source
        + "\n# Automatically inferred by UPDR; tautology keeps immutable clauses checkable.\n"
        + "\n".join(
            f"invariant [inferred_{i}] (done | !done) & ({clause})"
            for i, clause in enumerate(clauses)
        )
        + "\n"
    )


def scoped_check(original, instance, assumptions=None):
    """Implement assumptions with a temporary conjunction when cores are off."""
    if instance.use_cvc4 and assumptions:
        instance.z3solver.push()
        try:
            instance.z3solver.add(*assumptions)
            return original(instance, [])
        finally:
            instance.z3solver.pop()
    return original(instance, assumptions)


def cvc5_model_line(line, *, root=False):
    """Translate cvc5 1.3.1 model presentation to upstream's CVC4 presentation."""
    if root and line.strip() == "(":
        return "(model\n"
    cardinality = re.fullmatch(
        r"\s*; cardinality of ([A-Za-z_][\w.]*) is (\d+)\s*", line
    )
    if cardinality:
        return f"(declare-sort {cardinality[1]} 0)\n"
    # cvc5 annotates uninterpreted values with their sort; upstream records the
    # sort separately through the declaration and immediately following reps.
    line = re.sub(r"\(as (@[\w.!-]+) [A-Za-z_][\w.]*\)", r"\1", line)
    # UPDR reuses universe labels as fresh quantified-variable stems. Upstream
    # rejects @ in subsequent queries, so rename model values consistently.
    return re.sub(r"@([\w.!-]+)", MODEL_PREFIX + r"\1", line)


class ModelLines:
    """Track the outer model delimiter; nested multiline forms keep their shape."""

    def __init__(self):
        self.depth = 0

    def feed(self, line):
        root = self.depth == 0
        if root and line.strip() != "(":
            raise ValueError(f"Unsupported cvc5 model root: {line!r}")
        # The baseline signature contains only Bool and uninterpreted sorts,
        # hence model terms contain no strings or quoted identifiers.
        code = line.split(";", 1)[0]
        if '"' in code or "|" in code:
            raise ValueError("Quoted cvc5 model tokens are unsupported")
        self.depth += code.count("(") - code.count(")")
        if self.depth < 0:
            raise ValueError("Unbalanced cvc5 model")
        return cvc5_model_line(line, root=root)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mypyvy-root", type=Path, required=True)
    parser.add_argument("--cvc5", type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--solver-evidence", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    sys.path.insert(0, str(args.mypyvy_root.resolve() / "src"))
    import mypyvy
    import solver
    import solver_cvc4
    import syntax
    import updr
    import utils

    original_typecheck = mypyvy.typecheck_program

    def typecheck_and_expand(prog):
        if not original_typecheck(prog):
            return False
        for decl in prog.decls:
            if getattr(decl, "name", "") and decl.name.startswith(MODEL_PREFIX):
                raise ValueError(f"Reserved adapter symbol prefix: {decl.name}")
        # Upstream's finite-state evaluator does not evaluate definition calls.
        # Its own capture-avoiding expander preserves the safety formulas while
        # making them usable by UPDR's concrete model checks (both solvers).
        for inv in prog.invs():
            inv.expr = syntax.expand_macros(prog.scope, inv.expr)
        return True

    mypyvy.typecheck_program = typecheck_and_expand

    children = []
    evidence = args.solver_evidence.open("w") if args.solver_evidence else None
    if args.cvc5:
        binary = str(args.cvc5.resolve())

        def start_solver():
            command = [
                binary,
                "--lang=smt2",
                "--finite-model-find",
                "--produce-models",
                "--incremental",
                "--dag-thresh=0",
            ]
            if utils.args.timeout is not None:
                command += [
                    f"--tlimit-per={utils.args.timeout}",
                    f"--seed={utils.args.seed}",
                ]
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            children.append(proc)
            return proc

        solver.new_cvc4_process = start_solver
        solver_cvc4.new_cvc4_process = start_solver
        model_lines = ModelLines()

        def model_line(line):
            if evidence and "sat" not in recorded:
                evidence.write("; raw model: " + line)
            return model_lines.feed(line)

        solver_cvc4.cvc4_postprocess = model_line
        original_check = solver.check_with_cvc4

        def checked_query(proc, smt2):
            result, model = original_check(proc, smt2)
            if evidence:
                # The first SAT model and first UNSAT response suffice to show
                # the actual wire format, without retaining every large query.
                label = str(result)
                if label not in recorded:
                    evidence.write(f"; response: {label}\n{smt2}\n")
                    evidence.write(solver_cvc4._cvc4_last_model_response + "\n")
                    evidence.flush()
                    recorded.add(label)
            return result, model

        recorded = set()
        solver.check_with_cvc4 = checked_query
        original_solver_check = solver.Solver.check

        def check_with_assumptions(instance, assumptions=None):
            return scoped_check(original_solver_check, instance, assumptions)

        solver.Solver.check = check_with_assumptions
        if "--cvc4" not in arguments:
            arguments.insert(1, "--cvc4")
        if arguments and arguments[0] == "updr":
            arguments.insert(1, "--no-use-z3-unsat-cores")

    if args.certificate:
        original_search = updr.Frames.search

        def save_invariant(frames):
            frame = original_search(frames)
            source = Path(utils.args.filename).read_text()
            clauses = [str(expr) for expr in frame.summary()]
            args.certificate.write_text(certificate_text(source, clauses))
            return frame

        updr.Frames.search = save_invariant

    sys.argv = [str(args.mypyvy_root / "src/mypyvy.py"), *arguments]
    try:
        mypyvy.main()
    finally:
        for child in children:
            child.terminate()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if evidence:
            evidence.close()


if __name__ == "__main__":
    main()
