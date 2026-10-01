"""Diagnose Higher's role in the supplied Stack proof on fixed saved executions.

This is a proof ablation, not a production learner/backend option. Motion is
checked even after symbolic failure, to expose independent missing premises;
such motion results do not certify the program. Structural mode supplies an
explicit diagnostic invariant instead of claiming the learner discovered it.
"""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from time import perf_counter
from unittest.mock import patch

import z3

from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces, loop_store
from synthesis.cfg.tasks import task_spec
from synthesis.cfg.verification import (
    propose_summaries,
    symbolic_problem,
    verify_cfg_motion,
)
from synthesis.cfg.verified_synthesis import loop_regions
from synthesis.entry.verification_options import add_verification_options
from synthesis.experiment.motion_premise_ablation import (
    AUXILIARY_GEOMETRY_CHECKS,
    quantifier_count,
    without_extra_motion_premises,
)
from synthesis.experiment.run_logger import RunLogger
from synthesis.inference_lib.demo_store import InferenceVocabulary, InvInference
from synthesis.predicates.term import to_z3
from synthesis.util.on import using_higher_tolerance
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib import tower_geometry
from synthesis.verification_lib.cegis import _project
from synthesis.verification_lib.counterexamples import state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.motion_verification import MotionProblem
from synthesis.verification_lib.solver_options import using_unsat_core_minimization
from synthesis.verification_lib.symbolic_verify import discharge_vc


def structural_invariant(context):
    """One tower rooted at b0 with clear top b, plus separated singletons."""
    b, b0 = context.get_consts("b"), context.get_consts("b0")
    x, y = [fresh_const(context.BoxSort, n, avoid=(b, b0)) for n in ("x", "y")]
    on, scattered = context.ON_star, context.Scattered
    return z3.And(
        z3.ForAll([x], z3.Implies(on(x, b), x == b)),
        z3.ForAll([x], z3.Implies(on(b, x), on(x, b0))),
        z3.ForAll([x, y], z3.Implies(on(x, y), z3.Or(x == y, on(y, b0)))),
        z3.ForAll([x, y], z3.Or(on(x, y), on(y, x), scattered(x, y))),
    )


def uses_higher(expr):
    if z3.is_quantifier(expr):
        return uses_higher(expr.body())
    return (z3.is_app(expr) and str(expr.decl().name()) == "Higher") or any(
        uses_higher(child) for child in expr.children()
    )


def symbolic_checks(cfg, context, timeout_ms, *, logger=None):
    """Use the usual VCs without Higher-dependent countermodel realization."""
    records = []
    for size in (2, 3, 4, None):
        program, pre, post, active = symbolic_problem(cfg, context, size)
        for vc in program.VC_gen(pre, post, active):
            label = f"{size if size is not None else 'unbounded'}/{vc.kind}"
            if logger is not None:
                logger.set_status(phase="symbolic", current_vc=label)
            started = perf_counter()
            result = discharge_vc(vc, active, timeout_ms)
            seconds = perf_counter() - started
            model = result.model
            rows = dict(
                blocks=size,
                kind=vc.kind,
                status=result.status,
                reason=result.reason,
                queries=result.queries,
                seconds=seconds,
                inconclusive_query=(
                    ("validity" if result.queries == 1 else "premise_consistency")
                    if result.status == "unknown"
                    else None
                ),
                formula=vc.expr.sexpr(),
                model=str(model) if model is not None else None,
            )
            if model is not None and size is not None:
                objects = active.enum_blocks
                rows["relations"] = {
                    name: [
                        [
                            str(
                                model.eval(
                                    getattr(active, name)(a, b), model_completion=True
                                )
                            )
                            for b in objects
                        ]
                        for a in objects
                    ]
                    for name in ("ON_star", "Scattered")
                }
                rows["objects"] = list(map(str, objects))
                rows["bindings"] = {
                    n: str(model.eval(active.get_consts(n), model_completion=True))
                    for n in ("b", "b0", "b_prime")
                }
            records.append(rows)
            if logger is not None:
                logger.log_event(
                    "symbolic_vc",
                    f"{label}: {result.status}",
                    force=True,
                    seconds=seconds,
                    queries=result.queries,
                    inconclusive_query=rows["inconclusive_query"],
                    reason=result.reason,
                )
        if any(r["status"] != "valid" for r in records):
            break
    return dict(ok=all(r["status"] == "valid" for r in records), checks=records)


def run(args, logger):
    remove_all = args.case not in ("baseline", "inference-only")
    skipped = []
    query_audit = []
    original_check = MotionProblem.check

    def remaining_check(problem, obligation, violation=None):
        if (remove_all and obligation == "effect_Higher") or (
            args.without_height_premises
            and obligation in ("tower_height_entry", "tower_height_preserved")
        ):
            # Omitted properties do not count as valid. Collision, arm-clearance,
            # column-alignment and remaining relation checks are retained.
            skipped.append(f"{problem.block_v}/{obligation}")
            return None
        premises = tuple(problem.solver.assertions())
        result = original_check(problem, obligation, violation)
        record = dict(
            obligation=f"{problem.block_v}/{obligation}",
            kind="consistency" if violation is None else "violation",
            premise_quantifiers=quantifier_count(premises),
            violation_quantifiers=(
                0 if violation is None else quantifier_count((violation,))
            ),
            status=result.status,
        )
        if result.status == "refuted":
            query = z3.Solver()
            query.add(*premises, violation)
            path = "queries/" + record["obligation"].replace("/", "-") + ".smt2"
            logger.write_artifact(path, query.to_smt2())
            record["query_artifact"] = path
        query_audit.append(record)
        return result

    with ExitStack() as scope:
        scope.enter_context(
            using_unsat_core_minimization(getattr(args, "minimize_unsat_core", None))
        )
        scope.enter_context(using_higher_tolerance(0.001))
        if remove_all:
            scope.enter_context(
                patch.object(
                    HighLevelContext, "add_axiom_higher", lambda self, solver: None
                )
            )
        scope.enter_context(patch.object(MotionProblem, "check", remaining_check))
        if args.without_height_premises:
            # Keep the checked arm-clearance and column-alignment invariants;
            # remove only supported-height assumptions and their two VCs.
            scope.enter_context(
                patch.object(
                    tower_geometry,
                    "tower_height_invariant",
                    lambda problem, *, fields=None: z3.BoolVal(True),
                )
            )
        context = HighLevelContext()
        traces = load_traces(args.demos, require_valid=True)
        if any(t.task != "stack" or t.num_blocks != 4 for t in traces):
            raise ValueError("Expected validated four-block Stack demonstrations")
        definition = load_program("synthesis.examples.stack:build_program", context, 4)
        if any(
            t.metadata.get("fingerprint") != definition.metadata["fingerprint"]
            for t in traces
        ):
            raise ValueError("Demonstrations do not match the supplied Stack program")
        pre, post = task_spec("stack")
        if remove_all:
            assert pre.op == "and" and len(pre.args) == 2
            assert uses_higher(to_z3(pre.args[1], context))
            pre = pre.args[0]
            assert not uses_higher(to_z3(pre, context))
            assert not uses_higher(z3.And(*context.new_solver().assertions()))
        cfg = program_to_cfg(definition, [], pre, post)
        propose_summaries(cfg, context)
        store = loop_store(traces, loop_id="1", names=("b", "b0"))
        vocabulary = {
            "baseline": ("ON_star", "Higher", "Scattered", "equality"),
            "on-only": ("ON_star", "equality"),
        }.get(args.case, ("ON_star", "Scattered", "equality"))
        started = perf_counter()
        invariant = (
            structural_invariant(context)
            if args.case == "structural"
            else InvInference(
                store,
                "1",
                InferenceVocabulary(2, vocabulary, ("b", "b0")),
                context,
                minimizer="pyeda",
            )[0]
        )
        learning_seconds = perf_counter() - started
        assert uses_higher(invariant) == (args.case == "baseline")
        coverage = all(
            state_holds(invariant, _project(row, False)) for row in store.for_loop("1")
        )
        if not coverage:
            raise ValueError("Invariant excludes recorded runtime states")
        next(loop_regions(cfg)).invariant = invariant
        logger.write_artifact("invariant.smt2", invariant.sexpr())
        logger.log_event(
            "inferred",
            "Fixed dataset invariant ready",
            force=True,
            states=len(store),
            learning_seconds=learning_seconds,
        )
        symbolic = symbolic_checks(
            cfg, context, args.symbolic_timeout_ms, logger=logger
        )
        logger.write_artifact("symbolic.json", json.dumps(symbolic, indent=2))
        logger.log_event("symbolic", str(symbolic["ok"]), force=True)
        omissions = {}
        if args.without_extra_motion_premises:
            omissions = scope.enter_context(without_extra_motion_premises())
        motion = verify_cfg_motion(
            cfg,
            context,
            timeout_ms=args.motion_timeout_ms,
            supported_tower_model=True,
            table_surface_height=0.4,
            initial_arm=(
                None
                if args.without_extra_motion_premises
                else (1.3446426, 0.74911606, 0.5314612)
            ),
        )
        motion_data = dict(
            ok=bool(motion),
            checked_blocks=motion.checked_blocks,
            skipped_obligations=skipped,
            checks=[asdict(check) for check in motion.checks],
        )
        logger.write_artifact("motion.json", json.dumps(motion_data, indent=2))
        logger.write_artifact(
            "motion-query-audit.json", json.dumps(query_audit, indent=2)
        )
        core = [
            c
            for c in motion.checks
            if c.obligation.rsplit("/", 1)[-1] not in AUXILIARY_GEOMETRY_CHECKS
        ]
        auxiliary = [c for c in motion.checks if c not in core]
        summary = dict(
            case=args.case,
            states=len(store),
            coverage=coverage,
            invariant_source=(
                "diagnostic handwritten control"
                if args.case == "structural"
                else "partition learner / PyEDA"
            ),
            higher_removed_from_verification=remove_all,
            supported_height_premises=not (
                args.without_height_premises or args.without_extra_motion_premises
            ),
            extra_motion_premises=not args.without_extra_motion_premises,
            initial_arm_supplied=not args.without_extra_motion_premises,
            omitted_premise_calls=dict(omissions),
            learning_seconds=learning_seconds,
            symbolic_ok=symbolic["ok"],
            motion_ok=bool(motion),
            symbolic_statuses=dict(Counter(r["status"] for r in symbolic["checks"])),
            symbolic_failures=[
                {k: r[k] for k in ("blocks", "kind", "status", "reason")}
                for r in symbolic["checks"]
                if r["status"] != "valid"
            ],
            motion_statuses=dict(Counter(c.status for c in motion.checks)),
            core_motion_statuses=dict(Counter(c.status for c in core)),
            auxiliary_geometry_statuses=dict(Counter(c.status for c in auxiliary)),
            motion_query_counts=dict(
                total=len(query_audit),
                quantified_premises=sum(
                    bool(r["premise_quantifiers"]) for r in query_audit
                ),
                quantified_violation=sum(
                    bool(r["violation_quantifiers"]) for r in query_audit
                ),
            ),
            motion_failures=[
                dict(obligation=c.obligation, status=c.status, reason=c.reason)
                for c in motion.checks
                if c.status != "valid"
            ],
            skipped_obligations=skipped,
        )
        logger.write_artifact("summary.json", json.dumps(summary, indent=2))
        logger.finish("completed", ablation=summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", required=True)
    parser.add_argument(
        "--case",
        choices=("baseline", "inference-only", "no-higher", "on-only", "structural"),
        required=True,
    )
    premises = parser.add_mutually_exclusive_group()
    premises.add_argument("--without-height-premises", action="store_true")
    premises.add_argument(
        "--without-extra-motion-premises",
        action="store_true",
        help=(
            "Omit height/alignment/clearance assumptions, initial arm pose, and "
            "exact premise reassertions; retain all motion/auxiliary obligations"
        ),
    )
    parser.add_argument("--symbolic-timeout-ms", type=int, default=60000)
    add_verification_options(parser)
    parser.add_argument("--motion-timeout-ms", type=int, default=10000)
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args(argv)
    if min(args.symbolic_timeout_ms, args.motion_timeout_ms) <= 0:
        parser.error("Solver budgets must be positive")
    with RunLogger(
        args.output_dir,
        "higher-ablation",
        vars(args),
        slug=args.case
        + ("-no-height-premises" if args.without_height_premises else "")
        + ("-translated-only" if args.without_extra_motion_premises else ""),
    ) as logger:
        logger.progress_line(f"Higher ablation: {logger.run_dir}")
        try:
            summary = run(args, logger)
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", reason=str(exc))
            raise
    print(json.dumps(dict(run=str(logger.run_dir), **summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
