"""Shared command-line options for symbolic verification."""

from argparse import BooleanOptionalAction

from synthesis.verification_lib.wp_options import WPOptions


def add_verification_options(parser):
    parser.add_argument(
        "--wp-system",
        choices=("old", "new", "reduction"),
        default="old",
        help="old: existing WP/axioms; new: shared WP/relational axioms; reduction: shared WP/exact scene reduction",
    )
    parser.add_argument(
        "--wp-support-complete",
        action="store_true",
        help="Add relational support-completeness axiom C1 (requires --wp-system new)",
    )
    parser.add_argument(
        "--reduction-max-slots",
        type=int,
        default=64,
        help="Compilation budget for the derived countermodel bound; exceeding it returns UNKNOWN",
    )
    parser.add_argument(
        "--reduction-max-instances",
        type=int,
        default=200000,
        help="Reduction formula-construction budget; never a bound on program block counts",
    )
    parser.add_argument(
        "--minimize-unsat-core",
        action=BooleanOptionalAction,
        default=False,
        help="Minimize symbolic unsat cores (default: off); retain consistency checks",
    )


def wp_options_from_args(args):
    return WPOptions(
        wp_system=getattr(args, "wp_system", "old"),
        support_complete=getattr(args, "wp_support_complete", False),
        reduction_max_slots=getattr(args, "reduction_max_slots", 64),
        reduction_max_instances=getattr(args, "reduction_max_instances", 200000),
    )
