"""Shared command-line options for symbolic verification."""

from argparse import BooleanOptionalAction


def add_verification_options(parser):
    parser.add_argument(
        "--minimize-unsat-core",
        action=BooleanOptionalAction,
        default=False,
        help="Minimize symbolic unsat cores (default: off); retain consistency checks",
    )
