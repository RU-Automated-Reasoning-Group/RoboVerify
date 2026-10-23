"""Compatibility imports for historical diagnostics; collect via collect_demos.

The lateral program and observation-relative operands now execute in shared code.
This module retains the old factory shape for archived verification diagnostics.
"""

import sys

import numpy as np

from synthesis.cfg.program_source import load_program
from synthesis.mcmc.synthesis import make_roboverify_unstack_env as lateral_env
from synthesis.util.on import get_block_pos


def build_program(context, count):
    return load_program(
        "synthesis.examples.unstack:build_program",
        context,
        count,
        allow_observation_references=True,
    )


def positions(obs, count):
    return np.array([get_block_pos(obs, i) for i in range(count)])


def main():
    from synthesis.entry.collect_demos import main as collect

    return collect(["--task", "unstack", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
