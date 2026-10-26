"""Revalidate recorded archives under the settled criterion into sidecar reports.

For each archive, every trace's last full snapshot is restored into a fresh
environment, settled for ``--post-settling-steps`` unrecorded steps and scored
by the shared checker (``cfg/task_success.py``). The report is written next to
the archive as ``settled_validation.json``; the NPZ file is never modified.
Synthesis and verification entry points read the sidecar when an archive has no
stored settled result.
"""

import argparse
import json
from pathlib import Path

from synthesis.cfg.task_success import (
    SETTLING_STEPS,
    revalidate_archive,
    write_sidecar,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, nargs="+", required=True)
    parser.add_argument("--post-settling-steps", type=int, default=SETTLING_STEPS)
    parser.add_argument(
        "--no-write", action="store_true", help="Print the summary only"
    )
    args = parser.parse_args(argv)
    if args.post_settling_steps < 1:
        parser.error("--post-settling-steps must be positive")
    status = 0
    for path in args.demos:
        report = revalidate_archive(path, args.post_settling_steps)
        target = None if args.no_write else write_sidecar(path, report)
        print(
            json.dumps(
                dict(
                    archive=str(path),
                    passed=report["passed"],
                    total=len(report["traces"]),
                    failed_seeds=report["failed_seeds"],
                    sidecar=str(target) if target else None,
                )
            ),
            flush=True,
        )
        status = status or (0 if report["valid"] else 2)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
