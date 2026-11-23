"""Print a bounded summary of a baseline run without loading solver logs."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    result_path = directory / "result.json"
    path = result_path if result_path.exists() else directory / "status.json"
    data = json.loads(path.read_text())
    if data.get("schema_version", 1) != 1:
        raise ValueError("Unsupported baseline report schema")
    lines = [
        f"{data.get('family', 'mypyvy')}: {data['task']} / {data['solver']}",
        f"status: {data['status']}; elapsed: {data['elapsed_seconds']:.2f}s",
    ]
    if not result_path.exists():
        lines.append(
            f"phase: {data['current_phase']}; heartbeat: {data['heartbeat_utc']}; pid: {data['pid']}"
        )
        lines.append(
            "A running status is a heartbeat snapshot, not proof that the process is still alive."
        )
    if data.get("reason"):
        lines.append("reason: " + data["reason"][:500])
    for phase in data["phases"][:20]:
        lines.append(
            f"{phase['phase']}: {phase['status']} ({phase['elapsed_seconds']:.2f}s)"
        )
    lines.append(f"artifacts: {directory.resolve()}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    print(summarize(args.run))


if __name__ == "__main__":
    main()
