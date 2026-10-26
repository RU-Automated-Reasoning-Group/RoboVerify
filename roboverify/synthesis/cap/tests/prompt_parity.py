"""Every model request of a generation-only trial, for prompt-parity checks.

The golden file ``testdata/demonstrations_off_prompt_hashes.json`` was written
from the code before Revision 6 (demonstration input) existed. Regenerate it
only for an intentional prompt change:

    python -m synthesis.cap.tests.prompt_parity --write
"""

import argparse
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path

from ..config import RunConfig
from ..configs import config_frozen_v2 as frozen
from ..llm import ScriptedModel
from ..runner import run

GOLDEN = Path(__file__).with_name("testdata") / "demonstrations_off_prompt_hashes.json"
METHODS = ("cap_google_baseline", "cap_method_1", "cap_method_2")
MODES = ("dsl_only", "coordinate_predicates")
TASKS = tuple(frozen.TASKS)


def responses(method, task):
    """Scripted outputs that reach policy, helper and repair requests."""
    handle = "h0" if task in {"grid", "pyramid"} else "b0"
    if method == "cap_google_baseline":
        # An undefined name triggers a helper request even in debug mode.
        return [f"touch_input({handle})", "def touch_input(x):\n    Pick(x)\n"]
    return [
        f"if Pick({handle}):\n    pass\n",  # action in a condition: repair
        f"touch_input({handle})",
        "def touch_input(x):\n    Pick(x)\n",
        "def touch_input(x):\n    Pick(x)\n",
    ]


def rendered_prompts(method, mode, task, run_overrides=None, run_kwargs=None):
    """Prompts of one generate-only trial, in request order."""
    model = ScriptedModel(responses(method, task))
    with tempfile.TemporaryDirectory() as root:
        settings = frozen.run_settings(
            task, method=method, condition_mode=mode, output_root=root
        ) | dict(run_overrides or {})
        config = RunConfig(**settings)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config,
                frozen.model_config(),
                instruction=frozen.instruction(task),
                backend=model,
                generate_only=True,
                capture_stdout=False,
                **(run_kwargs or {}),
            )
    return model.prompts, result


def prompt_hashes(run_overrides=None):
    cases = []
    for task in TASKS:
        for method in METHODS:
            for mode in MODES:
                prompts, result = rendered_prompts(method, mode, task, run_overrides)
                cases.append(
                    {
                        "task": task,
                        "method": method,
                        "mode": mode,
                        "status": result["status"],
                        "sha256": [
                            hashlib.sha256(p.encode()).hexdigest() for p in prompts
                        ],
                    }
                )
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cases = prompt_hashes()
    if args.write:
        GOLDEN.write_text(json.dumps(cases, indent=1) + "\n")
    print(
        json.dumps(
            [
                (c["task"], c["method"], c["mode"], c["status"], len(c["sha256"]))
                for c in cases
            ]
        )
    )


if __name__ == "__main__":
    main()
