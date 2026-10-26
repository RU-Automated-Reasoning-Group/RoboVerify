"""Revision 6 demonstration input: loading, compaction, prompts and accounting."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from synthesis.cfg.reset import restore

from ..budget import Budget
from ..config import DemonstrationConfig, ModelConfig, RunConfig
from ..demonstrations import (
    SERIALIZATION_VERSION,
    DemonstrationError,
    compact,
    count_tokens,
    event_frames,
    gripper_commands,
    load_demonstrations,
    motion_waypoints,
    prepare,
    rdp,
)
from ..llm import Response, ScriptedModel
from ..recovery import RoundBudget
from ..runner import run
from . import prompt_parity
from .test_adapter import RobotFixture

# Archive contents that must never reach the model.
SECRETS = (
    "SECRET_PROGRAM_PickByName",
    "SECRET_EVENT_loop_head",
    "SECRET_TASK_SPEC_forall",
    "f1ngerpr1nt5ecret",
    "SECRET_SOURCE_module",
    "123.456",  # simulator snapshot values
)


def stack_trajectory(bases, seed):
    """A scripted two-block Stack demonstration: move o1 onto o0.

    Returns (states, actions). The gripper approaches o1 at a safe height,
    descends, closes, lifts 10 cm, transits, descends onto o0, opens and
    retreats 15 cm, with a few control steps per segment.
    """
    o0, o1 = (np.asarray(b, dtype=float) for b in bases)
    safe = 0.55
    waypoints = [
        ((1.1, 0.5, safe), 0.0),
        ((o1[0], o1[1], safe), 0.0),
        ((o1[0], o1[1], o1[2]), 0.0),
        ((o1[0], o1[1], o1[2]), -0.2),  # close
        ((o1[0], o1[1], o1[2] + 0.1), -0.2),  # lift
        ((o0[0], o0[1], o1[2] + 0.1), -0.2),  # transit
        ((o0[0], o0[1], o0[2] + 0.05), -0.2),  # descend onto o0
        ((o0[0], o0[1], o0[2] + 0.05), 0.2),  # open
        ((o0[0], o0[1], o0[2] + 0.2), 0.0),  # retreat
    ]
    gripper = [np.array(waypoints[0][0])]
    commands = []
    for (target, command), (previous, _) in zip(waypoints[1:], waypoints):
        steps = 1 if np.allclose(target, previous) else 6
        for k in range(1, steps + 1):
            gripper.append(
                np.asarray(previous) + (np.subtract(target, previous)) * k / steps
            )
            commands.append(command)
    states, held, block1 = [], False, o1.copy()
    for t, g in enumerate(gripper):
        if t and commands[t - 1] < 0:
            held = True
        if t and commands[t - 1] > 0:
            held = False
        if held:
            block1 = g.copy()
        state = np.zeros(10 + 12 * 2 + 3 * 2 + 3)
        state[:3] = g
        state[10:13] = o0
        state[22:25] = block1
        states.append(state)
    states = np.array(states)
    actions = np.zeros((len(states) - 1, 4))
    actions[:, :3] = np.clip(np.diff(states[:, :3], axis=0) * 10, -1, 1)
    actions[:, 3] = commands
    return states, actions


def write_archive(
    path,
    traces,
    *,
    task="stack",
    count=2,
    status="valid",
    tolerance=0.001,
    bindings=None,
):
    arrays, rows = {}, []
    for i, (seed, states, actions) in enumerate(traces):
        key = f"d{i}"
        arrays[key + "_states"] = states
        arrays[key + "_actions"] = actions
        arrays[key + "_indices"] = np.arange(len(states))
        arrays[key + "_gt"] = np.full((len(states), 5), 123.456)
        arrays[key + "_ctrl"] = np.full((len(states), 2), 123.456)
        initial = dict(bindings or {"b0": 0})
        rows.append(
            dict(
                seed=seed,
                task=task,
                num_blocks=count,
                bindings=[initial] * len(states),
                arrays=["ctrl"],
                events=[{"kind": "SECRET_EVENT_loop_head", "index": 0, "path": "1"}],
                metadata={
                    "status": status,
                    "higher_tolerance": tolerance,
                    "initial_bindings": initial,
                    "program": [{"instruction": "SECRET_PROGRAM_PickByName"}],
                    "task_spec": {"precondition": "SECRET_TASK_SPEC_forall"},
                    "fingerprint": "f1ngerpr1nt5ecret",
                    "source": "SECRET_SOURCE_module",
                },
            )
        )
    arrays["metadata"] = np.asarray(
        json.dumps(
            {"format": "roboverify-demonstrations", "version": 2, "traces": rows}
        )
    )
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


SCENES = [
    (1000, ((1.30, 0.60, 0.425), (1.20, 0.80, 0.425))),
    (1001, ((1.25, 0.70, 0.425), (1.35, 0.90, 0.425))),
]


def words(text):
    """Fixture tokenizer: whitespace-separated words, additive across a
    concatenation that ends in whitespace."""
    return len(text.split()), {"method": "fixture_words"}


class UsageModel(ScriptedModel):
    """Scripted responses whose provider usage is the fixture word count."""

    def generate(self, prompt, timeout):
        response = super().generate(prompt, timeout)
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 7}
        usage["total_tokens"] = usage["prompt_tokens"] + 7
        return Response(response.text, usage, "stop", response.raw)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.archive = self.root / "demonstrations.npz"
        write_archive(
            self.archive,
            [(seed, *stack_trajectory(bases, seed)) for seed, bases in SCENES],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **demonstrations):
        settings = {
            "enabled": True,
            "archive": str(self.archive),
            "token_counter": "utf8_bytes",
        } | demonstrations
        return RunConfig(
            task="stack",
            num_blocks=2,
            prompt_version="typed_v5",
            output_root=str(self.root / "runs"),
            save_video=False,
            demonstrations=settings,
        )


class ConfigTests(unittest.TestCase):
    def test_defaults_are_off_and_documented_values(self):
        settings = RunConfig().demonstrations
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.waypoint_tolerance, 0.01)
        self.assertEqual(settings.max_tokens, 131072)
        self.assertEqual(asdict(RunConfig())["demonstrations"]["enabled"], False)

    def test_invalid_settings_are_rejected(self):
        for bad in (
            {"waypoint_tolerance": 0},
            {"waypoint_tolerance": float("nan")},
            {"waypoint_tolerance": float("inf")},
            {"waypoint_tolerance": True},
            {"max_tokens": 0},
            {"max_tokens": 1.5},
            {"trace_ids": []},
            {"trace_ids": ["d0", "d0"]},
            {"trace_ids": ["0"]},
            {"token_counter": "guess"},
            {"archive_sha256": "abc"},
            {"enabled": True},  # no archive
            {"archive": "demos.json"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DemonstrationConfig(**bad)
        with self.assertRaises(ValueError):
            RunConfig(demonstrations={"unknown": 1})

    def test_enabling_requires_a_package_with_demonstration_templates(self):
        settings = {"enabled": True, "archive": "x.npz"}
        for version in ("legacy_v1", "typed_v3", "typed_v4"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                RunConfig(prompt_version=version, demonstrations=settings)
        RunConfig(prompt_version="typed_v5", demonstrations=settings)

    def test_typed_v5_is_typed_v4_plus_demonstration_templates(self):
        from ..prompt_package import package

        v5 = package("typed_v5")
        templates = v5.pop("demonstrations")
        self.assertEqual(v5, package("typed_v4"))
        self.assertEqual(templates["serialization_version"], SERIALIZATION_VERSION)


class PromptParityTests(unittest.TestCase):
    def test_demonstrations_off_leaves_every_prompt_unchanged(self):
        golden = json.loads(prompt_parity.GOLDEN.read_text())
        self.assertEqual(len(golden), 36)  # 6 tasks x 3 methods x 2 modes
        for overrides in ({}, {"prompt_version": "typed_v5"}):
            with self.subTest(overrides=overrides):
                self.assertEqual(prompt_parity.prompt_hashes(overrides), golden)


class CompactionTests(Fixture):
    def test_gripper_events_follow_command_switches(self):
        actions = np.zeros((6, 4))
        actions[1, 3] = 0.2  # open while open: no event
        actions[2:4, 3] = -0.2
        actions[4, 3] = 0.2
        commands = gripper_commands(actions)
        self.assertEqual(
            commands, ["open", "open", "open", "closed", "closed", "open", "open"]
        )
        self.assertEqual(event_frames(commands), [0, 3, 5, 6])

    def test_rdp_keeps_corners_and_drops_collinear_steps(self):
        path = np.array(
            [[0, 0, 0], [0, 0, 0.05], [0, 0, 0.1], [0.05, 0, 0.1], [0.1, 0, 0.1]]
        )
        self.assertEqual(rdp(path, 0.01), [0, 2, 4])
        self.assertEqual(rdp(path, 0.1), [0, 4])
        # A retraced vertical (lift and descend) is kept with segment distances.
        retrace = np.array(
            [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.1], [0, 0, 0]]
        )
        self.assertEqual(rdp(retrace, 0.01), [0, 2, 4])
        self.assertEqual(motion_waypoints(path, [0, 4], 0.01), [2])
        self.assertEqual(motion_waypoints(path, [0, 2, 4], 0.01), [])

    def test_frames_keep_pick_place_and_safe_height_structure(self):
        demonstrations, _ = load_demonstrations(self.config(trace_ids=["d0"]))
        frames = compact(demonstrations[0], 0.01)
        events = [f for f in frames if f.kind == "event"]
        waypoints = [f for f in frames if f.kind == "waypoint"]
        self.assertEqual(
            [f.command for f in events], ["open", "closed", "open", "open"]
        )
        self.assertEqual(events[1].holding, 1)
        self.assertIn("ON(o1,o0)", events[2].added)
        self.assertIn("Scattered(o0,o1)", events[2].removed)
        self.assertIn("Scattered(o0,o1)", events[0].relations)
        heights = [round(f.gripper[2], 3) for f in waypoints]
        # above o1 at the safe height, lift 10 cm, transit end, retreat start.
        self.assertIn(0.55, heights)
        self.assertIn(0.525, heights)
        self.assertTrue(all(f.blocks is None for f in waypoints))

    def test_larger_tolerance_merges_corners(self):
        demonstrations, _ = load_demonstrations(self.config(trace_ids=["d0"]))
        fine = compact(demonstrations[0], 0.01)
        coarse = compact(demonstrations[0], 0.5)
        self.assertLess(len(coarse), len(fine))
        self.assertEqual(
            [f.t for f in coarse if f.kind == "event"],
            [f.t for f in fine if f.kind == "event"],
        )


class ContextTests(Fixture):
    def test_rendered_block_is_deterministic_and_reveals_no_archive_internals(self):
        context = prepare(self.config(), ModelConfig())
        again = prepare(self.config(), ModelConfig())
        self.assertEqual(context.text, again.text)
        for secret in SECRETS:
            self.assertNotIn(secret, context.text)
        for internal in ("seed", "1000", "1001", "fingerprint", "loop", "while"):
            self.assertNotIn(internal, context.text.lower())
        self.assertTrue(
            context.text.startswith(f"DEMONSTRATIONS ({SERIALIZATION_VERSION})")
        )
        self.assertIn("waypoint_tolerance=0.010 m", context.text)
        self.assertIn("DEMONSTRATION 2/2", context.text)
        self.assertIn("inputs: b0=o0", context.text)
        self.assertRegex(context.text, r"gripper=\(\d\.\d{3},\d\.\d{3},\d\.\d{3}\)")
        manifest = context.manifest
        self.assertEqual(manifest["trace_ids"], ["d0", "d1"])
        self.assertEqual(manifest["serialization_version"], SERIALIZATION_VERSION)
        self.assertEqual(manifest["settings"]["waypoint_tolerance"], 0.01)
        self.assertEqual(
            manifest["archive"]["sha256"],
            hashlib.sha256(self.archive.read_bytes()).hexdigest(),
        )
        self.assertEqual(context.tokens, len(context.text.encode()))

    def test_selection_orders_the_block(self):
        context = prepare(self.config(trace_ids=["d1", "d0"]), ModelConfig())
        self.assertEqual(
            [r["seed"] for r in context.manifest["demonstrations"]], [1001, 1000]
        )
        first = prepare(self.config(trace_ids=["d1"]), ModelConfig())
        self.assertNotEqual(first.sha256, context.sha256)

    def test_tolerance_changes_the_input(self):
        fine = prepare(self.config(), ModelConfig())
        coarse = prepare(self.config(waypoint_tolerance=0.2), ModelConfig())
        self.assertIn("waypoint_tolerance=0.200 m", coarse.text)
        self.assertLess(coarse.tokens, fine.tokens)

    def test_incompatible_or_malformed_archives_are_rejected(self):
        cases = {
            "task": dict(task="unstack"),
            "status": dict(status="invalid"),
            "tolerance": dict(tolerance=0.002),
            "bindings": dict(bindings={"b0": 5}),
        }
        for name, options in cases.items():
            path = self.root / f"{name}.npz"
            write_archive(path, [(1, *stack_trajectory(SCENES[0][1], 1))], **options)
            with self.subTest(name=name), self.assertRaises(DemonstrationError):
                load_demonstrations(self.config(archive=str(path)))
        with self.assertRaises(DemonstrationError):
            load_demonstrations(self.config(trace_ids=["d7"]))
        with self.assertRaises(DemonstrationError):
            load_demonstrations(self.config(archive_sha256="0" * 64))
        with self.assertRaises(DemonstrationError):
            load_demonstrations(replace(self.config(), num_blocks=3))
        incomplete = self.root / "incomplete.npz"
        states, actions = stack_trajectory(SCENES[0][1], 1)
        write_archive(incomplete, [(1, states, actions[:-1])])
        with self.assertRaises(DemonstrationError):
            load_demonstrations(self.config(archive=str(incomplete)))
        with self.assertRaises(DemonstrationError):
            load_demonstrations(self.config(archive=str(self.root / "missing.npz")))

    def test_server_token_count_uses_tokenize_without_special_tokens(self):
        sent = []

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout):
            sent.append((request.full_url, json.loads(request.data)))
            return Reply(json.dumps({"count": 42, "max_model_len": 98304}).encode())

        settings = DemonstrationConfig(enabled=True, archive="x.npz")
        model = ModelConfig(
            provider="vllm", model="m", base_url="http://127.0.0.1:1/v1", api_key_env=""
        )
        count, measurement = count_tokens("text", settings, model, opener=opener)
        self.assertEqual(count, 42)
        self.assertEqual(sent[0][0], "http://127.0.0.1:1/tokenize")
        self.assertEqual(
            sent[0][1], {"model": "m", "prompt": "text", "add_special_tokens": False}
        )
        self.assertEqual(measurement["method"], "server_tokenize")


class RunnerTests(Fixture):
    def factory(self, config, snapshot):
        env = RobotFixture(config.num_blocks, success=True)
        if snapshot:
            restore(env, snapshot)
        return env

    def execute(self, config, responses, model=None, **options):
        backend = (model or ScriptedModel)(responses)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config,
                ModelConfig(),
                backend=backend,
                environment_factory=self.factory,
                capture_stdout=False,
                demonstration_token_counter=options.pop("token_counter", words),
                **options,
            )
        return result, backend

    def test_every_request_of_every_method_starts_with_the_same_block(self):
        contexts = set()
        for method, responses in (
            ("cap_google_baseline", ["touch(b0)", "def touch(x):\n    Pick(x)\n"]),
            ("cap_method_1", ["if Pick(b0):\n    pass\n", "Pick(b0)"]),
            ("cap_method_2", ["if Pick(b0):\n    pass\n", "Pick(b0)"]),
        ):
            config = replace(self.config(), method=method, prompt_version="typed_v5")
            result, backend = self.execute(config, responses)
            with self.subTest(method=method):
                self.assertGreaterEqual(len(backend.prompts), 2)
                block = (
                    Path(result["run_dir"])
                    / "artifacts/trial_001/demonstrations/context.txt"
                ).read_text()
                for prompt in backend.prompts:
                    self.assertTrue(prompt.startswith(block))
                    self.assertFalse(prompt[len(block) :].startswith("DEMONSTRATIONS"))
                contexts.add(block)
                self.assertEqual(
                    result["demonstration_context_sha256"],
                    hashlib.sha256(block.encode()).hexdigest(),
                )
                manifest = json.loads(
                    (
                        Path(result["run_dir"])
                        / "artifacts/trial_001/demonstrations/manifest.json"
                    ).read_text()
                )
                self.assertEqual(manifest["trace_ids"], ["d0", "d1"])
                metadata = json.loads(
                    (
                        Path(result["run_dir"])
                        / "artifacts/trial_001/lmp/call_0001/prompt_metadata.json"
                    ).read_text()
                )
                self.assertEqual(metadata["demonstration_tokens"], words(block)[0])
        self.assertEqual(len(contexts), 1)

    def test_demonstration_tokens_are_charged_outside_the_effort_budget(self):
        config = replace(self.config(), method="cap_method_2")
        result, backend = self.execute(
            config, ["if Pick(b0):\n    pass\n", "Pick(b0)"], model=UsageModel
        )
        block = result["demonstration_block_tokens"]
        calls = len(backend.prompts)
        provider = sum(len(p.split()) + 7 for p in backend.prompts)
        self.assertEqual(calls, 2)
        self.assertEqual(result["demonstration_tokens"], calls * block)
        self.assertEqual(result["known_tokens"], provider - calls * block)
        self.assertEqual(result["provider_total_tokens"], provider)
        # The same trial with demonstrations off charges exactly the effort part.
        off = replace(config, demonstrations=DemonstrationConfig())
        result_off, _ = self.execute(
            off, ["if Pick(b0):\n    pass\n", "Pick(b0)"], model=UsageModel
        )
        self.assertEqual(result_off["known_tokens"], result["known_tokens"])
        self.assertNotIn("demonstration_tokens", result_off)
        # An effort budget that the off trial exactly uses up still suffices.
        tight = replace(config, max_tokens=result_off["known_tokens"])
        result_tight, _ = self.execute(
            tight, ["if Pick(b0):\n    pass\n", "Pick(b0)"], model=UsageModel
        )
        self.assertIsNone(result_tight["budget_exhausted_by"])
        self.assertGreater(result_tight["provider_total_tokens"], tight.max_tokens)

    def test_oversized_block_fails_explicitly_before_any_request(self):
        config = replace(self.config(max_tokens=10), method="cap_method_2")
        result, backend = self.execute(config, ["Pick(b0)"])
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["budget_exhausted_by"], "demonstration_tokens")
        self.assertEqual(backend.prompts, [])
        self.assertIn("demonstrations.max_tokens", result["reason"])

    def test_evaluation_scene_cannot_be_a_demonstration(self):
        config = replace(self.config(), method="cap_method_2", seed=1001)
        result, backend = self.execute(config, ["Pick(b0)"])
        self.assertEqual(result["status"], "demonstration_rejected")
        self.assertEqual(backend.prompts, [])
        # Same block layout as the evaluation fixture, under another seed.
        same = self.root / "same.npz"
        fixture = ((0.0, 0.0, 0.025), (0.2, 0.0, 0.025))
        write_archive(same, [(77, *stack_trajectory(fixture, 77))])
        config = replace(self.config(archive=str(same)), method="cap_method_2", seed=3)
        result, backend = self.execute(config, ["Pick(b0)"])
        self.assertEqual(result["status"], "demonstration_rejected")
        self.assertIn("matches demonstration d0", result["reason"])

    def test_recovery_stops_when_the_demonstration_input_is_rejected(self):
        config = replace(
            self.config(max_tokens=10),
            method="cap_method_2",
            recovery={"enabled": True},
        )
        result, backend = self.execute(config, ["Pick(b0)"])
        self.assertEqual(result["stop_reason"], "demonstration_input_rejected")
        self.assertEqual(backend.prompts, [])


class RoundBudgetTests(unittest.TestCase):
    def test_reservations_exclude_the_block_but_the_context_includes_it(self):
        ledger = Budget(max_tokens=1000)
        ledger.charged_tokens = 0
        budget = RoundBudget(max_tokens=1000)
        budget.parent, budget.context_tokens, budget.output_tokens = ledger, 600, 100
        reservation = budget.reserve_request("x" * 50, demonstration_tokens=400)
        self.assertEqual(reservation["reserved_tokens"], 150)
        self.assertEqual(reservation["demonstration_tokens"], 400)
        budget.account_tokens(480, demonstration_tokens=400)
        self.assertEqual((budget.tokens, ledger.tokens), (80, 80))
        self.assertEqual(ledger.demonstration_tokens, 400)
        self.assertEqual(ledger.charged_tokens, 80)
        with self.assertRaises(Exception):
            budget.reserve_request("x" * 150, demonstration_tokens=400)
        self.assertEqual(budget.exhausted_by, "context_tokens")


if __name__ == "__main__":
    unittest.main()
