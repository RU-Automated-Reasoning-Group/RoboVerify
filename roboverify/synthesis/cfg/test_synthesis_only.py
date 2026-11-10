import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthesis.api.instructions import Skip
from synthesis.api.program import Program
from synthesis.cfg.candidate_traces import prepare_candidate
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_io import load_program_artifact, save_program_artifact
from synthesis.cfg.program_source import ProgramDefinition
from synthesis.cfg.straightline import StraightLineResult
from synthesis.cfg.tasks import task_identity, task_spec
from synthesis.cfg.test_collection import example_trace
from synthesis.cfg.verified_synthesis import VerifiedResult
from synthesis.entry.synthesize_cfg import main
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class SynthesisOnlyTests(unittest.TestCase):
    def trace(self):
        trace = example_trace()
        trace.metadata.update(task_spec=task_identity("stack"), fingerprint="teacher")
        return trace

    def test_synthesis_mode_saves_candidate_without_calling_verification(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "synthesis.entry.synthesize_cfg.load_traces", return_value=[self.trace()]
        ), patch(
            "synthesis.entry.synthesize_cfg.straight_line_synthesize",
            return_value=StraightLineResult(
                Program(1, [Skip(0)]), True, "test_oracle", 0, 1
            ),
        ) as search, patch(
            "synthesis.entry.synthesize_cfg.verified_synthesis"
        ) as verify, patch(
            "synthesis.entry.synthesize_cfg.prepare_candidate"
        ) as infer, contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(
                main(
                    [
                        "--mode",
                        "synthesize",
                        "--demos",
                        "unused.npz",
                        "--num-blocks",
                        "2",
                        "--search-strategy",
                        "contract",
                        "--output-dir",
                        directory,
                        "--distance",
                        "arc",
                        "--epsilon",
                        "0.003",
                        "--cem-std",
                        "0.004",
                    ]
                ),
                0,
            )
            verify.assert_not_called()
            infer.assert_not_called()
            budget = search.call_args.kwargs["budget"]
            self.assertEqual(
                (budget.distance, budget.epsilon, budget.cem_std), ("arc", 0.003, 0.004)
            )
            artifact = next(Path(directory).glob("cfg/*/artifacts/program.json"))
            payload = json.loads(artifact.read_text())
            self.assertEqual(payload["metadata"]["formal_verification"], "not_run")
            self.assertEqual(
                payload["metadata"]["synthesis_status"], "synthesized_unverified"
            )
            self.assertTrue(payload["metadata"]["accepted_by_cfg_search"])
            loaded = load_program_artifact(artifact, HighLevelContext(), 2)
            self.assertEqual(loaded.initial_bindings, {"b0": 0})

    def test_complete_program_survives_failed_post_quotient_search(self):
        from types import SimpleNamespace

        from synthesis.cfg.quotient import quotient
        from synthesis.cfg.region import BlockRegion
        from synthesis.cfg.test_id_first import ground_cfg
        from synthesis.entry.synthesize_cfg import _synthesize_only
        from synthesis.experiment.run_logger import RunLogger

        cfg = ground_cfg()
        cfg._task_demos = cfg.demos.for_node(cfg.order[0])
        expected = [
            str(instruction)
            for name in cfg.order
            for instruction in cfg.nodes[name].region.physical
        ]

        def realize(node, demos, post):
            if node.synthesis_approach == "id-first":
                return node.region, True
            return BlockRegion(None, (Skip(0),)), False

        args = SimpleNamespace(
            task="stack", num_blocks=4, refinements=0, predicate_seconds=5
        )
        with tempfile.TemporaryDirectory() as directory:
            with RunLogger(directory, "test", {}, slug="checkpoint") as logger:
                status = _synthesize_only(
                    cfg,
                    realize,
                    lambda candidate: [],
                    HighLevelContext(),
                    args,
                    logger,
                    quotient,
                )
                checkpoint = (
                    logger.artifact_dir()
                    / "checkpoints/pre-quotient-round-0000/program.json"
                )
                self.assertEqual(status, 2)
                loaded = load_program_artifact(checkpoint, HighLevelContext(), 4)
                self.assertEqual(loaded.initial_bindings, {"b0": 0})
                self.assertEqual(
                    [str(instruction) for instruction in loaded.program.instructions],
                    expected,
                )
                metadata = json.loads(checkpoint.read_text())["metadata"]
                self.assertEqual(
                    metadata["synthesis_status"], "search_complete_unrolled"
                )
                self.assertEqual(metadata["formal_verification"], "not_run")
                saved_cfg = json.loads(checkpoint.with_name("cfg.json").read_text())
                self.assertEqual(saved_cfg["order"], ["v0", "v1", "v2"])
                self.assertEqual(cfg.order, ["v0.loop"])

    def test_failed_block_attempts_retain_exact_parameters_and_entry_bindings(self):
        from types import SimpleNamespace

        from synthesis.api.instructions import MoveByName
        from synthesis.entry.synthesize_cfg import _save_block_candidate
        from synthesis.experiment.run_logger import RunLogger

        trace = self.trace()
        entries = [
            DemoSegment(0, 0, 1, trace, {"b0": 0, "source": source})
            for source in (0, 1)
        ]
        result = StraightLineResult(
            Program(
                1,
                [
                    MoveByName(
                        "source", "source", "b0", target_offset=[0.123456789, 0, 0.15]
                    )
                ],
            ),
            False,
            "budget_exhausted",
            float("inf"),
        )
        args = SimpleNamespace(
            task="stack", num_blocks=2, search_strategy="contract", demos="original.npz"
        )
        node = SimpleNamespace(
            name="body0",
            available_scope={"b0", "source"},
            synthesis_approach="relational",
        )
        with tempfile.TemporaryDirectory() as directory:
            with RunLogger(directory, "test", {}, slug="failed-candidate") as logger:
                for attempt in (0, 1):
                    _save_block_candidate(logger, result, node, entries, args, attempt)
                logger.finish("test_completed")
            paths = sorted(
                {
                    path.resolve()
                    for path in Path(directory).glob(
                        "test/*/artifacts/blocks/*/program.json"
                    )
                }
            )
            self.assertEqual(len(paths), 2)
            row = json.loads(paths[0].read_text())
            self.assertEqual(row["initial_bindings"], {"b0": 0})
            self.assertEqual(row["metadata"]["entry_bindings_required"], ["source"])
            self.assertEqual(row["metadata"]["distance"], None)
            self.assertFalse(row["metadata"]["accepted_by_block_search"])
            self.assertEqual(
                [entry["bindings"]["source"] for entry in row["metadata"]["entries"]],
                [0, 1],
            )
            self.assertEqual(row["instructions"][0]["offsets"][0], 0.123456789)

    def test_teacher_fingerprint_mismatch_requires_explicit_candidate_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            definition = ProgramDefinition(
                Program(1, [Skip(0)]), {"b0": 0, "fixed": 1}, "test"
            )
            save_program_artifact(path, definition)
            base = [
                "--mode",
                "verify",
                "--program",
                str(path),
                "--demos",
                "unused.npz",
                "--num-blocks",
                "2",
                "--output-dir",
                directory,
            ]
            with patch(
                "synthesis.entry.synthesize_cfg.load_traces",
                return_value=[self.trace()],
            ), patch(
                "synthesis.entry.synthesize_cfg.verified_synthesis"
            ) as verify, contextlib.redirect_stdout(
                io.StringIO()
            ):
                with self.assertRaisesRegex(
                    ValueError, "differs from the demonstration"
                ):
                    main(base)
                verify.assert_not_called()

                def stop(cfg, *args, **kwargs):
                    self.assertTrue(cfg._candidate_program)
                    self.assertEqual(cfg._initial_bindings["fixed"], 1)
                    self.assertTrue(kwargs["initial_candidate"])
                    return VerifiedResult(cfg, "test_stopped")

                verify.side_effect = stop
                self.assertEqual(main(base + ["--candidate-program"]), 2)
                verify.assert_called_once()

    def test_candidate_reload_uses_its_bindings_and_event_paths(self):
        original = self.trace()
        original.events = (
            dict(
                kind="instruction_start",
                path="teacher_path",
                index=0,
                bindings={"b0": 0},
            ),
        )
        candidate = copy.deepcopy(original)
        candidate.events = (
            dict(
                kind="instruction_start",
                path="0",
                index=0,
                bindings={"b0": 0, "fixed": 1},
            ),
            dict(
                kind="instruction_end",
                path="0",
                index=1,
                bindings={"b0": 0, "fixed": 1},
            ),
        )
        context = HighLevelContext()
        definition = ProgramDefinition(
            Program(1, [Skip(0)]), {"b0": 0, "fixed": 1}, "test"
        )
        cfg = program_to_cfg(
            definition, [DemoSegment(0, 0, 1, original)], *task_spec("stack")
        )
        cfg._candidate_program = True
        with patch(
            "synthesis.cfg.candidate_traces.record_execution", return_value=candidate
        ) as record:
            self.assertEqual(prepare_candidate(cfg, context), {})
        self.assertEqual(record.call_args.args[0].initial_bindings["fixed"], 1)
        rows = cfg.demos.for_node(cfg.order[0])
        self.assertEqual(len(rows), 1)
        self.assertIs(rows[0].trace, candidate)
        self.assertEqual((rows[0].t_start, rows[0].t_end), (0, 1))


if __name__ == "__main__":
    unittest.main()
