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
from synthesis.predicates.term import atom, ref
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class SynthesisOnlyTests(unittest.TestCase):
    def test_selected_alternative_and_extra_starts_are_reflected_in_saved_program(self):
        from synthesis.cfg.straightline import StraightLineAlternative

        training, extra = self.trace(), self.trace()
        extra.num_blocks = 3
        result = StraightLineResult(Program(1, [Skip(0)]), True, "test_oracle", 0, 1)
        result.alternatives = (StraightLineAlternative(Program(1, [Skip(1)]), 0, 1),)
        with tempfile.TemporaryDirectory() as directory, patch(
            "synthesis.entry.synthesize_cfg.load_traces",
            side_effect=[[training], [extra]],
        ), patch(
            "synthesis.entry.synthesize_cfg.straight_line_synthesize",
            return_value=result,
        ) as search, patch(
            "synthesis.cfg.robust_validation.validate_training_program",
            side_effect=[
                {"accepted": False, "passed": 1, "total": 2, "seconds": 0.1},
                {"accepted": True, "passed": 2, "total": 2, "seconds": 0.1},
            ],
        ) as validation, contextlib.redirect_stdout(
            io.StringIO()
        ):
            code = main(
                [
                    "--mode",
                    "synthesize",
                    "--demos",
                    "training.npz",
                    "--num-blocks",
                    "2",
                    "--search-strategy",
                    "contract",
                    "--contract-refinement",
                    "--validation-starts",
                    "extra.npz",
                    "--output-dir",
                    directory,
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(search.call_args.args[0]), 1)
            self.assertIs(search.call_args.args[0][0].trace, training)
            self.assertEqual(validation.call_count, 2)
            for call in validation.call_args_list:
                self.assertEqual([t.num_blocks for t in call.args[1]], [2, 3])
                self.assertIs(call.args[1][1], extra)
            artifact = next(Path(directory).glob("cfg/*/artifacts/program.json"))
            loaded = load_program_artifact(artifact, HighLevelContext(), 2)
            self.assertEqual(str(loaded.program.instructions[0]), str(Skip(1)))
            cfg = json.loads(artifact.with_name("cfg.json").read_text())
            self.assertEqual(cfg["nodes"]["v0"]["physical"], [str(Skip(1))])
            report = json.loads(
                artifact.with_name("training-selection-0.json").read_text()
            )
            self.assertEqual(report["selected_ranks"], [1])
            alternative = next(
                artifact.parent.glob("blocks/*/alternatives/candidate-0000.json")
            )
            saved = load_program_artifact(alternative, HighLevelContext(), 2)
            self.assertEqual(str(saved.program.instructions[0]), str(Skip(1)))
            self.assertEqual(
                json.loads(alternative.read_text())["metadata"]["alternative_rank"], 0
            )

    def test_local_search_success_does_not_override_failed_complete_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "synthesis.entry.synthesize_cfg.load_traces", return_value=[self.trace()]
        ), patch(
            "synthesis.entry.synthesize_cfg.straight_line_synthesize",
            return_value=StraightLineResult(
                Program(1, [Skip(0)]), True, "test_oracle", 0, 1
            ),
        ), patch(
            "synthesis.entry.synthesize_cfg._validate_synthesis_candidate",
            return_value={
                "accepted": False,
                "passed": 0,
                "total": 1,
                "demos": [{"reasons": ["settling_goal"]}],
            },
        ) as validation, contextlib.redirect_stdout(
            io.StringIO()
        ):
            code = main(
                [
                    "--mode",
                    "synthesize",
                    "--demos",
                    "unused.npz",
                    "--num-blocks",
                    "2",
                    "--search-strategy",
                    "contract",
                    "--contract-refinement",
                    "--output-dir",
                    directory,
                ]
            )
            self.assertEqual(code, 2)
            validation.assert_called_once()
            artifact = next(Path(directory).glob("cfg/*/artifacts/program.json"))
            metadata = json.loads(artifact.read_text())["metadata"]
            self.assertEqual(metadata["synthesis_status"], "simulation_rejected")
            self.assertTrue(metadata["accepted_by_cfg_search"])
            self.assertFalse(metadata["training_simulation_passed"])
            result = json.loads(
                artifact.parent.parent.joinpath("result.json").read_text()
            )
            self.assertEqual(result["simulation_evaluation"], "training_rejected")

    def test_contract_refinement_requires_search_and_explicit_metric_scales(self):
        for flags in (
            ["--contract-refinement"],
            ["--loop-body-distance", "arc"],
            [
                "--loop-body-distance",
                "arc",
                "--loop-body-theta",
                "nan",
                "--loop-body-epsilon",
                ".003",
            ],
            [
                "--search-strategy",
                "contract",
                "--contract-refinement",
                "--boundary-max-shift",
                "0",
            ],
        ):
            with self.subTest(flags=flags), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit) as error:
                main(["--mode", "synthesize", "--demos", "unused.npz", *flags])
            self.assertEqual(error.exception.code, 2)

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

    def test_only_accepted_empty_search_results_become_executable_identity(self):
        for accepted in (True, False):
            with self.subTest(
                accepted=accepted
            ), tempfile.TemporaryDirectory() as directory, patch(
                "synthesis.entry.synthesize_cfg.load_traces",
                return_value=[self.trace()],
            ), patch(
                "synthesis.entry.synthesize_cfg.straight_line_synthesize",
                return_value=StraightLineResult(
                    Program(0, []),
                    accepted,
                    "converged" if accepted else "budget_exhausted",
                    0,
                    1 if accepted else 0,
                ),
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                result = main(
                    [
                        "--mode",
                        "synthesize",
                        "--demos",
                        "unused.npz",
                        "--num-blocks",
                        "2",
                        "--search-strategy",
                        "contract",
                        "--refinements",
                        "0",
                        "--output-dir",
                        directory,
                    ]
                )
                self.assertEqual(result, 0 if accepted else 2)
                cfg_path = next(Path(directory).glob("cfg/*/artifacts/cfg.json"))
                artifact = cfg_path.with_name("program.json")
                cfg = json.loads(cfg_path.read_text())
                if accepted:
                    definition = load_program_artifact(artifact, HighLevelContext(), 2)
                    self.assertEqual(len(definition.program.instructions), 1)
                    identity = definition.program.instructions[0]
                    self.assertIsInstance(identity, Skip)
                    self.assertEqual(identity.skip_steps, 0)
                    trajectory = ["existing state"]
                    identity.eval(object(), trajectory)
                    self.assertEqual(trajectory, ["existing state"])
                    self.assertTrue(cfg["nodes"]["v0"]["physical"])
                else:
                    self.assertFalse(artifact.exists())
                    self.assertEqual(cfg["nodes"]["v0"]["physical"], [])

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
                post = atom("ON", ref("source"), ref("b0"))
                for attempt in (0, 1):
                    _save_block_candidate(
                        logger, result, node, entries, args, attempt, post
                    )
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
            self.assertEqual(row["metadata"]["target_postcondition"], str(post))
            self.assertEqual(row["metadata"]["target_postcondition_ast"]["op"], "ON")
            self.assertEqual(row["metadata"]["entries"][0]["entry_index"], 0)
            self.assertIsNone(row["metadata"]["entries"][0]["milestone_index"])
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
