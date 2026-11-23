import json
import tempfile
import unittest
from pathlib import Path

import z3

from synthesis.api.control import ControlConfig
from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    ReleaseByName,
    Skip,
    While,
)
from synthesis.api.program import Program
from synthesis.cfg.program_io import load_program_artifact, save_program_artifact
from synthesis.cfg.program_source import (
    ProgramDefinition,
    describe_program,
    guard_term,
    program_fingerprint,
)
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, conjunction, forall, negate, ref, to_z3
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class ProgramArtifactTests(unittest.TestCase):
    def definition(self, context):
        condition = conjunction(
            negate(atom("eq", ref("source"), ref("b0"))),
            forall(("u",), negate(atom("ON", ref("u"), ref("source")))),
        )
        move = MoveByName(
            "b",
            "Y_min",
            "Z_m",
            limit=41,
            target_offset=[0.123456789012345, -0.001234567890123, 0.15],
            control=ControlConfig(gain=17.25),
        )
        loop = While(
            to_z3(condition, context),
            [context.get_consts("source")],
            [
                PickByName("source"),
                move,
                ReleaseByName("Z_m", target_z=0.14000000000001),
                Assign("b", "source"),
            ],
            z3.BoolVal(True),
            max_iters=None,
        )
        loop.guard_term = condition
        return ProgramDefinition(
            Program(
                4,
                [
                    Get(
                        "source",
                        to_z3(condition, context),
                        [context.get_consts("source")],
                        guard_term=condition,
                    ),
                    Assign("b", "b0"),
                    loop,
                    Skip(0),
                ],
            ),
            {"b0": 0, "fixed_alias": 2},
            "test",
        )

    def test_exact_roundtrip_in_fresh_context_preserves_guards_and_floats(self):
        original = self.definition(HighLevelContext())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json"
            save_program_artifact(
                path, original, metadata={"formal_verification": "not_run"}
            )
            loaded = load_program_artifact(path, HighLevelContext(), 4)
        self.assertEqual(loaded.initial_bindings, original.initial_bindings)
        self.assertEqual(
            describe_program(loaded.program), describe_program(original.program)
        )
        self.assertEqual(
            program_fingerprint(loaded.program), program_fingerprint(original.program)
        )
        loop = loaded.program.instructions[2]
        self.assertIsNone(loop.max_iters)
        self.assertEqual(loop.body[1].target_offset[0].val, 0.123456789012345)
        self.assertEqual(loop.body[1].control.gain, 17.25)
        scene = Scene(
            {0: (0, 0, 0.425), 1: (0.2, 0, 0.425), 2: (0.4, 0, 0.425)},
            {"b0": 0, "source": 1},
        )
        self.assertTrue(evaluate(guard_term(loop), scene))
        scene.bindings["source"] = 0
        self.assertFalse(evaluate(guard_term(loop), scene))

    def test_tampering_and_missing_fixed_object_fail_closed(self):
        definition = self.definition(HighLevelContext())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json"
            payload = save_program_artifact(path, definition)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_program_artifact(path, HighLevelContext(), 2)
            payload["instructions"][2]["body"][1]["offsets"][0] += 0.01
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_program_artifact(path, HighLevelContext(), 4)

    def test_unspecified_offsets_freeze_their_runtime_zero_value(self):
        definition = ProgramDefinition(
            Program(1, [MoveByName("b0", "b0", "b0")]), {"b0": 0}, "test"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json"
            save_program_artifact(path, definition)
            loaded = load_program_artifact(path, HighLevelContext(), 2)
        self.assertEqual(
            [p.val for p in loaded.program.instructions[0].target_offset], [0.0] * 3
        )
        self.assertTrue(
            all(p.val is None for p in definition.program.instructions[0].target_offset)
        )


if __name__ == "__main__":
    unittest.main()
