import itertools
import random
import unittest

from synthesis.api.instructions import (
    Get,
    Move,
    MoveByName,
    Pick,
    Release,
    Skip,
)
from synthesis.api.program import Program
from synthesis.cfg.bindings import close_objects, enumerate_scoped, require_closed
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.mcmc.enumeration import enumerate_programs
from synthesis.predicates.scene import Scene
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class EnumerationTests(unittest.TestCase):
    def test_length_first_exhaustive_unique_structures(self):
        candidates = list(
            enumerate_programs({"Box": [1, 0, 1]}, [Pick, Move, Release, Skip, Pick], 2)
        )
        # Two Picks, eight independently referenced Moves, two Releases.
        self.assertEqual(len(candidates), 1 + 12 + 12**2)
        self.assertEqual(len({str(p) for p in candidates}), len(candidates))
        self.assertEqual(
            [len(p.instructions) for p in candidates], [0] + [1] * 12 + [2] * 144
        )
        moves = [
            p.instructions[0]
            for p in candidates
            if len(p.instructions) == 1 and isinstance(p.instructions[0], Move)
        ]
        operands = {tuple(o["val"] for o in m.get_operand()) for m in moves}
        self.assertEqual(operands, set(itertools.product([0, 1], repeat=3)))

    def test_lazy_and_independent_of_random_seed(self):
        state = random.getstate()
        first = list(
            itertools.islice(
                enumerate_programs({"Box": [0]}, [Pick, Move, Release], 1000), 6
            )
        )
        self.assertEqual(random.getstate(), state)
        second = list(
            itertools.islice(
                enumerate_programs({"Box": [0]}, [Pick, Move, Release], 1000), 6
            )
        )
        self.assertEqual(list(map(str, first)), list(map(str, second)))

    def test_parameters_and_prefix_are_not_shared(self):
        prefix = [Pick(0)]
        candidates = list(enumerate_programs({"Box": [0]}, [Move], 2, prefix=prefix))
        candidates[1].register_trainable_parameter()
        candidates[1].update_trainable_parameter([1, 2, 3])
        self.assertEqual(candidates[2].register_trainable_parameter(), [0.0] * 6)
        self.assertIsNot(candidates[1].instructions[0], prefix[0])
        self.assertIsNot(candidates[2].instructions[1], candidates[2].instructions[2])

    def test_named_enumeration_retains_binding_prefix(self):
        scene = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b0": 0, "demo_only": 1})
        segment = DemoSegment(
            0, 0, 1, DemoTrace((scene, scene), num_blocks=2), scene.bindings
        )
        initial = close_objects(
            Program(1, [Pick(1)]), [segment], {"b0"}, HighLevelContext()
        )
        for p in enumerate_scoped(initial, {"b0", "tbl"}, 1):
            self.assertIsInstance(p.instructions[0], Get)
            self.assertEqual(str(p.instructions[0]), str(initial.instructions[0]))
            require_closed(p.instructions, {"b0", "tbl"})
            self.assertTrue(
                all(
                    o["type"] == "BoxName" and o["val"] not in {"tbl", "demo_only"}
                    for i in p.instructions[1:]
                    for o in i.get_operand()
                )
            )

    def test_invalid_configuration(self):
        for operands, kinds, bound in [
            ({"Box": [0]}, [Pick], -1),
            ({"Box": []}, [Pick], 1),
            ({"Box": [0, "bad"]}, [Pick], 1),
            ({"Box": [0]}, [Get], 1),
            ({"Box": [0], "BoxName": ["b"]}, [Pick, MoveByName], 1),
        ]:
            with self.subTest(
                operands=operands, kinds=kinds, bound=bound
            ), self.assertRaises(ValueError):
                list(enumerate_programs(operands, kinds, bound))
