import unittest
from unittest.mock import patch

from synthesis.api.instructions import (
    Get,
    Move,
    Pick,
    PickByName,
    Release,
    While,
)
from synthesis.api.program import Program
from synthesis.cfg.bindings import mutate_ids, require_ids
from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.id_first import close_id_candidate
from synthesis.cfg.lower import lower
from synthesis.cfg.quotient import find_repetition, quotient
from synthesis.cfg.refine import refine_cfg
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.synthesize import synthesize_cfg
from synthesis.cfg.tasks import task_spec
from synthesis.predicates.classifier import learn_ground_classifier
from synthesis.predicates.language import Language
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, block_id, free_names, ref, to_z3
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def stacking_trace(count=4, shift=0):
    states = []
    for step in range(count):
        positions = {
            i: (
                (shift, 0, 0.425 + i * 0.05)
                if i <= step
                else (shift + 0.2 * i, 0, 0.425)
            )
            for i in range(count)
        }
        states.append(Scene(positions, {"b0": 0}))
    return DemoTrace(tuple(states), num_blocks=count)


def numeric_placement(source, target):
    return (
        Pick(source),
        Move(source, source, target, target_offset=[0, 0, 0.20]),
        Move(0, 0, target, target_offset=[0, 0, 0.20]),
        Move(0, 0, target, target_offset=[0, 0, 0.05]),
        Release(source, target_z=0.15),
    )


def ground_cfg(count=4):
    traces = [stacking_trace(count, shift) for shift in (0, 0.1)]
    pre, post = task_spec("stack")
    names = [f"v{i}" for i in range(count - 1)]
    labels = [
        atom("ON", block_id(i + 1), ref("b0") if i == 0 else block_id(i))
        for i in range(count - 2)
    ] + [post]
    cfg = RelationalCFG(
        {
            name: Node(name, BlockRegion(None, numeric_placement(i + 1, i)))
            for i, name in enumerate(names)
        },
        [Edge("entry", names[0], pre)]
        + [
            Edge(name, names[i + 1] if i + 1 < len(names) else "exit", labels[i])
            for i, name in enumerate(names)
        ],
        names,
        DemoAssignment(
            {
                name: [
                    DemoSegment(k, i, i + 1, trace, {"b0": 0})
                    for k, trace in enumerate(traces)
                ]
                for i, name in enumerate(names)
            }
        ),
        pre,
        post,
        initial_scope=frozenset({"b0"}),
        synthesis_approach="id-first",
    )
    cfg._task_demos = [
        DemoSegment(k, 0, count - 1, trace, {"b0": 0}) for k, trace in enumerate(traces)
    ]
    return cfg


class IDFirstTests(unittest.TestCase):
    def test_ids_are_literals_and_must_be_named_before_z3(self):
        term = atom("ON", block_id(1), ref("b0"))
        self.assertEqual(str(term), "ON(1, b0)")
        self.assertEqual(free_names(term), {"b0"})
        self.assertTrue(evaluate(term, stacking_trace().states[1]))
        with self.assertRaisesRegex(ValueError, "named before"):
            to_z3(term, HighLevelContext())
        for invalid in (-1, True, "1"):
            with self.assertRaises(ValueError):
                block_id(invalid)

    def test_mcmc_proposals_never_introduce_names_or_gets(self):
        from synthesis.mcmc.synthesis import preserved_global_rng, set_np_seed

        with preserved_global_rng():
            set_np_seed(5)
            program = Program(5, list(numeric_placement(1, 0)))
            for _ in range(80):
                program = mutate_ids(program, 4)
                require_ids(program.instructions, 4)
        with self.assertRaises(ValueError):
            mutate_ids(Program(1, [PickByName("b0")]), 4)
        with self.assertRaises(ValueError):
            mutate_ids(Program(1, [Pick(4)]), 4)

    def test_refinement_learns_concrete_milestones_without_binders(self):
        trace = stacking_trace()
        pre, post = task_spec("stack")
        cfg = RelationalCFG.initial(
            [DemoSegment(0, 0, 3, trace, {"b0": 0})],
            pre,
            post,
            {"b0"},
            synthesis_approach="id-first",
        )
        first = refine_cfg(cfg, "v0", [trace.states[0]], {"b0"})
        self.assertTrue(first, first.status)
        self.assertEqual(str(first.term), "ON(1, b0)")
        second = refine_cfg(cfg, "v0.1", list(trace.states[:2]), {"b0"})
        self.assertTrue(second, second.status)
        self.assertEqual(str(second.term), "ON(2, 1)")
        self.assertTrue(all(not e.binds for e in cfg.edges))
        self.assertTrue(
            all(
                set(row.bindings) == {"b0"}
                for rows in cfg.demos.segments.values()
                for row in rows
            )
        )

    def test_no_quantified_fallback_when_ground_vocabulary_has_no_separator(self):
        states = stacking_trace().states
        result = learn_ground_classifier(
            [states[1]],
            [states[0]],
            {"b0"},
            language=Language(relations=("eq",), max_depth=2, max_variables=8),
        )
        self.assertEqual(result.status, "no_separator")

    def test_quotient_generalizes_ids_and_preserves_base_xy_top_z(self):
        cfg = ground_cfg()
        self.assertIsNotNone(
            find_repetition(
                [
                    atom("ON", block_id(1), ref("b0")),
                    atom("ON", block_id(2), block_id(1)),
                ]
            )
        )
        self.assertTrue(quotient(cfg))
        self.assertEqual(len(cfg.order), 1)
        loop = cfg.nodes[cfg.order[0]].region
        self.assertEqual(loop.init, (("b", "b0"),))
        self.assertEqual(loop.update, (("b", "b_prime"),))
        self.assertEqual(loop.iteration_counts, (3, 3))
        self.assertEqual(cfg.postcondition, task_spec("stack")[1])
        body = loop.body[0].physical
        self.assertEqual(body[0].grab_box_name, "b_prime")
        self.assertEqual(
            (
                body[1].target_box_name_x,
                body[1].target_box_name_y,
                body[1].target_box_name_z,
            ),
            ("b_prime", "b_prime", "b"),
        )
        self.assertEqual(
            (
                body[3].target_box_name_x,
                body[3].target_box_name_y,
                body[3].target_box_name_z,
            ),
            ("b0", "b0", "b"),
        )
        self.assertEqual([p.numeric_val() for p in body[3].target_offset], [0, 0, 0.05])
        self.assertEqual(body[4].target_z_offset.numeric_val(), 0.15)
        self.assertIsNone(loop.invariant)
        close_id_candidate(cfg)
        program = lower(cfg, HighLevelContext(), physical=True)
        self.assertIsInstance(program.instructions[1], While)
        self.assertEqual(cfg._fixed_id_bindings, {})

    def test_search_is_numeric_until_quotient_and_returns_only_named_program(self):
        cfg = ground_cfg(3)
        seen = []

        def realize(node, demos, post):
            self.assertEqual(node.synthesis_approach, "id-first")
            require_ids(node.region.physical, 3)
            seen.append(node.name)
            return node.region, True

        def fold(candidate):
            self.assertEqual(seen, ["v0", "v1"])
            return quotient(candidate)

        result = synthesize_cfg(cfg, realize, lambda c: [], quotient=fold)
        self.assertTrue(result)
        self.assertIsInstance(cfg.nodes[cfg.order[0]].region, LoopRegion)
        program = lower(cfg, HighLevelContext(), physical=True)
        for inst in program.instructions[1].body:
            self.assertFalse(any(o["type"] == "Box" for o in inst.get_operand()))
        self.assertTrue(
            all(n.synthesis_approach == "relational" for n in cfg.nodes.values())
        )

    def test_unfolded_program_gets_fixed_aliases_not_arbitrary_witnesses(self):
        cfg = ground_cfg(3)
        # A shape mismatch conservatively retains a straight-line program.
        cfg.nodes["v1"].region.physical = cfg.nodes["v1"].region.physical[1:]
        self.assertFalse(quotient(cfg))
        close_id_candidate(cfg)
        self.assertEqual(cfg._fixed_id_bindings, {"block_1": 1, "block_2": 2})
        self.assertEqual(str(cfg.edges[1].label), "ON(block_1, b0)")
        context = HighLevelContext()
        self.assertIn("block_1", str(to_z3(cfg.precondition, context)))
        program = lower(cfg, context, physical=True)
        self.assertFalse(any(isinstance(i, Get) for i in program.instructions))
        self.assertTrue(
            all(
                o["type"] != "Box"
                for i in program.instructions
                for o in i.get_operand()
            )
        )
        for row in cfg._task_demos:
            self.assertEqual(row.bindings["block_2"], 2)
        original = str(cfg.precondition)
        close_id_candidate(cfg)
        self.assertEqual(str(cfg.precondition), original)

    def test_successful_straightline_boundary_is_named_even_without_a_fold(self):
        cfg = ground_cfg(2)
        result = synthesize_cfg(
            cfg,
            lambda node, demos, post: (node.region, True),
            lambda c: [],
            quotient=quotient,
        )
        self.assertTrue(result)
        self.assertEqual(cfg._fixed_id_bindings, {"block_1": 1})
        self.assertIsInstance(
            lower(cfg, HighLevelContext(), physical=True).instructions[0], PickByName
        )

    def test_named_boundary_runs_existing_symbolic_and_motion_verifiers(self):
        from synthesis.cfg.test_verification import placement_cfg
        from synthesis.cfg.verification import (
            propose_summaries,
            verify_cfg_motion,
            verify_cfg_symbolic,
        )
        from synthesis.predicates.term import substitute

        cfg = placement_cfg()
        mapping = {"a": block_id(1), "b": ref("b0")}
        cfg.precondition = substitute(cfg.precondition, mapping)
        cfg.postcondition = substitute(cfg.postcondition, mapping)
        cfg.edges = [
            Edge("entry", "v0", cfg.precondition),
            Edge("v0", "exit", cfg.postcondition),
        ]
        cfg.initial_scope = frozenset({"b0"})
        cfg.synthesis_approach = "id-first"
        body = numeric_placement(1, 0)
        body = (body[0], Move(1, 1, 1, target_offset=[0, 0, 0.3]), *body[2:])
        cfg.nodes["v0"].region = BlockRegion(None, body)
        first = Scene({0: (0, 0, 0), 1: (0.3, 0, -0.1)}, {"b0": 0})
        final = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, {"b0": 0})
        trace = DemoTrace((first, final), num_blocks=2)
        rows = [DemoSegment(0, 0, 1, trace, {"b0": 0})]
        cfg.demos = DemoAssignment({"v0": rows})
        result = synthesize_cfg(
            cfg, lambda node, demos, post: (node.region, True), lambda c: []
        )
        self.assertTrue(result)
        context = HighLevelContext()
        propose_summaries(cfg, context)
        symbolic = verify_cfg_symbolic(cfg, context, max_blocks=2)
        self.assertTrue(symbolic, str(symbolic))
        motion = verify_cfg_motion(
            cfg,
            context,
            initial_positions={"block_1": [0.3, 0, -0.1], "b0": [0, 0, 0]},
            initial_arm=[0.3, 0, 0.2],
        )
        self.assertTrue(motion, str(motion))

    def test_cli_selects_approach_and_enables_id_first_quotient(self):
        from contextlib import redirect_stdout
        from io import StringIO

        from synthesis.entry.synthesize_cfg import main

        for flags, expected, folding in [
            ([], "relational", False),
            (["--synthesis-approach", "id-first"], "id-first", True),
            (["--synthesis-approach", "relational", "--quotient"], "relational", True),
        ]:
            with self.subTest(approach=expected, flags=flags), patch(
                "synthesis.entry.synthesize_cfg.RunLogger"
            ) as logger, patch(
                "synthesis.entry.synthesize_cfg.run", return_value=0
            ) as run, redirect_stdout(
                StringIO()
            ):
                self.assertEqual(main(["--demos", "unused.npz", *flags]), 0)
                args = run.call_args.args[0]
                self.assertEqual(args.synthesis_approach, expected)
                self.assertEqual(args.quotient, folding)
                self.assertEqual(
                    logger.call_args.args[2]["synthesis_approach"], expected
                )

    def test_resynthesis_retains_id_first_and_prepare_sees_only_names(self):
        from synthesis.api.instructions import Skip
        from synthesis.cfg.test_verification import placement_cfg
        from synthesis.cfg.verified_synthesis import verified_synthesis

        cfg = placement_cfg()
        cfg.synthesis_approach = "id-first"
        aliases = {"a": 1, "b": 0, "b0": 0}
        before = Scene({0: (0, 0, 0), 1: (0.3, 0, -0.1)}, aliases)
        after = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, aliases)
        demo = DemoSegment(0, 0, 1, DemoTrace((before, after), num_blocks=2), aliases)
        cfg.demos.segments["v0"] = [demo]
        attempts, prepared = [], []

        def realize(node, demos, post):
            self.assertEqual(node.synthesis_approach, "id-first")
            attempts.append(len(demos))
            body = numeric_placement(1, 0)
            body = (body[0], Move(1, 1, 1, target_offset=[0, 0, 0.3]), *body[2:])
            return BlockRegion(None, body if len(attempts) > 1 else (Skip(0),)), True

        def prepare(candidate, revision):
            program = lower(candidate, HighLevelContext(), physical=True)
            self.assertFalse(
                any(
                    o["type"] == "Box"
                    for i in program.instructions
                    for o in i.get_operand()
                )
            )
            prepared.append(revision)
            return {}

        result = verified_synthesis(
            cfg,
            realize,
            lambda _: [],
            HighLevelContext(),
            prepare=prepare,
            demo_provider=lambda request: [demo],
            min_blocks=2,
            max_blocks=2,
            motion_options={
                "initial_positions": {
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "b0": [0, 0, 0],
                },
                "initial_arm": [0.3, 0, 0.2],
            },
        )
        self.assertTrue(result, f"{result.status}: {result.reason}")
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(prepared, [0, 1])

    def test_fixed_alias_does_not_capture_demonstration_only_name(self):
        cfg = ground_cfg(2)
        for row in cfg._task_demos:
            row.bindings["block_1"] = 0
        close_id_candidate(cfg)
        self.assertNotIn("block_1", cfg._fixed_id_bindings)
        self.assertEqual(list(cfg._fixed_id_bindings.values()), [1])


if __name__ == "__main__":
    unittest.main()
