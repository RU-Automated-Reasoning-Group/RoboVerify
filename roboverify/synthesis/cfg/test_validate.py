"""Temporal acceptance requires first-new <= last-old, including equality."""

import unittest
from unittest.mock import patch

from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.refine import refine_cfg
from synthesis.cfg.validate import validate_cfg, validate_split
from synthesis.predicates.enumerate import SearchResult
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, negate, ref


def milestones(old, new):
    return Scene(
        {
            0: (0, 0, 0),
            1: ((0, 0, 0.05) if new else (0.2, 0, 0)),
            2: (0.4, 0, 0.05 if old else 0),
        },
        {"a": 1, "b": 0, "c": 2},
    )


def example_cfg(sequence, *, refine=False):
    previous = negate(atom("Higher", ref("b"), ref("c")))
    following = atom("ON", ref("a"), ref("b"))
    trace = DemoTrace(tuple(milestones(*flags) for flags in sequence))
    cfg = RelationalCFG(
        {"left": Node("left"), "right": Node("right")},
        [
            Edge("entry", "left", boolean(True)),
            Edge("left", "right", previous),
            Edge("right", "exit", boolean(True) if refine else following),
        ],
        ["left", "right"],
        DemoAssignment(
            {
                "left": [DemoSegment(0, 0, 1, trace)],
                "right": [DemoSegment(0, 1, len(sequence) - 1, trace)],
            }
        ),
    )
    return cfg, following


class TemporalAcceptanceTests(unittest.TestCase):
    def test_accepts_overlap_and_equality_but_rejects_gaps(self):
        self.assertTrue(validate_split({0: 10}, {0: 20}))
        self.assertTrue(validate_split({0: 10}, {0: 10}))
        self.assertFalse(validate_split({0: 11}, {0: 10}))

    def test_requires_witnesses_for_every_demonstration(self):
        for starts, finishes in (
            ({}, {}),
            ({0: None}, {0: 10}),
            ({0: 10}, {0: None}),
            ({0: 10}, {1: 10}),
            ({0: 10, 1: 11}, {0: 20, 1: 10}),
        ):
            with self.subTest(starts=starts, finishes=finishes):
                self.assertFalse(validate_split(starts, finishes))

    def test_complete_cfg_rejects_a_gap_before_the_final_postcondition(self):
        cfg, _ = example_cfg(
            [(False, False), (True, False), (False, False), (False, True)]
        )
        self.assertFalse(validate_cfg(cfg))

    def test_complete_cfg_accepts_equality_and_persistence(self):
        for ending in (((True, False), (True, True)), ((True, True), (True, True))):
            with self.subTest(ending=ending):
                cfg, _ = example_cfg([(False, False), (True, False), *ending])
                self.assertTrue(validate_cfg(cfg))

    def test_refinement_rejects_a_gap_without_mutating_graph_or_demos(self):
        cfg, following = example_cfg(
            [
                (False, False),
                (True, False),
                (False, False),
                (False, True),
                (False, True),
            ],
            refine=True,
        )
        nodes, edges, order, demos = cfg.nodes, cfg.edges, cfg.order, cfg.demos.segments
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", following),
        ):
            result = refine_cfg(cfg, "right", [], {"a", "b", "c"})
        self.assertEqual(result.status, "validate_reject")
        for before, after in (
            (nodes, cfg.nodes),
            (edges, cfg.edges),
            (order, cfg.order),
            (demos, cfg.demos.segments),
        ):
            self.assertIs(before, after)

    def test_refinement_accepts_a_shared_boundary(self):
        cfg, following = example_cfg(
            [(False, False), (True, False), (True, True), (False, True)], refine=True
        )
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", following),
        ):
            result = refine_cfg(cfg, "right", [], {"a", "b", "c"})
        self.assertTrue(result)
        self.assertEqual(cfg.order, ["left", "right.0", "right.1"])
        self.assertTrue(validate_cfg(cfg))

    def test_gap_in_one_demo_rejects_the_whole_cfg(self):
        cfg, _ = example_cfg([(False, False), (True, False), (True, True)])
        bad, _ = example_cfg(
            [(False, False), (True, False), (False, False), (False, True)]
        )
        for name in cfg.order:
            cfg.demos.segments[name].extend(bad.demos.segments[name])
        self.assertFalse(validate_cfg(cfg))
