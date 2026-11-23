import copy
import unittest

from synthesis.api.instructions import Move
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.graph import Node
from synthesis.cfg.realization_cache import SuccessfulRealizationCache
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.predicates.term import boolean


class SuccessfulRealizationCacheTests(unittest.TestCase):
    def setup_case(self):
        trace = DemoTrace(tuple(range(5)))
        demos = [DemoSegment(0, 1, 3, trace, {"b0": 0}, entry_index=0)]
        region = BlockRegion(None, (Move(0, 0, 0, target_offset=[0.0, 0.0, 0.1]),))
        node = Node("body", region, frozenset({"b0"}), "id-first")
        post = boolean(True)
        cache = SuccessfulRealizationCache()
        cache.remember(node, demos, post, region, accepted=True)
        return cache, node, demos, post

    def test_exact_accepted_block_is_reused(self):
        cache, node, demos, post = self.setup_case()
        # A newly allocated list/segment describing the same archived entries
        # is still the same search input.
        self.assertIs(cache.get(node, [copy.copy(demos[0])], post), node.region)

    def test_changed_search_input_invalidates_acceptance(self):
        changes = {
            "post": lambda n, d: boolean(False),
            "scope": lambda n, d: setattr(n, "available_scope", frozenset()),
            "approach": lambda n, d: setattr(n, "synthesis_approach", "relational"),
            "start": lambda n, d: setattr(d[0], "t_start", 2),
            "end": lambda n, d: setattr(d[0], "t_end", 4),
            "entry": lambda n, d: setattr(d[0], "entry_index", 1),
            "milestone": lambda n, d: setattr(d[0], "milestone_index", 2),
            "handoff": lambda n, d: setattr(
                n, "_contract_successors", {(0, 0, 1, 3): {"b": 1}}
            ),
            "demo": lambda n, d: setattr(d[0], "demo_idx", 1),
            "bindings": lambda n, d: d[0].bindings.update(b0=1),
            "final_bindings": lambda n, d: setattr(d[0], "final_bindings", {"b0": 1}),
            "goal_bindings": lambda n, d: setattr(d[0], "goal_bindings", {"g": 1}),
            "trace": lambda n, d: setattr(d[0], "trace", copy.copy(d[0].trace)),
            "physical": lambda n, d: setattr(n, "region", BlockRegion(None, ())),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                cache, node, demos, post = self.setup_case()
                value = change(node, demos)
                if label == "post":
                    post = value
                self.assertIsNone(cache.get(node, demos, post))
                self.assertFalse(cache._entries)

    def test_failed_penalized_and_loop_realizations_are_not_cached(self):
        for kind in ("failed", "penalized", "loop"):
            with self.subTest(kind=kind):
                cache, node, demos, post = self.setup_case()
                region = (
                    LoopRegion(boolean(True), (), ()) if kind == "loop" else node.region
                )
                cache.remember(
                    node,
                    demos,
                    post,
                    region,
                    accepted=kind != "failed",
                    penalty=object() if kind == "penalized" else None,
                )
                node.region = region
                self.assertIsNone(cache.get(node, demos, post))

    def test_in_place_observation_snapshot_and_metadata_changes_invalidate(self):
        changes = {
            "observation": lambda t: t.states[2].__setitem__(0, 99),
            "frozen_entry": lambda t: t.states[0].__setitem__(0, 99),
            "snapshot": lambda t: t.snapshots[1].update(position=99),
            "initial_snapshot": lambda t: t.snapshots[0].update(position=99),
            "actions": lambda t: t.actions[0].append(99),
            "metadata": lambda t: t.metadata["holding_by_index"].update({1: 2}),
            "seed": lambda t: setattr(t, "seed", 99),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                cache, node, demos, post = self.setup_case()
                trace = demos[0].trace
                trace.states = tuple([float(i)] for i in range(5))
                trace.snapshots = tuple({"position": i} for i in range(5))
                trace.actions = ([0, 1, 2], [0, 1, 2, 3, 3])
                trace.metadata = {"holding_by_index": {1: None}}
                cache.remember(node, demos, post, node.region, accepted=True)
                self.assertIs(cache.get(node, demos, post), node.region)
                change(trace)
                self.assertIsNone(cache.get(node, demos, post))

    def test_new_penalty_and_replacement_node_invalidate_reuse(self):
        cache, node, demos, post = self.setup_case()
        self.assertIsNone(cache.get(copy.copy(node), demos, post))
        self.assertIsNone(cache.get(node, demos, post, penalty=object()))
        self.assertIsNone(cache.get(node, demos, post))


if __name__ == "__main__":
    unittest.main()
