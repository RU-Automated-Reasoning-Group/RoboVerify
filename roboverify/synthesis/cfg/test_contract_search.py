import unittest
from functools import partial
from unittest.mock import Mock, patch

import numpy as np
import z3

from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    ReleaseByName,
)
from synthesis.api.program import Program
from synthesis.cfg.bindings import require_closed, require_ids
from synthesis.cfg.contract_search import (
    check_contract_feasibility,
    entry_holding,
    make_contract_candidates,
)
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.search_feasibility import check_scene_feasibility
from synthesis.cfg.straightline import SearchBudget, straight_line_synthesize
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom
from synthesis.predicates.term import block_id as obj_id
from synthesis.predicates.term import boolean, ref
from synthesis.util import on
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def program(*instructions):
    return Program(len(instructions), list(instructions))


def observation(positions, gripper, aperture=0.08):
    obs = np.zeros(10 + 12 * len(positions))
    obs[:3] = gripper
    obs[3:5] = aperture / 2
    for index, position in enumerate(positions):
        obs[10 + 12 * index : 13 + 12 * index] = position
    return obs


def held_segment():
    start = observation([(0, 0, 0.425), (0.2, 0, 0.625)], (0.2, 0, 0.625), 0.05)
    end = observation([(0, 0, 0.425), (0, 0, 0.475)], (0, 0, 0.475), 0.05)
    trace = DemoTrace(
        (start, end),
        num_blocks=2,
        metadata={"initial_holding": 1},
    )
    return DemoSegment(0, 0, 1, trace, {"support": 0, "payload": 1})


class ReferenceBMCIntegrationTests(unittest.TestCase):
    def test_optional_operand_expansion_preserves_goals_on_its_deadline(self):
        from synthesis.search.contract import SearchResult

        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch(
                "synthesis.search.operand_neighborhood.expand_operand_neighborhood",
                return_value=SearchResult(
                    (), {"status": "deadline_exhausted"}, "deadline_exhausted"
                ),
            ) as expand:
                segment = held_segment()
                candidates = make_contract_candidates(
                    [segment],
                    atom("ON", ref("payload"), ref("support")),
                    program(),
                    segment.bindings,
                    max_length=1,
                    budget=SearchBudget(
                        search_strategy="contract",
                        contract_seconds=3,
                        contract_operand_neighborhood=enabled,
                    ),
                )
                generated = list(candidates)
                self.assertTrue(generated, candidates.reason)
                self.assertEqual(expand.call_count, int(enabled))
                self.assertTrue(
                    check_contract_feasibility(
                        generated[0], [segment], fixed_offsets=True
                    )
                )
                if enabled:
                    self.assertEqual(
                        candidates.stats["operand_neighborhood"]["status"],
                        "deadline_exhausted",
                    )

    def setUp(self):
        self.scene = Scene(
            {0: (0, 0, 0.425), 1: (0.3, 0.1, 0.425), 2: (0.5, -0.2, 0.625)},
            {"a": 0, "b": 1},
        )

    def check(self, candidate, expected, **kwargs):
        return check_scene_feasibility(
            candidate,
            self.scene,
            gripper=(0.2, 0.15, 0.7),
            fixed_offsets=True,
            goal=lambda symbols, aliases: z3.And(
                *[
                    series[symbols.T] == z3.RealVal(str(value))
                    for series, value in zip(
                        (symbols.ee_x, symbols.ee_y, symbols.ee_z), expected
                    )
                ]
            ),
            **kwargs,
        )

    def test_extrema_include_objects_absent_from_operands(self):
        p = program(MoveByName("gripper", "Y_min", "Z_m", target_offset=[0, 0, 0]))
        self.assertTrue(self.check(p, (0.2, -0.2, 0.625), initial_holding=None))
        wrong = self.check(p, (0.2, 0.0, 0.425), initial_holding=None)
        self.assertEqual(wrong.status, "unsat")

    def test_extrema_are_recomputed_at_each_instruction_entry(self):
        p = program(
            PickByName("b"),
            MoveByName("b", "b", "b", target_offset=[0, 0, 0.4]),
            MoveByName("gripper", "Y_min", "Z_m", target_offset=[0, 0, 0]),
            ReleaseByName("gripper", target_z=0.1),
        )
        self.assertTrue(self.check(p, (0.3, -0.2, 0.925), initial_holding=None))
        self.assertFalse(self.check(p, (0.3, -0.2, 0.725), initial_holding=None))

    def test_gripper_coordinates_are_frozen_for_the_whole_instruction(self):
        p = program(
            MoveByName("gripper", "gripper", "gripper", target_offset=[0.1, 0.2, 0.3])
        )
        self.assertTrue(self.check(p, (0.3, 0.35, 1.0), initial_holding=None))

    def test_alias_updates_and_get_prefix_are_preserved(self):
        context = HighLevelContext()
        condition = atom("eq", ref("chosen"), obj_id(1))
        # Get's executable Z3 condition uses a named existing alias; its typed
        # term is the finite-scene witness criterion used by the adapter.
        get = Get(
            "chosen",
            context.get_consts("chosen") == context.get_consts("b"),
            [context.get_consts("chosen")],
            guard_term=condition,
        )
        p = program(
            get,
            Assign("alias", "chosen"),
            MoveByName("alias", "alias", "alias", target_offset=[0, 0, 0]),
        )
        self.assertTrue(self.check(p, (0.3, 0.1, 0.425), initial_holding=None))
        self.assertFalse(
            self.check(
                program(MoveByName("a", "a", "a", target_offset=[0, 0, 0]), get),
                (0, 0, 0.425),
            )
        )

    def test_table_frame_is_explicit_and_not_a_physical_object(self):
        scene = Scene(dict(self.scene.positions, tbl=on.TABLE), {"table_alias": "tbl"})
        p = program(
            MoveByName(
                "table_alias", "table_alias", "table_alias", target_offset=[0, 0, 0.05]
            )
        )
        missing = check_scene_feasibility(p, scene, initial_holding=None)
        self.assertEqual(missing.status, "unsupported")
        result = check_scene_feasibility(
            p,
            scene,
            initial_holding=None,
            fixed_offsets=True,
            table_reference=(1.2, 0.7, 0.375),
            goal=lambda sym, bindings: z3.And(
                sym.ee_z[sym.T] == z3.RealVal("0.425"),
                sym.ee_x[sym.T] == z3.RealVal("1.2"),
                len(sym.block_names) == 3,
            ),
        )
        self.assertTrue(result)

    def test_recorded_grasp_offset_survives_move_and_rejects_false_boundary(self):
        scene = Scene({0: (0, 0, 0.606), 1: (0.2, 0, 0.5)}, {"payload": 0})
        p = program(
            MoveByName("gripper", "gripper", "gripper", target_offset=[0, 0, -0.1])
        )

        def check(goal):
            return check_scene_feasibility(
                p,
                scene,
                gripper=(0, 0, 0.6),
                initial_holding=0,
                fixed_offsets=True,
                goal=goal,
            )

        self.assertTrue(check(lambda s, _: s.bz["0"][s.T] == z3.RealVal("0.506")))
        # A support at .5 is not Higher than the payload at .506, even though
        # the gripper has reached .5. The previous snapping model accepted it.
        self.assertEqual(
            check(
                lambda s, _: s.bz["1"][s.T] + z3.RealVal("0.001") >= s.bz["0"][s.T]
            ).status,
            "unsat",
        )

    def test_move_kernel_retains_legacy_and_noisy_offset_policies(self):
        from synthesis.verification_lib.bmc_lib import (
            NoiseSpec,
            _encode_move,
            build_trace_symbols,
        )

        for noise, preserve, expected in (
            (None, False, "0.5"),
            (None, True, "0.506"),
            (NoiseSpec(0, 0, 0), False, "0.506"),
        ):
            with self.subTest(noise=noise, preserve=preserve):
                s = build_trace_symbols(("0",), 1)
                solver = z3.Solver()
                solver.add(
                    s.bx["0"][0] == 0,
                    s.by["0"][0] == 0,
                    s.bz["0"][0] == z3.RealVal("0.606"),
                    s.ee_x[0] == 0,
                    s.ee_y[0] == 0,
                    s.ee_z[0] == z3.RealVal("0.6"),
                    s.holding[0] == s.block_consts["0"],
                    _encode_move(
                        s,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        z3.RealVal("-0.106"),
                        noise,
                        preserve_grasp_offset=preserve,
                    ),
                )
                self.assertEqual(solver.check(), z3.sat)
                solver.add(s.bz["0"][1] != z3.RealVal(expected))
                self.assertEqual(solver.check(), z3.unsat)

    def test_unknown_holding_is_not_silently_set_empty(self):
        p = program(ReleaseByName("a", target_z=0.1))
        self.assertTrue(
            check_scene_feasibility(p, self.scene, initial_holding="unknown")
        )
        self.assertFalse(check_scene_feasibility(p, self.scene, initial_holding=None))


class ContractCFGIntegrationTests(unittest.TestCase):
    def test_shared_solver_proposals_pass_fixed_offset_bmc_on_all_entries(self):
        segments = []
        for index, error in enumerate((-0.00044, 0.001803, 0.00025)):
            start = observation(
                [(0, 0, 0.425), (0.2, 0, 0.6)], (0.2, 0, 0.6 - error), 0.05
            )
            end = observation([(0, 0, 0.425), (0.2, 0, 0.425)], (0.2, 0, 0.575))
            trace = DemoTrace(
                (start, end), num_blocks=2, metadata={"initial_holding": 1}
            )
            segments.append(DemoSegment(index, 0, 1, trace, {"base": 0, "payload": 1}))
        candidates = make_contract_candidates(
            segments,
            atom("Higher", ref("base"), ref("payload")),
            program(),
            {"base", "payload"},
            max_length=2,
            budget=SearchBudget(
                search_strategy="contract",
                contract_seconds=5,
                contract_solve_offsets=True,
                cem_iterations=0,
                contract_observation_proposals=False,
            ),
        )
        proposals = list(candidates)
        self.assertTrue(proposals, candidates.stats)
        self.assertGreater(candidates.stats.get("offset_solver_sat", 0), 0)
        for candidate in proposals:
            self.assertTrue(
                check_contract_feasibility(candidate, segments, fixed_offsets=True)
            )
            for segment in segments:
                first = segment.trace.states[0]
                result = check_scene_feasibility(
                    candidate,
                    Scene(
                        {i: on.get_block_pos(first, i) for i in range(2)},
                        segment.bindings,
                    ),
                    gripper=first[:3],
                    initial_holding=1,
                    fixed_offsets=True,
                    goal=lambda s, _: z3.And(
                        on.higher_z3(s.bz["0"][s.T], s.bz["1"][s.T]),
                        s.holding[s.T] == s.NONE,
                    ),
                )
                self.assertTrue(result, result.reason)

    def test_real_backend_finds_named_and_numeric_candidates(self):
        segment = held_segment()
        post = atom("ON", obj_id(1), obj_id(0))
        budget = SearchBudget(
            search_strategy="contract", contract_seconds=3, cem_iterations=0
        )
        for numeric in (False, True):
            with self.subTest(numeric=numeric):
                candidates = make_contract_candidates(
                    [segment],
                    post,
                    program(),
                    segment.bindings,
                    max_length=1,
                    budget=budget,
                    id_first=numeric,
                )
                self.assertEqual(candidates.status, "not_started")
                first = list(candidates)
                self.assertTrue(first, candidates.reason)
                self.assertGreater(candidates.stats.get("generated", 0), 0)
                frozen_stats = dict(candidates.stats)
                self.assertEqual([str(p) for p in first], [str(p) for p in candidates])
                self.assertEqual(candidates.stats, frozen_stats)
                for p in first:
                    if numeric:
                        require_ids(p.instructions, 2)
                    else:
                        require_closed(p.instructions, segment.bindings)
                    self.assertTrue(
                        check_contract_feasibility(p, [segment], fixed_offsets=True)
                    )

    def test_interior_open_hand_and_explicit_held_metadata(self):
        s = held_segment()
        trace = DemoTrace((s.trace.states[0],) * 3, num_blocks=2)
        interior = DemoSegment(0, 1, 2, trace, s.bindings)
        with self.assertRaisesRegex(ValueError, "unknown holding"):
            entry_holding(interior)
        trace.metadata["holding_by_index"] = {"1": 1}
        self.assertEqual(entry_holding(interior), 1)
        trace.metadata.clear()
        opened = s.trace.states[0].copy()
        opened[3:5] = 0.04
        trace.states = (opened,) * 3
        self.assertIsNone(entry_holding(interior))

    def test_real_backend_preserves_get_prefix_and_new_alias_scope(self):
        segment = held_segment()
        context = HighLevelContext()
        condition = atom("eq", ref("chosen"), obj_id(0))
        prefix = program(
            Get(
                "chosen",
                context.get_consts("chosen") == context.get_consts("support"),
                [context.get_consts("chosen")],
                guard_term=condition,
            ),
            Assign("target", "chosen"),
        )
        candidates = make_contract_candidates(
            [segment],
            atom("ON", ref("payload"), ref("target")),
            prefix,
            {"payload"},
            max_length=1,
            budget=SearchBudget(search_strategy="contract", contract_seconds=3),
        )
        generated = list(candidates)
        self.assertTrue(generated, candidates.reason)
        self.assertEqual(
            set(candidates.problem.examples[0].bindings) & {"chosen", "target"},
            {"chosen", "target"},
        )
        for candidate in generated:
            self.assertIsInstance(candidate.instructions[0], Get)
            self.assertIsInstance(candidate.instructions[1], Assign)
            self.assertEqual(
                require_closed(candidate.instructions, {"payload"}),
                {"payload", "chosen", "target"},
            )
            self.assertTrue(
                check_contract_feasibility(candidate, [segment], fixed_offsets=True)
            )

    def test_preparation_is_included_in_proposal_deadline(self):
        segment = held_segment()
        candidates = make_contract_candidates(
            [segment],
            boolean(True),
            program(),
            segment.bindings,
            max_length=1,
            budget=SearchBudget(search_strategy="contract", contract_seconds=1),
        )
        with patch(
            "synthesis.cfg.contract_search.perf_counter", side_effect=[0.0, 2.0]
        ):
            self.assertEqual(list(candidates), [])
        self.assertEqual(candidates.status, "deadline_exhausted")
        self.assertIn("preparation", candidates.reason)

    def test_proposal_adapter_preserves_the_external_wall_timeout(self):
        segment = held_segment()
        candidates = make_contract_candidates(
            [segment],
            boolean(True),
            program(),
            segment.bindings,
            max_length=1,
            budget=SearchBudget(search_strategy="contract"),
        )
        with patch.object(
            candidates, "_prepare", side_effect=TimeoutError("CLI wall budget")
        ):
            with self.assertRaisesRegex(TimeoutError, "CLI wall budget"):
                list(candidates)

    def test_bmc_rejection_precedes_every_rollout_and_cem(self):
        s = held_segment()
        rollout = Mock(side_effect=AssertionError("Rejected program must not execute"))
        rejected = program(PickByName("payload"), PickByName("support"))
        result = straight_line_synthesize(
            [s],
            boolean(True),
            rejected,
            None,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="contract", iterations=1, distance="mmd"
            ),
            candidates=[rejected],
            feasible=partial(check_contract_feasibility, segments=[s]),
        )
        self.assertEqual(result.status, "no_feasible_candidate")
        self.assertEqual((result.candidates_checked, result.bmc_rejected), (1, 1))
        self.assertFalse(result.ok)
        self.assertEqual(result.program.instructions, [])
        rollout.assert_not_called()

    def test_partial_adapter_supplies_checked_refinement_attempts(self):
        segment = held_segment()
        candidates = make_contract_candidates(
            [segment],
            boolean(False),
            program(),
            segment.bindings,
            max_length=1,
            budget=SearchBudget(search_strategy="contract", contract_seconds=3),
            refinement_fallback=True,
        )
        generated = list(candidates)
        self.assertTrue(generated, candidates.reason)
        self.assertEqual(candidates.stats["shortlist_size"], 0)
        self.assertEqual(candidates.stats["refinement_candidates"], len(generated))
        for candidate in generated:
            self.assertTrue(candidate._contract_refinement_only)
            self.assertTrue(
                check_contract_feasibility(candidate, [segment], fixed_offsets=True)
            )

    def test_partial_candidate_cannot_pass_even_with_low_imitation_distance(self):
        segment = held_segment()
        generated = program(
            MoveByName("support", "support", "support", target_offset=[0, 0, 0.05])
        )
        generated._contract_refinement_only = True
        events = []

        def feasible(candidate):
            events.append("bmc")
            return check_contract_feasibility(candidate, [segment])

        def rollout(candidate, demo):
            self.assertTrue(events)
            self.assertEqual(events[0], "bmc")
            events.append("rollout")
            self.assertEqual(len(candidate.instructions), 1)
            return demo.states

        result = straight_line_synthesize(
            [segment],
            boolean(True),
            program(PickByName("support")),
            None,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="contract",
                iterations=0,
                cem_iterations=0,
                distance="mmd",
                theta=1,
            ),
            candidates=[generated],
            feasible=feasible,
        )
        self.assertLess(result.distance, 1)
        self.assertFalse(result.ok)
        self.assertEqual(len(result.program.instructions), 1)
        self.assertIsInstance(result.program.instructions[0], MoveByName)
        self.assertTrue(result.program._contract_refinement_only)
        self.assertIn("rollout", events)

    def test_deterministic_cache_reuses_execution_across_distances_and_postscore(self):
        segment = held_segment()
        generated = program(
            MoveByName("support", "support", "support", target_offset=[0, 0, 0.05])
        )
        for cached, expected in ((False, 3), (True, 1)):
            with self.subTest(cache=cached):
                rollout = Mock(side_effect=lambda candidate, demo: demo.states)
                result = straight_line_synthesize(
                    [segment],
                    boolean(True),
                    program(),
                    None,
                    rollout=rollout,
                    budget=SearchBudget(
                        search_strategy="contract",
                        iterations=0,
                        cem_iterations=0,
                        distance="mmd",
                        theta=1,
                    ),
                    candidates=[generated],
                    feasible=lambda candidate: True,
                    cache_rollouts=cached,
                )
                self.assertTrue(result.ok)
                self.assertEqual(result.post_score, 1)
                self.assertEqual(rollout.call_count, expected)
                if cached:
                    self.assertEqual(result.search_stats["rollout_cache_hits"], 2)

    def test_deterministic_cache_invalidates_changed_cem_parameters(self):
        segment = held_segment()
        generated = program(
            MoveByName("support", "support", "support", target_offset=[0, 0, 0.05])
        )
        seen = []

        def rollout(candidate, demo):
            seen.append(tuple(candidate.register_trainable_parameter()))
            return demo.states

        result = straight_line_synthesize(
            [segment],
            boolean(True),
            program(),
            None,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="contract",
                iterations=0,
                cem_iterations=1,
                cem_samples=2,
                cem_elites=1,
                distance="mmd",
                theta=1,
            ),
            candidates=[generated],
            feasible=lambda candidate: True,
            cache_rollouts=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(result.search_stats["rollout_cache_misses"], 3)
        self.assertEqual(result.search_stats["rollout_cache_hits"], 7)

    def test_complete_only_cem_preserves_bmc_and_unoptimized_refinement(self):
        segment = held_segment()
        for enabled, partial, expected in (
            (False, True, 3),
            (True, True, 1),
            (False, False, 3),
            (True, False, 3),
        ):
            with self.subTest(enabled=enabled, partial=partial):
                candidate = program(
                    MoveByName(
                        "support", "support", "support", target_offset=[0, 0, 0.05]
                    )
                )
                candidate._contract_refinement_only = partial
                events, parameters = [], []

                def feasible(value):
                    events.append("bmc")
                    return True

                def rollout(value, demo):
                    self.assertEqual(events[0], "bmc")
                    events.append("rollout")
                    parameters.append(tuple(value.register_trainable_parameter()))
                    return demo.states

                result = straight_line_synthesize(
                    [segment],
                    boolean(True),
                    program(),
                    None,
                    rollout=rollout,
                    budget=SearchBudget(
                        search_strategy="contract",
                        iterations=0,
                        cem_iterations=1,
                        cem_samples=2,
                        cem_elites=1,
                        contract_cem_complete_only=enabled,
                        distance="mmd",
                        theta=1,
                    ),
                    candidates=[candidate],
                    feasible=feasible,
                    cache_rollouts=True,
                )
                self.assertEqual(len(parameters), expected)
                self.assertEqual(len(set(parameters)), expected)
                self.assertEqual(result.ok, not partial)
                self.assertEqual(
                    result.search_stats.get("cem_skipped_partial_candidates", 0),
                    int(enabled and partial),
                )
                if enabled and partial:
                    self.assertEqual(parameters, [(0.0, 0.0, 0.05)])
                    self.assertTrue(result.program._contract_refinement_only)

    def test_contract_retains_nearby_cem_trial_for_postscore_only(self):
        segment = held_segment()
        candidate = program(
            MoveByName("support", "support", "support", target_offset=[0, 0, 0])
        )
        for strategy, trial_offset, partial_candidate, expected in (
            ("contract", 0.002, False, True),
            ("contract", 0.02, False, False),
            ("contract", 0.002, True, False),
            ("enumeration", 0.002, False, False),
        ):
            candidate._contract_refinement_only = partial_candidate
            with self.subTest(
                strategy=strategy, offset=trial_offset, partial=partial_candidate
            ):
                rng = Mock()
                rng.normal.return_value = np.array([[trial_offset, 0, 0]])
                checked = []

                def feasible(value):
                    checked.append(value)
                    return True

                def rollout(value, demo):
                    self.assertTrue(checked, "BMC must precede every CEM rollout")
                    return np.array([[value.register_trainable_parameter()[0]]])

                with patch(
                    "synthesis.cfg.straightline.np.random.default_rng", return_value=rng
                ), patch(
                    "synthesis.cfg.straightline._features",
                    side_effect=lambda states, n: states,
                ), patch(
                    "synthesis.cfg.straightline.make_distance",
                    return_value=lambda states, samples: abs(states[0, 0]),
                ), patch(
                    "synthesis.cfg.straightline.postcondition_reached",
                    side_effect=lambda states, segment, post: states[0, 0] > 0.001,
                ):
                    result = straight_line_synthesize(
                        [segment],
                        boolean(True),
                        candidate,
                        None,
                        rollout=rollout,
                        budget=SearchBudget(
                            search_strategy=strategy,
                            iterations=0,
                            cem_iterations=1,
                            cem_samples=1,
                            cem_elites=1,
                            distance="mmd",
                            epsilon=0.005,
                            theta=0.03,
                        ),
                        candidates=[candidate],
                        feasible=feasible,
                        cache_rollouts=True,
                    )
                self.assertEqual(result.ok, expected)
                self.assertEqual(len(checked), 1)
                if expected:
                    self.assertEqual(
                        result.program.register_trainable_parameter()[0], trial_offset
                    )
                    self.assertAlmostEqual(result.distance, trial_offset)

    def test_exact_pool_keys_preserve_parameter_variants_and_default_behavior(self):
        from synthesis.cfg.program_source import program_fingerprint
        from synthesis.mcmc.search_core import CandidatePool

        first = program(MoveByName("a", "a", "a", target_offset=[0, 0, 0]))
        second = program(MoveByName("a", "a", "a", target_offset=[0.002, 0, 0]))
        self.assertEqual(str(first), str(second))
        for pool, count in (
            (CandidatePool(epsilon=0.005), 1),
            (CandidatePool(epsilon=0.005, key=program_fingerprint), 2),
        ):
            pool.add(first, 0)
            pool.add(second, 0.002)
            self.assertEqual(len(pool.entries), count)
            second.instructions[0].target_offset[0].val = 0.1
            if count == 2:
                self.assertEqual(
                    pool.entries[1].program.register_trainable_parameter()[0], 0.002
                )
            second.instructions[0].target_offset[0].val = 0.002

    def test_contract_only_optimizes_generated_candidates_not_initial_seed(self):
        segment = held_segment()
        initial = program()
        generated = program(
            MoveByName("support", "support", "support", target_offset=[0, 0, 0.05])
        )
        checked = []

        def feasible(candidate):
            checked.append(candidate)
            return check_contract_feasibility(candidate, [segment])

        def rollout(candidate, demo):
            self.assertEqual(len(candidate.instructions), 1)
            return demo.states

        result = straight_line_synthesize(
            [segment],
            boolean(True),
            initial,
            None,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="contract",
                iterations=0,
                cem_iterations=0,
                distance="mmd",
            ),
            candidates=[generated],
            feasible=feasible,
        )
        self.assertTrue(result.ok)
        self.assertEqual((result.candidates_checked, result.iterations), (1, 1))
        self.assertEqual(len(checked), 1)
        self.assertEqual(len(checked[0].instructions), 1)

    def test_contract_requires_bmc_and_keeps_imitation_scale_explicit(self):
        s = held_segment()
        with self.assertRaisesRegex(ValueError, "BMC screening"):
            straight_line_synthesize(
                [s],
                boolean(True),
                program(),
                None,
                rollout=lambda p, demo: demo.states,
                budget=SearchBudget(search_strategy="contract"),
            )
        self.assertEqual(SearchBudget(search_strategy="contract").distance, "kl_kde")
        with self.assertRaisesRegex(ValueError, "Contract proposal budgets"):
            SearchBudget(search_strategy="contract", contract_seconds=float("inf"))

    def test_unsupported_input_has_explicit_status(self):
        s = held_segment()
        s.trace.states = (
            Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.625)}, s.bindings),
        ) * 2
        candidates = make_contract_candidates(
            [s],
            boolean(True),
            program(),
            s.bindings,
            max_length=1,
            budget=SearchBudget(search_strategy="contract"),
        )
        self.assertEqual(list(candidates), [])
        self.assertEqual(candidates.status, "unsupported")
        self.assertIn("recorded gripper", candidates.reason)


if __name__ == "__main__":
    unittest.main()
