"""Numeric repair preserves CFG structure and all original acceptance gates."""

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np

from synthesis.api.instructions import Assign, Move, Release
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
from synthesis.cfg.endpoint_contract import EndpointResult
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.lower import lower
from synthesis.cfg.parameter_repair import repair_cfg_parameters
from synthesis.cfg.program_source import program_fingerprint
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.robust_validation import SharedActionBudget
from synthesis.cfg.search_feasibility import CandidateFeasibility
from synthesis.cfg.straightline import (
    SearchBudget,
    SegmentRollout,
    StraightLineResult,
    straight_line_synthesize,
)
from synthesis.predicates.scene import scene_from_obs
from synthesis.predicates.term import boolean
from synthesis.search.parameter_refinement import ParameterRefinementBudget


def parameters(program):
    return np.asarray(deepcopy(program).register_trainable_parameter())


def examples():
    obs = np.zeros(43)
    obs[3:5] = 0.04
    obs[10:13] = (0, 0, 0.425)
    obs[22:25] = (0, 0, 0.475)
    return [
        DemoSegment(
            index,
            0,
            1,
            DemoTrace(
                (obs.copy(), obs.copy()),
                num_blocks=2,
                metadata={"holding_by_index": {0: None, 1: None}},
            ),
            {"old": 0, "new": 1},
        )
        for index in range(2)
    ]


def block(name, instruction, *, reference=0.01):
    node = Node(
        name,
        BlockRegion(
            (Assign("logical_pointer", "old"),),
            (instruction,),
            frozenset({"export_" + name}),
        ),
    )
    node._contract_fit_budget = SearchBudget(
        search_strategy="contract",
        distance="arc",
        theta=0.03,
        epsilon=0.003,
        contract_endpoint_checks=True,
    )
    node._contract_fit_distance = reference
    return node


def chain(nodes):
    names = [node.name for node in nodes]
    return RelationalCFG(
        {node.name: node for node in nodes},
        [
            Edge(left, right, boolean(True))
            for left, right in zip(["entry"] + names, names + ["exit"])
        ],
        names,
        DemoAssignment({name: examples() for name in names}),
    )


def nested_cfg():
    before = block("before", Move(0, 1, 0, target_offset=[0.01, 0.02, 0.03]))
    first = block("release", Release(1, target_z=0.1))
    second = block("move", Move(1, 0, 1, target_offset=[0.04, 0.05, 0.06]))
    after = block("after", Release(0, target_z=0.2))
    body = chain([first, second])
    loop = LoopRegion(
        boolean(False),
        (),
        (first.region, second.region),
        init=(("pointer", "old"),),
        update=(("pointer", "new"),),
        body_cfg=body,
        iteration_limit=7,
    )
    cfg = chain([before, Node("loop", loop), after])
    return cfg, (before, first, second, after), loop


def local_success(segments, post, program, propose, **kwargs):
    return StraightLineResult(program, True, "converged", 0.011, 1.0)


class ParameterRepairTests(unittest.TestCase):
    def run_repair(self, cfg, validate, **options):
        return repair_cfg_parameters(
            cfg,
            None,
            validate,
            budget=options.pop("budget", ParameterRefinementBudget(max_trials=1)),
            env_factory=options.pop("env_factory", Mock()),
            **options,
        )

    def test_all_coordinates_map_across_nested_blocks_and_commit_exact_trial(self):
        cfg, nodes, loop = nested_cfg()
        originals = tuple(node.region for node in nodes)
        original_program = lower(cfg, None, physical=True)
        centre = parameters(original_program)
        original_region_fingerprints = [
            program_fingerprint(Program(len(region.physical), list(region.physical)))
            for region in originals
        ]
        frozen, seen = [], []

        def freeze(program, index, deadline):
            frozen.append(program_fingerprint(program))

        def validate(graph, index):
            program = lower(graph, None, physical=True)
            seen.append(parameters(program))
            self.assertEqual(program_fingerprint(program), frozen[-1])
            self.assertIs(loop.body[0], nodes[1].region)
            self.assertIs(loop.body[1], nodes[2].region)
            return {"accepted": len(seen) == 2 * len(centre)}

        with patch(
            "synthesis.cfg.parameter_repair.check_contract_feasibility",
            return_value=CandidateFeasibility(True, status="sat"),
        ) as bmc, patch(
            "synthesis.cfg.parameter_repair.straight_line_synthesize",
            side_effect=local_success,
        ):
            result = self.run_repair(
                cfg,
                validate,
                on_trial=freeze,
                budget=ParameterRefinementBudget(
                    max_trials=2 * len(centre), coordinate_multipliers=(1,)
                ),
            )
        self.assertTrue(result)
        self.assertEqual(len(seen), 16)
        for index, actual in enumerate(seen):
            expected = centre.copy()
            expected[index // 2] += 0.001 if index % 2 == 0 else -0.001
            np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            parameters(lower(cfg, None, physical=True)), seen[-1]
        )
        self.assertEqual(program_fingerprint(result.program), frozen[-1])
        self.assertEqual(bmc.call_count, 2 * len(seen))
        self.assertTrue(
            all(call.kwargs["fixed_offsets"] for call in bmc.call_args_list)
        )
        self.assertTrue(all(len(call.args[1]) == 1 for call in bmc.call_args_list))
        for node, original, fingerprint in zip(
            nodes, originals, original_region_fingerprints
        ):
            self.assertIs(node.region.symbolic, original.symbolic)
            self.assertIs(node.region.bindings, original.bindings)
            self.assertEqual(
                program_fingerprint(
                    Program(len(original.physical), list(original.physical))
                ),
                fingerprint,
            )
        self.assertIs(loop.guard, boolean(False))
        self.assertEqual(loop.init, (("pointer", "old"),))
        self.assertEqual(loop.update, (("pointer", "new"),))
        self.assertEqual(loop.max_iters, 7)

    def test_rejected_whole_program_restores_every_region_and_loop_body(self):
        cfg, nodes, loop = nested_cfg()
        originals = tuple(node.region for node in nodes)
        fingerprint = program_fingerprint(lower(cfg, None, physical=True))
        with patch(
            "synthesis.cfg.parameter_repair.check_contract_feasibility",
            return_value=CandidateFeasibility(True),
        ), patch("synthesis.cfg.parameter_repair.straight_line_synthesize") as local:
            result = self.run_repair(
                cfg,
                lambda *args: {"accepted": False},
                budget=ParameterRefinementBudget(max_trials=4),
            )
        self.assertFalse(result)
        local.assert_not_called()
        for node, original in zip(nodes, originals):
            self.assertIs(node.region, original)
        self.assertIs(loop.body[0], originals[1])
        self.assertIs(loop.body[1], originals[2])
        self.assertEqual(
            program_fingerprint(lower(cfg, None, physical=True)), fingerprint
        )

    def test_external_exception_restores_even_after_installation(self):
        for stage in ("bmc", "whole", "local"):
            with self.subTest(stage=stage):
                cfg, nodes, loop = nested_cfg()
                originals = tuple(node.region for node in nodes)
                bmc = Mock(return_value=CandidateFeasibility(True))
                validate = Mock(return_value={"accepted": True})
                local = Mock(side_effect=local_success)
                {"bmc": bmc, "whole": validate, "local": local}[stage].side_effect = (
                    TimeoutError("external deadline")
                )
                with patch(
                    "synthesis.cfg.parameter_repair.check_contract_feasibility", bmc
                ), patch(
                    "synthesis.cfg.parameter_repair.straight_line_synthesize", local
                ), self.assertRaisesRegex(
                    TimeoutError, "external deadline"
                ):
                    self.run_repair(cfg, validate)
                for node, original in zip(nodes, originals):
                    self.assertIs(node.region, original)
                self.assertIs(loop.body[0], originals[1])
                self.assertIs(loop.body[1], originals[2])

    def test_final_installation_exception_is_atomic(self):
        cfg, nodes, loop = nested_cfg()
        originals = tuple(node.region for node in nodes)
        gate_passed = [False]

        def local(*args, **kwargs):
            gate_passed[0] = True
            return local_success(*args, **kwargs)

        def lower_or_fail(*args, **kwargs):
            if gate_passed[0]:
                raise ValueError("final installation failed")
            return lower(*args, **kwargs)

        with patch(
            "synthesis.cfg.parameter_repair.lower", side_effect=lower_or_fail
        ), patch(
            "synthesis.cfg.parameter_repair.check_contract_feasibility",
            return_value=CandidateFeasibility(True),
        ), patch(
            "synthesis.cfg.parameter_repair.straight_line_synthesize", side_effect=local
        ), self.assertRaisesRegex(
            ValueError, "final installation failed"
        ):
            self.run_repair(cfg, lambda *args: {"accepted": True})
        for node, original in zip(nodes, originals):
            self.assertIs(node.region, original)
        self.assertIs(loop.body[0], originals[1])

    def test_fixed_bmc_rejection_precedes_any_physical_validation(self):
        cfg, nodes, _ = nested_cfg()
        original = nodes[0].region
        validate = Mock()
        with patch(
            "synthesis.cfg.parameter_repair.check_contract_feasibility",
            return_value=CandidateFeasibility(
                False, "fixed offsets infeasible", "unsat"
            ),
        ) as bmc, patch(
            "synthesis.cfg.parameter_repair.straight_line_synthesize"
        ) as local:
            result = self.run_repair(cfg, validate)
        self.assertFalse(result)
        self.assertEqual(result.history[0]["report"]["stage"], "fixed_bmc")
        self.assertTrue(bmc.call_args.kwargs["fixed_offsets"])
        validate.assert_not_called()
        local.assert_not_called()
        self.assertIs(nodes[0].region, original)

    def test_missing_original_gate_fails_closed(self):
        cfg, nodes, _ = nested_cfg()
        del nodes[0]._contract_fit_distance
        validate = Mock()
        with patch("synthesis.cfg.parameter_repair.check_contract_feasibility") as bmc:
            result = self.run_repair(cfg, validate)
        self.assertFalse(result)
        self.assertEqual(result.history[0]["report"]["stage"], "missing_local_gate")
        bmc.assert_not_called()
        validate.assert_not_called()

    def test_each_bmc_entry_checks_shared_deadline(self):
        cfg, nodes, _ = nested_cfg()
        original = nodes[0].region
        now = [10.0]
        validate = Mock()

        def first_entry(*args, **kwargs):
            now[0] = 12.0
            return CandidateFeasibility(True)

        with patch(
            "synthesis.cfg.parameter_repair.perf_counter", side_effect=lambda: now[0]
        ), patch(
            "synthesis.search.parameter_refinement.perf_counter",
            side_effect=lambda: now[0],
        ), patch(
            "synthesis.cfg.parameter_repair.check_contract_feasibility",
            side_effect=first_entry,
        ) as bmc:
            result = self.run_repair(
                cfg, validate, budget=ParameterRefinementBudget(seconds=1)
            )
        self.assertFalse(result)
        self.assertEqual(bmc.call_count, 1)
        self.assertEqual(len(bmc.call_args.args[1]), 1)
        self.assertLessEqual(bmc.call_args.kwargs["timeout_ms"], 1000)
        self.assertIn(result.status, ("deadline_exhausted", "budget_exhausted"))
        self.assertIs(nodes[0].region, original)
        validate.assert_not_called()

    def test_exhausted_shared_actions_prevent_bmc_and_physics(self):
        cfg, nodes, _ = nested_cfg()
        original = nodes[0].region
        validate = Mock()
        with patch("synthesis.cfg.parameter_repair.check_contract_feasibility") as bmc:
            result = self.run_repair(
                cfg, validate, action_budget=SharedActionBudget(1, 1)
            )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertFalse(result)
        self.assertIs(nodes[0].region, original)
        bmc.assert_not_called()
        validate.assert_not_called()

    def test_local_simulation_counts_actions_and_exact_cap_can_accept(self):
        for attempts, accepted in ((1, True), (2, False)):
            with self.subTest(attempts=attempts):
                cfg, nodes, _ = nested_cfg()
                original = nodes[0].region
                actions = SharedActionBudget(1)
                env = Mock()
                original_step = env.step

                def simulated_rollout(program, segment, factory, **kwargs):
                    environment = factory(segment.trace)
                    try:
                        for _ in range(attempts):
                            environment.step(np.zeros(4))
                    finally:
                        environment.close()

                def local(segments, post, program, propose, **kwargs):
                    kwargs["rollout"](program, segments[0])
                    return local_success(segments, post, program, propose, **kwargs)

                with patch(
                    "synthesis.cfg.parameter_repair.check_contract_feasibility",
                    return_value=CandidateFeasibility(True),
                ), patch(
                    "synthesis.cfg.parameter_repair.segment_rollout",
                    side_effect=simulated_rollout,
                ), patch(
                    "synthesis.cfg.parameter_repair.straight_line_synthesize",
                    side_effect=local,
                ):
                    result = self.run_repair(
                        cfg,
                        lambda *args: {"accepted": True},
                        env_factory=lambda trace: env,
                        action_budget=actions,
                    )
                self.assertEqual(bool(result), accepted)
                self.assertEqual(actions.used, 1)
                self.assertEqual(original_step.call_count, 1)
                env.close.assert_called_once()
                if not accepted:
                    self.assertEqual(result.status, "budget_exhausted")
                    self.assertIs(nodes[0].region, original)

    def test_real_local_search_preserves_theta_epsilon_endpoint_and_handoff_gates(self):
        cases = (
            ("theta", 0.04, 0.05, False, False, False),
            ("original epsilon", 0.02, 0.01, False, False, False),
            ("physical endpoint", 0.011, 0.01, True, False, False),
            ("actual handoff", 0.011, 0.01, False, True, False),
            ("all gates", 0.012, 0.01, False, False, True),
        )
        for name, distance, reference, endpoint_fails, handoff_fails, accepted in cases:
            with self.subTest(gate=name):
                node = block(
                    "body",
                    Move(0, 1, 0, target_offset=[0.01, 0.02, 0.03]),
                    reference=reference,
                )
                cfg = chain([node])
                original = node.region
                handoff = Mock(return_value=EndpointResult(not handoff_fails))
                factory = Mock(return_value=handoff)

                def rollout(candidate, segment, *args, **kwargs):
                    self.assertTrue(kwargs["audit_contract"])
                    self.assertEqual(kwargs["reset_mode"], "reset")
                    states = SegmentRollout()
                    states.observations = list(segment.states)
                    states.extend(
                        scene_from_obs(obs, 2, segment.bindings)
                        for obs in segment.states
                    )
                    states.contract_audit = {
                        "reasons": ["sampled_nonmover_frame"] if endpoint_fails else [],
                        "contract_bindings": dict(segment.bindings),
                        "endpoint": {
                            "known": True,
                            "holding": None,
                            "supports": [(1, 0)],
                        },
                        "instructions": [],
                    }
                    return states

                with patch(
                    "synthesis.cfg.parameter_repair.check_contract_feasibility",
                    return_value=CandidateFeasibility(True),
                ), patch(
                    "synthesis.cfg.parameter_repair.segment_rollout",
                    side_effect=rollout,
                ), patch(
                    "synthesis.cfg.parameter_repair.straight_line_synthesize",
                    wraps=straight_line_synthesize,
                ) as local, patch(
                    "synthesis.cfg.straightline.make_distance",
                    return_value=lambda *args, **kwargs: distance,
                ):
                    result = self.run_repair(
                        cfg,
                        lambda *args: {"accepted": True},
                        endpoint_acceptance=factory,
                    )
                self.assertEqual(bool(result), accepted)
                checked_budget = local.call_args.kwargs["budget"]
                self.assertEqual(checked_budget.theta, 0.03)
                self.assertEqual(checked_budget.epsilon, 0.003)
                self.assertEqual(checked_budget.distance, "arc")
                self.assertEqual(checked_budget.cem_iterations, 0)
                self.assertEqual(checked_budget.iterations, 0)
                self.assertTrue(checked_budget.contract_endpoint_checks)
                self.assertIs(local.call_args.kwargs["endpoint_acceptance"], handoff)
                factory.assert_called_once_with(node)
                if not accepted:
                    self.assertIs(node.region, original)
                    self.assertEqual(result.history[0]["report"]["stage"], "local_gate")
                if name in ("actual handoff", "all gates"):
                    self.assertEqual(
                        handoff.call_count, len(cfg.demos.for_node("body"))
                    )


if __name__ == "__main__":
    unittest.main()
