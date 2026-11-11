"""Physical-entry contract checks for milestone boundary proposals."""

import unittest
from unittest.mock import patch

import numpy as np

from synthesis.cfg.boundaries import (
    BoundaryBudget,
    _problem,
    point,
    propose_handoffs,
    truth_window,
)
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.predicates.term import atom, boolean, ref
from synthesis.search.contract import Action, ContractProblem

PICK_NEXT = Action("Pick", ("new",))
MILESTONE = atom("ON", ref("new"), ref("old"))


def handoff(*, known=True, aperture=0.08, broken=False, ready_at=2):
    positions = np.array([[0, 0, 0.425], [0, 0, 0.475], [0.2, 0, 0.425]])
    observations = []
    for index in range(5):
        obs = np.zeros(46)
        obs[:3] = [0, 0, 0.65 if index == 0 or index >= ready_at else 0.48]
        obs[3:5] = aperture / 2
        for obj, position in enumerate(positions):
            obs[10 + 12 * obj : 13 + 12 * obj] = position
        if broken and index == 2:
            obs[22] = 0.1  # ON is false here and true again at the later high pose.
        observations.append(obs)
    metadata = {"holding_by_index": {str(i): None for i in range(5)}} if known else {}
    trace = DemoTrace(tuple(observations), num_blocks=3, metadata=metadata)
    # The source/support roles change after the transfer. Picking 'new' at the
    # successor means object 2, not the just-placed object 1.
    before = DemoSegment(0, 0, 1, trace, {"old": 0, "new": 1}, entry_index=0)
    after = DemoSegment(0, 1, 4, trace, {"old": 1, "new": 2}, entry_index=0)
    return before, after


class BoundaryContractTests(unittest.TestCase):
    def test_initial_move_does_not_allow_mixed_iteration_hand_phases(self):
        row, nxt = handoff(ready_at=0)
        row.trace.metadata["holding_by_index"].update({"1": 2, "2": 2})
        for index in (1, 2):
            row.trace.states[index][34:37] = row.trace.states[index][:3]
        move = Action("Move", ("gripper", "gripper", "gripper"), (0, 0, 0.1))
        with patch.object(ContractProblem, "_make_actions", return_value=[move]):
            vectors, _ = propose_handoffs(
                [row],
                [nxt],
                boolean(True),
                prototypes=[point(nxt, 0)],
                initial_required=True,
            )
            self.assertEqual(vectors, [(1,)])
            vectors, _ = propose_handoffs(
                [row],
                [nxt],
                boolean(True),
                prototypes=[point(nxt, 0)],
                initial_required=True,
                phase_templates=[point(nxt, 0)],
            )
            self.assertEqual(vectors, [(3,)])

    def proposals(self, row, nxt, *, budget=None):
        # Restrict only the finite grammar in this test. Entry-state feasibility
        # and all envelope/holding checks use the production transition code.
        with patch.object(ContractProblem, "_make_actions", return_value=[PICK_NEXT]):
            return propose_handoffs(
                [row],
                [nxt],
                MILESTONE,
                budget=budget or BoundaryBudget(),
                prototypes=[point(nxt, 0)],
                initial_required=True,
            )

    def test_low_gripper_fails_next_pick_until_retreat_with_successor_bindings(self):
        row, nxt = handoff()
        low = _problem([point(nxt, 1)], actions=[PICK_NEXT])
        high = _problem([point(nxt, 2)], actions=[PICK_NEXT])
        self.assertIsNone(low.transition(low.initial, PICK_NEXT))
        self.assertIsNotNone(high.transition(high.initial, PICK_NEXT))
        # Using the old role binding would wrongly admit the old low cut.
        wrong_roles = _problem([point(row, 1)], actions=[PICK_NEXT])
        self.assertIsNotNone(wrong_roles.transition(wrong_roles.initial, PICK_NEXT))
        vectors, report = self.proposals(row, nxt)
        self.assertEqual(vectors, [(2,)])
        self.assertEqual(report["status"], "found")
        self.assertEqual(row.t_end, 1)
        self.assertEqual(nxt.t_start, 1)
        self.assertEqual(nxt.bindings["new"], 2)

    def test_unknown_closed_hand_does_not_become_empty(self):
        row, nxt = handoff(known=False, aperture=0.04)
        with self.assertRaisesRegex(ValueError, "unknown holding state"):
            _problem([point(nxt, 2)], actions=[PICK_NEXT])
        vectors, report = self.proposals(row, nxt)
        self.assertEqual(vectors, [])
        self.assertEqual(report["status"], "no_proposal")

    def test_observed_open_hand_can_supply_entry_evidence_without_contact_metadata(
        self,
    ):
        row, nxt = handoff(known=False)
        vectors, report = self.proposals(row, nxt)
        self.assertEqual(vectors, [(2,)])
        self.assertEqual(report["status"], "found")

    def test_cannot_jump_over_a_false_milestone_to_a_later_valid_pose(self):
        row, nxt = handoff(broken=True, ready_at=3)
        self.assertEqual(truth_window(row, MILESTONE, 3, 80), (1, 1))
        vectors, report = self.proposals(row, nxt)
        self.assertEqual(vectors, [])
        self.assertEqual(report["status"], "no_proposal")

    def test_maximum_shift_and_first_action_budget_are_respected(self):
        row, nxt = handoff(ready_at=3)
        vectors, report = self.proposals(
            row,
            nxt,
            budget=BoundaryBudget(max_shift_steps=1, max_first_actions=1),
        )
        self.assertEqual(vectors, [])
        self.assertEqual(report["actions_examined"], 1)
        vectors, report = self.proposals(
            row,
            nxt,
            budget=BoundaryBudget(max_shift_steps=2, max_first_actions=1),
        )
        self.assertEqual(vectors, [(3,)])
        self.assertEqual(report["actions_examined"], 1)

    def test_terminal_iteration_has_no_next_pick_requirement(self):
        row, _ = handoff(known=True, aperture=0.08, ready_at=99)
        vectors, report = propose_handoffs([row], [None], MILESTONE)
        self.assertEqual(vectors, [(4,)])
        self.assertEqual(report["status"], "terminal")
        self.assertEqual(report["actions_examined"], 0)

    def test_terminal_placement_still_requires_known_released_hand(self):
        row, _ = handoff(known=False, aperture=0.04, ready_at=99)
        vectors, report = propose_handoffs([row], [None], MILESTONE)
        self.assertEqual(vectors, [])

    def test_expired_deadline_returns_no_incomplete_vector(self):
        row, nxt = handoff()
        vectors, report = self.proposals(row, nxt, budget=BoundaryBudget(seconds=1e-9))
        self.assertEqual(vectors, [])
        self.assertEqual(report["status"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
