"""The contact observer follows active MuJoCo constraints, not penetration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from synthesis.cfg.endpoint_contract import EndpointRecorder, contact_state
from synthesis.cfg.test_endpoint_contract import observation, segment


def contact(first, second, *, address=0, distance=0.0002, vertical=False):
    return SimpleNamespace(
        geom1=first,
        geom2=second,
        efc_address=address,
        dist=distance,
        includemargin=0.001,
        frame=(0, 0, 1) if vertical else (1, 0, 0),
    )


def scene(contacts):
    names = [
        "object0",
        "object1",
        "left_gripper_finger_link",
        "right_gripper_finger_link",
        "table0",
    ]
    model = SimpleNamespace(
        ngeom=len(names), geom_name2id=names.index, geom_id2name=names.__getitem__
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(
            model=model, data=SimpleNamespace(contact=contacts, ncon=len(contacts))
        )
    )
    obs = observation([(0, 0, 0.425), (0, 0, 0.475)], empty=False)
    obs[:3] = (0, 0, 0.475)
    return env, obs


class ActiveContactTests(unittest.TestCase):
    def observe(self, contacts, forces):
        env, obs = scene(contacts)
        with patch(
            "synthesis.cfg.endpoint_contract._contact_wrench",
            side_effect=[np.array([f, 0, 0, 0, 0, 0]) for f in forces],
        ):
            return contact_state(env, obs, 2)

    def test_both_fingers_with_active_positive_gap_forces_establish_ownership(self):
        result = self.observe([contact(2, 1), contact(3, 1)], [180, 182])
        self.assertTrue(result["known"])
        self.assertEqual(result["holding"], 1)
        self.assertEqual(result["finger_objects"], [1])

    def test_inactive_near_contacts_do_not_establish_grasp_or_empty_hand(self):
        env, obs = scene([contact(2, 1, address=-1), contact(3, 1, address=-1)])
        with patch("synthesis.cfg.endpoint_contract._contact_wrench") as force:
            result = contact_state(env, obs, 2)
        force.assert_not_called()
        self.assertFalse(result["known"])
        self.assertIsNone(result["empty_evidence"])

    def test_included_constraints_without_positive_force_are_insufficient(self):
        result = self.observe([contact(2, 1), contact(3, 1)], [0, 0])
        self.assertFalse(result["known"])
        self.assertIsNone(result["empty_evidence"])

    def test_repeated_contacts_from_one_finger_do_not_establish_ownership(self):
        result = self.observe([contact(2, 1), contact(2, 1)], [180, 182])
        self.assertFalse(result["known"])

    def test_two_objects_remain_ambiguous(self):
        result = self.observe(
            [contact(2, 1), contact(3, 1), contact(2, 0)], [180, 182, 1]
        )
        self.assertFalse(result["known"])

    def test_support_requires_active_force_and_retains_vertical_geometry(self):
        contacts = [
            contact(1, 0, vertical=True),
            contact(0, 4, vertical=True),
        ]
        result = self.observe(contacts, [1, 2])
        self.assertEqual(set(result["supports"]), {(1, 0), (0, "tbl")})
        self.assertFalse(self.observe(contacts, [0, 0])["supports"])
        contacts[0].frame = (1, 0, 0)
        self.assertNotIn((1, 0), self.observe(contacts, [1, 2])["supports"])

    def test_penetration_without_active_constraint_is_not_support(self):
        result = self.observe(
            [contact(1, 0, address=-1, distance=-0.001, vertical=True)], []
        )
        self.assertFalse(result["supports"])

    def test_unavailable_or_nonfinite_force_remains_unknown_and_rejecting(self):
        for force in ([np.nan, 0, 0, 0, 0, 0], [1, 0]):
            with self.subTest(force=force):
                env, obs = scene([contact(2, 1)])
                recorder = EndpointRecorder(env, segment(obs, obs), obs)
                with patch(
                    "synthesis.cfg.endpoint_contract._contact_wrench",
                    return_value=force,
                ):
                    result = recorder._contact(obs)
                self.assertFalse(result["known"])
                self.assertFalse(result["contacts_available"])
                self.assertIn("contact_observation_unavailable", recorder.reasons)


if __name__ == "__main__":
    unittest.main()
